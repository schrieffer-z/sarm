import re
import os
import json
import heapq
import torch
import torch.nn as nn
import argparse
import datasets
from tqdm import tqdm
from types import SimpleNamespace
from collections import defaultdict
from transformers import AutoTokenizer
from typing import List, Optional, Union, Tuple
from transformers import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import logging
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, 
    LlamaRMSNorm, 
    LlamaRotaryEmbedding,
    LlamaPreTrainedModel
)

# Local
from sarm_llama import LlamaSARM
from sae import TopkSAE, pre_process, Normalized_MSE_loss, Masked_Normalized_MSE_loss

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Apply a pretrained Sparse Auto‑Encoder (SAE) to a dataset and extract high-activation contexts."
    )

    # === Core model & SAE params ===
    p.add_argument("--model_path", required=True, type=str, help="Path to SARM directory")
    p.add_argument("--tokenizer_path", required=True, type=str, help="Path to SARM's tokenizer")

    # === Dataset params ===
    p.add_argument("--dataset_name", required=True, type=str, help="Short name identifying the dataset kind (corpus / preference …)")
    p.add_argument("--data_path",   required=True, type=str, help="Path to the dataset root / file")
    p.add_argument("--split_index", required=True, type=int, help="Index of dataset split processed by this job (starting at 0)")
    p.add_argument("--split_num",   required=True, type=int, help="Total number of splits to shard the dataset")

    # === Runtime / extraction params ===
    p.add_argument("--batch_size",   default=4,   type=int,   help="Micro‑batch size (default: 4)")
    p.add_argument("--max_length",   default=1024, type=int, help="Maximum sequence length to feed LM")
    p.add_argument("--apply_threshold", default=5.0, type=float,
                   help="Activation threshold for a latent to be considered fired (default: 5.0)")
    p.add_argument("--device", default="cuda:0", type=str, help="Torch device, e.g. cuda:0 / cpu (default: cuda:0)")

    # === Output ===
    p.add_argument("--output_path", default="../contexts", type=str,
                   help="Directory to save extracted context JSON (default: ../contexts)")

    return p

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

class Preference(torch.utils.data.Dataset):
    def __init__(self, 
                 pref_data_path : str,
                 tokenizer: AutoTokenizer, 
                 max_length: int,
                 split_index: int, 
                 split_num: int,
        ):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        self.max_length = max_length
        self.pref_data_path = pref_data_path
        self.split_index = split_index
        self.split_num = split_num
        self.data = self.load_data()

    def load_data(self):
        ds = datasets.load_dataset('parquet', data_dir=self.pref_data_path)['train'].shuffle(seed=42)
        print(f'Spliting dataset of {len(ds)} pairs into {self.split_num} part.')
        split_pos = torch.linspace(0,len(ds),self.split_num+1)
        st, ed = int(split_pos[self.split_index]), int(split_pos[self.split_index+1])
        ds = datasets.Dataset.from_dict(ds[st:ed])
        print(f'This process is using index:{self.split_index}(dataset[{st}:{ed}]).')
        tokenizer = self.tokenizer
        def tokenize(sample):
            formated_text = tokenizer.apply_chat_template(sample['text'], tokenize=False, add_generation_prompt=False)
            if tokenizer.bos_token!=None:
                formated_text=formated_text.replace(tokenizer.bos_token, "")
            return tokenizer(
                formated_text, 
                truncation=True,
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length'
            )
        return datasets.Dataset.from_dict({'text':ds['chosen']+ds['rejected']}).map(tokenize, num_proc=8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_dataloader(
    dataset_name: str,
    folder_path: str, 
    tokenizer: AutoTokenizer, 
    batch_size: int, 
    max_length: int,
    keyword: str = 'text',
    split_index: int=0,
    split_num: int=1,
) -> torch.utils.data.DataLoader:
    if 'preference' in dataset_name.lower():
        dataset = Preference(folder_path, tokenizer, max_length, split_index, split_num)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


    def collate_fn(batch):
        input_ids = torch.stack([
                item[0] if dataset_name=='corpus'  
                else torch.tensor(item['input_ids']).squeeze()
                for item in batch
            ]
        )
        attention_mask = torch.stack([
                item[1] if dataset_name=='corpus'  
                else torch.tensor(item['attention_mask']).squeeze()
                for item in batch
            ]
        )
        return input_ids, attention_mask


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)
    return dataloader


def save_latent_dict(latent_context_map, output_path, threshold, max_length, max_per_token, lines):
    filtered_latent_context = {}
    for latent_dim, token_dict in latent_context_map.items():
        # # Skip latent token categories exceeding 32
        if len(token_dict) > 100:
            continue    
        total_contexts = sum(len(contexts) for contexts in token_dict.values())
        if total_contexts > 4:
            sorted_token_dict = {}
            for t_class, heap in token_dict.items():
                contexts_list = list(heap)
                contexts_list.sort(key=lambda x: x[0], reverse=True)
                sorted_token_dict[t_class] = [
                    {'context': ctx, 'activation': act} for act, ctx in contexts_list
                ]
            filtered_latent_context[latent_dim] = dict(sorted(sorted_token_dict.items()))

    total_latents = len(filtered_latent_context)
    sorted_latent_context = dict(sorted(filtered_latent_context.items()))

    output_data = {
        'total_latents': total_latents,
        'threshold': threshold,
        'max_length': max_length,
        'max_per_token': max_per_token,
        'lines': lines,
        'latent_context_map': sorted_latent_context,
    }
    save_json(output_data, output_path)
    return


def get_last_assistant_masks(input_ids):
    i=len(input_ids)-4
    pos = 0
    while i >= 0:
        if input_ids[i:i+4].tolist() == [128006, 78191, 128007, 271]:
            pos = i + 4
            break
        i -= 1
    
    assistant_masks = []
    for i in range(len(input_ids)):
        if i < pos:
            assistant_masks.append(0)
        else:
            assistant_masks.append(1)

    assert input_ids[-1]==128009
    return pos, assistant_masks


def parse_sarm_param(filename:str):
    latent = int(re.search(r"_Latent(\d+)_", filename).group(1))
    layer = int(re.search(r"_Layer(\d+)_", filename).group(1))
    k_value = int(re.search(r"_K(\d+)_", filename).group(1))

    assert "SARM" in filename or "Baseline" in filename 
    assert "TopK" in filename or "woTopK" in filename
    sarm_use_baseline = "Baseline" in filename
    sarm_use_topk = "TopK" in filename

    print(f"  SARM Parameters:  ".center(60, '='))
    print(f"Latent: {latent}")                
    print(f"Layer: {layer}")                  
    print(f"K: {k_value}")                    
    print(f"sarm_use_topk: {sarm_use_topk}")  

    ret = {
        "sae_latent_size": latent,
        "sae_hidden_state_source_layer": layer,
        "sae_k": k_value,
        'sarm_use_topk': sarm_use_topk,
    }
    if sarm_use_baseline: 
        ret.pop('sae_k')
        ret.pop('sarm_use_topk')
    return ret


class SARM4Latent(LlamaSARM):    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # Shuyi (aggregate latent)
        assistant_masks: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]


        h, _, _ = pre_process(hidden_states)
        sae_features = self.sae.pre_acts(h)
        latents_output = self.sae.get_latents(sae_features)
        if self.sarm_use_topk:
            sae_features = self.sae.get_latents(sae_features)
        

        logits = self.score(sae_features)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
        # ensure the last_token is <|eot_id|>
        assert ((input_ids[torch.arange(batch_size, device=logits.device), sequence_lengths]!=torch.ones(batch_size, device=logits.device)*128009).sum() == 0).item()
        
        # joint training
        rec_loss = None
        if self.sarm_train_mode==2:
            if not self.sarm_use_topk:
                sae_features_t = self.sae.get_latents(sae_features)
            h_hat = self.sae.decode(sae_features_t)
            rec_loss = Masked_Normalized_MSE_loss(h, h_hat, assistant_masks)
        elif self.sarm_train_mode==3 and not self.sae_use_sequence_level:
            h_d = h.detach()
            _, h_hat = self.sae(h_d)
            rec_loss = Masked_Normalized_MSE_loss(h_d, h_hat, assistant_masks)        
        elif self.sarm_train_mode==3 and self.sae_use_sequence_level:
            h_d = h.detach()
            sequence_lengths_t = sequence_lengths.view(-1,1,1)
            last_token_mask = torch.zeros([h_d.shape[0] ,1 ,h_d.shape[1]], device=h_d.device)
            last_token_mask.scatter_(-1, sequence_lengths_t, torch.ones_like(sequence_lengths_t, dtype=last_token_mask.dtype))
            
            # h_d -> (bs, seq_len, d), last_token_mask -> (bs, 1, seq_len)
            h_d = torch.matmul(last_token_mask.to(h_d.dtype), h_d) 
            
            _, h_hat = self.sae(h_d)
            rec_loss = Normalized_MSE_loss(h_d, h_hat)       


        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]


        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)
        if rec_loss is not None:
            loss = rec_loss

        return (latents_output, pooled_logits)

class Applier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        sarm_params = parse_sarm_param(cfg.model_path)
        tokenizer_path = cfg.model_path[:cfg.model_path.rfind('/')]
        self.sarm_title = tokenizer_path[tokenizer_path.rfind('/')+1:]
        if self.cfg.tokenizer_path is None:
            tokenizer_path = self.cfg.tokenizer_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.sarm = SARM4Latent.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            **sarm_params
        ).to(self.device)
        
        
        self.sarm.eval()
                
    @torch.no_grad()
    def get_context(
        self, 
        threshold: float = 3.0, 
        max_length: int = 1024, 
        max_per_token: int = 128, 
        lines: int = 4,  
        output_path=None
    ):
        self.context_title = f'{self.sarm_title}_{self.cfg.dataset_name}_threshold{threshold}_split-{self.cfg.split_index+1}-{self.cfg.split_num}'
        self.context_base_dir = self.cfg.output_path if self.cfg.output_path is not None else '../contexts/'
        
        self.dataloader = create_dataloader(self.cfg.dataset_name, self.cfg.data_path, self.tokenizer, self.cfg.batch_size, max_length, split_index=self.cfg.split_index, split_num=self.cfg.split_num)
        
        sentence_enders = [
            '?',    '.',   ';',    '!',
            ' ?',   ' .',  ' ;',   ' !',
            '<|eot_id|>' 
        ]
        sentence_enders_tokens = self.tokenizer.batch_encode_plus(sentence_enders, return_tensors='pt')['input_ids'][:,1].tolist()

        global_step_idx=0
        ckmap = defaultdict(lambda: defaultdict(list))
        latent_context_map = defaultdict(lambda: defaultdict(list))
        for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="SAE applying"):
            input_ids, attention_mask = batch[0].to(self.device), batch[1].to(self.device)
            with torch.no_grad():
                latents, _ = self.sarm(input_ids=input_ids, attention_mask=attention_mask)
            
            batch_size, seq_len, _ = latents.shape
            positions = (latents > threshold)

            for i in range(batch_size):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                st_pos, assistant_mask = get_last_assistant_masks(input_ids[i])

                assert tokens[-1] == '<|eot_id|>'
                assert st_pos == 0 or tokens[st_pos-4 : st_pos] == ['<|start_header_id|>', 'assistant', '<|end_header_id|>', 'ĊĊ']

                prev_pos = [st_pos]
                latent_indices = torch.nonzero(positions[i], as_tuple=False)
                for pos in range(st_pos, input_ids.shape[-1]):
                    if input_ids[i][pos] not in sentence_enders_tokens:
                        continue
                    
                    mask = (latent_indices[:, 0] == pos)
                    latents_at_pos = latent_indices[mask]
                    for activation in latents_at_pos:
                        _pos, latent_dim = activation.tolist()
                        assert _pos==pos

                        activation_value = latents[i, pos, latent_dim].item()
                        context_start_idx = len(prev_pos)-1
                        while(pos + 1 - prev_pos[context_start_idx]<10 and context_start_idx>0):
                            context_start_idx -= 1
                        raw = self.tokenizer.convert_tokens_to_ids(tokens[prev_pos[context_start_idx]:pos+1])
                        context_text = self.tokenizer.decode(raw).strip()
                        
                        assert len(ckmap)==len(latent_context_map)
                        heap = latent_context_map[latent_dim][tokens[pos]]
                        cmap = ckmap[latent_dim][tokens[pos]]
                        if context_text in cmap:
                            continue
                        heapq.heappush(cmap, context_text)
                        heapq.heappush(heap, (activation_value, context_text))
                        if len(heap) > max_per_token:
                            heapq.heappop(heap)
                        
                    prev_pos.append(pos + 1)
            save_ats = torch.linspace(0,len(self.dataloader),11).to(torch.int32)[1:]
            if (global_step_idx in save_ats):
                base_t = os.path.join(self.context_base_dir, 'tmp')
                title_t = self.context_title + f'@step{global_step_idx}.json'
                
                os.makedirs(base_t, exist_ok=True)
                output_path_tmp = os.path.join(base_t, title_t)
                save_latent_dict(latent_context_map, output_path_tmp, threshold, max_length, max_per_token, lines)
            global_step_idx += 1


        filtered_latent_context = {}
        for latent_dim, token_dict in latent_context_map.items():
            # # Skip latent token categories exceeding 32
            if len(token_dict) > 100:
                continue    
            total_contexts = sum(len(contexts) for contexts in token_dict.values())
            if total_contexts > lines:
                sorted_token_dict = {}
                for t_class, heap in token_dict.items():
                    contexts_list = list(heap)
                    contexts_list.sort(key=lambda x: x[0], reverse=True)
                    sorted_token_dict[t_class] = [
                        {'context': ctx, 'activation': act} for act, ctx in contexts_list
                    ]
                filtered_latent_context[latent_dim] = dict(sorted(sorted_token_dict.items()))

        total_latents = len(filtered_latent_context)
        sorted_latent_context = dict(sorted(filtered_latent_context.items()))

        output_data = {
            'total_latents': total_latents,
            'threshold': threshold,
            'max_length': max_length,
            'max_per_token': max_per_token,
            'lines': lines,
            'latent_context_map': sorted_latent_context,
        }

        save_json(output_data, os.path.join(self.context_base_dir, self.context_title+'.json'))



def main():
    args = build_arg_parser().parse_args()

    # Normalise paths and build cfg object expected by Applier classes -----------------
    cfg_dict = vars(args)

    # utils.Applier / SequenceApplier expect additional attributes; we fill sane defaults
    cfg_dict.setdefault("use_wandb", 0)
    cfg_dict.setdefault("pipe_project", [None, None, None])
    cfg_dict.setdefault("context_path", None)

    # Make sure requested paths are normalised
    for k in ["model_path", "data_path", "output_path"]:
        cfg_dict[k] = os.path.normpath(cfg_dict[k])

    cfg = SimpleNamespace(**cfg_dict)

    # -------------------------------------------------------------------------------
    applier = Applier(cfg)

    total_latents, save_path = applier.get_context(
        threshold=cfg.apply_threshold,
        max_length=cfg.max_length
    )

    print(f"[✓] Extracted contexts for {total_latents} latents & Saved at: {save_path}")


if __name__ == "__main__":
    main()
