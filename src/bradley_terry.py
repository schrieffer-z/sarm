########################
# This script is modified from the TRL package https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/reward_modeling.py
# This script is designed for the reward modeling with Mistral model which should be handled carefully because it does not have an official pad token
# If you have any question, feel free to send me an email via wx13@illinois.edu
########################
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# import evaluate
import numpy as np
import json
import torch
import os
import re
import torch.nn as nn
from datasets import load_dataset
from safetensors.torch import save_file, load_file
# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy, is_sagemaker_mp_enabled
from transformers.optimization import get_scheduler
from transformers.trainer import OptimizerNames

# local
from sarm_llama import LlamaBaseline, LlamaSARM
from sarm_gemma2 import Gemma2Baseline, Gemma2SARM
from sae import *

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    # SAE config从SAE模型文件名中解析出sae_latent_size和sae_hidden_state_source_layer
    sae_path: Optional[str] = field(
        default=None,
        metadata={"help": "the sae path to be merged in .safetensors(name of sae_path(e.g. _Latent16384_Layer8_K144) will be used to parse LatentSize and HiddenStateSourceLayer)."}
    )
    sae_use_sequence_level: Optional[bool] = field(
        default=False,
        metadata={"help": "whether or not to use sequence level in sae"}
    )
    sarm_train_mode: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "0: { lm.loss=None, sae.loss=None }"
                "1: { lm.loss=bt, sae.loss=None }"
                "2: { lm.loss=reconstruct + bt & sae.loss=reconstruct + bt }"
                "3: { lm.loss=bt & sae.loss=reconstruct + bt }"
            )
        }
    )
    sarm_rec_lambda: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "loss = bt_loss + lambda_rec_loss"
        }
    )
    sarm_use_topk: Optional[bool] = field(
        default=False,
        metadata={"help": "whether or not to use top k in rm"}
    )
    sarm_base_model: Optional[str] = field(
        default=None,
        metadata={"help": "RM Backbone type(from which arch's hidden states sae is trained). "}
    )
    # Baseline 
    sarm_use_baseline: Optional[bool] = field(
        default=False,
        metadata={"help": "whether or not to use baseine"}
    )
    # Resume Flag
    resume: Optional[bool] = field(
        default=False,
        metadata={"help": "whether or not to resume from certain ckpt"}
    )

    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    # for 8 GPU, the global batch size is 512
    gradient_accumulation_steps: Optional[int] = field(default=64)
    learning_rate: Optional[float] = field(default=2e-6)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=5,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_set_path: Optional[str] = field(
        default="hendrydong/preference_700K",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    eval_set_path: Optional[str] = field(
        default="hendrydong/preference_700K",
        metadata={"help": "The dir of the subset of the eval data to use"},
    )
    output_path: Optional[str] = field(
        default="./models/llama3_rm",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="paged_adamw_32bit",
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)

    save_only_model: Optional[bool] = field(
        default=True,
        metadata={"help": "Only save the model rather than lr_scheduler or optimizer state"},
    )
    save_every_steps: Optional[int] = field(
        default=25,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Eval the model every x steps"},
    )

    report_to: Optional[str] = field(
        default="none",
        metadata={"help": "The lr scheduler"},
    )



parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
script_args.model_name = os.path.normpath(script_args.model_name)


# Load the value-head model and tokenizer.
tokenizer_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = False)

# Adjusted according to the base model
# Need to do this for the models that don't have an official pad token.
if tokenizer.pad_token==None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print(tokenizer.padding_side)
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length



# Get the dataset
train_path = script_args.train_set_path
eval_path = script_args.eval_set_path
output_name = script_args.output_path


def build_dataset(tokenizer, train_path, eval_path):

    def tokenize(sample):
        sample['positive'] = tokenizer.apply_chat_template(sample['chosen'], tokenize=False, add_generation_prompt=False)
        if tokenizer.bos_token!=None:
            sample['positive']=sample['positive'].replace(tokenizer.bos_token, "")
        sample['negative'] = tokenizer.apply_chat_template(sample['rejected'], tokenize=False, add_generation_prompt=False)
        if tokenizer.bos_token!=None:
            sample['negative']=sample['negative'].replace(tokenizer.bos_token, "")

        tokenized_pos = tokenizer(sample['positive'], truncation=True)
        tokenized_neg = tokenizer(sample['negative'], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["assistant_masks_j"] = get_last_assistant_masks(tokenized_pos["input_ids"])
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        sample["assistant_masks_k"] = get_last_assistant_masks(tokenized_neg["input_ids"])
        return sample

    ds = load_dataset('parquet', data_files=train_path)['train'].shuffle(seed=42)
    ds = ds.map(tokenize, num_proc=8)

    eval_dataset = None

    train_dataset = ds
    eval_dataset = ds.select(range(500))
    return train_dataset, eval_dataset

train_dataset, eval_dataset = build_dataset(tokenizer, train_path, eval_path)
print("Training set: ", len(train_dataset), " Eval set: ", len(eval_dataset))


# Define the trainer
training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=script_args.eval_every_steps,
    save_strategy="steps",
    save_steps=script_args.save_every_steps,
    save_only_model=script_args.save_only_model,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.03,
    report_to=script_args.report_to
)


def parse_sae_params(filename):
    """
    从SAE模型文件名中解析出sae_latent_size和sae_hidden_state_source_layer
    """
    pattern = r'_Latent(\d+)_Layer(\d+)_K(\d+)_'
    match = re.search(pattern, filename)
    
    if not match:
        raise ValueError(f"Invalid SAE filename format: {filename}")
    
    ret = {
        "sae_latent_size": int(match.group(1)),
        "sae_hidden_state_source_layer": int(match.group(2)),
        "sae_k": int(match.group(3)),
        "sae_use_sequence_level": script_args.sae_use_sequence_level,
        'sarm_use_topk': script_args.sarm_use_topk,
        'sarm_train_mode': script_args.sarm_train_mode
    }
    if script_args.sarm_use_baseline: 
        ret.pop('sae_k')
        ret.pop('sae_use_sequence_level')
        ret.pop('sarm_use_topk')
    return ret


def merge_safetensor():
    if not os.path.exists(script_args.model_name+'-SARM'):
        import shutil
        shutil.copytree(script_args.model_name, script_args.model_name+'-SARM')

    weight_map = None
    safetensors_name = None
    weight_map_path = os.path.join(script_args.model_name, 'model.safetensors.index.json')
    if os.path.exists(weight_map_path):
        with open(weight_map_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)
    
    weights1=dict()
    for fname in os.listdir(script_args.model_name):
        if '.safetensors' in fname:
            safetensors_name = fname
            weights1 = load_file(os.path.join(script_args.model_name, fname))
            break
    
    weights2 = dict(torch.load(script_args.sae_path, weights_only=True, map_location=torch.device('cpu')))
    x = dict()
    for k, v in weights2.items():
        x.update({
            'sae.' + k : 
            v.cpu().to(torch.bfloat16).contiguous()
        })
    x.update({
        'score.weight': 
        torch.zeros([1,weights2['decoder.weight'].shape[1]]).to(torch.bfloat16)
    })
    weights1.update(x)
    

    target_path = script_args.model_name+"-SARM"
    # 修改model.safetensors.index.json的映射
    if weight_map is not None:
        for param_name in x.keys():
            weight_map["weight_map"].update({
                param_name: safetensors_name
            })
        weight_map["weight_map"].update({
                param_name: safetensors_name
            })
        with open(os.path.join(target_path, 'model.safetensors.index.json'), "w", encoding="ascii") as f:
            json.dump(weight_map, f, indent=2)

    save_file(
        weights1, 
        os.path.join(target_path, safetensors_name),
        metadata={"format": "pt"}
    )
        


if script_args.sarm_use_baseline:
    sae_kwargs = parse_sae_params(script_args.sae_path)
    if script_args.sarm_base_model=='gemma2':
        model = Gemma2Baseline.from_pretrained(
            script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", **sae_kwargs
        )
    elif script_args.sarm_base_model=='llama':
        model = LlamaBaseline.from_pretrained(
            script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", **sae_kwargs
        )
    else:
        raise ValueError(f"Invalid base model type: {script_args.sarm_base_model}")
elif script_args.sae_path is not None:
    sae_kwargs = parse_sae_params(script_args.sae_path)
    if not script_args.resume:
        merge_safetensor()
        loaded_model_path = script_args.model_name + "-SARM"
    else:
        loaded_model_path = script_args.model_name
    if script_args.sarm_base_model=='gemma2':
        model = Gemma2SARM.from_pretrained(
            loaded_model_path, num_labels=1, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", **sae_kwargs
        )
    elif script_args.sarm_base_model=='llama':
        model = LlamaSARM.from_pretrained(
            loaded_model_path, num_labels=1, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", **sae_kwargs
        )
    else:
        raise ValueError(f"Invalid base model type: {script_args.sarm_base_model}")
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    )



model.config.use_cache = not script_args.gradient_checkpointing
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

num_proc = 24  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        merged_features_for_assistant_masks = []

        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
            merged_features_for_assistant_masks.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["assistant_masks_j"],
                }
            )
            merged_features_for_assistant_masks.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["assistant_masks_k"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_for_assistant_masks = self.tokenizer.pad(
            merged_features_for_assistant_masks,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        assert batch["input_ids"].shape == batch["attention_mask"].shape
        assert batch["input_ids"].shape == batch_for_assistant_masks["attention_mask"].shape

        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "assistant_masks": batch_for_assistant_masks["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define the trainer
def compute_metrics(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    # We assume that the first sample is preferred by default in groundtruth
    result['accuracy'] = np.sum(
        pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
    return result

sarm_rec_lambda=script_args.sarm_rec_lambda
sarm_train_mode=script_args.sarm_train_mode
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        output = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], assistant_masks=inputs["assistant_masks"]
        )
        rewards = output['logits']
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        global sarm_train_mode, sarm_rec_lambda
        if sarm_train_mode==2 or sarm_train_mode==3:
            loss += sarm_rec_lambda * output['loss']

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, 
        max_length=script_args.max_length
    ),
)

print('*'*50)
mark = os.path.join(output_name, 'last_checkpoint')
print(mark)
print(os.path.exists(mark))
if not os.path.exists(mark):
    tokenizer.save_pretrained(output_name)
    trainer.train()   
    print("marking training is done by mkdir last checkpoint")
    os.makedirs(mark, exist_ok=True)