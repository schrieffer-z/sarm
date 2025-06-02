import torch
from torch import nn

def get_last_assistant_masks(input_ids):
    i=len(input_ids)-4
    while i >= 0:
        if input_ids[i:i+4] == [128006, 78191, 128007, 271]:
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
    return assistant_masks

def Normalized_MSE_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return (((x_hat - x) ** 2).mean(dim=-1) / (x**2).mean(dim=-1)).mean()

def pre_process(hidden_stats: torch.Tensor, eps: float = 1e-6) -> tuple:
    '''
    :param hidden_stats: Hidden states (shape: [batch, max_length, hidden_size]).
    :param eps: Epsilon value for numerical stability.
    '''
    mean = hidden_stats.mean(dim=-1, keepdim=True)
    std = hidden_stats.std(dim=-1, keepdim=True)
    x = (hidden_stats - mean) / (std + eps)
    return x, mean, std

class TopkSAE(nn.Module):
    '''
    TopK Sparse Autoencoder Implements:
    z = TopK(encoder(x - pre_bias) + latent_bias)
    x_hat = decoder(z) + pre_bias
    '''
    def __init__(
        self, hidden_size: int, latent_size: int, k: int
    ) -> None:
        '''
        :param hidden_size: Dimensionality of the input residual stream activation.
        :param latent_size: Number of latent units.
        :param k: Number of activated latents.
        '''

        # 'sae_pre_bias', 'sae_latent_bias', 'sae_encoder.weight', 'sae_decoder.weight'

        assert k <= latent_size, f'k should be less than or equal to {latent_size}'
        super(TopkSAE, self).__init__()
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)

        self.k = k
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        # "tied" init
        # self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.pre_bias
        return self.encoder(x) + self.latent_bias
    
    def get_latents(self, pre_acts: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(pre_acts, self.k, dim=-1)
        latents = torch.zeros_like(pre_acts)
        latents.scatter_(-1, topk.indices, topk.values)
        return latents

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = self.pre_acts(x)
        latents = self.get_latents(pre_acts)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias
    
    def forward(self, x: torch.Tensor) -> tuple:
        '''
        :param x: Input residual stream activation (shape: [batch_size, max_length, hidden_size]).
        :return:  latents (shape: [batch_size, max_length, latent_size]).
                  x_hat (shape: [batch_size, max_length, hidden_size]).
        '''
        latents = self.encode(x)
        x_hat = self.decode(latents)
        return latents, x_hat



