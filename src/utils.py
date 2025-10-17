import torch

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

def Masked_Normalized_MSE_loss(x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(torch.bfloat16)
    loss = ((x_hat - x) ** 2).mean(dim=-1) / (x**2).mean(dim=-1)
    assert loss.shape==mask.shape
    seq_loss = (mask * loss).sum(-1) / (mask.sum(-1))
    return seq_loss.mean()

def pre_process(hidden_stats: torch.Tensor, eps: float = 1e-6) -> tuple:
    '''
    :param hidden_stats: Hidden states (shape: [batch, max_length, hidden_size]).
    :param eps: Epsilon value for numerical stability.
    '''
    mean = hidden_stats.mean(dim=-1, keepdim=True)
    std = hidden_stats.std(dim=-1, keepdim=True)
    x = (hidden_stats - mean) / (std + eps)
    return x, mean, std