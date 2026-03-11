import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile(dynamic=True)
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.clamp(min=1e-9).unsqueeze(dim=1))
        return logits.sub_(torch.empty_like(logits).exponential_(1).log_()).argmax(dim=-1)
