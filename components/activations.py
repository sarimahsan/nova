import torch
import torch.nn as nn
import torch.nn.functional as F

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.square(F.relu(x))

class SiLU(nn.Module):
    def forward(self, x):
        return F.silu(x)
