"""Optional LoRA for the official WAM trainer (no PEFT dependency)."""
import os
import math
import torch
from torch import nn
from torch.nn import functional as F

class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.register_buffer('lora_alpha', torch.tensor(alpha, device=base.weight.device, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        y = self.base(x)
        z = F.linear(F.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return y + (z * (self.lora_alpha / self.lora_A.shape[0])).to(y.dtype)

def enabled():
    return os.getenv('WAM_LORA_ENABLED', '0') == '1'

def has_lora(model):
    return any(isinstance(m, LoRALinear) for m in model.modules())

def set_lora_trainable(model):
    model.eval()
    model.mot.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)

def inject_lora(model):
    if has_lora(model):
        raise RuntimeError('LoRA already installed')
    scope = os.getenv('WAM_LORA_SCOPE', 'action')
    rank = int(os.getenv('WAM_LORA_RANK', '16'))
    alpha = float(os.getenv('WAM_LORA_ALPHA', '32'))
    dropout = float(os.getenv('WAM_LORA_DROPOUT', '0'))
    if scope not in ('action', 'video', 'both') or rank < 1 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError('Invalid LoRA configuration')
    targets = {f'{a}.{b}' for a in ('self_attn','cross_attn') for b in ('q','k','v','o')}
    if os.getenv('WAM_LORA_FFN', '0') == '1':
        targets.update(('ffn.0','ffn.2'))
    for name in ('action','video') if scope == 'both' else (scope,):
        expert = getattr(model, name + '_expert')
        matches = [(n,m) for n,m in expert.named_modules() if isinstance(m,nn.Linear) and n.startswith('blocks.') and any(n.endswith('.'+t) for t in targets)]
        if not matches:
            raise RuntimeError(f'No LoRA targets in {name}')
        for n,m in matches:
            parent, child = n.rsplit('.',1)
            setattr(expert.get_submodule(parent), child, LoRALinear(m,rank,alpha,dropout))
        print(f'[LoRA] {name}: {len(matches)} layers, rank={rank}, alpha={alpha}', flush=True)
    set_lora_trainable(model)
    params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert params and all(n.endswith(('.lora_A','.lora_B')) for n,p in params)
    print(f'[LoRA] trainable={sum(p.numel() for n,p in params):,} / {sum(p.numel() for p in model.parameters()):,}', flush=True)
