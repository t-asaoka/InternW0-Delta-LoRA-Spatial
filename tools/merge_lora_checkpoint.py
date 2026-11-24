import argparse
from pathlib import Path
import torch

def merge(payload):
    state = payload['mot']
    prefixes = [k.removesuffix('.lora_A') for k in state if k.endswith('.lora_A')]
    if not prefixes:
        raise ValueError('No LoRA weights found')
    for p in prefixes:
        a,b,w,alpha = (state.pop(p+s) for s in ('.lora_A','.lora_B','.base.weight','.lora_alpha'))
        if p+'.weight' in state or w.dtype not in (torch.float32,torch.float16,torch.bfloat16):
            raise ValueError(f'Unexpected base weight: {p}')
        result = (w.float() + (b.float() @ a.float()) * (float(alpha)/a.shape[0])).to(w.dtype)
        if not torch.isfinite(result).all():
            raise ValueError(f'Nonfinite weight: {p}')
        state[p+'.weight'] = result
        if p+'.base.bias' in state:
            state[p+'.bias'] = state.pop(p+'.base.bias')
    if any('.lora_' in k or '.base.' in k for k in state):
        raise ValueError('Unmerged keys remain')
    print(f'Merged {len(prefixes)} layers')
    return payload

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src',type=Path); parser.add_argument('dst',type=Path)
    args = parser.parse_args()
    if args.dst.exists():
        raise FileExistsError(args.dst)
    payload = torch.load(args.src,map_location='cpu',weights_only=True,mmap=True)
    args.dst.parent.mkdir(parents=True,exist_ok=True)
    torch.save(merge(payload),args.dst)
