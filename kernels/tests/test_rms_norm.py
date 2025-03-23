import sys
sys.path.append('kernels/build')
import _kernels

import torch
from itertools import product
from typing import List


def bench_rms_norm(
    input,
    gemma,
    head_dim,
    ln_func,
    iter_warmup: int = 10,
    iter_total: int = 1000,
) -> List:
    for _ in range(iter_warmup):
        ln_func(input, gemma, head_dim)

    local_time_list = []
    result = []
    for _ in range(iter_total):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = ln_func(input, gemma, head_dim)
        end.record()
        torch.cuda.synchronize()
        local_time_list.append(start.elapsed_time(end))
        result.append(output)
    return sum(local_time_list) / len(local_time_list), result

def ref_torch_impl(
    input,
    gemma,
    head_dim,
) -> torch.Tensor:
    return torch.nn.functional.rms_norm(input, [input.size(-1)], gemma, 1e-6)

def ref_custom_impl(
    input,
    gemma,
    head_dim,
) -> torch.Tensor:
    input_clone = input.clone()
    _kernels.custom_rms_norm_forward(input_clone, gemma, head_dim, 1e-6)
    return input_clone

num_total_loading = 16380
head_dim = 128
num_heads = 40
parameters= (num_total_loading, head_dim * num_heads)
bsz, dim = parameters
input = torch.randn(bsz, head_dim, dtype=torch.float32).cuda()
gemma = torch.randn(head_dim, dtype=torch.float32).cuda()


torch_time, result_torch = bench_rms_norm(input, gemma, head_dim, ref_torch_impl)
custom_time, result_custom = bench_rms_norm(input, gemma, head_dim, ref_custom_impl)

for i in range(0, 1):
    print(f"num {i} torch.allclose(torch, custom): {(result_torch[i], result_custom[i])}")

print(f"Test args in [bsz,dim]: {parameters}")
print(f"Diffusers's Time: {torch_time} ms")
print(f"Customized's Time: {custom_time} ms")