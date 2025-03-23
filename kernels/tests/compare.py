import torch
import numpy as np
from typing import Optional, Tuple, List, Dict
import time

def compare_tensors(
    a: torch.Tensor, 
    b: torch.Tensor, 
    name_a: str = "tensor_a", 
    name_b: str = "tensor_b",
    abs_threshold: float = 1e-5,
    rel_threshold: float = 1e-5,
    max_print: int = 10,
    print_stats_only: bool = False,
    verbose: bool = False,
    return_stats: bool = False
) -> bool:
    """
    比较两个tensor并高效打印差异。
    
    参数:
        a, b: 要比较的两个tensor
        name_a, name_b: tensor的标识名称
        abs_threshold: 绝对误差阈值
        rel_threshold: 相对误差阈值
        max_print: 最大打印差异数量
        print_stats_only: 是否只打印统计信息
        verbose: 是否打印详细信息
        return_stats: 是否返回统计信息字典
        
    返回:
        bool: 如果tensor完全相等（在阈值范围内），则返回True
        dict: 如果return_stats=True，还会返回统计信息字典
    """
    # 检查输入有效性
    if a.shape != b.shape:
        raise ValueError(f"张量形状不匹配: {a.shape} vs {b.shape}")
    
    # 确保两个tensor在同一设备上
    if a.device != b.device:
        print(f"Warning: 张量在不同设备上，将 {name_b} 移到 {name_a} 所在设备 {a.device}")
        b = b.to(a.device)
    
    # 如果在GPU上，先计算差异再传回CPU进行分析
    start_time = time.time()
    is_cuda = a.device.type == 'cuda'
    
    # 计算绝对差异
    with torch.no_grad():
        abs_diff = torch.abs(a - b)
        
        # 计算相对差异，避免除零
        # 使用最大值作为分母来计算相对误差
        denominator = torch.maximum(torch.abs(a), torch.abs(b))
        # 防止除零：当分母接近0时，如果分子也接近0，相对误差为0；否则相对误差为1
        too_small_mask = denominator < 1e-10
        rel_diff = torch.zeros_like(abs_diff)
        rel_diff[~too_small_mask] = abs_diff[~too_small_mask] / denominator[~too_small_mask]
        rel_diff[too_small_mask & (abs_diff > 1e-10)] = 1.0
        
        # 找出差异超过阈值的元素
        diff_mask = (abs_diff > abs_threshold) | (rel_diff > rel_threshold)
        num_diff = torch.sum(diff_mask).item()
        
        # 计算统计信息
        max_abs_diff = torch.max(abs_diff).item()
        max_rel_diff = torch.max(rel_diff).item()
        avg_abs_diff = torch.mean(abs_diff).item()
        avg_rel_diff = torch.mean(rel_diff).item()
        
        # 如果有差异并且需要打印细节，找出最大的差异
        if num_diff > 0 and not print_stats_only:
            # 找到差异最大的元素
            flat_indices = torch.nonzero(diff_mask.flatten(), as_tuple=False).flatten()
            
            # 限制打印数量
            if len(flat_indices) > max_print:
                # 按绝对差异大小排序
                flat_abs_diff = abs_diff.flatten()
                sorted_indices = torch.argsort(flat_abs_diff[flat_indices], descending=True)
                flat_indices = flat_indices[sorted_indices[:max_print]]
            
            # 获取多维索引
            multi_indices = []
            for idx in flat_indices.cpu().numpy():
                multi_idx = np.unravel_index(idx, a.shape)
                multi_indices.append(multi_idx)
                
            # 收集差异信息
            diff_info = []
            for i, multi_idx in enumerate(multi_indices):
                flat_idx = flat_indices[i].item()
                a_val = a.flatten()[flat_idx].item()
                b_val = b.flatten()[flat_idx].item()
                abs_diff_val = abs_diff.flatten()[flat_idx].item()
                rel_diff_val = rel_diff.flatten()[flat_idx].item()
                
                diff_info.append({
                    "flat_idx": flat_idx,
                    "multi_idx": multi_idx,
                    "a_val": a_val,
                    "b_val": b_val,
                    "abs_diff": abs_diff_val,
                    "rel_diff": rel_diff_val
                })
        else:
            diff_info = []
    
    # 计算分析时间
    analysis_time = time.time() - start_time
    
    # 打印结果
    print("\n" + "="*50)
    print(f"Tensor Comparison: {name_a} vs {name_b}")
    print(f"Shape: {a.shape}, Total Elements: {a.numel()}")
    print(f"Device: {a.device}, Dtype: {a.dtype}")
    print(f"Analysis Time: {analysis_time*1000:.2f} ms")
    print("\nStatistics:")
    print(f"  Max Absolute Diff: {max_abs_diff:.6e}")
    print(f"  Max Relative Diff: {max_rel_diff:.6e}")
    print(f"  Avg Absolute Diff: {avg_abs_diff:.6e}")
    print(f"  Avg Relative Diff: {avg_rel_diff:.6e}")
    print(f"  Elements Differing: {num_diff} / {a.numel()} ({100*num_diff/a.numel():.4f}%)")
    
    # 打印详细差异
    if diff_info and not print_stats_only:
        print("\nDiffering Elements" + (f" (showing top {len(diff_info)} of {num_diff})" if num_diff > max_print else "") + ":")

        # 表头
        if verbose:
            print(f"{'Index':<12} {'Coordinates':<25} {name_a:<15} {name_b:<15} {'Abs Diff':<15} {'Rel Diff':<15}")
        else:
            print(f"{'Index':<12} {name_a:<15} {name_b:<15} {'Abs Diff':<15} {'Rel Diff':<15}")
        print("-" * 70)
        
        # 打印每个差异
        for info in diff_info:
            if verbose:
                coord_str = str(info["multi_idx"]).replace("(", "[").replace(")", "]")
                print(f"{info['flat_idx']:<12} {coord_str:<25} {info['a_val']:<15.6e} {info['b_val']:<15.6e} "
                      f"{info['abs_diff']:<15.6e} {info['rel_diff']:<15.6e}")
            else:
                print(f"{info['flat_idx']:<12} {info['a_val']:<15.6e} {info['b_val']:<15.6e} "
                      f"{info['abs_diff']:<15.6e} {info['rel_diff']:<15.6e}")
    
    # 结论
    print("\nComparison", "PASSED" if num_diff == 0 else "FAILED")
    print("="*50)
    
    # 准备返回
    stats = {
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "avg_abs_diff": avg_abs_diff,
        "avg_rel_diff": avg_rel_diff,
        "num_diff": num_diff,
        "total_elements": a.numel(),
        "diff_percentage": 100 * num_diff / a.numel(),
        "analysis_time_ms": analysis_time * 1000
    }
    
    if return_stats:
        return num_diff == 0, stats
    return num_diff == 0


def test_rmsnorm(batch_size=2, seq_len=8, hidden_size=128, head_dim=64):
    """测试RMSNorm实现，比较自定义实现与PyTorch原生实现"""
    # 导入自定义RMSNorm模块
    try:
        from sparsity.vefuser.optimization.kernels import _kernels
        has_custom_kernel = True
    except ImportError:
        print("Custom kernel module not found, using PyTorch implementation for both")
        has_custom_kernel = False
    
    # 生成测试数据
    x = torch.randn(batch_size * seq_len, hidden_size, 
                    dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    gamma = torch.ones(hidden_size, dtype=x.dtype, device=x.device)
    
    # PyTorch原生实现
    def pytorch_rmsnorm(x, gamma, epsilon=1e-5):
        # 计算均方根
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + epsilon)
        # 应用缩放参数
        return x_normed * gamma
    
    # 运行PyTorch实现
    x_ref = x.clone()
    x_torch = pytorch_rmsnorm(x_ref, gamma)
    
    # 运行自定义实现
    x_custom = x.clone()
    if has_custom_kernel:
        epsilon = 1e-5
        _kernels.custom_rms_norm_forward(x_custom, gamma, head_dim, epsilon)
    else:
        # 如果没有自定义核心，使用PyTorch实现
        x_custom = pytorch_rmsnorm(x_custom, gamma)
    
    # 比较结果
    return compare_tensors(
        x_custom, x_torch, 
        name_a="custom_rmsnorm", 
        name_b="torch_rmsnorm",
        abs_threshold=1e-5,
        rel_threshold=1e-5,
        verbose=True,
        return_stats=True
    )


if __name__ == "__main__":
    # 简单的测试案例
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([1.0, 2.0001, 2.9999, 4.1])
    compare_tensors(a, b, abs_threshold=1e-3, verbose=True)
    
    # 如果有GPU，测试RMSNorm
    if torch.cuda.is_available():
        try:
            is_equal, stats = test_rmsnorm()
            print(f"\nRMSNorm Test {'Passed' if is_equal else 'Failed'}")
            print(f"Stats: {stats}")
        except Exception as e:
            print(f"RMSNorm test failed with error: {e}")