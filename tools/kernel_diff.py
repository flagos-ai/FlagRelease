#!/usr/bin/env python3
"""
OpDiff2: Compare CUDA vs FlagGems kernel performance from nsys profiling CSV files.

Pipeline:
1. Parse all kernels from both CUDA and FlagGems CSV files
2. Map each kernel to an operator name using priority-based rules
3. Aggregate by operator, tracking original kernel names
4. Compare one-to-one: if kernel names differ -> SLOWER/FASTER, else -> NOT_REPLACED

Usage:
    python kernel_diff.py --cuda cuda.csv --flaggems flaggems.csv [--output report.md]
"""

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional


@dataclass
class KernelInfo:
    """Information about a kernel from profiling data."""
    name: str
    total_time_ns: int
    instances: int
    avg_ns: float
    percentage: float


@dataclass
class OperatorStats:
    """Aggregated statistics for an operator."""
    name: str
    total_time_ns: int = 0
    instances: int = 0
    kernel_names: Set[str] = field(default_factory=set)

    @property
    def total_time_s(self) -> float:
        return self.total_time_ns / 1e9

    @property
    def total_time_ms(self) -> float:
        return self.total_time_ns / 1e6


class KernelMapper:
    """Maps kernel names to operator names using pattern matching rules."""

    # Pattern rules: (regex_pattern, operator_name, priority)
    # Higher priority rules are checked first
    RULES = [
        # ============== FlagGems Specific Kernels (priority 100) ==============
        # Matmul
        (r'^mm_kernel_general_host_tma$', 'mm', 100),
        (r'^mm_kernel_general$', 'mm', 100),
        (r'^bmm_kernel$', 'bmm', 100),

        # Zeros and Fill
        (r'^zeros_kernel$', 'zeros', 100),
        (r'^ones_kernel$', 'ones', 100),
        (r'^fill_scalar_func_kernel', 'fill', 100),
        (r'^full_func_scalar_kernel', 'full', 100),

        # Elementwise ops
        (r'^softmax_kernel_inner$', 'softmax', 100),
        (r'^argmax_kernel_inner$', 'argmax', 100),
        (r'^true_div_func_kernel', 'div', 100),
        (r'^true_div_func_tensor_scalar', 'div', 100),
        (r'^fused_exponential_kernel', 'exponential', 100),
        (r'^sub_func_tensor_scalar_kernel', 'sub', 100),
        (r'^sub_func_scalar_tensor_kernel', 'sub', 100),
        (r'^sub_func_kernel', 'sub', 100),
        (r'^mul_func_kernel', 'mul', 100),
        (r'^mul_func_scalar_kernel', 'mul', 100),
        (r'^add_func_kernel', 'add', 100),
        (r'^pow_func_scalar_tensor_kernel', 'pow', 100),
        (r'^reciprocal_func_kernel', 'reciprocal', 100),
        (r'^sin_func_kernel', 'sin', 100),
        (r'^cos_func_kernel', 'cos', 100),
        (r'^exp_func_kernel', 'exp', 100),
        (r'^log_func_kernel', 'log', 100),
        (r'^sqrt_func_kernel', 'sqrt', 100),
        (r'^rsqrt_func_kernel', 'rsqrt', 100),
        (r'^abs_func_kernel', 'abs', 100),
        (r'^neg_func_kernel', 'neg', 100),
        (r'^relu_func_kernel', 'relu', 100),
        (r'^gelu_func_kernel', 'gelu', 100),
        (r'^silu_func_kernel', 'silu', 100),
        (r'^sigmoid_func_kernel', 'sigmoid', 100),
        (r'^tanh_func_kernel', 'tanh', 100),

        # Comparison ops
        (r'^le_func_kernel', 'le', 100),
        (r'^lt_func_kernel', 'lt', 100),
        (r'^lt_func_scalar_kernel', 'lt', 100),
        (r'^ge_func_kernel', 'ge', 100),
        (r'^gt_func_kernel', 'gt', 100),
        (r'^eq_func_kernel', 'eq', 100),
        (r'^ne_func_kernel', 'ne', 100),
        (r'^where_inner_kernel', 'where', 100),
        (r'^masked_fill_kernel', 'masked_fill', 100),

        # Copy and memory ops
        (r'^_copy_kernel_kernel', 'copy', 100),
        (r'^_to_copy_func_kernel', 'to_copy', 100),
        (r'^cat_copy_func_kernel', 'cat', 100),
        (r'^_index_jit_function$', 'index', 100),
        (r'^_scatter_jit_function$', 'scatter', 100),
        (r'^_gather_flaggems_jit_function$', 'gather', 100),
        (r'^arange_func$', 'arange', 100),
        (r'^rand_kernel$', 'rand', 100),

        # Reduction ops
        (r'^reduce_then_scan', 'scan', 100),
        (r'^compute_global_hist_kernel', 'histogram', 100),
        (r'^sweep$', 'sweep', 100),

        # ============== vLLM Specific Kernels (priority 95) ==============
        (r'vllm::topKPerRowDecode', 'topk_decode', 95),
        (r'vllm::topKPerRowPrefill', 'topk_prefill', 95),
        (r'vllm::topk_kernel', 'topk', 95),
        (r'vllm::cross_device_reduce', 'cross_device_reduce', 95),
        (r'vllm::moe::grouped_topk', 'moe_topk', 95),
        (r'vllm::moe::moe_align_block', 'moe_align', 95),
        (r'vllm::moe::moe_sum', 'moe_sum', 95),
        (r'vllm::moe::count_and_sort', 'moe_sort', 95),
        (r'vllm::act_and_mul_kernel', 'act_and_mul', 95),
        (r'vllm::concat_and_cache', 'concat_cache', 95),
        (r'vllm::indexer_k_quant', 'kv_quant', 95),
        (r'vllm::cp_gather_indexer', 'cp_gather', 95),
        (r'fused_moe_kernel', 'fused_moe', 95),
        (r'sparse_attn_fwd_kernel', 'sparse_attention', 95),
        (r'flash.*attention', 'flash_attention', 95),
        (r'flashmla', 'flash_mla', 95),

        # DeepGEMM
        (r'deep_gemm::sm90_fp8', 'deepgemm_fp8', 95),
        (r'deep_gemm::smxx_paged_mqa', 'deepgemm_mqa', 95),

        # ============== NCCL Kernels (priority 95) ==============
        (r'ncclDevKernel_AllGather', 'nccl_allgather', 95),
        (r'ncclDevKernel_AllReduce', 'nccl_allreduce', 95),
        (r'ncclDevKernel_Broadcast', 'nccl_broadcast', 95),
        (r'ncclDevKernel_Reduce', 'nccl_reduce', 95),
        (r'ncclDevKernel_ReduceScatter', 'nccl_reduce_scatter', 95),
        (r'ncclDevKernel_SendRecv', 'nccl_send_recv', 95),
        (r'two_shot_all_reduce', 'custom_allreduce', 95),

        # ============== CUDA/cuBLAS Kernels (priority 90) ==============
        # cuBLAS GEMM (nvjet)
        (r'^nvjet_tst_', 'mm', 90),
        (r'^sm90_xmma_gemm', 'mm', 90),
        (r'^cutlass.*gemm', 'mm', 90),
        (r'^volta_.*gemm', 'mm', 90),
        (r'^ampere_.*gemm', 'mm', 90),
        (r'^hopper_.*gemm', 'mm', 90),
        (r'^cublasLt::splitKreduce', 'mm_reduce', 85),

        # cuDNN ops
        (r'cunn_SoftMaxForward', 'softmax', 90),
        (r'cunn_SoftMaxBackward', 'softmax_bwd', 90),
        (r'cudnn.*BatchNorm', 'batch_norm', 90),
        (r'cudnn.*Conv', 'conv', 90),
        (r'cudnn.*Pool', 'pool', 90),

        # PyTorch ATen kernels
        (r'at::native.*CatArrayBatchedCopy', 'cat', 90),
        (r'at::native.*SoftMaxForward', 'softmax', 90),
        (r'at::native.*SoftMaxBackward', 'softmax_bwd', 90),
        (r'at::native.*reduce_kernel.*ArgMaxOps', 'argmax', 90),
        (r'at::native.*reduce_kernel.*ArgMinOps', 'argmin', 90),
        (r'at::native.*reduce_kernel.*SumOps', 'sum', 90),
        (r'at::native.*reduce_kernel.*MeanOps', 'mean', 90),
        (r'at::native.*reduce_kernel.*MaxOps', 'max', 90),
        (r'at::native.*reduce_kernel.*MinOps', 'min', 90),
        (r'at::native.*reduce_kernel.*ProdOps', 'prod', 90),

        # ATen elementwise with specific functors
        (r'DivFunctor', 'div', 85),
        (r'MulFunctor', 'mul', 85),
        (r'AddFunctor', 'add', 85),
        (r'SubFunctor', 'sub', 85),
        (r'at::native.*FillFunctor<signed char>', 'zeros', 90),
        (r'at::native.*FillFunctor<int>', 'fill', 85),
        (r'at::native.*FillFunctor<long>', 'fill', 85),
        (r'at::native.*FillFunctor<float>', 'fill', 85),
        (r'at::native.*FillFunctor<.*bfloat16>', 'fill', 85),
        (r'at::native.*FillFunctor<bool>', 'fill', 85),
        (r'at::native.*FillFunctor<unsigned char>', 'fill', 85),
        (r'at::native.*FillFunctor', 'fill', 80),

        # ATen copy operations
        (r'direct_copy_kernel_cuda.*BFloat16', 'copy_bf16', 85),
        (r'direct_copy_kernel_cuda', 'copy', 80),
        (r'bfloat16_copy_kernel', 'copy_bf16', 85),
        (r'LoadWithCast.*StoreWithCast', 'to_copy', 85),

        # ATen random/distribution
        (r'exponential_kernel', 'exponential', 85),
        (r'uniform_kernel', 'uniform', 85),
        (r'normal_kernel', 'normal', 85),
        (r'bernoulli_kernel', 'bernoulli', 85),
        (r'distribution_elementwise.*exponential', 'exponential', 85),
        (r'distribution_elementwise.*uniform', 'uniform', 85),

        # ATen index operations
        (r'at::native.*index_elementwise', 'index', 85),
        (r'at::native.*scatter_gather', 'scatter', 85),
        (r'at::native.*gather_kernel', 'gather', 85),
        (r'at::native.*index_kernel', 'index', 85),
        (r'at::native.*index_select', 'index_select', 85),
        (r'at::native.*index_add', 'index_add', 85),
        (r'at::native.*index_copy', 'index_copy', 85),

        # ATen comparison
        (r'at::native.*CompareFunctor', 'compare', 85),
        (r'at::native.*compare_scalar', 'compare', 85),
        (r'at::native.*where_kernel', 'where', 85),
        (r'at::native.*masked_fill', 'masked_fill', 85),

        # ATen math
        (r'at::native.*sin_kernel', 'sin', 85),
        (r'at::native.*cos_kernel', 'cos', 85),
        (r'at::native.*tan_kernel', 'tan', 85),
        (r'at::native.*exp_kernel', 'exp', 85),
        (r'at::native.*log_kernel', 'log', 85),
        (r'at::native.*sqrt_kernel', 'sqrt', 85),
        (r'at::native.*rsqrt_kernel', 'rsqrt', 85),
        (r'at::native.*pow_tensor', 'pow', 85),
        (r'at::native.*reciprocal', 'reciprocal', 85),

        # ATen activations
        (r'at::native.*relu', 'relu', 85),
        (r'at::native.*gelu', 'gelu', 85),
        (r'at::native.*silu', 'silu', 85),
        (r'at::native.*sigmoid', 'sigmoid', 85),
        (r'at::native.*tanh', 'tanh', 85),

        # ATen scan operations
        (r'at::native.*tensor_kernel_scan', 'cumsum', 85),
        (r'at::native.*cub.*Scan', 'scan', 85),
        (r'at::native.*cub.*Sort', 'sort', 85),
        (r'DeviceSegmentedRadixSort', 'sort', 85),

        # ============== Triton Fused Kernels (priority 80) ==============
        (r'triton_poi_fused.*softmax', 'softmax', 80),
        (r'triton_poi_fused.*layer_norm', 'layer_norm', 80),
        (r'triton_poi_fused.*rms_?norm', 'rms_norm', 80),
        (r'triton_poi_fused.*add.*mul.*rsqrt', 'rms_norm', 75),
        (r'triton_poi_fused.*mean.*mul.*pow.*rsqrt', 'rms_norm', 75),
        (r'triton_poi_fused.*silu', 'silu', 80),
        (r'triton_poi_fused.*gelu', 'gelu', 80),
        (r'triton_poi_fused.*mul.*silu.*slice', 'swiglu', 80),
        (r'triton_poi_fused.*embedding', 'embedding', 80),
        (r'triton_poi_fused.*all_reduce', 'allreduce_fused', 80),
        (r'triton_poi_fused', 'triton_fused', 50),
        (r'triton_red_fused', 'triton_reduce', 50),
        (r'triton_per_fused', 'triton_persistent', 50),

        # ============== Generic Patterns (priority 30-70) ==============
        (r'at::native.*elementwise', 'elementwise', 30),
        (r'at::native.*vectorized', 'vectorized', 30),
        (r'at::native.*unrolled', 'unrolled', 30),
        (r'CUDAFunctor_add', 'add', 70),
        (r'CUDAFunctorOnSelf_add', 'add', 70),
        (r'CUDAFunctorOnOther_add', 'add', 70),

        # Copy - unify different copy patterns
        (r'at::native.*direct_copy', 'copy', 75),
        (r'at::native.*CatArrayBatchedCopy_vectorized', 'cat', 85),
    ]

    def __init__(self):
        # Compile regex patterns and sort by priority (descending)
        self.compiled_rules = [
            (re.compile(pattern), op_name, priority)
            for pattern, op_name, priority in self.RULES
        ]
        self.compiled_rules.sort(key=lambda x: -x[2])

    def map_kernel(self, kernel_name: str) -> Tuple[str, int]:
        """Map a kernel name to (operator_name, priority)."""
        for pattern, op_name, priority in self.compiled_rules:
            if pattern.search(kernel_name):
                return op_name, priority
        return 'unknown', 0


def parse_csv(filepath: str) -> List[KernelInfo]:
    """Parse nsys stats CSV output file."""
    kernels = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the CSV header line
    csv_start = 0
    for i, line in enumerate(lines):
        if line.startswith('Time (%),'):
            csv_start = i
            break

    # Parse CSV data
    reader = csv.DictReader(lines[csv_start:])
    for row in reader:
        try:
            kernel = KernelInfo(
                name=row['Name'].strip('"'),
                total_time_ns=int(row['Total Time (ns)']),
                instances=int(row['Instances']),
                avg_ns=float(row['Avg (ns)']),
                percentage=float(row['Time (%)'])
            )
            kernels.append(kernel)
        except (KeyError, ValueError):
            continue

    return kernels


def aggregate_by_operator(
    kernels: List[KernelInfo],
    mapper: KernelMapper
) -> Dict[str, OperatorStats]:
    """Aggregate kernel statistics by mapped operator name."""
    op_stats: Dict[str, OperatorStats] = {}

    for kernel in kernels:
        op_name, _ = mapper.map_kernel(kernel.name)

        if op_name not in op_stats:
            op_stats[op_name] = OperatorStats(name=op_name)

        op_stats[op_name].total_time_ns += kernel.total_time_ns
        op_stats[op_name].instances += kernel.instances
        op_stats[op_name].kernel_names.add(kernel.name)

    return op_stats


def format_time(time_s: float) -> str:
    """Format time in human-readable format."""
    if time_s >= 1:
        return f"{time_s:.2f}s"
    elif time_s >= 0.001:
        return f"{time_s*1000:.2f}ms"
    else:
        return f"{time_s*1e6:.2f}us"


def generate_report(
    cuda_ops: Dict[str, OperatorStats],
    flaggems_ops: Dict[str, OperatorStats],
    output_file: Optional[str] = None
):
    """Generate comparison report."""
    lines = []

    # Calculate totals
    cuda_total_ns = sum(op.total_time_ns for op in cuda_ops.values())
    flaggems_total_ns = sum(op.total_time_ns for op in flaggems_ops.values())

    cuda_total_s = cuda_total_ns / 1e9
    flaggems_total_s = flaggems_total_ns / 1e9

    # Get all operator names
    all_ops = set(cuda_ops.keys()) | set(flaggems_ops.keys())

    # Build comparison data
    comparisons = []
    for op_name in all_ops:
        cuda_op = cuda_ops.get(op_name)
        flaggems_op = flaggems_ops.get(op_name)

        cuda_time_ns = cuda_op.total_time_ns if cuda_op else 0
        flaggems_time_ns = flaggems_op.total_time_ns if flaggems_op else 0
        cuda_instances = cuda_op.instances if cuda_op else 0
        flaggems_instances = flaggems_op.instances if flaggems_op else 0
        cuda_kernels = cuda_op.kernel_names if cuda_op else set()
        flaggems_kernels = flaggems_op.kernel_names if flaggems_op else set()

        cuda_time_s = cuda_time_ns / 1e9
        flaggems_time_s = flaggems_time_ns / 1e9

        cuda_pct = (cuda_time_ns / cuda_total_ns * 100) if cuda_total_ns > 0 else 0
        flaggems_pct = (flaggems_time_ns / flaggems_total_ns * 100) if flaggems_total_ns > 0 else 0

        # Determine status based on kernel names
        if cuda_time_ns > 0 and flaggems_time_ns > 0:
            # Both have this operator - check if kernels are different
            if cuda_kernels != flaggems_kernels:
                # Different kernels = replaced by FlagGems
                ratio = flaggems_time_ns / cuda_time_ns
                if ratio > 1.05:
                    status = "SLOWER"
                elif ratio < 0.95:
                    status = "FASTER"
                else:
                    status = "SIMILAR"
            else:
                # Same kernels = not replaced
                status = "NOT_REPLACED"
                ratio = 1.0
        elif cuda_time_ns > 0 and flaggems_time_ns == 0:
            status = "CUDA_ONLY"
            ratio = 0
        elif cuda_time_ns == 0 and flaggems_time_ns > 0:
            status = "FLAGGEMS_ONLY"
            ratio = float('inf')
        else:
            continue

        comparisons.append({
            'operator': op_name,
            'cuda_time_s': cuda_time_s,
            'flaggems_time_s': flaggems_time_s,
            'cuda_pct': cuda_pct,
            'flaggems_pct': flaggems_pct,
            'cuda_instances': cuda_instances,
            'flaggems_instances': flaggems_instances,
            'cuda_kernels': cuda_kernels,
            'flaggems_kernels': flaggems_kernels,
            'ratio': ratio,
            'status': status,
            'time_delta_s': flaggems_time_s - cuda_time_s
        })

    # Sort by FlagGems time (descending)
    comparisons.sort(key=lambda x: -x['flaggems_time_s'])

    # Generate report
    lines.append("# Operator Performance Comparison: CUDA vs FlagGems\n")

    # Summary
    lines.append("## Summary\n")
    lines.append(f"| Metric | CUDA | FlagGems |")
    lines.append(f"|--------|------|----------|")
    lines.append(f"| Total GPU Time | {format_time(cuda_total_s)} | {format_time(flaggems_total_s)} |")
    lines.append(f"| Overall Ratio | 1.00x | {flaggems_total_s/cuda_total_s:.2f}x |")
    lines.append(f"| Unique Operators | {len(cuda_ops)} | {len(flaggems_ops)} |")
    lines.append("")

    # Count by status
    status_counts = defaultdict(int)
    for c in comparisons:
        status_counts[c['status']] += 1

    lines.append("### Status Breakdown\n")
    lines.append("| Status | Count | Description |")
    lines.append("|--------|-------|-------------|")
    lines.append(f"| SLOWER | {status_counts['SLOWER']} | Replaced by FlagGems, slower |")
    lines.append(f"| FASTER | {status_counts['FASTER']} | Replaced by FlagGems, faster |")
    lines.append(f"| SIMILAR | {status_counts['SIMILAR']} | Replaced by FlagGems, similar perf |")
    lines.append(f"| NOT_REPLACED | {status_counts['NOT_REPLACED']} | Same kernel in both |")
    lines.append(f"| CUDA_ONLY | {status_counts['CUDA_ONLY']} | Only in CUDA baseline |")
    lines.append(f"| FLAGGEMS_ONLY | {status_counts['FLAGGEMS_ONLY']} | Only in FlagGems |")
    lines.append("")

    # Detailed table
    lines.append("## Detailed Comparison\n")
    lines.append("| Operator | CUDA Time | %CUDA | FlagGems Time | %FG | Ratio | CUDA Inst | FG Inst | Status |")
    lines.append("|----------|-----------|-------|---------------|-----|-------|-----------|---------|--------|")

    for c in comparisons:
        if c['ratio'] == float('inf'):
            ratio_str = "N/A"
        elif c['ratio'] == 0:
            ratio_str = "N/A"
        else:
            ratio_str = f"{c['ratio']:.2f}x"

        lines.append(
            f"| {c['operator']} | {format_time(c['cuda_time_s'])} | {c['cuda_pct']:.1f}% | "
            f"{format_time(c['flaggems_time_s'])} | {c['flaggems_pct']:.1f}% | {ratio_str} | "
            f"{c['cuda_instances']} | {c['flaggems_instances']} | {c['status']} |"
        )

    lines.append("")

    # Kernel name details for replaced operators
    replaced = [c for c in comparisons if c['status'] in ('SLOWER', 'FASTER', 'SIMILAR')]
    if replaced:
        lines.append("## Replaced Operators - Kernel Details\n")
        lines.append("| Operator | Status | CUDA Kernel(s) | FlagGems Kernel(s) |")
        lines.append("|----------|--------|----------------|---------------------|")
        for c in replaced:
            cuda_k = ', '.join(sorted(c['cuda_kernels']))[:80]
            fg_k = ', '.join(sorted(c['flaggems_kernels']))[:80]
            if len(cuda_k) > 80:
                cuda_k = cuda_k[:77] + "..."
            if len(fg_k) > 80:
                fg_k = fg_k[:77] + "..."
            lines.append(f"| {c['operator']} | {c['status']} | {cuda_k} | {fg_k} |")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by opdiff2.py*")

    report = "\n".join(lines)

    # Output
    print(report)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare CUDA vs FlagGems kernel performance from nsys CSV files.'
    )
    parser.add_argument(
        '--cuda', '-c',
        required=True,
        help='Path to CUDA baseline kernel CSV file'
    )
    parser.add_argument(
        '--flaggems', '-f',
        required=True,
        help='Path to FlagGems kernel CSV file'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output markdown file path (optional)'
    )

    args = parser.parse_args()

    # Parse CSV files
    print(f"Parsing CUDA CSV: {args.cuda}")
    cuda_kernels = parse_csv(args.cuda)
    print(f"  Found {len(cuda_kernels)} kernels")

    print(f"Parsing FlagGems CSV: {args.flaggems}")
    flaggems_kernels = parse_csv(args.flaggems)
    print(f"  Found {len(flaggems_kernels)} kernels")

    # Map kernels to operators
    mapper = KernelMapper()

    print("Aggregating by operator...")
    cuda_ops = aggregate_by_operator(cuda_kernels, mapper)
    flaggems_ops = aggregate_by_operator(flaggems_kernels, mapper)

    print(f"  CUDA operators: {len(cuda_ops)}")
    print(f"  FlagGems operators: {len(flaggems_ops)}")
    print()

    # Generate report
    generate_report(cuda_ops, flaggems_ops, args.output)


if __name__ == '__main__':
    main()
