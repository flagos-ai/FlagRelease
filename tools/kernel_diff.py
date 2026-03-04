#!/usr/bin/env python3
"""
Kernel Comparison Tool: CUDA vs FlagGems
Maps kernels to unified names and categorizes performance differences.
"""

import re
import csv
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# =============================================================================
# MAPPING RULES: Map kernel names to unified operator names
# =============================================================================

# Regex patterns for CUDA kernels -> unified name
CUDA_KERNEL_PATTERNS = [
    # GEMM operations (nvjet, cublas, cutlass)
    (r'nvjet_tst_\d+x\d+_\d+x\d+.*', 'GEMM'),
    (r'sm90_xmma_gemm.*', 'GEMM'),
    (r'cutlass.*gemm.*', 'GEMM'),
    (r'cublasLt::splitKreduce_kernel.*', 'GEMM_splitK_reduce'),
    (r'internal::gemvx::kernel.*', 'GEMV'),

    # Softmax
    (r'cunn_SoftMaxForward.*', 'Softmax'),

    # Reduction operations
    (r'at::native::reduce_kernel.*ArgMaxOps.*', 'ArgMax'),
    (r'at::native::reduce_kernel.*', 'Reduce'),

    # Elementwise copy operations
    (r'at::native::elementwise_kernel.*direct_copy_kernel_cuda.*BFloat16.*', 'Copy_bf16'),
    (r'at::native::elementwise_kernel.*direct_copy_kernel_cuda.*', 'Copy'),
    (r'at::native::unrolled_elementwise_kernel.*direct_copy_kernel_cuda.*float\).*', 'Copy_float'),
    (r'at::native::unrolled_elementwise_kernel.*direct_copy_kernel_cuda.*long\).*', 'Copy_long'),
    (r'at::native::unrolled_elementwise_kernel.*direct_copy_kernel_cuda.*int\).*', 'Copy_int'),
    (r'at::native::unrolled_elementwise_kernel.*direct_copy_kernel_cuda.*', 'Copy'),
    (r'at::native::.*CatArrayBatchedCopy.*', 'Cat'),
    (r'at::native::.*CatArrayBatchedCopy_vectorized.*', 'Cat_vectorized'),
    (r'at::native::bfloat16_copy_kernel_cuda.*', 'Copy_bf16_cast'),

    # Division
    (r'at::native::elementwise_kernel.*DivFunctor.*', 'Div'),
    (r'at::native::vectorized_elementwise_kernel.*DivFunctor.*', 'Div'),
    (r'at::native::unrolled_elementwise_kernel.*DivFunctor.*', 'Div'),

    # Multiplication
    (r'at::native::elementwise_kernel.*MulFunctor.*', 'Mul'),
    (r'at::native::vectorized_elementwise_kernel.*MulFunctor.*', 'Mul'),
    (r'at::native::vectorized_elementwise_kernel.*AUnaryFunctor.*MulFunctor.*', 'Mul_scalar'),
    (r'at::native::vectorized_elementwise_kernel.*BUnaryFunctor.*MulFunctor.*', 'Mul_scalar'),

    # Addition/Subtraction
    (r'at::native::unrolled_elementwise_kernel.*CUDAFunctor_add<int>.*', 'Add_int'),
    (r'at::native::unrolled_elementwise_kernel.*CUDAFunctorOnSelf_add<int>.*', 'Add_inplace_int'),
    (r'at::native::vectorized_elementwise_kernel.*CUDAFunctorOnOther_add<long>.*', 'Add_scalar_long'),
    (r'at::native::vectorized_elementwise_kernel.*CUDAFunctorOnOther_add<float>.*', 'Add_scalar_float'),

    # Fill operations
    (r'at::native::vectorized_elementwise_kernel.*FillFunctor<signed char>.*', 'Fill_int8'),
    (r'at::native::vectorized_elementwise_kernel.*FillFunctor<int>.*', 'Fill_int'),
    (r'at::native::vectorized_elementwise_kernel.*FillFunctor<long>.*', 'Fill_long'),
    (r'at::native::vectorized_elementwise_kernel.*FillFunctor<float>.*', 'Fill_float'),
    (r'at::native::vectorized_elementwise_kernel.*FillFunctor<c10::BFloat16>.*', 'Fill_bf16'),
    (r'at::native::vectorized_elementwise_kernel.*FillFunctor<bool>.*', 'Fill_bool'),
    (r'at::native::vectorized_elementwise_kernel.*FillFunctor<unsigned char>.*', 'Fill_uint8'),
    (r'at::native::elementwise_kernel.*FillFunctor<bool>.*', 'Fill_bool'),
    (r'at::native::unrolled_elementwise_kernel.*FillFunctor<long>.*', 'Fill_long'),

    # Exponential/Random distribution
    (r'at::native::.*distribution_elementwise_grid_stride_kernel.*exponential.*', 'Exponential'),
    (r'at::native::.*distribution_elementwise_grid_stride_kernel.*uniform.*', 'Rand'),

    # Scatter/Gather
    (r'at::native::_scatter_gather_elementwise_kernel.*<\(bool\)1.*', 'Scatter'),
    (r'at::native::_scatter_gather_elementwise_kernel.*<\(bool\)0.*', 'Gather_scatter'),
    (r'at::native::vectorized_gather_kernel.*', 'Gather'),

    # Index operations
    (r'at::native::index_elementwise_kernel.*OpaqueType<\(int\)2>.*', 'Index_2byte'),
    (r'at::native::index_elementwise_kernel.*OpaqueType<\(int\)4>.*', 'Index_4byte'),
    (r'at::native::index_elementwise_kernel.*', 'Index'),

    # Masked fill
    (r'at::native::vectorized_elementwise_kernel.*masked_fill_kernel.*', 'MaskedFill'),

    # Trigonometric
    (r'at::native::vectorized_elementwise_kernel.*sin_kernel_cuda.*', 'Sin'),
    (r'at::native::vectorized_elementwise_kernel.*cos_kernel_cuda.*', 'Cos'),

    # Power
    (r'at::native::elementwise_kernel.*pow_tensor_tensor_kernel.*', 'Pow'),

    # Reciprocal
    (r'at::native::vectorized_elementwise_kernel.*reciprocal_kernel_cuda.*', 'Reciprocal'),

    # Comparison
    (r'at::native::elementwise_kernel.*CompareFunctor.*', 'Compare'),
    (r'at::native::vectorized_elementwise_kernel.*compare_scalar_kernel.*', 'Compare_scalar'),

    # Where/Conditional
    (r'at::native::vectorized_elementwise_kernel.*where_kernel_impl.*long.*', 'Where_long'),
    (r'at::native::elementwise_kernel.*where_kernel_impl.*float.*', 'Where_float'),
    (r'at::native::vectorized_elementwise_kernel.*where_kernel_impl.*', 'Where'),

    # Arange
    (r'elementwise_kernel_with_index.*arange_cuda_out.*', 'Arange'),

    # Sort operations
    (r'at_cuda_detail::cub::DeviceSegmentedRadixSortKernel.*<\(bool\)1.*', 'RadixSort_descend'),
    (r'at_cuda_detail::cub::DeviceSegmentedRadixSortKernel.*<\(bool\)0.*', 'RadixSort_ascend'),
    (r'at::native::.*fill_reverse_indices_kernel.*', 'Sort_fill_indices'),

    # Scan
    (r'at::native::tensor_kernel_scan_innermost_dim.*', 'CumSum'),
]

# Regex patterns for FlagGems kernels -> unified name
FLAGGEMS_KERNEL_PATTERNS = [
    # GEMM operations
    (r'^mm_kernel_general$', 'GEMM'),
    (r'^mm_kernel_general_host_tma$', 'GEMM'),
    (r'^bmm_kernel$', 'BMM'),

    # Softmax
    (r'^softmax_kernel_inner$', 'Softmax'),

    # ArgMax
    (r'^argmax_kernel_inner$', 'ArgMax'),

    # Copy operations
    (r'^_copy_kernel_kernel_rank_\d+$', 'Copy'),
    (r'^cat_copy_func_kernel_\d+$', 'Cat'),

    # Division
    (r'^true_div_func_kernel_rank_\d+$', 'Div'),
    (r'^true_div_func_tensor_scalar_kernel_rank_\d+$', 'Div_scalar'),

    # Multiplication
    (r'^mul_func_kernel_rank_\d+$', 'Mul'),
    (r'^mul_func_scalar_kernel_rank_\d+$', 'Mul_scalar'),

    # Subtraction (maps to Add in some cases)
    (r'^sub_func_tensor_scalar_kernel_rank_\d+$', 'Sub_scalar'),
    (r'^sub_func_scalar_tensor_kernel_rank_\d+$', 'Sub_scalar_tensor'),
    (r'^sub_func_kernel_rank_\d+$', 'Sub'),

    # Fill operations
    (r'^fill_scalar_func_kernel_rank_\d+$', 'Fill'),
    (r'^zeros_kernel$', 'Zeros'),
    (r'^ones_kernel$', 'Ones'),
    (r'^full_func_scalar_kernel_rank_\d+$', 'Full'),

    # Exponential/Random
    (r'^fused_exponential_kernel_f32$', 'Exponential'),
    (r'^rand_kernel$', 'Rand'),

    # Scatter/Gather
    (r'^_scatter_jit_function$', 'Scatter'),
    (r'^_gather_flaggems_jit_function$', 'Gather'),

    # Index
    (r'^_index_jit_function$', 'Index'),

    # Masked fill
    (r'^masked_fill_kernel_kernel_rank_\d+$', 'MaskedFill'),

    # Trigonometric
    (r'^sin_func_kernel_rank_\d+$', 'Sin'),
    (r'^cos_func_kernel_rank_\d+$', 'Cos'),

    # Power
    (r'^pow_func_scalar_tensor_kernel_rank_\d+$', 'Pow'),

    # Reciprocal
    (r'^reciprocal_func_kernel_rank_\d+$', 'Reciprocal'),

    # Comparison
    (r'^lt_func_kernel_rank_\d+$', 'LessThan'),
    (r'^le_func_kernel_rank_\d+$', 'LessEqual'),
    (r'^lt_func_scalar_kernel_rank_\d+$', 'LessThan_scalar'),

    # Where/Conditional
    (r'^where_inner_kernel_rank_\d+$', 'Where'),

    # Arange
    (r'^arange_func$', 'Arange'),

    # Sort operations (FlagGems specific)
    (r'^sweep$', 'RadixSort_sweep'),
    (r'^compute_global_hist_kernel$', 'RadixSort_histogram'),
    (r'^reduce_then_scan_block_scan_kernel_row$', 'Scan_block'),
    (r'^reduce_then_scan_block_sum_kernel_row$', 'Scan_sum'),
    (r'^reduce_then_scan_root_scan_kernel_row$', 'Scan_root'),
]

# Kernels that exist in both CUDA and FlagGems (not replaced)
COMMON_KERNEL_PATTERNS = [
    # Attention kernels
    (r'sm90::fwd::sparse_attn_fwd_kernel.*', 'SparseAttention'),

    # MoE kernels
    (r'^fused_moe_kernel$', 'FusedMoE'),

    # VLLM specific kernels
    (r'vllm::cross_device_reduce_1stage.*', 'CrossDeviceReduce'),
    (r'two_shot_all_reduce_kernel_inplace.*', 'TwoShotAllReduce'),
    (r'vllm::topk_kernel.*', 'TopK'),
    (r'vllm::topKPerRowDecode.*', 'TopKDecode'),
    (r'vllm::topKPerRowPrefill.*', 'TopKPrefill'),
    (r'vllm::concat_and_cache_mla_kernel.*', 'ConcatCacheMLA'),
    (r'vllm::indexer_k_quant_and_cache_kernel.*', 'IndexerKQuantCache'),
    (r'vllm::act_and_mul_kernel.*', 'ActAndMul'),
    (r'vllm::moe::grouped_topk_fused_kernel.*', 'MoEGroupedTopK'),
    (r'vllm::moe::moe_align_block_size_kernel.*', 'MoEAlignBlockSize'),
    (r'vllm::moe::moe_align_block_size_small_batch_expert_kernel.*', 'MoEAlignBlockSizeSmall'),
    (r'vllm::moe::moe_sum_kernel.*', 'MoESum'),
    (r'vllm::moe::count_and_sort_expert_tokens_kernel.*', 'MoECountSortTokens'),
    (r'vllm::cp_gather_indexer_k_quant_cache_kernel.*', 'CPGatherIndexerKQuantCache'),

    # Deep GEMM kernels
    (r'deep_gemm::sm90_fp8_paged_mqa_logits.*', 'DeepGEMM_FP8_PagedMQA'),
    (r'deep_gemm::sm90_fp8_mqa_logits.*', 'DeepGEMM_FP8_MQA'),
    (r'deep_gemm::smxx_paged_mqa_logits_metadata.*', 'DeepGEMM_PagedMQA_Metadata'),

    # NCCL kernels
    (r'ncclDevKernel_AllGather_RING_LL.*', 'NCCL_AllGather'),
    (r'ncclDevKernel_AllReduce_Sum_f32_RING_LL.*', 'NCCL_AllReduce'),

    # Quantization kernels
    (r'per_token_group_quant_8bit_kernel.*', 'PerTokenGroupQuant8bit'),

    # Convert kernels
    (r'^_convert_req_index_to_global_index_kernel$', 'ConvertReqIndexToGlobal'),

    # Triton fused kernels (common)
    (r'^triton_poi_fused_5$', 'Triton_fused_5'),
    (r'^triton_poi_fused_6$', 'Triton_fused_6'),
    (r'^triton_poi_fused_4$', 'Triton_fused_4'),
    (r'^triton_poi_fused_3$', 'Triton_fused_3'),
    (r'^triton_red_fused_3$', 'Triton_red_fused_3'),
    (r'^triton_red_fused_2$', 'Triton_red_fused_2'),
    (r'^triton_poi_fused_mul_silu_slice_1$', 'Triton_MulSiluSlice_1'),
    (r'^triton_poi_fused_mul_silu_slice_0$', 'Triton_MulSiluSlice_0'),
    (r'^triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_2$', 'Triton_RMSNorm_2'),
    (r'^triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0$', 'Triton_RMSNorm_0'),
    (r'^triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_1$', 'Triton_RMSNorm_1'),
    (r'^triton_per_fused__to_copy_add_mean_moe_forward_shared_mul_pow_rsqrt_0$', 'Triton_MoE_RMSNorm'),
    (r'^triton_poi_fused_mul_unsqueeze_view_7$', 'Triton_MulUnsqueezeView_7'),
    (r'^triton_poi_fused_mul_unsqueeze_view_6$', 'Triton_MulUnsqueezeView_6'),
    (r'^triton_poi_fused_add_all_reduce_bitwise_and_bitwise_not_bitwise_or_embedding_ge_lt_masked_fill_mul_sub_unsqueeze_0$', 'Triton_EmbeddingMask'),
    (r'^triton_poi_fused_add_all_reduce_mul_1$', 'Triton_AddAllReduceMul'),
]

# Mapping for equivalent operations between CUDA and FlagGems
EQUIVALENT_OPS = {
    # GEMM equivalents
    'GEMM': ['GEMM', 'BMM'],
    'GEMM_splitK_reduce': ['GEMM'],
    'GEMV': ['GEMM'],

    # Softmax
    'Softmax': ['Softmax'],

    # ArgMax
    'ArgMax': ['ArgMax'],

    # Copy operations
    'Copy': ['Copy'],
    'Copy_bf16': ['Copy'],
    'Copy_float': ['Copy'],
    'Copy_long': ['Copy'],
    'Copy_int': ['Copy'],
    'Copy_bf16_cast': ['Copy'],
    'Cat': ['Cat'],
    'Cat_vectorized': ['Cat'],

    # Division
    'Div': ['Div', 'Div_scalar'],

    # Multiplication
    'Mul': ['Mul'],
    'Mul_scalar': ['Mul_scalar'],

    # Addition related
    'Add_int': ['Sub_scalar'],  # FlagGems uses sub for some add ops
    'Add_inplace_int': ['Sub_scalar'],
    'Add_scalar_long': ['Sub_scalar_tensor'],
    'Add_scalar_float': ['Sub_scalar_tensor'],

    # Fill operations
    'Fill_int8': ['Fill', 'Zeros', 'Full'],
    'Fill_int': ['Fill', 'Zeros', 'Full'],
    'Fill_long': ['Fill', 'Zeros', 'Full'],
    'Fill_float': ['Fill', 'Zeros', 'Full'],
    'Fill_bf16': ['Fill', 'Zeros', 'Full'],
    'Fill_bool': ['Fill', 'Zeros', 'Full'],
    'Fill_uint8': ['Fill', 'Zeros', 'Full'],

    # Exponential/Random
    'Exponential': ['Exponential'],
    'Rand': ['Rand'],

    # Scatter/Gather
    'Scatter': ['Scatter'],
    'Gather': ['Gather'],
    'Gather_scatter': ['Gather', 'Scatter'],

    # Index
    'Index': ['Index'],
    'Index_2byte': ['Index'],
    'Index_4byte': ['Index'],

    # Masked fill
    'MaskedFill': ['MaskedFill'],

    # Trigonometric
    'Sin': ['Sin'],
    'Cos': ['Cos'],

    # Power
    'Pow': ['Pow'],

    # Reciprocal
    'Reciprocal': ['Reciprocal'],

    # Comparison
    'Compare': ['LessThan', 'LessEqual'],
    'Compare_scalar': ['LessThan_scalar'],

    # Where
    'Where': ['Where'],
    'Where_long': ['Where'],
    'Where_float': ['Where'],

    # Arange
    'Arange': ['Arange'],

    # Sort operations
    'RadixSort_descend': ['RadixSort_sweep', 'RadixSort_histogram', 'Scan_block', 'Scan_sum', 'Scan_root'],
    'RadixSort_ascend': ['RadixSort_sweep', 'RadixSort_histogram', 'Scan_block', 'Scan_sum', 'Scan_root'],
    'Sort_fill_indices': ['RadixSort_sweep'],
    'CumSum': ['Scan_block', 'Scan_sum', 'Scan_root'],
}


def parse_csv(filepath: str) -> List[Dict]:
    """Parse nsys kernel CSV file."""
    kernels = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the header line
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('Time (%),'):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find header in {filepath}")

    # Parse CSV from header
    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        if row.get('Name'):
            kernels.append({
                'name': row['Name'].strip('"'),
                'time_pct': float(row['Time (%)']) if row['Time (%)'] else 0,
                'total_time_ns': int(row['Total Time (ns)']) if row['Total Time (ns)'] else 0,
                'instances': int(row['Instances']) if row['Instances'] else 0,
                'avg_ns': float(row['Avg (ns)']) if row['Avg (ns)'] else 0,
            })
    return kernels


def match_kernel_name(kernel_name: str, patterns: List[Tuple[str, str]]) -> Optional[str]:
    """Match kernel name against patterns and return unified name."""
    for pattern, unified_name in patterns:
        if re.search(pattern, kernel_name):
            return unified_name
    return None


def classify_kernel(kernel_name: str) -> Tuple[str, str]:
    """
    Classify a kernel and return (unified_name, source).
    source: 'cuda', 'flaggems', 'common', 'unknown'
    """
    # Check common patterns first
    unified = match_kernel_name(kernel_name, COMMON_KERNEL_PATTERNS)
    if unified:
        return unified, 'common'

    # Check CUDA patterns
    unified = match_kernel_name(kernel_name, CUDA_KERNEL_PATTERNS)
    if unified:
        return unified, 'cuda'

    # Check FlagGems patterns
    unified = match_kernel_name(kernel_name, FLAGGEMS_KERNEL_PATTERNS)
    if unified:
        return unified, 'flaggems'

    return kernel_name, 'unknown'


def aggregate_kernels(kernels: List[Dict]) -> Dict[str, Dict]:
    """Aggregate kernels by unified name."""
    aggregated = defaultdict(lambda: {
        'total_time_ns': 0,
        'time_pct': 0.0,
        'instances': 0,
        'raw_kernels': []
    })

    for kernel in kernels:
        unified_name, source = classify_kernel(kernel['name'])
        agg = aggregated[unified_name]
        agg['total_time_ns'] += kernel['total_time_ns']
        agg['time_pct'] += kernel['time_pct']
        agg['instances'] += kernel['instances']
        agg['raw_kernels'].append(kernel['name'])
        agg['source'] = source

    return dict(aggregated)


def find_equivalent_fg_ops(cuda_op: str, fg_ops: Dict[str, Dict]) -> List[str]:
    """Find equivalent FlagGems operations for a CUDA operation."""
    if cuda_op in EQUIVALENT_OPS:
        return [op for op in EQUIVALENT_OPS[cuda_op] if op in fg_ops]
    return [cuda_op] if cuda_op in fg_ops else []


def compare_kernels(cuda_file: str, fg_file: str, ratio_threshold: float = 0.1) -> List[Dict]:
    """
    Compare CUDA and FlagGems kernels.

    Categories:
    - FlagGems_faster: FlagGems is >10% faster
    - FlagGems_slower: FlagGems is >10% slower
    - Similar: Within 10% of each other
    - Not_replaced: Same kernel in both
    - Only_CUDA: Only in CUDA
    - Only_FlagGems: Only in FlagGems
    """
    cuda_kernels = parse_csv(cuda_file)
    fg_kernels = parse_csv(fg_file)

    cuda_agg = aggregate_kernels(cuda_kernels)
    fg_agg = aggregate_kernels(fg_kernels)

    results = []
    processed_cuda_ops = set()
    processed_fg_ops = set()

    # Step 1: Group CUDA ops by their FlagGems equivalents
    # This handles multiple CUDA ops mapping to the same FlagGems op
    fg_to_cuda_mapping = defaultdict(list)  # fg_op -> list of cuda_ops

    for cuda_op, cuda_data in cuda_agg.items():
        source = cuda_data.get('source', 'unknown')
        if source == 'cuda':
            fg_equivalents = find_equivalent_fg_ops(cuda_op, fg_agg)
            if fg_equivalents:
                # Use frozenset as key to group by same FG equivalents
                fg_key = tuple(sorted(fg_equivalents))
                fg_to_cuda_mapping[fg_key].append(cuda_op)

    # Step 2: Process grouped CUDA->FlagGems mappings (sum multiple CUDA ops)
    for fg_key, cuda_ops in fg_to_cuda_mapping.items():
        fg_equivalents = list(fg_key)

        # Sum all CUDA ops that map to this FlagGems op(s)
        cuda_time = sum(cuda_agg[op]['total_time_ns'] for op in cuda_ops)
        cuda_pct = sum(cuda_agg[op]['time_pct'] for op in cuda_ops)
        cuda_instances = sum(cuda_agg[op]['instances'] for op in cuda_ops)

        # Sum all FlagGems equivalents
        fg_time = sum(fg_agg[op]['total_time_ns'] for op in fg_equivalents)
        fg_pct = sum(fg_agg[op]['time_pct'] for op in fg_equivalents)
        fg_instances = sum(fg_agg[op]['instances'] for op in fg_equivalents)

        for op in cuda_ops:
            processed_cuda_ops.add(op)
        for op in fg_equivalents:
            processed_fg_ops.add(op)

        # Determine category based on ratio
        if cuda_time > 0:
            ratio = fg_time / cuda_time
        else:
            ratio = 1.0

        if abs(ratio - 1.0) <= ratio_threshold:
            category = 'Similar'
        elif ratio < 1.0 - ratio_threshold:
            category = 'FlagGems_faster'
        else:
            category = 'FlagGems_slower'

        # Build operator name
        if len(cuda_ops) == 1:
            op_name = cuda_ops[0]
        else:
            # Multiple CUDA ops, use common prefix or list them
            op_name = ','.join(sorted(cuda_ops))

        if fg_equivalents and fg_equivalents[0] != cuda_ops[0]:
            op_name = f"{op_name} -> {','.join(fg_equivalents)}"

        results.append({
            'Operator': op_name,
            'CUDA_Time_ns': cuda_time,
            'CUDA_Pct': cuda_pct,
            'FlagGems_Time_ns': fg_time,
            'FlagGems_Pct': fg_pct,
            'CUDA_Instances': cuda_instances,
            'FlagGems_Instances': fg_instances,
            'Category': category,
        })

    # Step 3: Process 'common' source (not replaced) - same kernel in both
    for cuda_op, cuda_data in cuda_agg.items():
        source = cuda_data.get('source', 'unknown')
        if source == 'common' and cuda_op not in processed_cuda_ops:
            if cuda_op in fg_agg:
                fg_data = fg_agg[cuda_op]
                processed_cuda_ops.add(cuda_op)
                processed_fg_ops.add(cuda_op)

                results.append({
                    'Operator': cuda_op,
                    'CUDA_Time_ns': cuda_data['total_time_ns'],
                    'CUDA_Pct': cuda_data['time_pct'],
                    'FlagGems_Time_ns': fg_data['total_time_ns'],
                    'FlagGems_Pct': fg_data['time_pct'],
                    'CUDA_Instances': cuda_data['instances'],
                    'FlagGems_Instances': fg_data['instances'],
                    'Category': 'Not_replaced',
                })
            else:
                processed_cuda_ops.add(cuda_op)
                results.append({
                    'Operator': cuda_op,
                    'CUDA_Time_ns': cuda_data['total_time_ns'],
                    'CUDA_Pct': cuda_data['time_pct'],
                    'FlagGems_Time_ns': 0,
                    'FlagGems_Pct': 0,
                    'CUDA_Instances': cuda_data['instances'],
                    'FlagGems_Instances': 0,
                    'Category': 'Only_CUDA',
                })

    # Step 4: Process remaining CUDA ops (unknown source or no FG equivalent)
    for cuda_op, cuda_data in cuda_agg.items():
        if cuda_op in processed_cuda_ops:
            continue

        source = cuda_data.get('source', 'unknown')

        # Check if exists in FlagGems with same name
        if cuda_op in fg_agg:
            fg_data = fg_agg[cuda_op]
            processed_cuda_ops.add(cuda_op)
            processed_fg_ops.add(cuda_op)

            results.append({
                'Operator': cuda_op,
                'CUDA_Time_ns': cuda_data['total_time_ns'],
                'CUDA_Pct': cuda_data['time_pct'],
                'FlagGems_Time_ns': fg_data['total_time_ns'],
                'FlagGems_Pct': fg_data['time_pct'],
                'CUDA_Instances': cuda_data['instances'],
                'FlagGems_Instances': fg_data['instances'],
                'Category': 'Not_replaced',
            })
        else:
            processed_cuda_ops.add(cuda_op)
            results.append({
                'Operator': cuda_op,
                'CUDA_Time_ns': cuda_data['total_time_ns'],
                'CUDA_Pct': cuda_data['time_pct'],
                'FlagGems_Time_ns': 0,
                'FlagGems_Pct': 0,
                'CUDA_Instances': cuda_data['instances'],
                'FlagGems_Instances': 0,
                'Category': 'Only_CUDA',
            })

    # Step 5: Process FlagGems-only operations
    for fg_op, fg_data in fg_agg.items():
        if fg_op not in processed_fg_ops:
            source = fg_data.get('source', 'unknown')
            if source == 'flaggems':
                results.append({
                    'Operator': fg_op,
                    'CUDA_Time_ns': 0,
                    'CUDA_Pct': 0,
                    'FlagGems_Time_ns': fg_data['total_time_ns'],
                    'FlagGems_Pct': fg_data['time_pct'],
                    'CUDA_Instances': 0,
                    'FlagGems_Instances': fg_data['instances'],
                    'Category': 'Only_FlagGems',
                })

    return results


def format_time(ns: int) -> str:
    """Format time in ns to human readable format."""
    if ns >= 1e9:
        return f"{ns/1e9:.2f}s"
    elif ns >= 1e6:
        return f"{ns/1e6:.2f}ms"
    elif ns >= 1e3:
        return f"{ns/1e3:.2f}us"
    else:
        return f"{ns}ns"


def print_category_table(categories: Dict[str, List[Dict]]):
    """Print a formatted ASCII table of category summary."""
    # Calculate data for each category
    cat_order = ['FlagGems_faster', 'FlagGems_slower', 'Similar', 'Not_replaced', 'Only_CUDA', 'Only_FlagGems']

    rows = []
    for cat in cat_order:
        items = categories.get(cat, [])
        count = len(items)
        total_cuda = sum(r['CUDA_Time_ns'] for r in items)
        total_fg = sum(r['FlagGems_Time_ns'] for r in items)

        cuda_str = format_time(total_cuda) if total_cuda > 0 else '-'
        fg_str = format_time(total_fg) if total_fg > 0 else '-'

        rows.append((cat, count, cuda_str, fg_str))

    # Column widths
    col1_w = max(len('Category'), max(len(r[0]) for r in rows)) + 2
    col2_w = max(len('Count'), max(len(str(r[1])) for r in rows)) + 2
    col3_w = max(len('CUDA Total'), max(len(r[2]) for r in rows)) + 2
    col4_w = max(len('FG Total'), max(len(r[3]) for r in rows)) + 2

    # Build table
    top_border = f"┌{'─' * col1_w}┬{'─' * col2_w}┬{'─' * col3_w}┬{'─' * col4_w}┐"
    header_sep = f"├{'─' * col1_w}┼{'─' * col2_w}┼{'─' * col3_w}┼{'─' * col4_w}┤"
    bottom_border = f"└{'─' * col1_w}┴{'─' * col2_w}┴{'─' * col3_w}┴{'─' * col4_w}┘"

    def make_row(c1, c2, c3, c4):
        return f"│{c1:^{col1_w}}│{c2:^{col2_w}}│{c3:^{col3_w}}│{c4:^{col4_w}}│"

    print()
    print(top_border)
    print(make_row('Category', 'Count', 'CUDA Total', 'FG Total'))
    print(header_sep)

    for i, (cat, count, cuda_str, fg_str) in enumerate(rows):
        print(make_row(cat, str(count), cuda_str, fg_str))
        if i < len(rows) - 1:
            print(header_sep)

    print(bottom_border)


def write_csv(results: List[Dict], output_file: str):
    """Write results to CSV file."""
    fieldnames = [
        'Operator', 'CUDA_Time_ns', 'CUDA_Pct', 'FlagGems_Time_ns', 'FlagGems_Pct',
        'CUDA_Instances', 'FlagGems_Instances', 'Category'
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Format percentages
            output_row = {
                'Operator': row['Operator'],
                'CUDA_Time_ns': row['CUDA_Time_ns'],
                'CUDA_Pct': f"{row['CUDA_Pct']:.2f}%",
                'FlagGems_Time_ns': row['FlagGems_Time_ns'],
                'FlagGems_Pct': f"{row['FlagGems_Pct']:.2f}%",
                'CUDA_Instances': row['CUDA_Instances'],
                'FlagGems_Instances': row['FlagGems_Instances'],
                'Category': row['Category'],
            }
            writer.writerow(output_row)


def print_summary(results: List[Dict]):
    """Print summary statistics."""
    categories = defaultdict(list)
    for r in results:
        categories[r['Category']].append(r)

    print("\n" + "="*80)
    print("SUMMARY BY CATEGORY")
    print("="*80)

    # Print formatted table
    print_category_table(categories)

    # Print detailed info for each category
    for cat in ['FlagGems_faster', 'FlagGems_slower', 'Similar', 'Not_replaced', 'Only_CUDA', 'Only_FlagGems']:
        items = categories.get(cat, [])
        if items:
            total_cuda = sum(r['CUDA_Time_ns'] for r in items)
            total_fg = sum(r['FlagGems_Time_ns'] for r in items)
            print(f"\n{cat}: {len(items)} operators")
            print(f"  Total CUDA time: {format_time(total_cuda)}")
            print(f"  Total FlagGems time: {format_time(total_fg)}")

            # Top 5 by time
            sorted_items = sorted(items, key=lambda x: max(x['CUDA_Time_ns'], x['FlagGems_Time_ns']), reverse=True)[:5]
            print("  Top 5:")
            for item in sorted_items:
                cuda_t = item['CUDA_Time_ns']
                fg_t = item['FlagGems_Time_ns']
                if cuda_t > 0:
                    ratio = fg_t / cuda_t
                    ratio_str = f"{ratio:.3f}"
                elif fg_t > 0:
                    ratio_str = "0"
                else:
                    ratio_str = "-"
                print(f"    - {item['Operator']}: CUDA={format_time(cuda_t)}, "
                      f"FG={format_time(fg_t)}, Ratio={ratio_str}")


def print_flaggems_faster_kernels(cuda_file: str, fg_file: str, ratio_threshold: float = 0.1):
    """Print original kernel names for FlagGems_faster category."""
    cuda_kernels = parse_csv(cuda_file)
    fg_kernels = parse_csv(fg_file)

    cuda_agg = aggregate_kernels(cuda_kernels)
    fg_agg = aggregate_kernels(fg_kernels)

    print("\n" + "="*80)
    print("FlagGems_faster: Original Kernel Names")
    print("="*80)

    faster_ops = []

    for cuda_op, cuda_data in cuda_agg.items():
        source = cuda_data.get('source', 'unknown')
        if source == 'common':
            continue

        # Find equivalent FlagGems operation(s)
        fg_equivalents = find_equivalent_fg_ops(cuda_op, fg_agg)

        if source == 'cuda' and fg_equivalents:
            fg_time = sum(fg_agg[op]['total_time_ns'] for op in fg_equivalents)
            cuda_time = cuda_data['total_time_ns']

            if cuda_time > 0:
                ratio = fg_time / cuda_time
                if ratio < 1.0 - ratio_threshold:
                    # FlagGems is faster
                    fg_raw_kernels = []
                    for op in fg_equivalents:
                        fg_raw_kernels.extend(fg_agg[op]['raw_kernels'])

                    faster_ops.append({
                        'unified_name': cuda_op,
                        'cuda_kernels': cuda_data['raw_kernels'],
                        'fg_kernels': fg_raw_kernels,
                        'cuda_time': cuda_time,
                        'fg_time': fg_time,
                        'ratio': ratio,
                    })

    # Sort by speedup (lowest ratio = most speedup)
    faster_ops.sort(key=lambda x: x['ratio'])

    for op in faster_ops:
        print(f"\n[{op['unified_name']}] Ratio: {op['ratio']:.3f} (CUDA: {format_time(op['cuda_time'])}, FG: {format_time(op['fg_time'])})")
        print("  CUDA kernels:")
        for k in set(op['cuda_kernels']):
            print(f"    - {k}")
        print("  FlagGems kernels:")
        for k in set(op['fg_kernels']):
            print(f"    - {k}")


def main():
    parser = argparse.ArgumentParser(description='Compare CUDA and FlagGems kernel performance')
    parser.add_argument('--cuda', required=True, help='CUDA kernel CSV file')
    parser.add_argument('--flaggems', required=True, help='FlagGems kernel CSV file')
    parser.add_argument('--output', default='comparison.csv', help='Output CSV file')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Ratio threshold for Similar category (default: 0.1 = 10%%)')

    args = parser.parse_args()

    print(f"Comparing kernels:")
    print(f"  CUDA: {args.cuda}")
    print(f"  FlagGems: {args.flaggems}")

    results = compare_kernels(args.cuda, args.flaggems, args.threshold)

    # Sort by total time (max of CUDA and FlagGems)
    results.sort(key=lambda x: max(x['CUDA_Time_ns'], x['FlagGems_Time_ns']), reverse=True)

    write_csv(results, args.output)
    print(f"\nResults written to: {args.output}")

    print_summary(results)

    # Print FlagGems_faster original kernel names
    print_flaggems_faster_kernels(args.cuda, args.flaggems, args.threshold)


if __name__ == '__main__':
    main()
