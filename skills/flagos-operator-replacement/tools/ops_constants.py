#!/usr/bin/env python3
"""
共享常量 — 算子分组定义

供 operator_optimizer.py / diagnose_ops.py 等脚本复用，避免双份维护。
"""

# =============================================================================
# OOT 高层算子（plugin 场景独立控制）
# =============================================================================

OOT_OPERATORS = [
    "silu_and_mul",      # SiLU 激活 + 逐元素乘
    "rms_norm",          # RMS 归一化
    "rotary_embedding",  # 旋转位置编码
    "fused_moe",         # 融合 MoE
    "attention_backend", # Attention 实现选择
]


OPERATOR_GROUPS = {
    "compute": [
        "addmm", "mm", "bmm", "linear", "matmul",
        "conv2d", "conv_depthwise2d",
    ],
    "memory": [
        "copy_", "zero_", "zeros", "ones", "ones_like", "full", "fill_scalar_",
        "clone", "to_copy", "empty_like", "new_zeros", "new_ones",
    ],
    "math": [
        "cos", "sin", "pow_scalar", "reciprocal", "exp", "log", "sqrt", "rsqrt",
        "abs", "neg", "tanh", "sigmoid", "gelu", "silu", "relu",
        "add", "sub", "mul", "div", "add_scalar", "sub_scalar", "mul_scalar",
        "div_scalar",
        "floor_div",
    ],
    "index": [
        "gather", "scatter", "scatter_add_0", "index", "index_select",
        "embedding", "slice_scatter", "select_scatter",
    ],
    "reduce": [
        "cumsum", "sort", "sort_stable", "argmax", "arange_start",
        "sum", "mean", "max", "min", "softmax", "log_softmax",
        "layer_norm", "group_norm",
    ],
}

# 运行时函数名 -> aten 算子名 映射（常见的不一致项）
RUNTIME_TO_ATEN_MAP = {
    "arange_start": "arange.start",
    "arange_start_step": "arange.start_step",
    "add_scalar": "add.Scalar",
    "sub_scalar": "sub.Scalar",
    "mul_scalar": "mul.Scalar",
    "div_scalar": "div.Scalar",
    "pow_scalar": "pow.Scalar",
    "pow_tensor_scalar": "pow.Tensor_Scalar",
    "fill_scalar_": "fill_.Scalar",
    "scatter_add_0": "scatter_add",
    "sort_stable": "sort.stable",
    "to_copy": "_to_copy",
    "conv_depthwise2d": "_conv_depthwise2d",
    "new_zeros": "new_zeros",
    "new_ones": "new_ones",
    "floor_div": "floor_divide",
}

# aten 算子名 -> 运行时函数名 反向映射
ATEN_TO_RUNTIME_MAP = {v: k for k, v in RUNTIME_TO_ATEN_MAP.items()}

# =============================================================================
# 算子性能影响力分级（LLM 推理场景先验知识）
# =============================================================================
# high:   推理热路径核心算子，FlagGems 实现可能有显著性能差异
# medium: 有一定调用频次，偶尔影响性能
# low:    几乎不影响推理吞吐

OP_RISK_LEVELS = {
    "high": [
        "addmm", "mm", "bmm", "linear", "matmul",  # compute 核心
        "softmax", "layer_norm", "rms_norm",          # reduce 高频
    ],
    "medium": [
        "embedding", "gather", "index_select",        # index 常用
        "gelu", "silu", "sigmoid",                     # 激活函数
        "sum", "mean", "log_softmax",                  # reduce 次要
        "conv2d",                                      # compute 次要
    ],
    "low": [
        # memory 组
        "copy_", "zero_", "zeros", "ones", "ones_like", "full", "fill_scalar_",
        "clone", "to_copy", "empty_like", "new_zeros", "new_ones",
        # math 基础运算
        "cos", "sin", "pow_scalar", "reciprocal", "exp", "log", "sqrt", "rsqrt",
        "abs", "neg", "tanh", "relu",
        "add", "sub", "mul", "div", "add_scalar", "sub_scalar", "mul_scalar", "div_scalar",
        "floor_div",
        # index 低频
        "scatter", "scatter_add_0", "index", "slice_scatter", "select_scatter",
        # reduce 低频
        "cumsum", "sort", "sort_stable", "argmax", "arange_start",
        "max", "min", "group_norm",
        # compute 低频
        "conv_depthwise2d",
    ],
}
