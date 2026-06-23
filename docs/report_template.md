> **自动化流程产出镜像版本：**
> - V1：tree版本=基础版：只带flagtree不开启任何flagos组件
> - V2：tree+gems=Pro版：开启flaggems且性能达到V1的80%，与V1的精度误差在5%以内
> - V3：tree+gems+plugin=Max版：在V2的基础上安装使用plugin，且性能达到V1的80%，与V1的精度误差在5%以内
> - V4：tree+gems+plugin=Flag-express版：在V3的基础上，性能表现超过V1版本
> - V5：tree+gems(应开尽开)+plugin=Royal Megamaster交付版本：携带了所有的FlagOS组件，所有算子能开尽开，只要服务能够顺利启动就ok

# 基本信息

| 项目 | 内容 |
|------|------|
| 开始时间 | 2026.06.04 08:00 |
| gems+tree版本上传时间 | 2026.06.04 14:00 |
| plugin上传时间 | 2026.06.04 16:30 |
| 模型 | LLM-Research/Llama-3.2-1B-Instruct |
| 权重数制 | bfloat16 |
| 计算数制（默认权重数制） | bfloat16 |
| 推理框架后端 | vllm |
| 推理框架后端版本 | 0.13.0 |
| 推理框架插件plugin-FL | 0.1.0 |
| FlagGems版本 | 5.0.1rc0 |
| Flagtree版本 | 0.5.0 |
| FlagCX版本 |  |
| 厂商 | Ascend |
| GPU | 910B : 8 x 64GB |
| 容器 | Llama-3.2-1B-Instruct_flagos |
| release自动化工具版本 | v0.1.0 |

# 算子替换列表

## 初始算子替换列表（默认全开）
共替换13个算子
[DEBUG] flag_gems.ops.zeros: GEMS ZEROS
[DEBUG] flag_gems.ops.arange: GEMS ARANGE
[DEBUG] flag_gems.ops.div: GEMS TRUE_DIVIDE
[DEBUG] flag_gems.ops.reciprocal: GEMS RECIPROCAL
[DEBUG] flag_gems.ops.mul: GEMS MUL
[DEBUG] flag_gems.ops.cos: GEMS COS
[DEBUG] flag_gems.ops.sin: GEMS SIN
[DEBUG] flag_gems.ops.cat: GEMS CAT
[DEBUG] flag_gems.ops.mm: GEMS MM
[DEBUG] flag_gems.ops.addmm: GEMS ADDMM
[DEBUG] flag_gems.ops.bmm: GEMS BMM
[DEBUG] flag_gems.ops.rms_norm: GEMS RMS_NORM
[DEBUG] flag_gems.ops.softmax: GEMS SOFTMAX

## enable 算子数（flaggems include 列表）
```yaml
include:
  - add
  - addmm
  - arange
  - bmm
  - cat
  - cos
  - div
  - mm
  - mul
  - reciprocal
  - rms_norm
  - sin
  - softmax
  - sub
  - zeros
```

## 发布算子替换列表（达标版本）
共替换8个算子
[DEBUG] flag_gems.ops.zeros: GEMS ZEROS
[DEBUG] flag_gems.ops.arange: GEMS ARANGE
[DEBUG] flag_gems.ops.div: GEMS TRUE_DIVIDE
[DEBUG] flag_gems.ops.reciprocal: GEMS RECIPROCAL
[DEBUG] flag_gems.ops.mul: GEMS MUL
[DEBUG] flag_gems.ops.cos: GEMS COS
[DEBUG] flag_gems.ops.sin: GEMS SIN
[DEBUG] flag_gems.ops.cat: GEMS CAT

## 禁用算子
- addmm
- bmm
- mm
- rms_norm
- softmax

# 评测结果

## 精度评测

### V1
| 数据集 | 评测条数 | 正确率(%) | 开启算子数 | FlagOS配置 |
|--------|---------|-----------|-----------|-----------|
| GPQA_Diamond | 50 | 23.0 | - | - |

### V2
| 数据集 | 评测条数 | 正确率(%) | 开启算子数 | FlagOS配置 |
|--------|---------|-----------|-----------|-----------|
| GPQA_Diamond | 50 | 23.0 | 8 | Pro |

### V3
| 数据集 | 评测条数 | 正确率(%) | 开启算子数 | FlagOS配置 |
|--------|---------|-----------|-----------|-----------|
| GPQA_Diamond | 50 | 21.0 | 8 | Max |

### V4
| 数据集 | 评测条数 | 正确率(%) | 开启算子数 | FlagOS配置 |
|--------|---------|-----------|-----------|-----------|
| GPQA_Diamond | - | - | - | - |

### V5
| 数据集 | 评测条数 | 正确率(%) | 开启算子数 | FlagOS配置 |
|--------|---------|-----------|-----------|-----------|
| GPQA_Diamond | 50 | 22.0 | 29 | Royal Megamaster |

### 结果对比
| 对比项 | 结果 |
|--------|------|
| V1 VS V2 | 精度偏差 0.0% |
| V1 VS V3 | 精度偏差 2.0% |
| V1 VS V4 | - |
| V2 VS V3 | 精度偏差 2.0% |

## 性能评测

### V1
| 模型名 | 厂商 | TFLOPS（单卡） | 卡数 | TFLOPS（单卡） × 卡数 | 4k-1k 64并发 - mean TTFT（ms） | 4k-1k 64并发 - P99 TTFT（ms） | 4k-1k 64并发 - output toks/s | 4k-1k 64并发 - total tok/s | 4k-1k 64并发 - Mean TPOT (ms) | 开算子数 | FlagOS配置 | 单算力吞吐 |
|--------|------|---------------|------|---------------------|------|------|------|------|------|------|------|------|
| LLM-Research/Llama-3.2-1B-Instruct | Ascend | 296 | 8 | 2368 | 11402 | 22015 | 808.54 | 4109.7 | 63.27 | 0 | - | 1.735515 |

### V2
| 模型名 | 厂商 | TFLOPS（单卡） | 卡数 | TFLOPS（单卡） × 卡数 | 4k-1k 64并发 - mean TTFT（ms） | 4k-1k 64并发 - P99 TTFT（ms） | 4k-1k 64并发 - output toks/s | 4k-1k 64并发 - total tok/s | 4k-1k 64并发 - Mean TPOT (ms) | 开算子数 | FlagOS配置 | 单算力吞吐 |
|--------|------|---------------|------|---------------------|------|------|------|------|------|------|------|------|
| LLM-Research/Llama-3.2-1B-Instruct | Ascend | 296 | 8 | 2368 | 11402 | 22015 | 720.5 | 3650.2 | 63.27 | 8 | Pro | 1.541470 |

### V3
| 模型名 | 厂商 | TFLOPS（单卡） | 卡数 | TFLOPS（单卡） × 卡数 | 4k-1k 64并发 - mean TTFT（ms） | 4k-1k 64并发 - P99 TTFT（ms） | 4k-1k 64并发 - output toks/s | 4k-1k 64并发 - total tok/s | 4k-1k 64并发 - Mean TPOT (ms) | 开算子数 | FlagOS配置 | 单算力吞吐 |
|--------|------|---------------|------|---------------------|------|------|------|------|------|------|------|------|
| LLM-Research/Llama-3.2-1B-Instruct | Ascend | 296 | 8 | 2368 | - | - | - | - | - | - | Max | - |

### V4
| 模型名 | 厂商 | TFLOPS（单卡） | 卡数 | TFLOPS（单卡） × 卡数 | 4k-1k 64并发 - mean TTFT（ms） | 4k-1k 64并发 - P99 TTFT（ms） | 4k-1k 64并发 - output toks/s | 4k-1k 64并发 - total tok/s | 4k-1k 64并发 - Mean TPOT (ms) | 开算子数 | FlagOS配置 | 单算力吞吐 |
|--------|------|---------------|------|---------------------|------|------|------|------|------|------|------|------|
| LLM-Research/Llama-3.2-1B-Instruct | Ascend | 296 | 8 | 2368 | - | - | - | - | - | - | Flag-express | - |

### V5
| 模型名 | 厂商 | TFLOPS（单卡） | 卡数 | TFLOPS（单卡） × 卡数 | 4k-1k 64并发 - mean TTFT（ms） | 4k-1k 64并发 - P99 TTFT（ms） | 4k-1k 64并发 - output toks/s | 4k-1k 64并发 - total tok/s | 4k-1k 64并发 - Mean TPOT (ms) | 开算子数 | FlagOS配置 | 单算力吞吐 |
|--------|------|---------------|------|---------------------|------|------|------|------|------|------|------|------|
| LLM-Research/Llama-3.2-1B-Instruct | Ascend | 296 | 8 | 2368 | 11402 | 22015 | 690.3 | 3480.1 | 63.27 | 29 | Royal Megamaster | 1.469637 |

### 结果对比
| 对比项 | 结果 |
|--------|------|
| V1 VS V2 | 性能比 88.8% |
| V1 VS V3 | - |
| V1 VS V4 | - |
| V2 VS V3 | - |

# 流程耗时与消费

| 项目 | 内容 |
|------|------|
| 流程耗时 | 6h 0m 0s |
| 流程消费 | — |

# 发布信息

- Harbor 镜像
  - V1：（阶段一手动发布）
  - V2：harbor.baai.ac.cn/flagrelease-public/flagrelease-ascend-release-model_llama-3.2-1b-instruct-tree_0.5.0-gems_5.0.1rc0-cx_none-python_3.11.14-torch_npu-2.8.0-pcp_cann2.8.0-gpu_ascend001-arc_arm64-driver_25.2.3:202506041400-v2
  - V3：-
  - V4：-
  - V5：-

- ModelScope: https://www.modelscope.cn/models/FlagRelease/Llama-3.2-1B-Instruct-ascend-FlagOS
- HuggingFace: https://huggingface.co/FlagRelease/Llama-3.2-1B-Instruct-ascend-FlagOS

# 结论

- 流自动化程结论：✅ 流程已达标
- gems+tree上传正常：✅
- Plugin 上传正常：❌ 不合格
- 该模型目前是否达到正常模型发布标准（是否安装plugin成功）：否

# 提交的 Issue
1. [性能下降] Operator performance degradation on Ascend
  - 描述：During performance benchmarking on model LLM-Research/Llama-3.2-1B-Instruct, the following operators were identified as causing significant performance degradation (<80% of native baseline).
  - 复现步骤：
    1. Set up environment on Ascend
    2. Install dependencies (PyTorch + Triton + FlagGems + vLLM)
    3. Run performance benchmark with V1 (native) and V2 (FlagGems enabled)
    4. Compare V1 vs V2 throughput
    5. Run operator search optimization
2. [性能下降] Operator performance degradation: addmm, bmm, mm, rms_norm, softmax on Ascend
  - 描述：During performance benchmarking, the following operators were identified as causing significant performance degradation.
  - 复现步骤：
    1. Set up environment on Ascend
    2. Install dependencies
    3. Run performance benchmark
    4. Compare V1 vs V2 throughput
    5. Run operator search optimization to identify problematic operators
3. [Plugin 错误] vllm-plugin-FL error on ascend
  - 描述：vllm-plugin-FL error encountered on model LLM-Research/Llama-3.2-1B-Instruct. Related components: vllm-plugin-FL.
  - 复现步骤：
    1. Set up environment on ascend
    2. Install dependencies (PyTorch + Triton + FlagGems + vLLM)
    3. Install vllm-plugin-FL (git clone + pip install --no-build-isolation)
    4. Start vLLM inference service with plugin enabled
    5. Observe plugin-related errors

---

报告生成时间：2026.06.23
