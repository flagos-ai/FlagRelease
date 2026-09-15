---
name: flagos-offline-inference
description: 离线推理模型适配验证，完成模型分析、推理脚本编写、flag_gems 接入、GPU 验证、README 沉淀
version: 1.0.0
triggers:
  - 离线推理
  - offline inference
  - 模型适配
  - inference 跑通
  - 推理验证
depends_on: []
next_skill: null
provides:
  - inference.script_path
  - inference.input_file
  - inference.output_file
  - inference.status
  - inference.gpu_device
---

# 离线推理模型适配 Skill

针对非 vLLM 类模型（如 embedding、分类、生成、扩散等），完成离线推理跑通验证与 flag_gems 接入，产出可复现的推理流程供评测组使用。

**不涉及**：vllm、在线服务、benchmark、plugin、性能评测。

---

# 用户输入

| 参数 | 必填 | 说明 |
|------|------|------|
| `CONTAINER` | 是 | 目标容器名（已存在）或待创建的容器名 |
| `MODEL_ID` | 是 | HuggingFace 模型 ID（如 `thaonguyen217/farm_molecular_representation`） |
| `MODEL_PATH` | 是 | 宿主机模型路径 |
| `IMAGE` | 否 | 基础镜像地址（已有容器时不需要） |

---

# 执行流程

## 步骤 1 — 环境准备

### 已有容器

```bash
docker inspect --type=container $CONTAINER > /dev/null 2>&1
```

确认容器运行中，检查模型路径是否已挂载。

### 从镜像创建

```bash
docker run -d \
  --name $CONTAINER \
  --gpus all \
  --ipc=host \
  --shm-size=128g \
  -v /mnt:/mnt \
  -v /data:/data \
  -v /etc/localtime:/etc/localtime:ro \
  $IMAGE sleep infinity
```

### 依赖检查

进入容器后确认基础依赖：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
```

缺失依赖按模型需求安装（使用阿里云镜像源）。

---

## 步骤 2 — 模型分析

**目标**：确定模型架构、推理方式、输入输出格式。

1. 读取模型目录结构：
   ```bash
   docker exec $CONTAINER ls $CONTAINER_MODEL_PATH/
   ```

2. 读取 `config.json` 确定模型架构（model_type、architectures）

3. 读取模型仓库 README（如有）确定官方推荐推理方式

4. 检查是否有官方推理脚本（`inference.py`、`predict.py`、`run.py` 等）

5. 确定：
   - 模型加载方式（AutoModel / 特定 Model 类）
   - 输入格式（文本/图片/SMILES/音频等）
   - 输出格式（embedding/logits/生成文本等）
   - tokenizer 类型
   - 特殊依赖

---

## 步骤 3 — 推理脚本编写

**产出**：`/root/run_inference.py`

### 强制规范

1. **文件顶部** flag_gems 接入（try/except 保护）：
   ```python
   try:
       import flag_gems
       flag_gems.enable(record=True, once=True, unused=[], path="/root/gems.txt")
   except ImportError:
       pass
   ```

2. **统一 argparse 接口**（至少包含）：
   - `--model_path`：模型目录路径（必填）
   - `--input_file`：输入文件路径（必填）
   - `--output_file`：输出保存路径（可选）
   - `--device`：推理设备，默认 `cuda:0`
   - `--batch_size`：批量大小，默认 32
   - 可根据模型特性添加额外参数

3. **GPU 验证**：打印实际使用的 device，确认模型和 tensor 在 CUDA 上

4. **禁止硬编码**：输入数据、模型路径、输出路径全部通过参数传入

5. **代码结构**：
   - `parse_args()` → 参数解析
   - `main()` → 加载模型 → 读取输入 → 推理 → 输出结果
   - 保持简洁，不过度封装

参考骨架见 `templates/run_inference_skeleton.py`。

---

## 步骤 4 — 测试数据准备

**产出**：`/root/test_input.xxx`（格式取决于模型输入类型）

| 模型输入类型 | 测试文件格式 | 文件名 |
|-------------|-------------|--------|
| 文本 | `.txt`，每行一条 | `test_input.txt` |
| JSON 结构化 | `.json`，数组格式 | `test_input.json` |
| CSV 表格 | `.csv`，含表头 | `test_input.csv` |
| 图片 | 图片文件 + 路径列表 | `test_input.txt`（路径列表） |

要求：
- 3-5 条有代表性的样本
- 覆盖典型场景
- 数据来源于模型官方示例或领域常识
- 独立文件，支持后续替换为评测数据

---

## 步骤 5 — GPU 推理验证

实际运行推理脚本：

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /root/run_inference.py \
  --model_path $CONTAINER_MODEL_PATH \
  --input_file /root/test_input.xxx \
  --device cuda:0 \
  --output_file /root/output_xxx"
```

### 验证清单

- [ ] 推理成功完成，无报错
- [ ] 输出结果合理（非全零、非 NaN）
- [ ] 确认 GPU 实际使用（非 CPU fallback）
- [ ] `gems.txt` 已生成（flag_gems 记录）
- [ ] 输出文件已保存（如指定了 --output_file）

### 常见问题处理

| 问题 | 处理方式 |
|------|---------|
| CUDA OOM | 减小 batch_size |
| 缺少依赖 | pip install（阿里云镜像） |
| tokenizer 报错 | 检查输入格式是否匹配 |
| 模型加载失败 | 检查权重文件完整性 |

---

## 步骤 6 — README 沉淀

**产出**：`/root/README.md`

使用 `templates/README_TEMPLATE.md` 格式，必须包含：

1. 项目说明（模型用途、架构）
2. 环境准备（容器创建命令、依赖安装）
3. 模型路径
4. 推理入口文件
5. 输入输出说明（格式、示例）
6. 命令行参数表
7. 完整执行命令
8. GPU 使用说明
9. flag_gems 接入位置与记录的算子
10. 输入样例
11. 输出结果示例（实际运行结果）
12. 跑通状态
13. 文件清单
14. 注意事项

**目标**：评测组基于 README 可独立复现推理流程。

---

# 产出文件清单

| 文件 | 位置 | 说明 |
|------|------|------|
| `run_inference.py` | `/root/` | 推理脚本 |
| `test_input.xxx` | `/root/` | 测试输入数据 |
| `README.md` | `/root/` | 完整文档 |
| `gems.txt` | `/root/` | flag_gems 算子记录（运行时生成） |
| `output_xxx` | `/root/` | 推理输出（运行时生成） |

---

# 质量检查清单

执行完成前逐项确认：

- [ ] `run_inference.py` 可独立运行，无硬编码路径
- [ ] 测试数据为独立文件，可替换
- [ ] GPU 推理已验证（打印了 device 信息）
- [ ] flag_gems 已接入且 gems.txt 有记录
- [ ] 输出结果合理
- [ ] README 包含完整执行命令
- [ ] README 包含实际输出示例
- [ ] 评测组可基于 README 独立复现

---

# 与主流程的关系

此 Skill 独立于 CLAUDE.md 中定义的 1-13 步主工作流。适用于：
- 宣传类模型适配
- 非 vLLM 模型验证
- 评测前的推理跑通确认

不触发性能评测、算子调优、镜像发布等后续流程。
