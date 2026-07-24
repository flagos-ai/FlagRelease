# {{MODEL_NAME}} - 离线推理验证

## 项目说明

{{模型用途和架构的简要描述}}

## 环境准备

### 创建容器

```bash
docker run -d \
  --name {{CONTAINER}} \
  --gpus all \
  --ipc=host \
  --shm-size=128g \
  -v /mnt:/mnt \
  -v /data:/data \
  -v /etc/localtime:/etc/localtime:ro \
  {{IMAGE}} \
  sleep infinity
```

### 安装依赖

```bash
pip install {{依赖列表}}
```

## 模型路径

```
{{CONTAINER_MODEL_PATH}}
```

模型来源：HuggingFace `{{MODEL_ID}}`

## 推理入口文件

```
/root/run_inference.py
```

## 输入输出说明

### 输入格式

{{输入格式描述}}

示例（`/root/test_input.xxx`）：
```
{{输入样例}}
```

### 输出格式

{{输出格式描述，包含 shape、dtype 等}}

## 命令行参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | 是 | - | 模型目录路径 |
| `--input_file` | 是 | - | 输入文件路径 |
| `--output_file` | 否 | None | 输出保存路径 |
| `--device` | 否 | cuda:0 | 推理设备 |
| `--batch_size` | 否 | 32 | 批量大小 |
| {{额外参数}} | | | |

## 完整执行命令

```bash
python /root/run_inference.py \
  --model_path {{CONTAINER_MODEL_PATH}} \
  --input_file /root/test_input.xxx \
  --device cuda:0 \
  --output_file /root/output_xxx
```

## GPU 使用说明

- 推理默认使用 `cuda:0`
- 可通过 `--device` 参数指定 GPU 设备
- 实际验证 GPU：{{GPU_NAME}}
- 所有输入 tensor 和模型参数均在 CUDA 设备上运行

## flag_gems 接入位置

位于 `/root/run_inference.py` 文件顶部：

```python
try:
    import flag_gems
    flag_gems.enable(record=True, once=True, unused=[], path="/root/gems.txt")
except ImportError:
    pass
```

flag_gems 记录文件：`/root/gems.txt`

记录的算子：
{{算子列表}}

## 输入样例

文件 `/root/test_input.xxx`：

{{输入样例表格或内容}}

## 输出结果示例

```
{{实际运行的输出结果}}
```

## 跑通状态

**已跑通** - {{DATE}}

- {{验证项1}}
- {{验证项2}}
- flag_gems 记录正常生成
- GPU 推理验证通过

## 文件清单

| 文件 | 用途 |
|------|------|
| `/root/run_inference.py` | 推理脚本 |
| `/root/test_input.xxx` | 测试输入数据 |
| `/root/gems.txt` | flag_gems 算子记录 |
| `/root/README.md` | 本文档 |
| `/root/output_xxx` | 推理输出 |

## 注意事项

{{注意事项列表}}

- 不使用 vllm、不启动在线服务、不做 benchmark
- 评测组使用时替换 `--input_file` 指向评测数据即可
