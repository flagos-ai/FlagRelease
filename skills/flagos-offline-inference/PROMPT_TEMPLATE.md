# 离线推理模型适配 — 提示词模板

## 使用方式

将下方模板中的 `{{变量}}` 替换为实际值后，作为 Claude Code 的输入提示词。

---

## 精简提示词（推荐）

适用于已有容器的场景：

```
在容器 "{{CONTAINER}}" 中完成模型 {{MODEL_ID}} 的离线推理适配。

模型路径：{{MODEL_PATH}}

按照 skills/flagos-offline-inference/SKILL.md 执行。
```

---

## 完整提示词（含镜像创建）

适用于需要从镜像创建容器的场景：

```
使用镜像 {{IMAGE}} 创建容器 "{{CONTAINER}}"，完成模型 {{MODEL_ID}} 的离线推理适配。

模型路径：{{MODEL_PATH}}

按照 skills/flagos-offline-inference/SKILL.md 执行。
```

---

## 详细提示词（不依赖 SKILL.md 时使用）

当 Claude 无法读取 SKILL.md 或需要独立使用时：

```
在容器 "{{CONTAINER}}" 中完成模型 {{MODEL_ID}} 的离线推理适配验证。

模型路径：{{MODEL_PATH}}
基础镜像：{{IMAGE}}（已有容器时删除此行）

任务目标：离线 inference 跑通 + flag_gems 接入 + GPU 验证 + README 沉淀。
不使用 vllm，不启动在线服务，不做 benchmark。

执行步骤：
1. 环境准备：确认容器可用，安装缺失依赖
2. 模型分析：读取 config.json 和官方文档，确定架构、推理方式、输入输出格式
3. 推理脚本：编写 /root/run_inference.py
   - 顶部加入 flag_gems（try/except 保护，path="/root/gems.txt"）
   - 统一 argparse 接口：--model_path, --input_file, --output_file, --device, --batch_size
   - 禁止硬编码路径或数据
4. 测试数据：创建 /root/test_input.xxx（独立文件，3-5 条代表性样本）
5. GPU 验证：实际运行推理，确认 CUDA 设备，保存输出
6. README：生成 /root/README.md，包含完整执行命令、输入输出说明、实际输出示例

产出规范：
- 所有文件放在容器 /root 下
- 不修改原始模型仓库
- 测试数据独立于代码，支持替换
- README 需保证评测组可独立复现
```

---

## 变量说明

| 变量 | 必填 | 示例 |
|------|------|------|
| `{{CONTAINER}}` | 是 | `thaonguyen` |
| `{{MODEL_ID}}` | 是 | `thaonguyen217/farm_molecular_representation` |
| `{{MODEL_PATH}}` | 是 | `/mnt/data/models/farm_molecular_representation` |
| `{{IMAGE}}` | 否 | `nvcr.io/nvidia/pytorch:26.04-py3` |

---

## 与原始提示词的对比

| 维度 | 原始提示词 | 优化后 |
|------|-----------|--------|
| 长度 | ~1500 字 | 精简版 ~50 字 / 详细版 ~300 字 |
| 重复 | "不使用 vllm" 出现 3 次 | 仅声明 1 次 |
| 规范 | 内联在提示词中 | 抽离到 SKILL.md |
| 可复用性 | 每次重写 | 填变量即可 |
| 遗漏风险 | 高（手写易遗漏） | 低（SKILL.md 有检查清单） |
