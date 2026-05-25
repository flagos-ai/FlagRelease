# FlagRelease — FlagOS 自动化发布框架

FlagOS 模型迁移与发布的全自动化工作流框架，基于 Claude Code 驱动，支持 NVIDIA 和华为昇腾平台。

## 功能概览

从容器准备到镜像发布的 13 步全自动流程，零人工交互：

| 步骤 | 名称 | 说明 |
|------|------|------|
| 1 | 容器准备 | 自动识别容器/镜像 + 模型权重搜索 + 工具部署 |
| 2 | 环境检测 | 场景分类（native / vllm_flaggems / vllm_plugin_flaggems） |
| 3 | 启服务 | V1(native) + V2(flagos) 启动验证 |
| 4 | 精度评测 | V1/V2 GPQA Diamond 对比 |
| 5 | 精度算子调优 | 条件触发，分组排查定位问题算子 |
| 6 | 性能评测 | V1/V2 benchmark 对比 |
| 7 | 性能算子调优 | 条件触发，逐个禁用直到达标 |
| 8 | 自动发布 | 打包 + Harbor 上传（合格公开/不合格私有） |
| 9-13 | Plugin 验证 | 安装 → 启服务 → 精度 → 性能 → 发布 |

## 项目结构

```
├── skills/                     # 各步骤的 Skill 定义和工具脚本
│   ├── flagos-container-preparation/
│   ├── flagos-pre-service-inspection/
│   ├── flagos-service-startup/
│   ├── flagos-performance-testing/
│   ├── flagos-operator-replacement/
│   ├── flagos-eval-comprehensive/
│   ├── flagos-component-install/
│   ├── flagos-plugin-install/
│   ├── flagos-release/
│   ├── flagos-issue-reporter/
│   ├── flagos-log-analyzer/
│   └── shared/
├── prompts/                    # 流水线启动脚本
│   ├── run_pipeline.sh         # 单模型流水线
│   └── run_batch.sh            # 批量执行
├── shared/                     # 共享工具（报告生成、context 更新等）
├── CLAUDE.md                   # Claude Code 项目指令
└── settings.local.json         # 权限预配置
```

## 快速开始

```bash
# 单模型流水线（容器模式）
bash prompts/run_pipeline.sh --container <container_name> --model <model_name>

# 单模型流水线（镜像模式）
bash prompts/run_pipeline.sh --image <image:tag> --model <model_name>

# 批量执行
bash prompts/run_batch.sh
```

## 支持平台

- NVIDIA GPU（推荐：vllm >= 0.7.3 + flaggems >= 5.1.0 + flagtree >= 0.5.0）
- 华为昇腾 NPU（vllm_ascend 适配层）

## 环境要求

- Claude Code CLI
- Docker
- 环境变量：`HARBOR_USER`、`HARBOR_PASSWORD`、`MODELSCOPE_TOKEN`、`HF_TOKEN`、`GITHUB_TOKEN`（按需）

## 分支说明

| 分支 | 用途 |
|------|------|
| main | 稳定版本 |
| huawei | 华为昇腾适配开发 |
| test | 集成测试 |
| hygon | 海光适配开发 |
| metax | 沐曦适配开发 |
