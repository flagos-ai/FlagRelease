# Skills 概览

本文档整理了 FlagOS GPU 性能测试自动化框架中所有 Skill 的功能说明和执行顺序。

**支持双执行模式**：
- **Host 模式**：Claude Code 在宿主机，通过 `docker exec` 操作容器
- **Container 模式**：Claude Code 在容器内，直接执行命令

**支持多入口**：
- 已有容器 → 直接接入
- 已有镜像 → 创建容器
- README 链接 → 解析后创建容器

---

## 统一工作目录

**核心设计**：所有操作在统一挂载的 `/flagos-workspace` 目录下进行，宿主机可实时访问日志和结果。

```
宿主机: /data/flagos-workspace/<model_name>/
                      ↓ 挂载
容器内: /flagos-workspace/
    ├── scripts/              # 自动化脚本
    │   ├── benchmark_runner.py
    │   ├── performance_compare.py
    │   ├── operator_optimizer.py
    │   ├── operator_search.py
    │   ├── eval_monitor.py
    │   └── ...
    ├── results/              # 最终交付物
    │   ├── native_performance.json
    │   ├── flagos_performance.json
    │   ├── flagos_optimized.json
    │   ├── gpqa_native.json
    │   ├── gpqa_flagos.json
    │   ├── operator_config.json
    │   └── performance_compare.csv
    ├── traces/               # 每步留痕（JSON）
    ├── logs/                 # 运行日志
    ├── config/               # 使用的配置快照
    ├── perf/                 # 性能测试配置
    └── shared/
        └── context.yaml
```

---

## 工作流程图

### 新模型迁移发布

```
1 下载模型+容器准备   镜像/容器就绪 + 权重检查 + 环境检测 + 工具部署
        ↓
2 启服务             V1(native) + V2(flagos) 启动验证 → 异常自动 issue
        ↓
3 精度评测           V1/V2 GPQA Diamond 对比 → 异常自动 issue + ≤3 轮算子优化
        ↓
4 性能评测           V1/V2 4k1k benchmark 对比 → 异常自动 issue + ≤3 轮算子优化
        ↓
5 自动发布           打包 + 上传 → qualified 公开 / 不合格私有
        ↓
→ 报告整理收尾
```

**三版结果文件**：
- `native_performance.json` — V1 (Native，无 FlagGems)
- `flagos_performance.json` — V2 (FlagGems)
- `flagos_optimized.json` — V3 (Optimized FlagGems，仅 V2 不达标时产出)

**自动化**：步骤1~5全自动执行，零交互。仅网络失败时需用户介入：
- 网络失败时（pip 自动加阿里云镜像重试，其他操作询问代理）
- 5发布凭证通过环境变量自动读取（Harbor 登录、`MODELSCOPE_TOKEN`、`HF_TOKEN`、`GITHUB_TOKEN`）

**发布条件判定**：`qualified = service_ok AND accuracy_ok AND performance_ok`
- qualified → 公开发布；不合格 → 私有发布
- 提交了 issue 但优化成功 → 仍算合格

---

## Skills 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     主流程 (顺序执行)                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1 下载模型+容器准备         镜像/容器就绪 + 环境检测 + 工具部署       │
│     (container-preparation + pre-service-inspection)                │
│         ↓                                                           │
│  2 启服务                    V1/V2 启动验证 → 异常自动 issue          │
│     (service-startup)                                               │
│         ↓                                                           │
│  3 精度评测                  V1/V2 GPQA 对比 → issue + ≤3 轮优化     │
│     (eval-comprehensive)                                            │
│         ↓                                                           │
│  4 性能评测                  V1/V2 benchmark 对比 → issue + ≤3 轮优化│
│     (performance-testing)                                           │
│         ↓                                                           │
│  5 自动发布                  条件判定 → 打包 + 上传（公开/私有）       │
│     (flagos-release)                                                │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                     独立工具 (按需调用)                               │
├─────────────────────────────────────────────────────────────────────┤
│  operator-replacement       算子替换 + 分组二分搜索优化               │
│  component-install          组件安装/升级（FlagGems、FlagTree）       │
│  log-analyzer               日志分析 + 失败恢复指引                   │
│  issue-reporter             问题自动归因 + GitHub issue 提交          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Skills 详细说明

### 1 flagos-container-preparation + flagos-pre-service-inspection (下载模型+容器准备)

| 属性 | 说明 |
|------|------|
| **功能** | 自动识别入口类型（容器名/镜像/URL），检测 GPU，创建或接入容器；环境检测（env_type + FlagGems 控制方式 + 组件版本）；工具脚本部署 |
| **依赖** | 无 (流程起点) |
| **触发词** | `container preparation`, `prepare container`, `容器准备`, `环境准备` |

**三种入口**：

| 入口 | 用户提供什么 | 系统做什么 |
|------|-------------|-----------|
| 已有容器 | 容器名/ID | docker inspect → 验证 → 接入 |
| 已有镜像 | 镜像地址 + 模型信息 | docker run 创建 |
| README | URL 链接 | WebFetch → 解析 → docker pull + run |

---

### 2 flagos-service-startup (启服务)

| 属性 | 说明 |
|------|------|
| **功能** | V1(native) + V2(flagos) 启动验证、健康检查、异常自动提交 issue |
| **依赖** | 步骤1 |
| **触发词** | `service startup`, `start service`, `启动服务`, `health check` |

**启动模式**：default / native（关闭 FlagGems）/ flagos（启用 FlagGems）
**异常处理**：FlagGems 模式启动失败 → 提交 `operator-crash` issue → 标记 `workflow.service_ok: false` → 跳过3/4直接到5

---

### 3 flagos-eval-comprehensive (精度评测)

| 属性 | 说明 |
|------|------|
| **功能** | V1/V2 GPQA Diamond 精度对比（5% 阈值）+ 异常自动 issue + ≤3 轮算子优化 |
| **依赖** | 步骤2 |
| **触发词** | `精度评测`, `GPQA`, `fast gpqa`, `comprehensive eval`, `远端评测`, `FlagRelease`, `flageval` |

**异常处理**：V2精度下降 >5% → 提交 `accuracy-degraded` issue → 算子优化（≤3 轮）→ 超限标记 `workflow.accuracy_ok: false`

---

### 4 flagos-performance-testing (性能评测)

| 属性 | 说明 |
|------|------|
| **功能** | V1/V2 4k1k benchmark 对比 + 异常自动 issue + ≤3 轮算子优化 |
| **依赖** | 步骤3 |
| **触发词** | `性能测试`, `benchmark`, `vllm bench`, `吞吐量测试` |

**异常处理**：V2/V1 任一并发级别 <80% → 提交 `performance-degraded` issue → 算子优化（≤3 轮）→ 超限标记 `workflow.performance_ok: false`

**脚本**：
- `benchmark_runner.py`：测试入口，支持 `--strategy quick/fast/comprehensive/fixed` + 可选 `--final-burst`
- `performance_compare.py`：三版对比 + CSV 生成（`--format markdown` 标准三列表格输出）

---

### 5 flagos-release (自动发布)

| 属性 | 说明 |
|------|------|
| **功能** | 发布条件判定（qualified → 公开 / 不合格 → 私有）+ Docker 镜像 commit/tag/push Harbor + README 生成 + ModelScope/HuggingFace 上传 |
| **依赖** | 步骤4 |
| **触发词** | `发布`, `镜像上传`, `镜像打包`, `模型发布`, `release`, `publish` |

**条件判定**：`qualified = service_ok AND accuracy_ok AND performance_ok` → 公开发布；否则私有发布

---

### flagos-operator-replacement (算子替换 + 优化) — 独立工具

| 属性 | 说明 |
|------|------|
| **功能** | 被动排除（评测报错）+ 主动分组二分搜索优化（性能驱动） |
| **依赖** | 无 (可随时调用) |
| **触发词** | `operator replacement`, `replace operator`, `算子替换`, `算子优化` |

**脚本**：
- `operator_optimizer.py`：分组二分搜索引擎、算子列表自动发现
- `operator_search.py`：全自动搜索编排（next→toggle→restart→benchmark→update）

---

### flagos-log-analyzer (日志分析) — 独立工具

| 属性 | 说明 |
|------|------|
| **功能** | 分析日志，分类错误，推断服务状态，提供失败恢复指引 |
| **依赖** | 无 (可随时调用) |
| **触发词** | `log analysis`, `analyze logs`, `日志分析` |

**脚本**：
- `log_analyzer.py`：日志分析与诊断（analyze 单文件 / scan 目录扫描）

---

### flagos-issue-reporter (问题自动提交) — 独立工具

| 属性 | 说明 |
|------|------|
| **功能** | 收集算子/框架问题数据，格式化 Bug Report，提交 GitHub issue（gh CLI / API / 手动降级） |
| **依赖** | 无 (可随时调用) |
| **触发词** | `提交 issue`, `submit issue`, `report bug`, `自动报告` |

**五种 issue 类型**：`operator-crash`、`accuracy-zero`、`accuracy-degraded`、`performance-degraded`、`flagtree-error`

**脚本**：
- `issue_reporter.py`：问题收集/格式化/提交（collect / format / submit / full）

---

## 自动化程度

### 无需人工介入的环节

| 环节 | 说明 |
|------|------|
| GPU 检测 | 自动检测 10 种 GPU 厂商 |
| 入口类型判断 | 自动识别容器名/镜像/URL |
| FlagGems 集成方式 | 运行时多维探测 |
| FlagGems 启停方法 | 从探测结果推导 |
| 性能对比判断 | 自动计算比例 |
| 是否需要算子优化 | 自动判断 < 80% 触发 |
| 算子优化搜索 | 全自动搜索（≤3 轮），超限自动标记不合格 |
| 报告生成 | 自动生成 |
| 发布条件判定 | qualified → 公开 / 不合格 → 私有 |

### 需要人工介入的环节

1. 网络失败时（pip 自动加阿里云镜像重试，其他操作询问代理）

**注意**：5发布所需凭证（Harbor 登录、`MODELSCOPE_TOKEN`、`HF_TOKEN`、`GITHUB_TOKEN`）均通过环境变量自动读取，无需人工提供。

---

## 数据流

```
┌──────────────────────────────┐
│ container-preparation (1)    │──写入──┐
│ + pre-service-inspection     │        │
│ (容器准备 + 环境检测)         │        │
└──────────────────────────────┘        ↓
                                ┌─────────────────┐
                                │ context.yaml    │
                                │ (共享上下文)     │
                                └─────────────────┘
                                        ↑
┌──────────────────────────────┐        │
│ service-startup (2)         │──追加──┤
│ → V1/V2 启动验证             │        │
│ → 异常 → issue-reporter      │        │
└──────────────────────────────┘        │
         │                              ↑
         ↓                              │
┌──────────────────────────────┐        │
│ eval-comprehensive (3)       │──追加──┤
│ → V1/V2 精度对比             │        │
│ → 异常 → issue + ≤3 轮优化   │        │
└──────────────────────────────┘        │
         │                              ↑
         ↓                              │
┌──────────────────────────────┐        │
│ performance-testing (4)      │──追加──┘
│ → V1/V2 性能对比             │
│ → 异常 → issue + ≤3 轮优化   │
│ → performance_compare.csv    │
└──────────────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│ flagos-release (5)           │
│ → 条件判定 qualified?        │
│ → 打包 + 上传（公开/私有）    │
└──────────────────────────────┘

独立工具:
┌──────────────────┐  ┌──────────────────┐
│ log-analyzer     │  │ issue-reporter   │
│ + 失败恢复指引    │  │ + 自动 issue 提交 │
└──────────────────┘  └──────────────────┘
```

---

## GPU 厂商支持

| 厂商 | 检测命令 | 可见设备环境变量 |
|------|----------|------------------|
| NVIDIA | `nvidia-smi` | `CUDA_VISIBLE_DEVICES` |
| 华为 (Ascend) | `npu-smi info` | `ASCEND_RT_VISIBLE_DEVICES` |
| 海光 (Hygon) | `hy-smi` | `HIP_VISIBLE_DEVICES` |
| 摩尔线程 | `mthreads-gmi` | `MUSA_VISIBLE_DEVICES` |
| 昆仑芯 | `xpu-smi` | `XPU_VISIBLE_DEVICES` |
| 天数 | `ixsmi` | `CUDA_VISIBLE_DEVICES` |
| 沐曦 | `mx-smi` | `CUDA_VISIBLE_DEVICES` |
| 清微智能 | `tsm_smi` | `TXDA_VISIBLE_DEVICES` |
| 寒武纪 | `cnmon` | `MLU_VISIBLE_DEVICES` |
| 平头哥 | - | `CUDA_VISIBLE_DEVICES` |

---

## 关键配置文件

| 文件 | 容器内路径 | 用途 |
|------|-----------|------|
| `context.yaml` | `/flagos-workspace/shared/context.yaml` | Skill 间共享上下文 |
| `perf_config.yaml` | `/flagos-workspace/scripts/config/perf_config.yaml` | 性能测试配置 |
| `operator_config.json` | `/flagos-workspace/results/operator_config.json` | 算子优化状态 |
| `skills/*/SKILL.md` | 项目目录内 | Skill 定义文件 |

---

## 宿主机常用命令

```bash
# 实时查看服务日志
tail -f /data/flagos-workspace/<model>/logs/*.log

# 查看性能测试结果
cat /data/flagos-workspace/<model>/results/native_performance.json
cat /data/flagos-workspace/<model>/results/flagos_performance.json
cat /data/flagos-workspace/<model>/results/performance_compare.csv

# 查看精度评测结果
cat /data/flagos-workspace/<model>/results/gpqa_native.json
cat /data/flagos-workspace/<model>/results/gpqa_flagos.json

# 查看流程留痕
ls /data/flagos-workspace/<model>/traces/

# 查看评测进度
tail -f /data/flagos-workspace/<model>/logs/eval_gpqa_progress.log

# 查看算子优化状态
cat /data/flagos-workspace/<model>/results/operator_config.json

# 搜索错误日志
grep -ri "error" /data/flagos-workspace/<model>/logs/
```
