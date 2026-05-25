# test 分支改进说明

> 基于 `main` 分支的增量改动，共 3 次提交，涉及 7 个文件。

---

## 一、流水线启动脚本增强（`prompts/run_pipeline.sh`）

### 新增 `--model-path` 参数
- 支持用户显式指定宿主机模型路径，跳过自动搜索逻辑
- 镜像模式下验证路径存在性；容器模式下自动忽略并给出警告
- 参数解析从固定位置改为 `while/case` 循环，支持任意顺序的可选参数

### 宿主机 Python 依赖预检
- 启动前自动检测 `pyyaml`、`huggingface_hub` 是否已安装
- 缺失时按阿里云 → 清华 → 腾讯镜像源顺序自动安装，全部失败则终止

### 兜底发布代码缩进修复
- 修正 `if/else` 分支内缩进错位问题，确保兜底 Harbor 发布逻辑正确执行

---

## 二、项目指令文档优化（`CLAUDE.md`）

### 路径统一
- 宿主机路径前缀从 `/data/` 统一为 `/mnt/data/`，与实际部署环境一致

### 算子调优触发条件明确化
- 步骤 5/7 增加 `env_type≠native` 前置条件，native 场景无 FlagGems 不触发调优

### Plugin 流程规则完善
- 新增 `plugin_workflow.crash_stopped` 状态标记
- 明确 Issue 路由方式：通过 `issue_reporter.py full --type plugin-error` 提交，仅保存本地文件
- 新增 Plugin 不达标发布规则：先提 issue，再私有上传 Harbor，不更新 README
- 新增镜像 tag 规则：追加 `-plugin` 后缀
- 禁止 Plugin 流程重新检测 GPU

### 约束规则重组
- 将原有 30 条约束按类别重新分组：工具使用、环境、GPU 资源管理、流程、算子控制、数据完整性
- 编号重排为 1-32，逻辑更清晰

### Plugin 算子控制策略升级
- 从黑名单（`VLLM_FL_FLAGOS_BLACKLIST`）改为白名单（`VLLM_FL_FLAGOS_WHITELIST`）
- 引入 `persist_op_config.py --auto` 将算子配置固化到容器 `/etc/environment`
- 步骤 10-12 优先使用 `start_service.sh --mode flagos` 自动加载固化变量

### 场景说明精简
- 移除 `vllm_flaggems` 和 `vllm_plugin_flaggems` 的冗长实现细节
- 改为引用对应 SKILL.md 文件，避免文档重复维护

---

## 三、报告生成工具增强（`shared/generate_report.py`）

### 新增 `--summary` 模式
- 输出精简摘要（含调优过程），适合快速查看流程状态

### `ledger_steps()` 方法重写
- 兼容 dict 格式（`key=step_number`）和 list 格式的台账数据
- 自动从 traces 补充 pending 步骤的实际状态
- 修正 `duration=0` 但有时间戳的步骤，自动计算耗时

### 新增 `_resolve_disabled_ops()` 辅助函数
- 统一解析禁用算子列表，支持多数据源 fallback：
  - `optimization.disabled_ops` → `operator_config.json` → `perf.disabled_ops`
- 兼容字符串和列表格式
- 自动区分精度禁用和性能禁用算子
- 提取搜索日志和优化后 ratio

### 环境类型显示优化
- Plugin 流程下区分显示主流程和 Plugin 环境类型

---

## 四、权限配置文件（`settings.local.json`）

- 新增并纳入版本管理，包含 Claude Code 运行所需的完整权限白名单
- 覆盖 docker、pip、curl、nvidia-smi、git、模型平台 CLI 等常用命令
- 支持宿主机文件操作（`/data/flagos-workspace/`）和 SSH 相关操作

---

## 五、容器准备脚本安全性修复（`setup_workspace.sh`）

- context 写入参数从字符串拼接改为数组构建 + `printf %q` 转义
- 防止模型名或路径中的空格/特殊字符导致命令注入

---

## 六、发布模块改进（`skills/flagos-release/tools/`）

### 配置生成（`config.py`）
- 评测指标名称支持多模式映射（`gpqa_diamond` → `GPQA_Diamond`、`erqa` → `ERQA`、`aime24` → `Aime24`）
- README 中的模型路径改为用户下载路径格式 `/data/{Model}-FlagOS`，与 ModelScope 下载命令一致

### 发布阶段（`publish.py`）
- 评测指标表格匹配增加模糊查找（忽略大小写、下划线、空格、括号）
- Plugin 评测指标名统一为 `GPQA_Diamond`，移除 `GPQA (plugin)` / `GPQA (tuned)` 等非标准命名
