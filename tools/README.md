# Release Notes Generator

统一的 Release Notes 生成工具，用于 flagos-ai 组织下的各个仓库。

## 功能特性

- 自动获取两个版本间的 merged PR 列表
- 根据 PR 标题前缀或标签自动分类
- 生成符合主流开源社区标准的 Markdown 格式
- 支持自定义配置（分类规则、模板）
- 显示贡献者统计
- 支持首次发布（从仓库创建开始）

## 安装要求

### 必需依赖

- **Python 3.8+**
- **GitHub CLI (`gh`)** - 已安装并登录
- **PyYAML** - Python 库

```bash
# 安装 PyYAML
pip install pyyaml

# 确保 gh 已安装并登录
gh auth status
```

### 验证安装

```bash
# 检查 gh 是否安装
which gh
gh --version

# 检查 Python 和 PyYAML
python3 -c "import yaml; print('PyYAML OK')"
```

## 使用方法

### 基本用法

```bash
# 生成两个版本间的 release notes
python3 generate_release_notes.py \
    --repo flagos-ai/FlagScale \
    --from v0.9.0 \
    --to v1.0.0-alpha.0

# 保存到文件
python3 generate_release_notes.py \
    --repo flagos-ai/FlagScale \
    --from v0.9.0 \
    --to v1.0.0-alpha.0 \
    -o release_notes.md

# 首次发布（没有前置版本）
python3 generate_release_notes.py \
    --repo flagos-ai/FlagScale \
    --to v0.1.0 \
    --first-release
```

### 命令行参数

| 参数 | 说明 | 必需 |
|------|------|------|
| `--repo` | 仓库名称，格式：`owner/repo` | 是 |
| `--from` | 起始版本（tag、分支或 commit） | 否（与 `--first-release` 二选一） |
| `--to` | 结束版本（tag、分支或 commit） | 是 |
| `--first-release` | 首次发布，从仓库创建开始 | 否 |
| `-o, --output` | 输出文件路径（默认输出到 stdout） | 否 |
| `--config` | 自定义配置文件路径 | 否 |
| `--method` | PR 获取方法：`date` 或 `commit` | 否（默认 `date`） |

### 使用自定义配置

```bash
python generate_release_notes.py \
    --repo flagos-ai/FlagScale \
    --from v0.9.0 \
    --to v1.0.0 \
    --config release_config.yaml
```

## PR 分类规则

工具根据 PR 标题前缀或标签自动分类：

| 前缀 | 分类 |
|------|------|
| `[Feature]`, `[Model]`, `[Add]` | New Features |
| `[Fix]`, `[BugFix]` | Bug Fixes |
| `[Perf]`, `[Optimization]` | Performance |
| `[Doc]`, `[Docs]` | Documentation |
| `[CI]`, `[CICD]` | CI/Infrastructure |
| `[Refactor]`, `[Improve]` | Improvements |
| `[Breaking]` | Breaking Changes |
| `[Test]`, `[Benchmark]` | Testing |
| `[MLU]`, `[CUDA]`, `[NPU]` | Hardware Support |

## 输出格式示例

### 常规发布

```markdown
# Release v1.0.0

**Changes since v0.9.0**

## New Features
- Add new model support ([#123](https://github.com/owner/repo/pull/123)) by @author

## Bug Fixes
- Fix memory leak ([#124](https://github.com/owner/repo/pull/124)) by @author

## Contributors
Thanks to all contributors who made this release possible:
- @author1 (3 PRs)
- @author2
```

### 首次发布

```markdown
# Release v0.1.0

**Initial Release**

## New Features
- Initial implementation ([#1](https://github.com/owner/repo/pull/1)) by @author

## Contributors
Thanks to all contributors who made this release possible:
- @author1
```

## 配置文件说明

配置文件使用 YAML 格式，主要包含：

- `categories`: 分类规则定义
- `hardware_prefixes`: 硬件相关前缀列表
- `output`: 输出格式配置
- `header_template`: Release notes 头部模板

详细配置示例请参考 `release_config.yaml`。

## 示例：为 flagos-ai 仓库生成 Release Notes

### FlagScale

```bash
python generate_release_notes.py \
    --repo flagos-ai/FlagScale \
    --from v0.9.0 \
    --to v1.0.0-alpha.0 \
    -o flagscale_release.md
```

### FlagTree

```bash
python generate_release_notes.py \
    --repo flagos-ai/flagtree \
    --from v0.3.0 \
    --to v0.4.0 \
    -o flagtree_release.md
```

## 注意事项

1. 首次运行需要确保 `gh` 已登录并有权限访问目标仓库
2. 如果两个版本间 PR 数量过多（>500），可能需要调整 `--limit` 参数
3. 对于私有仓库，需要有相应的访问权限

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

Apache License 2.0
