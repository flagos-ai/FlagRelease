#!/bin/bash
# FlagOS 迁移结果汇总 — 调用 Claude Code 交互式分析
#
# 用法:
#   bash summarize.sh <batch_log_path>                              # 批量模式：交互式
#   bash summarize.sh --model <model_dir>                           # 单模型模式
#   bash summarize.sh --model <model_dir> --result-json result.json # 单模型结构化结果
#   bash summarize.sh <batch_log_path> --print --output report.md   # 非交互，直接输出文件
#
# 默认启动交互式 Claude 会话，可以实时对话追问。加 --print 则为非交互模式直接输出。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROMPT_FILE="${SCRIPT_DIR}/summarize_batch_results.md"
WORKSPACE="/data/flagos-workspace"
OUTPUT_FILE=""
BATCH_LOG=""
MODEL_DIR=""
MODEL_NAME=""
TASK_FILE=""
PRINT_MODE=false
RESULT_JSON=""
RESULT_TARGET=""
RESULT_VENDOR="unknown"
RESULT_OUTCOME=""
RESULT_EXIT_CODE=""
RESULT_ELAPSED_SECONDS=""

show_help() {
    echo "FlagOS 迁移结果汇总工具"
    echo ""
    echo "用法:"
    echo "  bash summarize.sh <batch_log_path> [选项]       批量模式"
    echo "  bash summarize.sh --model <model_dir> [选项]    单模型模式"
    echo ""
    echo "参数:"
    echo "  <batch_log_path>         批次日志文件路径（批量模式）"
    echo "  --model, -m DIR          模型目录路径（单模型模式，与 batch_log 二选一）"
    echo "  --result-json FILE       生成供进度汇报使用的结构化单模型结果"
    echo "  --task-file, -t FILE     任务列表文件（明确本次批量的模型范围）"
    echo "  --print, -p              非交互模式（直接输出报告，不启动对话）"
    echo "  --output, -o FILE        报告输出到文件（隐含 --print）"
    echo "  --workspace, -w DIR      工作空间根目录（默认 /data/flagos-workspace）"
    echo "  --help, -h               显示帮助"
    echo ""
    echo "模式说明:"
    echo "  默认(交互式): 启动 Claude 对话会话，数据读取自动进行，你可以实时追问"
    echo "  --print:      非交互，Claude 自动完成分析并输出报告"
    echo ""
    echo "示例:"
    echo "  bash summarize.sh batch_console.log                          # 交互式分析"
    echo "  bash summarize.sh batch_console.log --print -o report.md     # 输出到文件"
    echo "  bash summarize.sh --model /data/flagos-workspace/tiiuae/Falcon-H1-Tiny-90M-Instruct"
}

if [ $# -lt 1 ]; then
    show_help
    exit 1
fi

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --print|-p) PRINT_MODE=true; shift ;;
        --output|-o) OUTPUT_FILE="$2"; PRINT_MODE=true; shift 2 ;;
        --workspace|-w) WORKSPACE="$2"; shift 2 ;;
        --model|-m) MODEL_DIR="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --result-json) RESULT_JSON="$2"; shift 2 ;;
        --target) RESULT_TARGET="$2"; shift 2 ;;
        --vendor) RESULT_VENDOR="$2"; shift 2 ;;
        --outcome) RESULT_OUTCOME="$2"; shift 2 ;;
        --exit-code) RESULT_EXIT_CODE="$2"; shift 2 ;;
        --elapsed-seconds) RESULT_ELAPSED_SECONDS="$2"; shift 2 ;;
        --task-file|-t) TASK_FILE="$2"; shift 2 ;;
        --help|-h) show_help; exit 0 ;;
        -*) echo "未知参数: $1"; exit 1 ;;
        *)
            if [ -z "$BATCH_LOG" ]; then
                BATCH_LOG="$1"
            else
                echo "未知参数: $1"; exit 1
            fi
            shift ;;
    esac
done

# 结构化单模型分析模式：由专用分析器调用 Claude JSON Schema 输出并原子落盘。
# 该模式与原有交互式/Markdown 详细报告完全隔离。
if [ -n "$RESULT_JSON" ]; then
    if [ -z "$MODEL_DIR" ] || [ -z "$RESULT_OUTCOME" ] || [ -z "$RESULT_EXIT_CODE" ] || [ -z "$RESULT_ELAPSED_SECONDS" ]; then
        echo "错误: --result-json 需要同时提供 --model、--outcome、--exit-code、--elapsed-seconds" >&2
        exit 2
    fi
    if [ -z "$MODEL_NAME" ]; then
        MODEL_NAME="$(basename "$MODEL_DIR")"
    fi
    ANALYZER="${SCRIPT_DIR}/model_result_analyzer.py"
    RESULT_PROMPT="${SCRIPT_DIR}/summarize_model_result.md"
    if [ ! -f "$ANALYZER" ] || [ ! -f "$RESULT_PROMPT" ]; then
        echo "错误: 单模型结构化分析器或提示词不存在" >&2
        exit 2
    fi
    exec python3 "$ANALYZER" \
        --model-dir "$MODEL_DIR" \
        --model-name "$MODEL_NAME" \
        --result-json "$RESULT_JSON" \
        --prompt-file "$RESULT_PROMPT" \
        --target "$RESULT_TARGET" \
        --vendor "$RESULT_VENDOR" \
        --outcome "$RESULT_OUTCOME" \
        --exit-code "$RESULT_EXIT_CODE" \
        --elapsed-seconds "$RESULT_ELAPSED_SECONDS" \
        --timeout-seconds "${FLAGOS_MODEL_ANALYSIS_TIMEOUT_SECONDS:-900}" \
        --claude-command "${CLAUDE_COMMAND:-claude}"
fi

# 检测 Claude Code
if ! command -v claude &>/dev/null; then
    echo "错误: 未找到 claude 命令，请先安装 Claude Code"
    exit 1
fi

# 检测提示词文件
if [ ! -f "$PROMPT_FILE" ]; then
    echo "错误: 提示词文件不存在: $PROMPT_FILE"
    exit 1
fi

# 确定运行模式
RUN_MODE=""
if [ -n "$MODEL_DIR" ]; then
    if [ ! -d "$MODEL_DIR" ]; then
        echo "错误: 模型目录不存在: $MODEL_DIR"
        exit 1
    fi
    if [ ! -f "$MODEL_DIR/config/context_snapshot.yaml" ]; then
        echo "警告: 模型目录下未找到 config/context_snapshot.yaml"
    fi
    MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
    RUN_MODE="single"
elif [ -n "$BATCH_LOG" ]; then
    if [ ! -f "$BATCH_LOG" ]; then
        if [ -f "$(pwd)/${BATCH_LOG}" ]; then
            BATCH_LOG="$(pwd)/${BATCH_LOG}"
        else
            echo "错误: 批次日志文件不存在: $BATCH_LOG"
            exit 1
        fi
    fi
    BATCH_LOG="$(cd "$(dirname "$BATCH_LOG")" && pwd)/$(basename "$BATCH_LOG")"
    RUN_MODE="batch"
else
    echo "错误: 必须指定 <batch_log_path> 或 --model <model_dir>"
    show_help
    exit 1
fi

echo "═══════════════════════════════════════"
echo "  FlagOS 迁移结果汇总"
echo "═══════════════════════════════════════"
if [ "$RUN_MODE" = "single" ]; then
    echo "  运行模式: 单模型"
    echo "  模型目录: ${MODEL_DIR}"
else
    echo "  运行模式: 批量"
    echo "  批次日志: ${BATCH_LOG}"
    echo "  工作空间: ${WORKSPACE}"
    if [ -n "$TASK_FILE" ]; then
        echo "  任务列表: ${TASK_FILE}"
    fi
fi
if $PRINT_MODE; then
    echo "  交互模式: 非交互（直接输出）"
    echo "  输出文件: ${OUTPUT_FILE:-终端}"
else
    echo "  交互模式: 交互式（可实时对话）"
fi
echo "═══════════════════════════════════════"
echo ""

# 读取提示词
PROMPT_CONTENT=$(cat "$PROMPT_FILE")

# 构建 HEADER
if [ "$RUN_MODE" = "single" ]; then
    FULL_PROMPT="${PROMPT_CONTENT//\{\{BATCH_LOG_PATH\}\}/（单模型模式，无批次日志）}"

    HEADER="运行模式: 单模型分析
模型目录: ${MODEL_DIR}

仅分析上述单个模型目录。不需要搜索其他模型。
- 板块A：仅输出该模型的概览信息、耗时和费用（不需要跨模型汇总表）
- 板块B：输出该模型的完整流程详述
- 板块C：仅分析该模型遇到的问题
"
else
    FULL_PROMPT="${PROMPT_CONTENT//\{\{BATCH_LOG_PATH\}\}/$BATCH_LOG}"

    HEADER="运行模式: 批量分析
工作空间根目录: ${WORKSPACE}
批次日志路径: ${BATCH_LOG}
"
    if [ -n "$TASK_FILE" ]; then
        TASK_FILE="$(cd "$(dirname "$TASK_FILE")" && pwd)/$(basename "$TASK_FILE")"
        MODELS_LIST=""
        while IFS='|' read -r T M || [ -n "$T" ]; do
            T=$(echo "$T" | xargs)
            M=$(echo "$M" | xargs)
            [[ -z "$T" || "$T" == \#* ]] && continue
            MODELS_LIST="${MODELS_LIST}  - ${M}\n"
        done < "$TASK_FILE"
        HEADER="${HEADER}
任务列表文件: ${TASK_FILE}
本次批量执行的模型列表（仅分析以下模型，不要分析其他模型）:
$(echo -e "$MODELS_LIST")
"
    fi
fi

HEADER="${HEADER}
请输出所有板块的完整内容（板块A + 板块B + 板块C）。
板块A包含结果总表+耗时汇总+费用汇总+Harbor镜像验证+Issue产出汇总。
板块B需要逐模型读取 traces/*.json 和 logs/claude_full_*.log 分析流程和根因。

格式要求：
- A2/A3 的 V5 列必须包含：V5配置来源（V5=V2/V5=V2+N个扩展/跳过）+ V5精度（分数和rel_drop%）
- A2/A3 V5精度取数优先级：results/gpqa_v5.json > eval.v5_score > operator_config_v5.json.v5_score > fallback推断（无禁用算子或扩展0个时V5精度=V2精度）
- A3 逐模型列出 V1~V5 各阶段状态，标明是否由网络问题导致失败
- A3.1 Harbor镜像发布验证：读取各模型 traces/08_release.json、traces/13_plugin_publish.json（或13_v3_release.json）、traces/15_v5_publish.json 中的 harbor_push action，逐模型验证每个成功V阶段是否正确上传镜像，标注异常
- A4 耗时按真实步骤名逐列统计（容器准备/环境检测/服务启动/精度评测/精度调优/性能评测/性能调优/打包发布/Plugin安装/Plugin服务/Plugin精度/Plugin性能/Plugin发布/V4减算子/V5扩展/V5发布）
- A6 Issue产出汇总：读取 results/issue_data_*.json 和 results/issue_*_flagos-ai_*.md，列出每个issue的类型、目标组件(FlagGems/vllm-plugin-FL)、问题摘要、涉及算子、产出阶段
- 板块B每个模型必须列出 V1~V5 全部五个阶段，跳过的写明原因（如「V3精度不达标，V4被跳过」）
- 板块B V5阶段必须写明配置来源、精度分数及数据来源（实测/继承V2）
"

FULL_PROMPT="${HEADER}
${FULL_PROMPT}"

# 部署权限白名单（参考 run_pipeline.sh 的处理逻辑）
# 查找项目根目录的 settings.local.json 并部署到 .claude/
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f "${PROJECT_ROOT}/settings.local.json" ]; then
    mkdir -p .claude && cp "${PROJECT_ROOT}/settings.local.json" .claude/settings.local.json
    echo "  ✓ 已部署权限白名单: ${PROJECT_ROOT}/settings.local.json"
fi

# 构建 --add-dir 参数
ADD_DIRS=()
if [ "$RUN_MODE" = "single" ]; then
    ADD_DIRS+=(--add-dir "$MODEL_DIR")
else
    ADD_DIRS+=(--add-dir "$WORKSPACE")
    BATCH_LOG_DIR="$(dirname "$BATCH_LOG")"
    if [ "$BATCH_LOG_DIR" != "$WORKSPACE" ]; then
        ADD_DIRS+=(--add-dir "$BATCH_LOG_DIR")
    fi
fi

if $PRINT_MODE; then
    # 非交互模式：直接输出结果
    CLAUDE_ARGS=(-p "$FULL_PROMPT" --output-format text --permission-mode auto --max-turns 3 "${ADD_DIRS[@]}")
    if [ -n "$OUTPUT_FILE" ]; then
        claude "${CLAUDE_ARGS[@]}" > "$OUTPUT_FILE"
        echo ""
        echo "✓ 汇总报告已写入: ${OUTPUT_FILE}"
    else
        claude "${CLAUDE_ARGS[@]}"
    fi
else
    # 交互式模式：启动对话会话，提示词作为系统指令
    # 写入临时文件避免命令行参数长度限制
    PROMPT_TMPFILE=$(mktemp /tmp/flagos_summarize_prompt.XXXXXX.md)
    echo "$FULL_PROMPT" > "$PROMPT_TMPFILE"
    trap "rm -f '$PROMPT_TMPFILE'" EXIT

    echo "正在启动交互式 Claude 会话..."
    echo "提示: 输入「开始分析」即可执行，完成后可继续追问。"
    echo "输入 'exit' 或按 Ctrl+C 退出。"
    echo ""
    claude --system-prompt-file "$PROMPT_TMPFILE" --permission-mode auto "${ADD_DIRS[@]}" \
        "请按照系统指令开始分析，完成后我可能会追问。"
fi
