#!/usr/bin/env python3
"""stream_filter.py — 解析 claude --output-format stream-json 事件流，实时打印进度信息
并可选地将关键步骤信息写入 pipeline.log（人可读的流程执行记录）。

默认精简模式：只显示步骤标记、关键结果、重要命令，过滤噪音。
--verbose 模式：恢复全量输出，用于调试。

用法:
  claude -p "..." --output-format stream-json | python3 stream_filter.py
  claude -p "..." --output-format stream-json | python3 stream_filter.py --pipeline-log /path/to/pipeline.log
  claude -p "..." --output-format stream-json | python3 stream_filter.py --verbose
  claude -p "..." --output-format stream-json | python3 stream_filter.py --no-color
"""

import sys
import os
import json
import re
import argparse
import time
from datetime import datetime


# ============================================================================
# pipeline.log 匹配规则（不受 --verbose 影响，始终按规则写入）
# ============================================================================

# 步骤标记: [步骤1] ~ [步骤13]（5/7为条件触发的算子调优步骤，9-13为条件触发的 plugin 流程）
RE_STEP = re.compile(r'\[步骤\d{1,2}\]')

# 段完成标记: [段1] ... / [段2] ... / [段3] ... / [段4] ...
RE_SEG_DONE = re.compile(r'\[段[1234]\].*(?:完成|结束|已同步)')

# 段边框标记: ╔ ╚ ║ ┌ └ │ 开头的行（段分隔框）
RE_SEG_BORDER = re.compile(r'^\s*[╔╚║┌└│]')

# 关键结果行: ✓ / ✗ / ⚠ 开头（去掉前导空格后）
RE_RESULT = re.compile(r'^\s*[✓✗⚠]')

# 流程分隔线: ===
RE_SEPARATOR = re.compile(r'={3,}')

# 关键判定词
RE_VERDICT = re.compile(
    r'qualified|达标|不达标|偏差|\bratio\b|精度|性能|service_ok|accuracy_ok|performance_ok|'
    r'env_type=|flaggems=|公开发布|私有发布',
    re.IGNORECASE
)

# 时间戳前缀检测: [2026-04-09 15:10:07] 格式
RE_HAS_TIMESTAMP = re.compile(r'^\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}')


# ============================================================================
# 精简模式 — 终端显示过滤规则
# ============================================================================

# 英文填充语（Claude 的自言自语）
RE_FILLER = re.compile(
    r'^(Let me|Now let me|Continuing|Good[,.]|Need to|I\'ll|Let\'s|'
    r'Checking|Looking|Searching|Trying|Running|'
    r'The |This |That |It |Same |Another |'
    r'Now |OK[,.]|Alright|Starting|First[, ]|'
    r'Container |Model |Weight |Docker |Service |'
    r'No |Yes |Got |Found |Moving|Verify|'
    r'Wait |A |An |Process |FlagGems is |vLLM |'
    r'I |We |My |Our |'
    r'Good |Great |Perfect |Done |'
    r'Hmm|Interesting|Actually|However|'
    r'Still |Already |Also |Just |'
    r'Here |There |These |Those |'
    r'After |Before |During |While |'
    r'Since |Because |So |But |And |Or |'
    r'Performance |Accuracy |Config |'
    r'```)',
    re.IGNORECASE
)

# 关键信号词 — 包含这些的行始终显示
SIGNAL_KEYWORDS = [
    '步骤', '✓', '✗', '⚠', '达标', '不达标', 'qualified', 'ratio=', 'ratio ',
    'env_type', 'flaggems=', '偏差', '性能', '精度', '公开发布', '私有发布',
    'service_ok', 'accuracy_ok', 'performance_ok', '报告', '迁移报告',
    '耗时', 'TPS', 'V1', 'V2', 'V3', '算子',
    'ops,', 'ops)', 'ops ', 'excluded', 'disabled', 'round ', 'deviation', 'baseline',
]

# 关键 Bash 命令关键词 — 只有匹配这些的才在终端显示
# 注意：docker exec 本身太宽泛（会匹配 ls/find/cat/heredoc 写入等探测命令）
# 所以只匹配 docker exec + 关键脚本/工具的组合
SHOW_COMMANDS = [
    # 容器生命周期
    'docker inspect', 'docker start', 'docker restart',
    'docker commit', 'docker tag', 'docker push',
    # 关键工具脚本（docker exec 内执行）
    'setup_workspace', 'inspect_env', 'toggle_flaggems',
    'wait_for_service', 'fast_gpqa', 'benchmark_runner',
    'performance_compare', 'operator_search', 'operator_optimizer', 'diagnose_ops',
    'eval_monitor', 'install_component', 'issue_reporter',
    # 外部服务
    'modelscope', 'huggingface-cli',
    # GPU / 服务
    'nvidia-smi', 'vllm serve', 'sglang',
]

# docker exec 中应隐藏的探测/写入命令（精简模式下，优先于 SHOW_COMMANDS）
HIDE_DOCKER_PATTERNS = [
    'cat >', 'cat >>', 'cat <<',  # heredoc 写入（context.yaml, trace）
    '"ls ', '"find ', '"grep ', '"head ', '"tail ',  # 探测命令
    '"mkdir ', '"cp ', '"mv ', '"rm ', '"kill ', '"pkill ',  # 文件/进程操作
    '"pgrep ', '"sleep ', '"echo ',  # 辅助
    'python3 -c',  # 内联 python 脚本
    'pip show', 'pip list',  # 包查询
    '"cat /root/', '"cat /workspace/', '"cat /README',  # 文件读取
]

# pipeline.log 中记录的关键命令（不受 verbose 影响）
LOG_COMMANDS = [
    'setup_workspace', 'inspect_env', 'toggle_flaggems',
    'wait_for_service', 'fast_gpqa', 'benchmark_runner',
    'performance_compare', 'operator_search', 'operator_optimizer', 'diagnose_ops',
    'docker commit', 'docker tag', 'docker push',
    'modelscope upload', 'huggingface-cli upload',
]

# 完全隐藏的工具类型（精简模式下）
HIDE_TOOLS = {'Read', 'Write', 'Edit', 'Glob', 'Grep', 'TaskCreate', 'TaskUpdate', 'TaskList', 'Agent', 'AskUserQuestion'}


# ============================================================================
# ANSI 颜色
# ============================================================================

class Colors:
    """ANSI 颜色码，可通过 enabled 开关全局关闭"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f'{code}{text}\033[0m'

    def blue_bold(self, text: str) -> str:
        return self._wrap('\033[1;34m', text)

    def green(self, text: str) -> str:
        return self._wrap('\033[32m', text)

    def red(self, text: str) -> str:
        return self._wrap('\033[31m', text)

    def yellow(self, text: str) -> str:
        return self._wrap('\033[33m', text)

    def gray(self, text: str) -> str:
        return self._wrap('\033[90m', text)


# ============================================================================
# 进度条
# ============================================================================

# 步骤定义：id, 显示名, 进度条标签
PIPELINE_STEPS = [
    ('1', '容器准备',   '[1]'),
    ('2', '环境检测',   '[2]'),
    ('3', '服务启动',   '[3]'),
    ('4', '精度评测',   '[4]'),
    ('5', '精度调优',   '[5]'),
    ('6', '性能评测',   '[6]'),
    ('7', '性能调优',   '[7]'),
    ('8', '打包发布',   '[8]'),
    ('9', 'Plugin安装', '[9]'),
    ('10', 'Plugin服务', '[10]'),
    ('11', 'Plugin精度', '[11]'),
    ('12', 'Plugin性能', '[12]'),
    ('13', 'Plugin发布', '[13]'),
]

# 匹配 [步骤X] 开始 / 完成 / 失败 / 跳过（支持两位数步骤号）
RE_STEP_START = re.compile(r'\[步骤(\d{1,2})\].*(?:开始|启动)')
RE_STEP_DONE = re.compile(r'\[步骤(\d{1,2})\].*(?:完成|结束)')
RE_STEP_FAIL = re.compile(r'\[步骤(\d{1,2})\].*失败')
RE_STEP_SKIP = re.compile(r'\[步骤(\d{1,2})\].*跳过')


class ProgressBar:
    """13 步进度条，实时刷新在终端（5/7为条件触发的算子调优步骤，9-13为条件触发的 plugin 流程）"""

    # 状态符号
    SYM_PENDING = '○'
    SYM_RUNNING = '●'
    SYM_DONE = '✓'
    SYM_FAIL = '✗'
    SYM_SKIP = '⚠'

    def __init__(self, colors: Colors, enabled: bool = True, start_step: int = 1,
                 load_durations: str = None):
        self.colors = colors
        self.enabled = enabled
        # 每步状态: pending / running / done / fail
        self.states = ['pending'] * len(PIPELINE_STEPS)
        self.step_start_times = [None] * len(PIPELINE_STEPS)
        self.step_durations = [None] * len(PIPELINE_STEPS)
        self.workflow_start = None
        self.last_render = ''
        # 分段执行：将 start_step 之前的步骤标记为 done
        if start_step > 1:
            for i in range(min(start_step - 1, len(PIPELINE_STEPS))):
                self.states[i] = 'done'
        # 加载前序段的耗时数据
        if load_durations and os.path.isfile(load_durations):
            try:
                with open(load_durations) as f:
                    prev = json.load(f)
                for k, v in prev.items():
                    idx = int(k) - 1
                    if 0 <= idx < len(self.step_durations) and v is not None:
                        self.step_durations[idx] = v
            except Exception:
                pass

    def _normalize(self, step_id: str) -> int:
        """步骤标识 → 索引 (0-based)"""
        for i, (sid, _, _label) in enumerate(PIPELINE_STEPS):
            if sid == step_id:
                return i
        return -1

    def on_step_start(self, step_id: str):
        idx = self._normalize(step_id)
        if idx < 0:
            return
        self.states[idx] = 'running'
        self.step_start_times[idx] = time.time()
        if self.workflow_start is None:
            self.workflow_start = time.time()
        self.render()

    def on_step_done(self, step_id: str):
        idx = self._normalize(step_id)
        if idx < 0:
            return
        self.states[idx] = 'done'
        if self.step_start_times[idx]:
            self.step_durations[idx] = time.time() - self.step_start_times[idx]
        self.render()

    def on_step_fail(self, step_id: str):
        idx = self._normalize(step_id)
        if idx < 0:
            return
        self.states[idx] = 'fail'
        if self.step_start_times[idx]:
            self.step_durations[idx] = time.time() - self.step_start_times[idx]
        self.render()

    def on_step_skip(self, step_id: str):
        idx = self._normalize(step_id)
        if idx < 0:
            return
        self.states[idx] = 'skip'
        if self.step_start_times[idx]:
            self.step_durations[idx] = time.time() - self.step_start_times[idx]
        self.render()

    def process_text(self, text: str):
        """从 assistant 文本中检测步骤状态变化"""
        for line in text.split('\n'):
            # 终态优先：完成/失败/跳过 优先于 开始
            # 避免步骤名含"启动"时误匹配 START（如 "[步骤10] Plugin 服务启动 — 完成"）
            m = RE_STEP_DONE.search(line)
            if m:
                self.on_step_done(m.group(1))
                continue
            m = RE_STEP_FAIL.search(line)
            if m:
                self.on_step_fail(m.group(1))
                continue
            m = RE_STEP_SKIP.search(line)
            if m:
                self.on_step_skip(m.group(1))
                continue
            m = RE_STEP_START.search(line)
            if m:
                self.on_step_start(m.group(1))

    def _format_duration(self, seconds: float) -> str:
        if seconds is None:
            return ''
        m = int(seconds) // 60
        s = int(seconds) % 60
        if m > 0:
            return f'{m}m{s}s'
        return f'{s}s'

    def render(self):
        """渲染一行紧凑进度到终端（不覆盖，每次状态变化追加一行）"""
        if not self.enabled:
            return

        c = self.colors
        parts = []
        done_count = sum(1 for s in self.states if s in ('done', 'fail', 'skip'))
        total = len(PIPELINE_STEPS)

        for i, (sid, name, label) in enumerate(PIPELINE_STEPS):
            state = self.states[i]
            dur = self.step_durations[i]
            dur_str = f'({self._format_duration(dur)})' if dur else ''

            if state == 'done':
                parts.append(c.green(f'{self.SYM_DONE}{label}{dur_str}'))
            elif state == 'fail':
                parts.append(c.red(f'{self.SYM_FAIL}{label}{dur_str}'))
            elif state == 'skip':
                parts.append(c.yellow(f'{self.SYM_SKIP}{label}{dur_str}'))
            elif state == 'running':
                elapsed = ''
                if self.step_start_times[i]:
                    elapsed = f'({self._format_duration(time.time() - self.step_start_times[i])})'
                parts.append(c.yellow(f'{self.SYM_RUNNING}{label}{elapsed}'))
            else:
                parts.append(c.gray(f'{self.SYM_PENDING}{label}'))

        pct = int(done_count / total * 100)
        elapsed_total = ''
        if self.workflow_start:
            elapsed_total = f' 总{self._format_duration(time.time() - self.workflow_start)}'

        line = f'  ⏱ {" ".join(parts)}  {pct}%{elapsed_total}'

        # 只在内容变化时输出（避免重复行）
        # 比较时去掉 ANSI 颜色码和动态耗时，只比较状态结构
        state_key = tuple(self.states)
        if not hasattr(self, '_last_state_key') or state_key != self._last_state_key:
            print(line, flush=True)
            self._last_state_key = state_key
            self.last_render = line

    def render_summary(self):
        """渲染各步骤耗时汇总（段结束时调用）"""
        if not self.enabled:
            return
        if not any(d is not None for d in self.step_durations):
            return
        c = self.colors
        print('', flush=True)
        print('各步骤耗时:', flush=True)
        for i, (sid, name, _label) in enumerate(PIPELINE_STEPS):
            state = self.states[i]
            dur = self.step_durations[i]
            if state == 'pending':
                continue
            dur_str = self._format_duration(dur) if dur else '-'
            sym = {'done': '✓', 'fail': '✗', 'skip': '⚠'}.get(state, '●')
            line = f'  {sym} 步骤{sid} {name}: {dur_str}'
            if state == 'done':
                line = c.green(line)
            elif state == 'fail':
                line = c.red(line)
            elif state == 'skip':
                line = c.yellow(line)
            print(line, flush=True)

    def save_durations(self, filepath: str):
        """将当前所有步骤耗时写入 JSON 文件，供后续段加载"""
        if not filepath:
            return
        data = {}
        for i, (sid, _, _) in enumerate(PIPELINE_STEPS):
            data[sid] = self.step_durations[i]
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass


# ============================================================================
# 过滤函数
# ============================================================================

def should_log(line: str) -> bool:
    """判断一行文本是否应写入 pipeline.log"""
    stripped = line.strip()
    if not stripped:
        return False
    if RE_STEP.search(stripped):
        return True
    if RE_SEG_DONE.search(stripped):
        return True
    if RE_SEG_BORDER.match(stripped):
        return True
    if RE_RESULT.match(stripped):
        return True
    if RE_SEPARATOR.match(stripped):
        return True
    if RE_VERDICT.search(stripped):
        return True
    return False


def should_display_line(line: str) -> bool:
    """精简模式：判断一行文本是否值得在终端显示"""
    stripped = line.strip()
    # 过滤空白和纯点号
    if not stripped or stripped in ('.', '..', '...'):
        return False
    # 段边框行 → 始终显示
    if RE_SEG_BORDER.match(stripped):
        return True
    # 包含关键信号词 → 始终显示（优先于填充语和纯英文过滤）
    if any(kw in stripped for kw in SIGNAL_KEYWORDS):
        return True
    # 英文填充语 → 过滤
    if RE_FILLER.match(stripped):
        return False
    # 步骤分隔线
    if RE_SEPARATOR.match(stripped) or stripped.startswith('━') or stripped.startswith('---'):
        return True
    # markdown 表格行
    if stripped.startswith('|'):
        return True
    # 纯英文句子（无中文字符）→ 大概率是 Claude 自言自语
    # 匹配大写字母或数字开头（如 "8x H20 visible..."）
    if re.match(r'^[A-Z0-9]', stripped) and not re.search(r'[\u4e00-\u9fff]', stripped):
        return False
    # 其余保留（中文内容、报告等）
    return True


def should_show_command(cmd: str) -> bool:
    """精简模式：判断 Bash 命令是否值得在终端显示"""
    # docker exec 中的探测/写入命令 → 隐藏（优先判定）
    if 'docker exec' in cmd:
        if any(pat in cmd for pat in HIDE_DOCKER_PATTERNS):
            return False
    # 宿主机辅助命令 → 隐藏
    cmd_stripped = cmd.strip()
    if cmd_stripped.startswith(('kill ', 'pkill ', 'sleep ', 'mkdir ', 'cp ', 'mv ',
                                'rm ', 'ls ', 'cat ', 'echo ', 'find ', 'grep ',
                                'head ', 'tail ', 'ln ', 'chmod ', 'chown ',
                                'WORKFLOW_START=', 'STEP', 'date ')):
        return False
    # 复合命令以辅助开头（如 "kill -9 xxx; sleep 3; ..."）
    if cmd_stripped.startswith(('kill -', 'sleep ', 'STEP')):
        return False
    # 匹配关键命令
    return any(kw in cmd for kw in SHOW_COMMANDS)


_PHASE_BANNERS = [
    ('vllm serve',           '🚀 服务启动中...'),
    ('-m sglang',            '🚀 服务启动中（SGLang）...'),
    ('wait_for_service',     '⏳ 等待服务就绪...'),
    ('fast_gpqa',            '📊 精度评测运行中（GPQA Diamond）...'),
    ('benchmark_runner',     '📊 性能评测运行中（Benchmark）...'),
    ('operator_search.py run', '🔧 性能算子调优中（自动搜索）...'),
    ('diagnose_ops.py',      '🔧 精度算子诊断中...'),
    ('toggle_flaggems.py --action enable',  '🔄 启用 FlagGems...'),
    ('toggle_flaggems.py --action disable', '🔄 关闭 FlagGems...'),
    ('toggle_flaggems.py --action modify-enable', None),  # 动态生成，见 _detect_phase_banner
    ('docker commit',        '📦 打包镜像中...'),
    ('docker push',          '📦 推送镜像中...'),
]
_last_phase_banner = ''


def _extract_ops_summary(cmd: str) -> str:
    """从 toggle_flaggems modify-enable 命令中提取算子摘要"""
    m = re.search(r"--disabled-ops\s+['\"]?([^'\"]+)['\"]?", cmd)
    if m:
        ops = m.group(1).strip().rstrip("'\"")
        if len(ops) > 40:
            count = len(ops.split(','))
            ops = ops[:35] + f'...共{count}个'
        return f'🔄 启用 FlagGems（排除 {ops}）...'
    m = re.search(r"--enabled-ops\s+['\"]?([^'\"]+)['\"]?", cmd)
    if m:
        ops = m.group(1).strip().rstrip("'\"")
        if len(ops) > 40:
            count = len(ops.split(','))
            ops = ops[:35] + f'...共{count}个'
        return f'🔄 启用 FlagGems（仅 {ops}）...'
    return '🔄 调整 FlagGems 算子...'


def _detect_phase_banner(cmd: str) -> str:
    """检测命令是否对应关键阶段，返回标题（连续相同标题不重复）"""
    global _last_phase_banner
    # 停止/检查命令不触发阶段标题
    if any(p in cmd for p in ('pkill', 'kill -', 'pgrep', 'docker restart', 'docker stop')):
        return ''
    for pattern, banner in _PHASE_BANNERS:
        if pattern in cmd:
            if banner is None:
                banner = _extract_ops_summary(cmd)
            if banner != _last_phase_banner:
                _last_phase_banner = banner
                return banner
            return ''
    return ''


def add_timestamp(line: str) -> str:
    """如果行没有时间戳前缀，自动添加"""
    stripped = line.strip()
    if RE_HAS_TIMESTAMP.match(stripped):
        return stripped
    # 段边框行不加时间戳（保持对齐）
    if RE_SEG_BORDER.match(stripped):
        return stripped
    if line.startswith('  ') and not RE_STEP.search(line):
        return stripped
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f'[{ts}] {stripped}'


def colorize_line(line: str, colors: Colors) -> str:
    """给终端输出行添加颜色"""
    stripped = line.strip()
    if not stripped:
        return line
    # 段边框（╔╚║┌└│）→ 醒目蓝色粗体
    if RE_SEG_BORDER.match(stripped):
        return colors.blue_bold(line)
    # 步骤标记
    if RE_STEP.search(stripped):
        return colors.blue_bold(line)
    # 成功
    if stripped.startswith('✓') or stripped.lstrip().startswith('✓'):
        return colors.green(line)
    # 失败
    if stripped.startswith('✗') or stripped.lstrip().startswith('✗'):
        return colors.red(line)
    # 警告
    if stripped.startswith('⚠') or stripped.lstrip().startswith('⚠'):
        return colors.yellow(line)
    # 命令
    if stripped.lstrip().startswith('▶'):
        return colors.gray(line)
    # 分隔线
    if stripped.startswith('━') or (RE_SEPARATOR.match(stripped) and len(stripped) > 10):
        return colors.gray(line)
    return line


# ============================================================================
# PipelineLogger（不变）
# ============================================================================

class PipelineLogger:
    """管理 pipeline.log 的写入"""

    def __init__(self, log_path: str = None):
        self.log_path = log_path
        self.log_file = None
        self.header_written = False

    def open(self):
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self.log_file = open(self.log_path, 'a', encoding='utf-8')
            # 分段执行时，段2/3 追加写入已有文件，跳过重复 header
            if self.log_file.tell() > 0:
                self.header_written = True

    def close(self):
        if self.log_file:
            self.log_file.close()

    def write_line(self, line: str):
        if not self.log_file:
            return
        self.log_file.write(line + '\n')
        self.log_file.flush()

    def write_header(self, model: str = '', container: str = ''):
        """写入流程开始头部"""
        if self.header_written or not self.log_file:
            return
        self.header_written = True
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.write_line(f'[{ts}] ===== FlagOS 迁移流程开始 =====')
        if model:
            self.write_line(f'  模型: {model}')
        if container:
            self.write_line(f'  容器: {container}')
        self.write_line('')

    def write_footer(self, duration_ms: int = 0, cost_usd: float = 0):
        """写入流程结束尾部"""
        if not self.log_file:
            return
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.write_line('')
        self.write_line(f'[{ts}] ===== FlagOS 迁移流程结束 =====')
        if duration_ms:
            minutes = duration_ms // 60000
            seconds = (duration_ms % 60000) // 1000
            self.write_line(f'  总耗时: {minutes}m {seconds}s')
        if cost_usd:
            self.write_line(f'  费用: ${cost_usd:.2f}')

    def process_text(self, text: str):
        """处理 assistant 文本块，提取关键行写入日志"""
        if not self.log_file:
            return
        for line in text.split('\n'):
            if should_log(line):
                logged = add_timestamp(line)
                self.write_line(logged)


def extract_model_container(prompt_text: str):
    """从 prompt 文本中提取模型名和容器名"""
    model = container = ''
    m = re.search(r'模型[名]?\s*[:：]\s*(\S+)', prompt_text)
    if m:
        model = m.group(1).rstrip('，,')
    m = re.search(r'容器[名]?\s*[:：]\s*(\S+)', prompt_text)
    if m:
        container = m.group(1).rstrip('，,')
    return model, container


# ============================================================================
# 终端日志记录器
# ============================================================================

class TerminalLogger:
    """记录所有终端输出到文件（去除 ANSI 颜色码）"""

    def __init__(self, log_path: str = None):
        self.log_path = log_path
        self.log_file = None

    def open(self):
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self.log_file = open(self.log_path, 'a', encoding='utf-8')

    def close(self):
        if self.log_file:
            self.log_file.close()

    def write(self, text: str, end: str = '\n'):
        """写入文本到日志文件（去除 ANSI 颜色码）"""
        if not self.log_file:
            return
        # 去除 ANSI 颜色码
        clean_text = re.sub(r'\x1b\[[0-9;]*m', '', text)
        self.log_file.write(clean_text + end)
        self.log_file.flush()


# 全局终端日志记录器（由 main() 初始化）
_terminal_logger = None


# ============================================================================
# 终端输出函数
# ============================================================================

def out(text: str, colors: Colors = None, end: str = '\n'):
    """输出到终端，可选着色，同时写入终端日志"""
    if colors:
        text = colorize_line(text, colors)
    print(text, end=end, flush=True)
    # 同时写入终端日志
    if _terminal_logger:
        _terminal_logger.write(text, end=end)


def format_result_compact(stdout: str, is_error: bool, last_cmd: str = '') -> str:
    """精简模式：压缩工具结果为一行摘要"""
    if not stdout:
        return ''
    lines = [l for l in stdout.strip().split('\n') if l.strip()]
    if not lines:
        return ''
    if is_error:
        # 错误：显示第一行有意义的错误信息
        for l in lines:
            s = l.strip()
            if s and len(s) > 3 and not s.startswith('{') and not s.startswith('['):
                return f'    \u2717 {s[:150]}'
        return f'    \u2717 {lines[0].strip()[:150]}'
    # 评测/性能结果：提取关键数值行
    if last_cmd and ('fast_gpqa' in last_cmd or 'benchmark_runner' in last_cmd or 'performance_compare' in last_cmd):
        key_lines = []
        for l in lines:
            s = l.strip()
            if re.search(r'得分|score|TPS|throughput|ratio|accuracy|正确率', s, re.IGNORECASE):
                key_lines.append(s)
        if key_lines:
            summary = ' | '.join(s.lstrip('| ').rstrip() for s in key_lines[:3])
            if len(lines) > 5:
                return f'    {summary[:140]}  ({len(lines)} lines)'
            return f'    {summary[:150]}'
    # 过滤无意义的单字符/括号结果
    last = lines[-1].strip()
    if len(last) <= 2 or last in ('}', ']', '{', '[', 'OK', 'ok'):
        # 尝试找一行有意义的
        for l in reversed(lines):
            s = l.strip()
            if len(s) > 3 and s not in ('}', ']', '{', '['):
                last = s
                break
    if len(lines) == 1:
        return f'    {last[:150]}'
    if len(lines) <= 5:
        return f'    {last[:150]}'
    return f'    {last[:120]}  ({len(lines)} lines)'


# ============================================================================
# 主循环
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Claude stream-json filter with pipeline.log')
    parser.add_argument('--pipeline-log', help='Path to pipeline.log for writing progress')
    parser.add_argument('--verbose', action='store_true', help='详细模式：显示全量输出（同旧版行为）')
    parser.add_argument('--no-color', action='store_true', help='关闭 ANSI 颜色')
    parser.add_argument('--start-step', type=int, default=1,
                        help='分段执行时的起始步骤编号（1-13），之前的步骤标记为已完成')
    parser.add_argument('--cost-file', help='写入本段费用和耗时的文件路径（供 run_pipeline.sh 汇总）')
    parser.add_argument('--terminal-log', help='记录处理后的终端输出到文件（去除 ANSI 颜色码）')
    parser.add_argument('--load-durations', help='加载前序段的步骤耗时 JSON 文件')
    parser.add_argument('--durations-file', help='段结束时写入本段步骤耗时到此 JSON 文件')
    args = parser.parse_args()

    verbose = args.verbose
    use_color = (not args.no_color) and sys.stdout.isatty()
    colors = Colors(enabled=use_color)

    global _terminal_logger
    _terminal_logger = TerminalLogger(args.terminal_log)
    _terminal_logger.open()

    logger = PipelineLogger(args.pipeline_log)
    logger.open()

    progress = ProgressBar(colors, enabled=not verbose, start_step=args.start_step,
                           load_durations=args.load_durations)

    # 跟踪上一个 tool_use 是否应该显示结果（精简模式用）
    last_tool_visible = False
    last_cmd = ''

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if verbose:
                    out(line)
                continue

            if not isinstance(event, dict):
                continue

            etype = event.get("type", "")

            # --- system init ---
            if etype == "system" and event.get("subtype") == "init":
                continue

            # --- assistant 消息 ---
            if etype == "assistant":
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                for block in message.get("content", []):
                    btype = block.get("type", "")

                    if btype == "text":
                        text = block["text"]

                        # pipeline.log 始终处理
                        if not logger.header_written and RE_STEP.search(text):
                            logger.write_header()
                        logger.process_text(text)

                        # 进度条更新
                        progress.process_text(text)

                        # --- 终端输出 ---
                        if verbose:
                            # verbose 模式：全量输出
                            out(text, end="")
                        else:
                            # 精简模式：逐行过滤
                            for tline in text.split('\n'):
                                if should_display_line(tline):
                                    # 步骤开始时插入分隔线
                                    if RE_STEP.search(tline) and '开始' in tline:
                                        sep = '━' * 50
                                        out(sep, colors)
                                    out(tline, colors)
                                    # 步骤开始标记后也加分隔线
                                    if RE_STEP.search(tline) and '开始' in tline:
                                        sep = '━' * 50
                                        out(sep, colors)

                    elif btype == "tool_use":
                        name = block.get("name", "")
                        inp = block.get("input", {})

                        if name == "Bash":
                            cmd = inp.get('command', '')[:200]

                            # 提前更新进度条状态（不输出可见文本，步骤标记由 Claude 文本输出驱动）
                            STEP1_CMD_HINTS = ['nvidia-smi', 'docker inspect', 'docker run', 'setup_workspace', 'check_model_local']
                            if any(kw in cmd for kw in STEP1_CMD_HINTS) and progress.states[0] == 'pending':
                                progress.on_step_start('1')

                            # pipeline.log 记录关键命令
                            if args.pipeline_log and logger.log_file:
                                if any(kw in cmd for kw in LOG_COMMANDS):
                                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    logger.write_line(f'[{ts}]   ▶ {cmd}')

                            # --- 终端输出 ---
                            if verbose:
                                out(f"\n  ▶ {cmd}")
                                last_tool_visible = True
                                last_cmd = cmd
                            else:
                                # 关键阶段自动注入醒目标题（仅对可见命令）
                                show_cmd = should_show_command(cmd)
                                banner = _detect_phase_banner(cmd) if show_cmd else ''
                                if banner:
                                    out('', colors)
                                    out(f'  ▸ {banner}', colors)
                                    out('', colors)
                                    if logger.log_file:
                                        logger.write_line(add_timestamp(f'▸ {banner}'))

                                if show_cmd:
                                    out(f"  ▶ {cmd}", colors)
                                    last_tool_visible = True
                                    last_cmd = cmd
                                else:
                                    last_tool_visible = False

                        elif name in HIDE_TOOLS:
                            if verbose:
                                if name in ("Read", "Write", "Edit", "Glob", "Grep"):
                                    path = inp.get("file_path", inp.get("pattern", ""))
                                    out(f"\n  ▶ [{name}] {path}")
                                else:
                                    out(f"\n  ▶ [{name}]")
                            last_tool_visible = verbose
                        else:
                            if verbose:
                                out(f"\n  ▶ [{name}]")
                            last_tool_visible = verbose

            # --- tool result ---
            elif etype == "user":
                result = event.get("tool_use_result", {})
                if isinstance(result, str):
                    stdout = result
                    is_error = False
                elif isinstance(result, dict):
                    stdout = result.get("stdout", "")
                    is_error = result.get("is_error", False)
                else:
                    stdout = ""
                    is_error = False

                if stdout:
                    if verbose:
                        # verbose：前 3 行预览
                        lines = stdout.strip().split("\n")
                        preview = "\n".join(lines[:3])
                        if len(lines) > 3:
                            preview += f"\n    ... ({len(lines)} lines total)"
                        out(f"    {preview}")
                    elif last_tool_visible:
                        # 精简模式：只显示可见命令的压缩结果
                        compact = format_result_compact(stdout, is_error, last_cmd)
                        if compact:
                            out(compact, colors)

            # --- 流程结束 ---
            elif etype == "result":
                # 将所有未完成的 running 步骤标记为 done（兜底）
                for i, s in enumerate(progress.states):
                    if s == 'running':
                        progress.states[i] = 'done'
                        if progress.step_start_times[i]:
                            progress.step_durations[i] = time.time() - progress.step_start_times[i]
                progress.render()

                # 输出各步骤耗时汇总
                progress.render_summary()

                # 保存耗时数据供后续段加载
                progress.save_durations(args.durations_file)

                dur = event.get("duration_ms", 0) or 0
                cost = event.get("total_cost_usd", 0) or 0
                minutes = int(dur // 60000)
                seconds = int((dur % 60000) // 1000)
                out(f"\n{'━' * 50}", colors)
                out(f"完成 — 耗时 {minutes}m {seconds}s, 费用 ${cost:.2f}", colors)
                logger.write_footer(duration_ms=dur, cost_usd=cost)

                # 写入费用文件供 run_pipeline.sh 汇总
                if args.cost_file:
                    try:
                        with open(args.cost_file, 'w') as cf:
                            cf.write(f"{cost:.4f}\n")
                    except Exception:
                        pass

    finally:
        if _terminal_logger:
            _terminal_logger.close()
        logger.close()


if __name__ == '__main__':
    main()
