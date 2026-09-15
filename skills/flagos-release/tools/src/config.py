# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
配置管理模块
从 context.yaml 加载配置，并提供配置验证和自动填充
"""
import os
import sys
import json
import subprocess
from dataclasses import dataclass, field
from typing import List
import yaml

from .chip_detector import ChipDetector, ChipVendor, VENDOR_NAMES, sanitize_docker_tag

# 芯片厂商×型号统一规范表（项目 shared/ 目录）。用于命名后缀与厂商归一化。
# 缺失时降级：naming_suffix 回退原 vendor 名，normalize 回退原值（不改变旧行为）。
_chip_spec = None
try:
    _shared_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "shared")
    )
    if _shared_dir not in sys.path:
        sys.path.insert(0, _shared_dir)
    import chip_spec as _chip_spec  # type: ignore
except Exception:
    _chip_spec = None


@dataclass
class ChipConfig:
    """芯片配置"""
    # 芯片厂商，默认自动检测
    vendor: str = "auto"
    # Harbor 仓库地址
    harbor_registry: str = "harbor.baai.ac.cn/flagrelease-public"
    # 以下为内部使用，自动检测填充
    auto_generate_tag: bool = True
    tree: str = "none"
    gems_version: str = ""
    cx: str = "none"
    date_tag: str = ""
    # date_tag 的跨进程一致种子（原始 workflow_start 字符串，ISO8601）。
    # 由 from_context 从 context 的 timing.workflow_start 读入，auto_fill_config
    # 据此生成 date_tag，确保 push/README/upload 等多进程发布段拿到同一时间戳。
    date_tag_seed: str = ""
    driver_version: str = ""
    sdk_version: str = ""
    torch_version: str = ""
    python_version: str = ""
    gpu_model: str = ""


@dataclass
class PublishConfig:
    """发布阶段配置"""
    enabled: bool = True
    # 镜像发布
    tag_image: bool = True
    push_harbor: bool = True
    # README 生成
    generate_readme: bool = True
    readme_output_path: str = "./README.md"
    # 模型发布
    publish_modelscope: bool = True
    modelscope_model_id: str = ""
    modelscope_token: str = ""
    publish_huggingface: bool = True
    huggingface_repo_id: str = ""
    huggingface_token: str = ""
    # 权重文件上传
    upload_weights: bool = True
    weights_dir: str = ""
    # 自动读取评测结果目录（步骤4/5产出），填入 README
    results_dir: str = ""
    # 仓库可见性：恒为私有。发布点(publish.py)已硬编码私有，不再读此字段决定可见性，
    # 保留仅为向后兼容。禁止改 False 或据此恢复公开发布分支。
    private: bool = True
    # 已有的 Harbor 镜像地址（跳过 commit/tag/push）
    existing_harbor_image: str = ""
    # 步骤8 已发布的仓库 ID（plugin 模式下用于更新 README）
    base_modelscope_model_id: str = ""
    base_huggingface_repo_id: str = ""
    # 内部使用
    image_source: str = ""
    image_target_tag: str = ""
    harbor_path: str = ""
    readme_script_path: str = ""
    upload_files: List[str] = field(default_factory=list)


@dataclass
class ModelInfo:
    """模型信息配置"""
    # 必填：模型来源
    source_of_model_weights: str = ""  # 如 "Qwen/Qwen3-8B"
    # 可选：模型介绍
    new_model_introduction: str = ""
    # 可选：评测结果
    evaluation_results: List[dict] = field(default_factory=list)
    # 以下全部自动生成
    output_name: str = ""
    vendor: str = ""
    docker_version: str = ""
    ubuntu_version: str = ""
    flagrelease_name: str = ""
    flagrelease_name_pre: str = ""
    image_harbor_path: str = ""
    # README 中 docker pull 命令展示的镜像地址覆盖：
    # V4 发布时 README 仍推荐 V3(Max) 镜像（V4 是减算子实验版，交付推荐用 V3），
    # 由 from_context 从 versions.v3.image_url 回填。为空则回退 image_harbor_path。
    readme_image_override: str = ""
    container_run_cmd: str = ""
    serve_start_cmd: str = ""
    serve_infer_cmd: str = ""
    canonical_model_path: str = ""


@dataclass
class PipelineConfig:
    """完整的流水线配置"""
    input_type: str = "container"
    container_name: str = ""
    host_workspace_base: str = ""  # /data/flagos-workspace/<model>，由 context.yaml workspace.host_path 填充
    config_persisted: bool = False
    plugin_image_mode: bool = False  # plugin 模式：镜像 tag 追加 -plugin，仓库名追加 -plugin
    plugin_qualified: bool = False   # plugin 精度达标(accuracy_ok)即为 True→更新 README；性能不门控
    version_tag: str = "v2"          # 发布版本标签：v1/v2/v3/v4
    also_tag: str = ""               # 额外镜像 tag 版本（V2=V3 同镜像双 tag 场景）
    incompatible_tag: str = ""       # 不适配标记名（设置后只打标记不发布版本镜像）

    # 各阶段配置
    chip: ChipConfig = field(default_factory=ChipConfig)
    publish: PublishConfig = field(default_factory=PublishConfig)
    model_info: ModelInfo = field(default_factory=ModelInfo)

    # 执行哪些阶段
    stages_to_run: List[str] = field(default_factory=lambda: ["publish"])


def load_config_from_context(context_path: str) -> PipelineConfig:
    """从 FlagOS context.yaml 自动构建发布配置，无需手写 YAML 配置文件。

    context.yaml 是 FlagOS 工作流各步骤的共享状态，包含容器名、模型信息、
    评测结果、GPU 信息等。本函数将这些字段映射为 PipelineConfig。
    """
    with open(context_path, 'r', encoding='utf-8') as f:
        ctx = yaml.safe_load(f)

    config = PipelineConfig()
    config.input_type = 'container'
    config.container_name = ctx.get('container', {}).get('name', '')
    config.stages_to_run = ['publish']

    # ---- model_info ----
    model = ctx.get('model', {})
    config.model_info.source_of_model_weights = model.get('name', '')

    # evaluation_results
    ev = ctx.get('eval', {})
    if ev.get('v1_score') is not None and ev.get('v2_score') is not None:
        mode = ev.get('mode', 'gpqa_diamond')
        mode_to_metric = {'gpqa_diamond': 'GPQA_Diamond', 'erqa': 'ERQA', 'aime24': 'Aime24'}
        method = mode_to_metric.get(mode, 'GPQA_Diamond')
        config.model_info.evaluation_results = [
            {'metric': method, 'origin': ev['v1_score'], 'flagos': ev['v2_score']}
        ]

    # serve_start_cmd + container_run_cmd
    # 核心原则：README 中下载路径、docker run 挂载路径、vllm serve 模型路径三者必须一致
    # 统一使用 canonical_model_path 作为唯一模型路径
    import re
    svc = ctx.get('service', {})
    runtime = ctx.get('runtime', {})
    commands = ctx.get('commands', {})
    model_short = model.get('name', '').split('/')[-1] if model.get('name') else ''
    flagrelease_name = f"{model_short}-FlagOS" if model_short else ''
    canonical_model_path = f"/data/{flagrelease_name}" if flagrelease_name else model.get('container_path', '/data/model')

    if commands.get('serve_start'):
        serve_cmd = commands['serve_start']
        # 用正则替换 vllm serve 后的模型路径参数（第一个非 -- 参数）
        serve_cmd = re.sub(r'(vllm\s+serve\s+)\S+', rf'\1{canonical_model_path}', serve_cmd)
        # 替换端口为默认 8000
        serve_cmd = re.sub(r'--port\s+\d+', '--port 8000', serve_cmd)
        config.model_info.serve_start_cmd = serve_cmd
    else:
        tp = runtime.get('tp_size') or 1
        max_model_len = svc.get('max_model_len', '')
        cmd_parts = [f"vllm serve {canonical_model_path}",
                     f"--host 0.0.0.0 --port 8000",
                     f"--tensor-parallel-size {tp}",
                     f"--served-model-name {model_short}" if model_short else None,
                     "--trust-remote-code"]
        if max_model_len:
            cmd_parts.append(f"--max-model-len {max_model_len}")
        config.model_info.serve_start_cmd = " \\\n".join(p for p in cmd_parts if p)

    # container_run_cmd (优先从 context commands 读取实际命令)
    if commands.get('container_run'):
        run_cmd = commands['container_run']
        # 替换镜像为目标镜像占位符
        image_name = ctx.get('image', {}).get('name', '')
        if image_name:
            run_cmd = run_cmd.replace(image_name, '{{IMAGE}}')
        # 兜底：image.name 与实际 run 命令中的镜像不一致时替换落空，
        # 基础镜像名会原样流进 README（pull/run 不一致），按 harbor 镜像 token 定位兜底替换
        if '{{IMAGE}}' not in run_cmd:
            run_cmd = re.sub(r'harbor\S+', '{{IMAGE}}', run_cmd, count=1)
        # 移除 workspace 挂载（-v ...:/flagos-workspace）
        run_cmd = re.sub(r'\s*-v\s+\S+:/flagos-workspace', '', run_cmd)
        # 替换所有模型相关挂载为 -v /data:/data（canonical_model_path 在 /data/ 下，天然可达）
        container_path = model.get('container_path', '')
        local_path = model.get('local_path', '')
        if container_path:
            run_cmd = re.sub(r'-v\s+\S+:' + re.escape(container_path), '-v /data:/data', run_cmd)
        elif local_path:
            run_cmd = re.sub(r'-v\s+' + re.escape(local_path) + r':\S+', '-v /data:/data', run_cmd)
        # 如果命令中没有 -v /data:/data（原始命令无模型挂载或替换未命中），追加
        if '-v /data:/data' not in run_cmd and '-v /data:' not in run_cmd:
            run_cmd = re.sub(r'(docker\s+run\s+)', r'\1-v /data:/data ', run_cmd)
        # 替换容器名为通用名
        run_cmd = re.sub(r'--name[= ]\S+', '--name flagos', run_cmd)
        config.model_info.container_run_cmd = run_cmd
    else:
        config.model_info.container_run_cmd = (
            "docker run -itd --gpus=all --network=host "
            "-v /data:/data --name flagos {{IMAGE}}"
        )

    # 天数(Iluvatar)：ixsmi 由宿主机 corex 驱动提供，基础镜像的 /usr/local/corex 下
    # corex-<ver> 目录不含该工具，必须 bind-mount 进容器，否则容器内检不到 GPU
    # （平台校验报"容器内没有 ixsmi"）。对齐既有厂商做法：Ascend 挂 npu-smi、
    # Cambricon 挂 cnmon、Hygon 挂 /opt/hyhal。挂宿主机软链路径而非具体版本目录，
    # 可跨 corex 版本。刻意放在 if/else 之后统一注入：commands.container_run 缺失
    # 走兜底命令时同样补上，不留"换个口子又漏挂"的缺口；已含 ixsmi 则不重复注入。
    if ((ctx.get('gpu', {}) or {}).get('vendor', '') == 'iluvatar'
            and 'ixsmi' not in config.model_info.container_run_cmd):
        config.model_info.container_run_cmd = re.sub(
            r'\{\{IMAGE\}\}',
            '-v /usr/local/corex/bin/ixsmi:/usr/local/corex/bin/ixsmi {{IMAGE}}',
            config.model_info.container_run_cmd, count=1)

    # 保存 canonical_model_path 供模板使用
    config.model_info.canonical_model_path = canonical_model_path

    # ---- chip ----
    gpu = ctx.get('gpu', {})
    config.chip.vendor = gpu.get('vendor', 'auto')

    # flagtree 版本从 inspection 或 environment 读取
    inspection = ctx.get('inspection', {})
    flag_packages = inspection.get('flag_packages', {})
    environment = ctx.get('environment', {})
    flagtree_ver = flag_packages.get('flagtree', '') or ''
    if not flagtree_ver and environment.get('has_flagtree'):
        flagtree_ver = environment.get('flagtree_version', '')
    if flagtree_ver:
        config.chip.tree = flagtree_ver

    # date_tag 跨进程一致种子：run-scoped 的 workflow_start（见 auto_fill_config /
    # _tag_timestamp_from_seed）。发布分多进程执行（push/双tag/README/upload/一致性
    # 重试），各进程各自 now() 会错开时间戳，导致 README 的 docker pull 指向 Harbor
    # 上未推送的 tag。改由此稳定种子统一，缺失时 auto_fill 回退 now()。
    config.chip.date_tag_seed = str((ctx.get('timing', {}) or {}).get('workflow_start', '') or '')

    # ---- publish ----
    config.publish.tag_image = True
    config.publish.push_harbor = True
    # 统一私有发布（发布点已硬编码私有，不留公开口子），达标与否在总结报告中注明
    workflow = ctx.get('workflow', {})
    config.publish.private = True
    config.config_persisted = workflow.get('config_persisted', False)
    config.publish.upload_weights = True
    # 优先用 local_path（宿主机路径），其次 container_path（容器内路径）
    # 镜像模式下 local_path 是用户提供的宿主机路径，一定能访问
    # 容器模式下两者可能相同（容器内路径），宿主机未必能访问，publish.py 有 docker cp 兜底
    config.publish.weights_dir = model.get('local_path', '') or model.get('container_path', '')
    config.publish.publish_modelscope = False
    config.publish.publish_huggingface = False

    # 如果 context 中已有已发布的 Harbor 镜像地址，跳过 commit/tag/push
    # 只读 image.registry_url（步骤8成功后写入），不读 image.tag（源镜像信息）
    image_section = ctx.get('image', {})
    existing_registry_url = str(image_section.get('registry_url', ''))
    if existing_registry_url and '/' in existing_registry_url:
        config.publish.existing_harbor_image = existing_registry_url

    # token 从宿主机环境变量读取，若不存在则尝试从容器内获取
    config.publish.modelscope_token = os.environ.get('MODELSCOPE_TOKEN', '')
    config.publish.huggingface_token = os.environ.get('HF_TOKEN', '')

    if (not config.publish.modelscope_token or not config.publish.huggingface_token
            or not os.environ.get('HARBOR_USER') or not os.environ.get('HARBOR_PASSWORD')) and config.container_name:
        try:
            result = subprocess.run(
                ["docker", "exec", config.container_name, "cat", "/flagos-workspace/.env"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                env_map = {}
                for line in result.stdout.strip().splitlines():
                    if '=' in line and not line.startswith('#'):
                        k, v = line.split('=', 1)
                        env_map[k.strip()] = v.strip()
                if not config.publish.modelscope_token:
                    config.publish.modelscope_token = env_map.get('MODELSCOPE_TOKEN', '')
                if not config.publish.huggingface_token:
                    config.publish.huggingface_token = env_map.get('HF_TOKEN', '')
                if not os.environ.get('HARBOR_USER') and 'HARBOR_USER' in env_map:
                    os.environ['HARBOR_USER'] = env_map['HARBOR_USER']
                if not os.environ.get('HARBOR_PASSWORD') and 'HARBOR_PASSWORD' in env_map:
                    os.environ['HARBOR_PASSWORD'] = env_map['HARBOR_PASSWORD']
        except Exception:
            pass

    # 有 token 则启用对应平台上传
    config.publish.publish_modelscope = bool(config.publish.modelscope_token)
    config.publish.publish_huggingface = bool(config.publish.huggingface_token)

    # ===== 需求 D（用户 2026-07-20 定稿）：V2 精度不达标时不对外发布 =====
    # 非 plugin 模式(步骤8 V2 发布)：仅当 V2 精度达标(workflow.accuracy_ok=true) 才创建
    # ModelScope/HuggingFace 仓库并上传权重；V2 精度不达标 → 仅 Harbor 私有镜像(过程产物)，
    # 不对外发布。若后续 V3(plugin)达标，由步骤13的 full-publish 兜底补发对外仓库。
    # 说明：plugin 模式(步骤13)不走此分支，其 README/发布门控由 plugin_qualified 决定。
    if not config.plugin_image_mode:
        v2_accuracy_ok = bool(workflow.get('accuracy_ok'))
        if not v2_accuracy_ok:
            config.publish.publish_modelscope = False
            config.publish.publish_huggingface = False
    # results_dir 用于 README 自动读取评测结果
    workspace = ctx.get('workspace', {})
    container_workspace = workspace.get('container_path', '/flagos-workspace')
    host_workspace = workspace.get('host_path', '')
    # 优先使用宿主机路径（脚本在宿主机执行），回退到容器路径
    if host_workspace:
        config.publish.results_dir = os.path.join(host_workspace, "results")
    else:
        config.publish.results_dir = f"{container_workspace}/results"

    # 宿主机工作目录（数据回传目标，应为 /data/flagos-workspace/<model> 格式）
    config.host_workspace_base = workspace.get('host_path') or ''

    # 始终读取 release 段（步骤8产出），plugin 模式下用于定位原仓库
    # 即使 plugin_workflow.triggered 未设置，--plugin-mode 参数也可能启用 plugin 模式
    release_section = ctx.get('release', {})
    ms_url = str(release_section.get('modelscope_url', ''))
    hf_url = str(release_section.get('huggingface_url', ''))
    if ms_url:
        parts = ms_url.rstrip('/').split('/models/')
        if len(parts) == 2:
            config.publish.base_modelscope_model_id = parts[1]
    if hf_url:
        parts = hf_url.rstrip('/').split('huggingface.co/')
        if len(parts) == 2:
            config.publish.base_huggingface_repo_id = parts[1]
        if not config.publish.base_huggingface_repo_id:
            parts = hf_url.rstrip('/').split('hf-mirror.com/')
            if len(parts) == 2:
                config.publish.base_huggingface_repo_id = parts[1]

    # plugin_qualified → 决定是否更新 README。
    # 现行规则（用户 2026-07 定稿）：精度是唯一硬闸门，性能不门控。
    # 只要 plugin 精度达标（accuracy_ok=true）即视为合格、应更新 README；
    # plugin_workflow.qualified 含性能门控(performance_ok)，不能用作 README 门控，
    # 否则精度达标但性能<80%的 V3 会被误判"不达标"跳过 README
    # （历史事故：DeepSeek-R1-0528 精度62%达标、性能77.9%，README被误跳）。
    plugin_wf = ctx.get('plugin_workflow', {})
    if plugin_wf.get('accuracy_ok', False) or plugin_wf.get('qualified', False):
        config.plugin_qualified = True

    # V4 发布时 README 仍推荐 V3(Max) 镜像：从 versions.v3 读取已发布的 V3 镜像地址存起来，
    # 供 publish 阶段在 version_tag==v4 时覆盖 README 的 docker pull 命令。
    # 此处 version_tag 尚未设置（main.py 在 from_context 之后才设），故无条件读取、
    # 由 publish 侧按 version_tag 决定是否启用。字段名兼容 image_url / harbor_image。
    v3_node = ctx.get('versions', {}).get('v3', {}) or {}
    v3_image = str(v3_node.get('image_url', '') or v3_node.get('harbor_image', '') or '').strip()
    if v3_image:
        config.model_info.readme_image_override = v3_image

    return config


def validate_config(config: PipelineConfig) -> List[str]:
    """验证配置是否完整"""
    errors = []

    if not config.container_name:
        errors.append("container_name is required (from context.yaml container.name)")

    if 'publish' in config.stages_to_run and config.publish.enabled:
        if not config.model_info.source_of_model_weights:
            errors.append("model_info.source_of_model_weights is required (e.g., 'Qwen/Qwen3-8B')")

        if config.publish.publish_modelscope and not config.publish.modelscope_token:
            errors.append("publish.modelscope_token is required (or set MODELSCOPE_TOKEN env)")

        if config.publish.publish_huggingface and not config.publish.huggingface_token:
            errors.append("publish.huggingface_token is required (or set HF_TOKEN env)")

    return errors


def _extract_model_name(source: str) -> str:
    """从模型来源提取模型名称"""
    if not source:
        return ""
    if "/" in source:
        return source.split("/")[-1]
    return source


def _clean_model_name_for_tag(name: str) -> str:
    """清理模型名称用于生成 tag"""
    import re
    clean = re.sub(r'[^a-zA-Z0-9.\-]', '-', name.lower())
    clean = re.sub(r'-+', '-', clean).strip('-')
    return clean


def _read_json_field(filepath: str, field: str):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get(field)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def _tag_timestamp_from_seed(seed: str) -> str:
    """把 run-scoped 的 workflow_start(ISO8601) 转成 tag 时间戳 YYYYmmddHHMM。

    发布分多进程执行（push / 双tag / README / upload / 一致性重试），若各进程各自
    datetime.now() 生成 date_tag，时间戳会错开，导致 README 的 docker pull 指向
    Harbor 上不存在的 tag。改用 context 里 run-scoped 的稳定 workflow_start 作种子，
    各进程得到同一 tag。种子缺失/解析失败返回 ""，调用方回退 now()（不劣于旧行为）。
    """
    if not seed:
        return ""
    import datetime
    # fromisoformat 在 <3.11 不接受结尾 'Z'，先剥除；带偏移量(+08:00)可直接解析
    s = str(seed).strip().rstrip("Zz")
    try:
        return datetime.datetime.fromisoformat(s).strftime("%Y%m%d%H%M")
    except (ValueError, TypeError):
        return ""


def auto_fill_config(config: PipelineConfig) -> PipelineConfig:
    """根据环境检测自动填充配置中的空字段"""
    import datetime
    import re

    # 确定容器名称
    container = config.container_name

    # 创建检测器
    detector = ChipDetector(container_name=container if container else None)

    # 解析 vendor
    vendor = None
    if config.chip.vendor and config.chip.vendor != "auto":
        try:
            vendor = ChipVendor(config.chip.vendor)
        except ValueError:
            pass

    # 检测环境信息
    try:
        env_info = detector.detect_environment(vendor=vendor)
    except Exception:
        env_info = None

    # ==================== 芯片和系统信息 ====================
    if env_info:
        if config.chip.vendor == "auto" and env_info.vendor:
            config.chip.vendor = env_info.vendor.value

        if not config.model_info.vendor and env_info.vendor_cn_name:
            config.model_info.vendor = env_info.vendor_cn_name

        if not config.model_info.docker_version and env_info.docker_version:
            config.model_info.docker_version = env_info.docker_version

        if not config.model_info.ubuntu_version:
            if env_info.os_name and env_info.os_version:
                config.model_info.ubuntu_version = f"{env_info.os_name} {env_info.os_version}"
            elif env_info.os_version:
                config.model_info.ubuntu_version = env_info.os_version

        if not config.chip.driver_version and env_info.driver_version:
            config.chip.driver_version = env_info.driver_version
        if not config.chip.sdk_version and env_info.sdk_version:
            config.chip.sdk_version = env_info.sdk_version
        if not config.chip.torch_version and env_info.torch_version:
            config.chip.torch_version = env_info.torch_version
        if not config.chip.python_version and env_info.python_version:
            config.chip.python_version = env_info.python_version
        if not config.chip.gpu_model and env_info.gpu_model:
            config.chip.gpu_model = env_info.gpu_model

        if env_info.flaggems_version:
            config.chip.gems_version = env_info.flaggems_version
        if env_info.flagtree_version:
            config.chip.tree = env_info.flagtree_version

    # ==================== 模型名称 ====================
    model_name = _extract_model_name(config.model_info.source_of_model_weights)
    vendor_name = config.chip.vendor or "unknown"
    # 厂商名归一到规范 key（huawei→ascend、tianshu→iluvatar 等），
    # 再取规范命名后缀，保证命名/tag/报告三处统一。
    if _chip_spec and vendor_name and vendor_name != "unknown":
        try:
            vendor_name = _chip_spec.normalize_vendor(vendor_name) or vendor_name
        except Exception:
            pass
    config.chip.vendor = vendor_name  # 回写归一化结果，供后续 tag 生成复用
    # 命名后缀：规范表可用则查表，否则回退归一化后的 vendor 名
    naming_vendor = vendor_name
    if _chip_spec and vendor_name and vendor_name != "unknown":
        try:
            naming_vendor = _chip_spec.naming_suffix(vendor_name) or vendor_name
        except Exception:
            pass

    if not config.model_info.output_name and model_name:
        # 全部厂商统一 xxx-{vendor}-FlagOS（含 nvidia，按规范表要求）
        config.model_info.output_name = f"{model_name}-{naming_vendor}"

    if not config.model_info.flagrelease_name and config.model_info.output_name:
        suffix = "-FlagOS"
        config.model_info.flagrelease_name = f"{config.model_info.output_name}{suffix}"

    if not config.model_info.flagrelease_name_pre and model_name:
        match = re.match(r'^([A-Za-z]+\d*)', model_name)
        if match:
            config.model_info.flagrelease_name_pre = match.group(1)
        else:
            config.model_info.flagrelease_name_pre = model_name.split('-')[0]

    # ==================== 镜像 tag ====================
    # V3 (Max) 发布到 flagrelease-project 仓库（交付 SVT 验收），其余版本（V1/V2/V4）走 public
    if getattr(config, 'version_tag', None) == "v3":
        if config.chip.harbor_registry == "harbor.baai.ac.cn/flagrelease-public":
            config.chip.harbor_registry = "harbor.baai.ac.cn/flagrelease-project"

    if not config.chip.date_tag:
        # 优先用跨进程一致的 workflow_start 种子；缺失/解析失败才回退 now()（不劣于旧行为）
        tag = _tag_timestamp_from_seed(config.chip.date_tag_seed) or datetime.datetime.now().strftime("%Y%m%d%H%M")
        # 根据 version_tag 决定后缀
        version_tag = getattr(config, 'version_tag', None)
        if version_tag:
            version_suffix_map = {"v1": "-v1", "v2": "-v2", "v3": "-v3", "v4": "-v4"}
            suffix = version_suffix_map.get(version_tag, "")
        elif config.plugin_image_mode:
            suffix = "-plugin"  # 向后兼容：未设置 version_tag 但设置了 plugin_image_mode
        else:
            suffix = ""
        config.chip.date_tag = f"{tag}{suffix}"

    if not config.publish.image_target_tag and config.publish.existing_harbor_image:
        config.publish.image_target_tag = config.publish.existing_harbor_image

    if not config.publish.image_target_tag and config.chip.auto_generate_tag:
        from .chip_detector import ChipVersionInfo, generate_image_tag as _generate_tag
        # ChipVendor 枚举可能未收录新增厂商(zhenwu/arm/sunrise/enflame)，构造失败时
        # 兜底为 None——vendor_name 已显式传给 _generate_tag，info.vendor 仅作兜底不影响命名。
        try:
            _chip_vendor_enum = (
                ChipVendor(vendor_name)
                if vendor_name and vendor_name != "unknown" else None
            )
        except ValueError:
            _chip_vendor_enum = None
        chip_info = ChipVersionInfo(
            vendor=_chip_vendor_enum,
            driver_version=config.chip.driver_version,
            sdk_version=config.chip.sdk_version,
            torch_backend=env_info.torch_backend if env_info and env_info.torch_backend else "",
            torch_version=config.chip.torch_version,
            python_version=config.chip.python_version,
            gpu_model=config.chip.gpu_model,
            arch=env_info.arch if env_info and env_info.arch else "amd64",
        ) if vendor_name and vendor_name != "unknown" else None

        if chip_info:
            # 委托 get_image_name.sh 采集容器实际版本生成镜像名。
            # 采集失败不应中断整个 auto_fill（后续还有 harbor_path/仓库ID/命令等填充），
            # 留空 tag 交由 validate_config 报缺失，行为不比旧字符串拼接脆弱。
            try:
                config.publish.image_target_tag = _generate_tag(
                    info=chip_info,
                    model_name=model_name or "unknown",
                    harbor_registry=config.chip.harbor_registry,
                    tree=config.chip.tree,
                    gems_version=config.chip.gems_version,
                    cx=config.chip.cx,
                    date_tag=config.chip.date_tag,
                    container_name=config.container_name,
                    vendor_name=vendor_name,
                )
            except Exception as e:
                print(f"  ⚠ 自动生成镜像 tag 失败，留空待手动指定: {e}")

    if not config.publish.harbor_path and config.publish.image_target_tag:
        config.publish.harbor_path = config.publish.image_target_tag

    if not config.model_info.image_harbor_path and config.publish.image_target_tag:
        config.model_info.image_harbor_path = config.publish.image_target_tag

    # ==================== ModelScope / HuggingFace ID ====================
    if not config.publish.modelscope_model_id and config.model_info.flagrelease_name:
        config.publish.modelscope_model_id = f"FlagRelease/{config.model_info.flagrelease_name}"

    if not config.publish.huggingface_repo_id and config.model_info.flagrelease_name:
        config.publish.huggingface_repo_id = f"FlagRelease/{config.model_info.flagrelease_name}"

    # ==================== Plugin 模式覆盖 ====================
    if config.plugin_image_mode:
        # 指向步骤8原仓库，不创建新仓库
        if config.publish.base_modelscope_model_id:
            config.publish.modelscope_model_id = config.publish.base_modelscope_model_id
        if config.publish.base_huggingface_repo_id:
            config.publish.huggingface_repo_id = config.publish.base_huggingface_repo_id

        # 用 plugin 评测分数覆盖 evaluation_results
        results_dir = config.publish.results_dir
        if results_dir and os.path.isdir(results_dir):
            plugin_path = os.path.join(results_dir, "gpqa_plugin.json")
            native_path = os.path.join(results_dir, "gpqa_native.json")
            if os.path.exists(plugin_path):
                plugin_score = _read_json_field(plugin_path, "score")
                native_score = _read_json_field(native_path, "score")
                if plugin_score is not None:
                    config.model_info.evaluation_results = [{
                        "metric": "GPQA (plugin)",
                        "origin": native_score if native_score is not None else "N/A",
                        "flagos": plugin_score,
                    }]

    # ==================== 命令 ====================
    if config.model_info.container_run_cmd and config.publish.image_target_tag:
        config.model_info.container_run_cmd = config.model_info.container_run_cmd.replace(
            '{{IMAGE}}', config.publish.image_target_tag
        )

    if not config.model_info.serve_infer_cmd:
        _model_short = config.model_info.source_of_model_weights.split('/')[-1] if config.model_info.source_of_model_weights else "flagOS"
        config.model_info.serve_infer_cmd = f'''curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{_model_short}",
    "messages": [{{"role": "user", "content": "你好"}}]
  }}' '''

    # ==================== 上传文件列表 ====================
    if not config.publish.upload_files:
        config.publish.upload_files = [config.publish.readme_output_path]

    if config.publish.upload_weights and not config.publish.weights_dir:
        pass  # weights_dir 必须在配置中显式指定

    return config
