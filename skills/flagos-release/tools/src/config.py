"""
配置管理模块
从 context.yaml 加载配置，并提供配置验证和自动填充
"""
import os
import json
import subprocess
from dataclasses import dataclass, field
from typing import List
import yaml

from .chip_detector import ChipDetector, ChipVendor, VENDOR_NAMES, sanitize_docker_tag


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
    # 仓库可见性
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
    container_run_cmd: str = ""
    serve_start_cmd: str = ""
    serve_infer_cmd: str = ""


@dataclass
class PipelineConfig:
    """完整的流水线配置"""
    input_type: str = "container"
    container_name: str = ""
    host_workspace_base: str = ""  # /data/flagos-workspace/<model>，由 context.yaml workspace.host_path 填充
    config_persisted: bool = False
    plugin_image_mode: bool = False  # plugin 模式：镜像 tag 追加 -plugin，仓库名追加 -plugin
    plugin_qualified: bool = False   # plugin 精度+性能均达标时为 True，否则跳过 README 更新

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

    # serve_start_cmd
    svc = ctx.get('service', {})
    runtime = ctx.get('runtime', {})
    commands = ctx.get('commands', {})
    model_short = model.get('name', '').split('/')[-1] if model.get('name') else ''
    flagrelease_name = f"{model_short}-FlagOS" if model_short else ''
    user_model_path = f"/data/{flagrelease_name}" if flagrelease_name else model.get('container_path', '')

    if commands.get('serve_start'):
        import re
        serve_cmd = commands['serve_start']
        # 替换模型路径为用户下载路径
        container_path = model.get('container_path', '')
        if container_path and user_model_path:
            serve_cmd = serve_cmd.replace(container_path, user_model_path)
        # 替换端口为默认 8000
        serve_cmd = re.sub(r'--port\s+\d+', '--port 8000', serve_cmd)
        config.model_info.serve_start_cmd = serve_cmd
    else:
        port = svc.get('port', 8000)
        tp = runtime.get('tp_size') or 1
        max_model_len = svc.get('max_model_len', '')
        cmd_parts = [f"vllm serve {user_model_path}",
                     f"--host 0.0.0.0 --port 8000",
                     f"--tensor-parallel-size {tp}",
                     f"--served-model-name {model_short}" if model_short else None,
                     "--trust-remote-code"]
        if max_model_len:
            cmd_parts.append(f"--max-model-len {max_model_len}")
        config.model_info.serve_start_cmd = " \\\n".join(p for p in cmd_parts if p)

    # container_run_cmd (优先从 context commands 读取实际命令)
    if commands.get('container_run'):
        import re
        run_cmd = commands['container_run']
        # 替换镜像为 {{IMAGE}} 占位符（镜像通常是最后一个非选项参数或可通过已知镜像名匹配）
        image_name = ctx.get('image', {}).get('name', '')
        if image_name:
            run_cmd = run_cmd.replace(image_name, '{{IMAGE}}')
        # 移除 workspace 挂载（-v ...:/flagos-workspace）
        run_cmd = re.sub(r'\s*-v\s+\S+:/flagos-workspace', '', run_cmd)
        # 替换模型挂载为简化的 -v /data:/data
        container_path = model.get('container_path', '')
        local_path = model.get('local_path', '')
        if container_path and local_path:
            run_cmd = re.sub(r'-v\s+\S+:' + re.escape(container_path), '-v /data:/data', run_cmd)
        # 替换容器名为通用名
        run_cmd = re.sub(r'--name[= ]\S+', '--name flagos', run_cmd)
        config.model_info.container_run_cmd = run_cmd
    else:
        config.model_info.container_run_cmd = (
            "docker run -itd --gpus=all --network=host "
            "-v /data:/data --name flagos {{IMAGE}}"
        )

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

    # ---- publish ----
    config.publish.tag_image = True
    config.publish.push_harbor = True
    # 统一私有发布，达标与否在总结报告中注明
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

    # plugin_workflow.qualified → plugin_qualified
    plugin_wf = ctx.get('plugin_workflow', {})
    if plugin_wf.get('qualified', False):
        config.plugin_qualified = True

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

    if not config.model_info.output_name and model_name:
        if vendor_name == "nvidia":
            config.model_info.output_name = model_name
        else:
            config.model_info.output_name = f"{model_name}-{vendor_name}"

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
    if not config.chip.date_tag:
        tag = datetime.datetime.now().strftime("%Y%m%d%H%M")
        config.chip.date_tag = f"{tag}-plugin" if config.plugin_image_mode else tag

    if not config.publish.image_target_tag and config.chip.auto_generate_tag:
        from .chip_detector import ChipVersionInfo, generate_image_tag as _generate_tag
        chip_info = ChipVersionInfo(
            vendor=ChipVendor(vendor_name) if vendor_name and vendor_name != "unknown" else None,
            driver_version=config.chip.driver_version,
            sdk_version=config.chip.sdk_version,
            torch_backend=env_info.torch_backend if env_info and env_info.torch_backend else "",
            torch_version=config.chip.torch_version,
            python_version=config.chip.python_version,
            gpu_model=config.chip.gpu_model,
            arch=env_info.arch if env_info and env_info.arch else "amd64",
        ) if vendor_name and vendor_name != "unknown" else None

        if chip_info:
            config.publish.image_target_tag = _generate_tag(
                info=chip_info,
                model_name=model_name or "unknown",
                harbor_registry=config.chip.harbor_registry,
                tree=config.chip.tree,
                gems_version=config.chip.gems_version,
                cx=config.chip.cx,
                date_tag=config.chip.date_tag,
            )

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
        infer_model_name = model_short or "flagOS"
        config.model_info.serve_infer_cmd = f'''curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{infer_model_name}",
    "messages": [{{"role": "user", "content": "你好"}}]
  }}' '''

    # ==================== 上传文件列表 ====================
    if not config.publish.upload_files:
        config.publish.upload_files = [config.publish.readme_output_path]

    if config.publish.upload_weights and not config.publish.weights_dir:
        pass  # weights_dir 必须在配置中显式指定

    return config
