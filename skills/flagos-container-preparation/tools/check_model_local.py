#!/usr/bin/env python3
"""
check_model_local.py — 模型权重搜索、校验与自动下载（宿主机 + 容器内）

支持三种运行模式:
  host      — 宿主机搜索+宿主机下载（默认，向后兼容）
  container — 宿主机编排入口：先在容器内搜索，再检查宿主机+挂载，未找到则容器内下载
  internal  — 容器内执行（由 container 模式自动调用）

用法:
    # 宿主机模式（默认，与旧版完全一致）
    python3 check_model_local.py --model "Qwen/Qwen3-8B" --output-json
    python3 check_model_local.py --model "Qwen/Qwen3-8B" --download-dir /mnt/data/models
    python3 check_model_local.py --model "Qwen3-8B" --no-download --output-json

    # 容器模式（编排层在宿主机执行）
    python3 check_model_local.py --model "Qwen/Qwen3-8B" --mode container --container my_container --output-json

    # 容器内模式（container 模式自动调用，一般不手动使用）
    python3 check_model_local.py --model "Qwen/Qwen3-8B" --mode internal --output-json

退出码: 0=找到有效权重(本地或下载), 1=未找到, 2=参数错误
"""

import argparse
import json
import os
import re
import subprocess
import sys

DEFAULT_SEARCH_PATHS = ["/data", "/nfs", "/share", "/models", "/home"]
CONTAINER_SEARCH_PATHS = ["/data", "/models", "/root", "/home", "/workspace", "/mnt", "/opt"]
DEFAULT_MAX_DEPTH = 4
SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", ".cache", ".trash"}

# 权重文件排除模式
EXCLUDE_BIN = re.compile(r"^(optimizer|training_args|scheduler)", re.IGNORECASE)

# 单个权重文件最小合理大小（1 MB），低于此阈值视为下载中断的残文件
MIN_WEIGHT_FILE_SIZE = 1 * 1024 * 1024

# 容器内下载路径优先级：优先使用已挂载的宿主机卷，避免写入 overlay
PREFERRED_MOUNT_PREFIXES = ["/data", "/mnt", "/nfs", "/share"]
CONTAINER_DEFAULT_DOWNLOAD_DIR = "/data/models"


def parse_model_identifier(model_input: str) -> dict:
    """从用户输入解析模型名称和组织信息。"""
    result = {"model_name": "", "org": "", "input_type": "name", "raw": model_input}

    model_input = model_input.strip().rstrip("/")

    # ModelScope URL
    ms_match = re.match(r"https?://modelscope\.cn/models/([^/]+)/([^/]+)", model_input)
    if ms_match:
        result["org"] = ms_match.group(1)
        result["model_name"] = ms_match.group(2)
        result["input_type"] = "modelscope_url"
        return result

    # HuggingFace URL
    hf_match = re.match(r"https?://huggingface\.co/([^/]+)/([^/]+)", model_input)
    if hf_match:
        result["org"] = hf_match.group(1)
        result["model_name"] = hf_match.group(2)
        result["input_type"] = "huggingface_url"
        return result

    # org/name 格式
    if "/" in model_input and not model_input.startswith("http"):
        parts = model_input.rsplit("/", 1)
        result["org"] = parts[0]
        result["model_name"] = parts[1]
        return result

    # 纯模型名
    result["model_name"] = model_input
    return result


def read_config_model_name(dir_path: str) -> str:
    """读取目录下 config.json 的 _name_or_path 字段，提取模型名。"""
    config_path = os.path.join(dir_path, "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        # _name_or_path 通常是 "org/model-name" 或绝对路径
        name_or_path = config.get("_name_or_path", "")
        if name_or_path:
            # 取最后一段路径作为模型名
            return name_or_path.rstrip("/").rsplit("/", 1)[-1]
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return ""


def has_weight_files(dir_path: str) -> bool:
    """快速检查目录是否包含权重文件（不做完整校验）。"""
    try:
        for entry in os.listdir(dir_path):
            entry_lower = entry.lower()
            if entry_lower.endswith(".safetensors") or entry_lower.endswith(".bin"):
                if not entry_lower.startswith(("training_args", "optimizer", "scheduler")):
                    return True
    except PermissionError:
        pass
    return False


def search_model_dirs(model_name: str, search_paths: list, max_depth: int) -> list:
    """在宿主机路径下搜索目录名匹配的模型目录。

    三种匹配策略（按优先级）：
    1. 精确匹配：目录名 == model_name（大小写不敏感）
    2. 包含匹配：目录名包含 model_name
    3. config 匹配：目录名不匹配，但 config.json 中 _name_or_path 包含模型名
    """
    exact_matches = []
    contain_matches = []
    config_matches = []
    model_lower = model_name.lower()

    for root_path in search_paths:
        if not os.path.isdir(root_path):
            continue

        for dirpath, dirnames, filenames in os.walk(root_path):
            # 计算当前深度
            depth = dirpath[len(root_path):].count(os.sep)
            if depth >= max_depth:
                dirnames.clear()
                continue

            # 跳过隐藏目录和排除目录
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and d not in SKIP_DIRS
            ]

            for d in dirnames:
                d_lower = d.lower()
                full_path = os.path.join(dirpath, d)
                if d_lower == model_lower:
                    exact_matches.append(full_path)
                elif model_lower in d_lower:
                    contain_matches.append(full_path)

            # 策略 3：当前目录有 config.json + 权重文件，检查 config 内容
            # 仅对目录名未匹配的目录执行（避免重复）
            dir_basename = os.path.basename(dirpath).lower()
            if dir_basename != model_lower and model_lower not in dir_basename:
                if "config.json" in filenames and has_weight_files(dirpath):
                    config_name = read_config_model_name(dirpath)
                    if config_name and model_lower in config_name.lower():
                        config_matches.append(dirpath)

    return exact_matches, contain_matches, config_matches


def check_index_completeness(dir_path: str, weight_files: list, weight_format: str) -> dict:
    """对比 index.json 分片清单，检查权重文件是否齐全。

    返回 {"complete": bool, "missing": [...], "index_file": str or None}
    """
    # 确定 index 文件名
    if weight_format == "safetensors":
        index_name = "model.safetensors.index.json"
    elif weight_format == "pytorch_bin":
        index_name = "pytorch_model.bin.index.json"
    else:
        return {"complete": True, "missing": [], "index_file": None}

    index_path = os.path.join(dir_path, index_name)
    if not os.path.isfile(index_path):
        # 无 index 文件说明是单文件模型或非分片格式，跳过检查
        return {"complete": True, "missing": [], "index_file": None}

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"complete": True, "missing": [], "index_file": index_name}

    # weight_map 的值是分片文件名，取唯一集合
    weight_map = index_data.get("weight_map", {})
    expected_files = set(weight_map.values())
    actual_files = set(weight_files)
    missing = sorted(expected_files - actual_files)

    return {
        "complete": len(missing) == 0,
        "missing": missing,
        "index_file": index_name,
    }


def check_truncated_files(dir_path: str, weight_files: list) -> list:
    """检查是否有疑似下载中断的残文件（< MIN_WEIGHT_FILE_SIZE）。"""
    truncated = []
    for fname in weight_files:
        full_path = os.path.join(dir_path, fname)
        try:
            size = os.path.getsize(full_path)
            if size < MIN_WEIGHT_FILE_SIZE:
                truncated.append({"file": fname, "size_bytes": size})
        except OSError:
            truncated.append({"file": fname, "size_bytes": -1})
    return truncated


def validate_model_dir(dir_path: str) -> dict:
    """校验目录是否包含有效模型权重。"""
    result = {
        "valid": False,
        "config_json": False,
        "weight_format": "none",
        "weight_files": [],
        "weight_count": 0,
        "total_size_gb": 0.0,
        "tokenizer": False,
        "completeness": {},
        "truncated_files": [],
    }

    try:
        entries = os.listdir(dir_path)
    except PermissionError:
        return result

    entries_lower = {e.lower(): e for e in entries}

    # config.json
    result["config_json"] = "config.json" in entries_lower

    # tokenizer
    result["tokenizer"] = any(
        k in entries_lower
        for k in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
    )

    # 权重文件
    safetensors = []
    bins = []
    total_size = 0

    for entry in entries:
        entry_lower = entry.lower()
        full_path = os.path.join(dir_path, entry)

        if entry_lower.endswith(".safetensors") and not entry_lower.startswith("training_args"):
            safetensors.append(entry)
            try:
                total_size += os.path.getsize(full_path)
            except OSError:
                pass

        elif entry_lower.endswith(".bin") and not EXCLUDE_BIN.match(entry):
            bins.append(entry)
            try:
                total_size += os.path.getsize(full_path)
            except OSError:
                pass

    # 优先 safetensors
    if safetensors:
        result["weight_format"] = "safetensors"
        result["weight_files"] = sorted(safetensors)
    elif bins:
        result["weight_format"] = "pytorch_bin"
        result["weight_files"] = sorted(bins)

    result["weight_count"] = len(result["weight_files"])
    result["total_size_gb"] = round(total_size / (1024 ** 3), 2)

    # 完整性校验：对比 index.json 分片清单
    result["completeness"] = check_index_completeness(
        dir_path, result["weight_files"], result["weight_format"]
    )

    # 残文件检测：单文件 < 1MB
    result["truncated_files"] = check_truncated_files(dir_path, result["weight_files"])

    # valid = config.json + 至少一个权重文件 + 分片齐全 + 无残文件
    result["valid"] = (
        result["config_json"]
        and result["weight_count"] > 0
        and result["completeness"]["complete"]
        and len(result["truncated_files"]) == 0
    )

    return result


DEFAULT_DOWNLOAD_DIR = "/mnt/data/models"


def check_network(container=None) -> bool:
    """检测网络连通性（宿主机或容器内）"""
    for url in ["https://modelscope.cn", "https://pypi.org"]:
        try:
            if container:
                cmd = ["docker", "exec", container, "curl", "--connect-timeout", "5",
                       "-s", "-o", "/dev/null", "-w", "%{http_code}", url]
            else:
                cmd = ["curl", "--connect-timeout", "5", "-s", "-o", "/dev/null",
                       "-w", "%{http_code}", url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip().strip("'\"") in ("200", "301", "302"):
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    return False


def ensure_modelscope_cli(container=None) -> bool:
    """确保 modelscope CLI 可用，未安装则自动安装"""
    try:
        if container:
            cmd_check = ["docker", "exec", container, "bash", "-c", "PATH=/opt/conda/bin:$PATH modelscope --help"]
            cmd_install = ["docker", "exec", container, "bash", "-c", "PATH=/opt/conda/bin:$PATH pip install modelscope"]
        else:
            cmd_check = ["modelscope", "--help"]
            cmd_install = ["pip", "install", "modelscope"]

        subprocess.run(cmd_check, capture_output=True, timeout=10)
        return True
    except FileNotFoundError:
        print("  modelscope CLI 未安装，正在自动安装...")
        result = subprocess.run(cmd_install, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("  modelscope 安装成功")
            return True
        else:
            print(f"  modelscope 安装失败: {result.stderr}", file=sys.stderr)
            return False
    except subprocess.TimeoutExpired:
        return True


def download_in_container(container: str, model_id: str, container_download_path: str) -> dict:
    """优先在容器内下载模型到挂载路径"""
    result = {"success": False, "path": container_download_path, "error": "", "method": "container"}

    # 容器内网络检测
    if not check_network(container):
        result["error"] = "容器内网络不通"
        print(f"  容器内网络不通，将尝试宿主机下载", file=sys.stderr)
        return result

    # 容器内确保 modelscope
    if not ensure_modelscope_cli(container):
        result["error"] = "容器内 modelscope 安装失败"
        print(f"  容器内 modelscope 不可用，将尝试宿主机下载", file=sys.stderr)
        return result

    cmd = [
        "docker", "exec", container, "bash", "-c",
        f"PATH=/opt/conda/bin:$PATH modelscope download --model {model_id} --local_dir {container_download_path}"
    ]
    print(f"\n>>> 在容器内下载模型: {model_id}")
    print(f"    容器: {container}")
    print(f"    目标路径: {container_download_path}")

    try:
        proc = subprocess.run(cmd, timeout=7200, capture_output=False)
        if proc.returncode == 0:
            result["success"] = True
            print(f"\n✓ 容器内下载完成: {container_download_path}")
        else:
            result["error"] = f"容器内下载失败，退出码 {proc.returncode}"
            print(f"\n✗ {result['error']}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        result["error"] = "容器内下载超时（2小时）"
        print(f"\n✗ {result['error']}", file=sys.stderr)
    except Exception as e:
        result["error"] = str(e)
        print(f"\n✗ 容器内下载异常: {e}", file=sys.stderr)

    return result


def download_from_modelscope(model_id: str, download_path: str, container=None,
                              container_download_path=None) -> dict:
    """下载模型权重。优先容器内下载，失败 fallback 到宿主机。

    Args:
        model_id: ModelScope 模型 ID，格式 "org/model_name"
        download_path: 宿主机下载目标路径
        container: 容器名（可选，提供时优先在容器内下载）
        container_download_path: 容器内下载路径（可选）

    Returns:
        {"success": bool, "path": str, "error": str, "method": str}
    """
    # 优先容器内下载
    if container and container_download_path:
        result = download_in_container(container, model_id, container_download_path)
        if result["success"]:
            return result
        print(f"\n  容器内下载失败，fallback 到宿主机...")

    # 宿主机下载
    result = {"success": False, "path": download_path, "error": "", "method": "host"}

    if not check_network():
        result["error"] = "网络不通，无法下载模型。请检查网络连接或手动下载模型到宿主机"
        print(f"\n✗ {result['error']}", file=sys.stderr)
        return result

    if not ensure_modelscope_cli():
        result["error"] = "modelscope CLI 安装失败，请手动执行: pip install modelscope"
        print(f"\n✗ {result['error']}", file=sys.stderr)
        return result

    os.makedirs(download_path, exist_ok=True)

    cmd = ["modelscope", "download", "--model", model_id, "--local_dir", download_path]
    print(f"\n>>> 在宿主机下载模型: {model_id}")
    print(f"    目标路径: {download_path}")
    print(f"    命令: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, timeout=7200, capture_output=False)
        if proc.returncode == 0:
            result["success"] = True
            print(f"\n✓ 宿主机下载完成: {download_path}")
        else:
            result["error"] = f"modelscope download 退出码 {proc.returncode}"
            print(f"\n✗ 下载失败: 退出码 {proc.returncode}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        result["error"] = "下载超时（2小时）"
        print(f"\n✗ {result['error']}", file=sys.stderr)
    except Exception as e:
        result["error"] = str(e)
        print(f"\n✗ 下载异常: {e}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Container mode helpers
# ---------------------------------------------------------------------------

def get_container_mounts(container_name: str) -> list:
    """通过 docker inspect 获取容器的卷挂载信息。

    Returns:
        [{"source": "/host/path", "destination": "/container/path", "type": "bind|volume"}, ...]
    """
    cmd = [
        "docker", "inspect", container_name,
        "--format", "{{json .Mounts}}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if proc.returncode != 0:
            print(f"  警告: docker inspect 失败: {proc.stderr.strip()}", file=sys.stderr)
            return []
        mounts_raw = json.loads(proc.stdout.strip())
        return [
            {
                "source": m.get("Source", ""),
                "destination": m.get("Destination", ""),
                "type": m.get("Type", ""),
            }
            for m in mounts_raw
        ]
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        print(f"  警告: 解析容器挂载信息失败: {e}", file=sys.stderr)
        return []


def find_host_path_in_container(host_path: str, mounts: list) -> str:
    """检查宿主机路径是否通过挂载可在容器内访问。

    Returns:
        容器内路径（如 /data/models/Qwen3-8B），未找到返回空字符串。
    """
    host_path = os.path.normpath(host_path)
    for mount in mounts:
        src = os.path.normpath(mount["source"])
        dst = mount["destination"]
        if host_path == src or host_path.startswith(src + os.sep):
            relative = os.path.relpath(host_path, src)
            container_path = os.path.join(dst, relative) if relative != "." else dst
            return container_path
    return ""


def choose_download_path(mounts: list, model_name: str) -> str:
    """从容器挂载卷中选择最佳下载路径，避免写入 overlay。

    优先级: /data > /mnt > /nfs > /share，选中后拼接 models/<model_name>。
    如果挂载点本身已经是模型专属目录（末段匹配模型名），直接使用，避免嵌套。
    无可用挂载卷时 fallback 到 /data/models/<model_name>。
    """
    model_lower = model_name.lower()
    for prefix in PREFERRED_MOUNT_PREFIXES:
        for mount in mounts:
            dst = mount["destination"]
            if dst == prefix or dst.startswith(prefix + "/"):
                # 挂载点末段已是模型名 → 直接用，避免 /data/models/X/models/X 嵌套
                if os.path.basename(dst.rstrip("/")).lower() == model_lower:
                    return dst
                return os.path.join(dst, "models", model_name)
    return os.path.join(CONTAINER_DEFAULT_DOWNLOAD_DIR, model_name)


def docker_exec_run(container_name: str, cmd_str: str, proxy: str = "",
                    timeout: int = 30) -> subprocess.CompletedProcess:
    """在容器内执行命令，自动添加 PATH 前缀和可选代理。"""
    env_flags = []
    if proxy:
        env_flags.extend(["-e", f"http_proxy={proxy}", "-e", f"https_proxy={proxy}"])
    full_cmd = ["docker", "exec"] + env_flags + [
        container_name, "bash", "-c",
        f"PATH=/opt/conda/bin:$PATH {cmd_str}",
    ]
    return subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)


def search_model_in_container(container_name: str, model_input: str,
                              search_paths: list, max_depth: int) -> dict:
    """在容器内搜索模型权重。

    通过 docker cp 将本脚本复制到容器 /tmp/，以 --mode internal 运行。
    """
    # 复制自身到容器
    script_path = os.path.abspath(__file__)
    try:
        cp_proc = subprocess.run(
            ["docker", "cp", script_path, f"{container_name}:/tmp/check_model_local.py"],
            capture_output=True, text=True, timeout=10,
        )
        if cp_proc.returncode != 0:
            return {"error": f"docker cp 失败: {cp_proc.stderr.strip()}"}
    except Exception as e:
        return {"error": f"docker cp 异常: {e}"}

    # 在容器内执行搜索（不下载）
    paths_arg = ",".join(search_paths)
    cmd = (
        f'python3 /tmp/check_model_local.py '
        f'--model "{model_input}" '
        f'--mode internal '
        f'--search-paths "{paths_arg}" '
        f'--max-depth {max_depth} '
        f'--no-download --output-json'
    )
    try:
        proc = docker_exec_run(container_name, cmd, timeout=120)
        if proc.returncode not in (0, 1):
            return {"error": f"容器内搜索失败 (exit {proc.returncode}): {proc.stderr.strip()}"}
        # 解析 JSON 输出
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return {"error": f"容器内搜索输出非法 JSON: {proc.stdout[:500]}"}
    except subprocess.TimeoutExpired:
        return {"error": "容器内搜索超时 (120s)"}
    except Exception as e:
        return {"error": f"容器内搜索异常: {e}"}


def download_in_container(container_name: str, model_id: str, download_path: str,
                          proxy: str = "") -> dict:
    """在容器内通过 modelscope download 下载模型。"""
    result = {"success": False, "path": download_path, "error": ""}

    # 确保目标目录存在
    try:
        docker_exec_run(container_name, f"mkdir -p '{download_path}'", timeout=10)
    except Exception:
        pass

    # 检查 modelscope CLI 是否可用
    check = docker_exec_run(container_name, "which modelscope", timeout=10)
    if check.returncode != 0:
        print("  容器内未安装 modelscope CLI，尝试安装...")
        install_cmd = "pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
        install_proc = docker_exec_run(container_name, install_cmd, proxy=proxy, timeout=600)
        if install_proc.returncode != 0:
            result["error"] = f"modelscope 安装失败: {install_proc.stderr[:300]}"
            print(f"  ✗ {result['error']}", file=sys.stderr)
            return result
        print("  ✓ modelscope CLI 安装成功")

    # 执行下载
    dl_cmd = f"modelscope download --model '{model_id}' --local_dir '{download_path}'"
    print(f"\n>>> 容器内未找到权重，开始从 ModelScope 下载: {model_id}")
    print(f"    容器: {container_name}")
    print(f"    目标路径: {download_path}")

    try:
        proc = docker_exec_run(container_name, dl_cmd, proxy=proxy, timeout=7200)
        if proc.returncode == 0:
            result["success"] = True
            print(f"\n✓ 容器内下载完成: {download_path}")
        else:
            result["error"] = f"modelscope download 退出码 {proc.returncode}: {proc.stderr[:300]}"
            print(f"\n✗ 容器内下载失败: {result['error']}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        result["error"] = "下载超时（2小时）"
        print(f"\n✗ {result['error']}", file=sys.stderr)
    except Exception as e:
        result["error"] = str(e)
        print(f"\n✗ 下载异常: {e}", file=sys.stderr)

    return result


def run_container_mode(args, parsed: dict) -> dict:
    """容器模式：在容器内搜索模型，必要时在容器内下载。

    流程:
    1. 在容器内搜索
    2. 在宿主机搜索 + 检查挂载关系
    3. 如均未找到 → 容器内下载（优先挂载卷路径）
    """
    container = args.container
    model_name = parsed["model_name"]

    output = {
        "model_input": args.model,
        "parsed": parsed,
        "mode": "container",
        "container": container,
        "container_search": {"found": False, "best_match": None, "candidates": []},
        "host_search": {
            "found": False, "best_match": None, "candidates": [],
            "mounted_in_container": False, "container_mount_path": "",
        },
        "download": None,
        "final_container_path": None,
        "final_host_path": None,
    }

    # 获取容器挂载信息
    mounts = get_container_mounts(container)
    output["mounts"] = [
        {"source": m["source"], "destination": m["destination"]}
        for m in mounts
    ]

    # --- Step 1: 容器内搜索 ---
    container_search_paths = CONTAINER_SEARCH_PATHS
    if args.search_paths:
        container_search_paths = [p.strip() for p in args.search_paths.split(",") if p.strip()]

    print(f"[1/3] 在容器 {container} 内搜索模型 {model_name} ...")
    container_result = search_model_in_container(
        container, args.model, container_search_paths, args.max_depth,
    )

    if "error" in container_result:
        print(f"  ✗ 容器内搜索失败: {container_result['error']}", file=sys.stderr)
        output["container_search"]["error"] = container_result["error"]
    else:
        output["container_search"]["found"] = container_result.get("found", False)
        output["container_search"]["best_match"] = container_result.get("best_match")
        output["container_search"]["candidates"] = container_result.get("candidates", [])

    if output["container_search"]["found"]:
        container_path = output["container_search"]["best_match"]
        print(f"  ✓ 容器内找到模型: {container_path}")
        output["final_container_path"] = container_path
        # 尝试推算宿主机路径
        for mount in mounts:
            dst = mount["destination"]
            if container_path == dst or container_path.startswith(dst + "/"):
                relative = os.path.relpath(container_path, dst)
                host_path = os.path.join(mount["source"], relative) if relative != "." else mount["source"]
                output["final_host_path"] = host_path
                break
        return output

    # --- Step 2: 宿主机搜索 + 挂载检查 ---
    host_search_paths = DEFAULT_SEARCH_PATHS
    print(f"[2/3] 在宿主机搜索模型 {model_name} ...")
    exact, contain, config = search_model_dirs(model_name, host_search_paths, args.max_depth)

    host_candidates = []
    for path in exact:
        info = validate_model_dir(path)
        host_candidates.append({"path": path, "match_type": "exact", **info})
    for path in contain:
        info = validate_model_dir(path)
        host_candidates.append({"path": path, "match_type": "contains", **info})
    for path in config:
        info = validate_model_dir(path)
        host_candidates.append({"path": path, "match_type": "config", **info})

    MATCH_PRIORITY = {"exact": 0, "contains": 1, "config": 2}
    valid_host = [c for c in host_candidates if c["valid"]]
    host_best = None
    if valid_host:
        valid_host.sort(key=lambda c: (MATCH_PRIORITY.get(c["match_type"], 9), -c["total_size_gb"]))
        host_best = valid_host[0]["path"]

    output["host_search"]["candidates"] = host_candidates
    output["host_search"]["found"] = host_best is not None
    output["host_search"]["best_match"] = host_best

    if host_best:
        container_mount_path = find_host_path_in_container(host_best, mounts)
        if container_mount_path:
            print(f"  ✓ 宿主机找到模型: {host_best} → 容器内挂载于 {container_mount_path}")
            output["host_search"]["mounted_in_container"] = True
            output["host_search"]["container_mount_path"] = container_mount_path
            output["final_container_path"] = container_mount_path
            output["final_host_path"] = host_best
            return output
        else:
            print(f"  ⚠ 宿主机找到模型: {host_best}，但未挂载到容器（无法直接使用）")
            output["host_search"]["mounted_in_container"] = False
            output["final_host_path"] = host_best
    else:
        print(f"  ✗ 宿主机也未找到模型")

    # --- Step 3: 容器内下载 ---
    if args.no_download:
        print("  跳过下载（--no-download）")
        return output

    model_id = ""
    if parsed["org"]:
        model_id = f"{parsed['org']}/{parsed['model_name']}"
    elif "/" in args.model and not args.model.startswith("http"):
        model_id = args.model

    if not model_id:
        print(f"\n✗ 无法自动下载: 缺少组织名，请使用 org/model 格式（如 Qwen/Qwen3-8B）", file=sys.stderr)
        output["download"] = {"success": False, "error": "缺少组织名，无法构建 ModelScope model ID"}
        return output

    download_path = choose_download_path(mounts, model_name)
    print(f"[3/3] 在容器内下载模型到: {download_path} ...")

    dl_result = download_in_container(container, model_id, download_path, proxy=args.proxy or "")
    output["download"] = dl_result

    if dl_result["success"]:
        # 验证下载结果
        verify_result = search_model_in_container(
            container, args.model,
            [os.path.dirname(download_path)], max_depth=2,
        )
        if not verify_result.get("error") and verify_result.get("found"):
            output["final_container_path"] = verify_result["best_match"]
            print(f"  ✓ 下载校验通过: {verify_result['best_match']}")
        else:
            output["final_container_path"] = download_path
            print(f"  ⚠ 下载完成但校验未确认，使用下载路径: {download_path}", file=sys.stderr)

        # 推算宿主机路径
        final_cp = output["final_container_path"]
        for mount in mounts:
            dst = mount["destination"]
            if final_cp and (final_cp == dst or final_cp.startswith(dst + "/")):
                relative = os.path.relpath(final_cp, dst)
                output["final_host_path"] = os.path.join(mount["source"], relative) if relative != "." else mount["source"]
                break

    return output


def main():
    parser = argparse.ArgumentParser(description="模型权重搜索、校验与自动下载")
    parser.add_argument("--model", required=True, help="模型名 / ModelScope URL / HuggingFace URL")
    parser.add_argument("--mode", choices=["host", "container", "internal"], default="host",
                        help="运行模式: host=宿主机(默认), container=容器编排, internal=容器内执行")
    parser.add_argument("--container", default=None, help="容器名称（--mode container 时必填）")
    parser.add_argument("--proxy", default=None, help="下载代理地址（如 http://proxy:port）")
    parser.add_argument("--search-paths", default=None, help="搜索根目录，逗号分隔")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH, help="搜索目录深度")
    parser.add_argument("--output-json", action="store_true", help="JSON 格式输出")
    parser.add_argument("--download-dir", default=None, help="自动下载目标目录")
    parser.add_argument("--no-download", action="store_true", help="禁用自动下载，仅搜索本地")
    parser.add_argument("--container-model-path", default=None, help="容器内模型下载路径（如 /data/models/Qwen3-8B）")
    args = parser.parse_args()

    # 参数校验
    if args.mode == "container" and not args.container:
        print("Error: --mode container 需要 --container 参数", file=sys.stderr)
        sys.exit(2)

    # 解析模型标识
    parsed = parse_model_identifier(args.model)
    if not parsed["model_name"]:
        print("Error: 无法解析模型名称", file=sys.stderr)
        sys.exit(2)

    # ---- container 模式 ----
    if args.mode == "container":
        output = run_container_mode(args, parsed)
        found = output.get("final_container_path") is not None

        if args.output_json:
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(f"\nModel: {parsed['model_name']} (mode: container, container: {args.container})")
            if found:
                print(f"✓ 容器内模型路径: {output['final_container_path']}")
                if output.get("final_host_path"):
                    print(f"  宿主机路径: {output['final_host_path']}")
            else:
                print("✗ 未找到有效模型权重")
        sys.exit(0 if found else 1)

    # ---- host / internal 模式（逻辑相同，仅默认路径不同）----
    if args.search_paths:
        search_paths = [p.strip() for p in args.search_paths.split(",") if p.strip()]
    elif args.mode == "internal":
        search_paths = CONTAINER_SEARCH_PATHS
    else:
        search_paths = DEFAULT_SEARCH_PATHS

    if args.download_dir is None:
        download_dir = CONTAINER_DEFAULT_DOWNLOAD_DIR if args.mode == "internal" else DEFAULT_DOWNLOAD_DIR
    else:
        download_dir = args.download_dir

    # 搜索
    exact_matches, contain_matches, config_matches = search_model_dirs(
        parsed["model_name"], search_paths, args.max_depth
    )

    # 校验所有候选
    candidates = []
    for path in exact_matches:
        info = validate_model_dir(path)
        candidates.append({
            "path": path,
            "match_type": "exact",
            **info,
        })
    for path in contain_matches:
        info = validate_model_dir(path)
        candidates.append({
            "path": path,
            "match_type": "contains",
            **info,
        })
    for path in config_matches:
        info = validate_model_dir(path)
        candidates.append({
            "path": path,
            "match_type": "config",
            **info,
        })

    # 选择 best_match: valid=true 中，exact > contains > config，权重大小优先
    MATCH_PRIORITY = {"exact": 0, "contains": 1, "config": 2}
    valid_candidates = [c for c in candidates if c["valid"]]
    best_match = None
    if valid_candidates:
        valid_candidates.sort(
            key=lambda c: (MATCH_PRIORITY.get(c["match_type"], 9), -c["total_size_gb"])
        )
        best_match = valid_candidates[0]["path"]

    output = {
        "model_input": args.model,
        "parsed": parsed,
        "mode": args.mode,
        "found": best_match is not None,
        "candidates": candidates,
        "best_match": best_match,
        "download": None,
    }

    # 未找到本地权重时，自动从 ModelScope 下载
    if best_match is None and not args.no_download:
        model_id = ""
        if parsed["org"]:
            model_id = f"{parsed['org']}/{parsed['model_name']}"
        elif "/" in args.model and not args.model.startswith("http"):
            model_id = args.model

        if model_id:
            # 避免嵌套：download_dir 末段已是模型名时直接用
            download_dir = args.download_dir
            if os.path.basename(download_dir.rstrip("/")).lower() == parsed["model_name"].lower():
                download_path = download_dir
            else:
                download_path = os.path.join(download_dir, parsed["model_name"])
            # 容器内下载路径：优先用户指定，其次从 download_dir 推算
            container_dl_path = args.container_model_path
            if not container_dl_path and args.container:
                container_dl_path = os.path.join(args.download_dir, parsed["model_name"])
            dl_result = download_from_modelscope(
                model_id, download_path,
                container=args.container,
                container_download_path=container_dl_path,
            )
            output["download"] = dl_result

            if dl_result["success"]:
                # 校验下载结果
                dl_validation = validate_model_dir(download_path)
                if dl_validation["valid"]:
                    output["found"] = True
                    output["best_match"] = download_path
                    output["candidates"].append({
                        "path": download_path,
                        "match_type": "downloaded",
                        **dl_validation,
                    })
                else:
                    print(f"  警告: 下载完成但校验未通过", file=sys.stderr)
                    output["download"]["validation"] = dl_validation
        else:
            print(f"\n✗ 无法自动下载: 缺少组织名，请使用 org/model 格式（如 Qwen/Qwen3-8B）", file=sys.stderr)
            output["download"] = {"success": False, "error": "缺少组织名，无法构建 ModelScope model ID"}

    if args.output_json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        # 人类可读输出
        print(f"Model: {parsed['model_name']} (org: {parsed['org'] or 'N/A'}, mode: {args.mode})")
        print(f"Input type: {parsed['input_type']}")
        print(f"Search paths: {', '.join(search_paths)}")
        print(f"Candidates found: {len(candidates)}")
        if best_match:
            print(f"\n✓ Best match: {best_match}")
            best_info = next((c for c in candidates if c["path"] == best_match), None)
            if best_info:
                print(f"  Format: {best_info['weight_format']}, Files: {best_info['weight_count']}, Size: {best_info['total_size_gb']} GB")
                print(f"  Tokenizer: {'yes' if best_info['tokenizer'] else 'no'}")
                if best_info.get("match_type") == "downloaded":
                    print(f"  Source: ModelScope (自动下载)")
        else:
            print("\n✗ No valid model weights found.")
            if candidates:
                print("  Partial matches:")
                for c in candidates[:5]:
                    issues = []
                    if not c["config_json"]:
                        issues.append("missing config.json")
                    if c["weight_count"] == 0:
                        issues.append("no weight files")
                    if not c["completeness"].get("complete", True):
                        missing = c["completeness"]["missing"]
                        issues.append(f"missing {len(missing)} shard(s): {', '.join(missing[:3])}")
                    if c["truncated_files"]:
                        names = [t["file"] for t in c["truncated_files"]]
                        issues.append(f"truncated: {', '.join(names[:3])}")
                    print(f"    {c['path']} — {'; '.join(issues) or 'unknown'}")

    sys.exit(0 if best_match else 1)


if __name__ == "__main__":
    main()
