"""
芯片版本检测模块
支持多种国产AI芯片的版本信息自动检测
"""
import subprocess
import re
import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from enum import Enum


class ChipVendor(Enum):
    """芯片厂商枚举"""
    NVIDIA = "nvidia"          # 英伟达 (nvidia-smi)
    METAX = "metax"            # 沐曦 (mx-smi)
    MTHREADS = "mthreads"      # 摩尔线程 (mthreads-gmi)
    ILUVATAR = "iluvatar"      # 天数智芯 (ixsmi)
    ASCEND = "ascend"          # 华为昇腾 (npu-smi)
    HYGON = "hygon"            # 海光 DCU (hy-smi)
    KUNLUNXIN = "kunlunxin"    # 昆仑芯 (xpu-smi)
    CAMBRICON = "cambricon"    # 寒武纪 (cnmon)
    TSINGMICRO = "tsingmicro"  # 清微智能 (tsm_smi)


# 芯片厂商名称映射（英文）
VENDOR_NAMES = {
    ChipVendor.NVIDIA: "nvidia",
    ChipVendor.METAX: "metax",
    ChipVendor.MTHREADS: "mthreads",
    ChipVendor.ILUVATAR: "iluvatar",
    ChipVendor.ASCEND: "ascend",
    ChipVendor.HYGON: "hygon",
    ChipVendor.KUNLUNXIN: "kunlunxin",
    ChipVendor.CAMBRICON: "cambricon",
    ChipVendor.TSINGMICRO: "tsingmicro",
}


# 芯片型号 → 编码映射表（用于镜像命名）
# 按厂商分组，每组内更具体的模式排在前面避免误匹配
GPU_MODEL_CODE_MAP = {
    "nvidia": [
        ("a100", "nvidia001"),
        ("a800", "nvidia002"),
        ("h20", "nvidia003"),
        ("h100", "nvidia004"),
        ("h800", "nvidia005"),
    ],
    "arm": [
        ("graviton4", "arm001"),
    ],
    "metax": [
        ("c550", "metax001"),
    ],
    "tsingmicro": [
        ("rex1032", "tsingmicro001"),
    ],
    "ascend": [
        ("ascend910", "ascend001"),
        ("atlas800", "ascend001"),
        ("910b", "ascend001"),
    ],
    "hygon": [
        ("bw1000", "hygon001"),
    ],
    "mthreads": [
        ("s5000", "mthreads001"),
    ],
    "kunlunxin": [
        ("p900", "kunlunxin002"),
        ("p800", "kunlunxin001"),
    ],
    "iluvatar": [
        ("tg-v200", "iluvatar002"),
        ("tgv200", "iluvatar002"),
        ("bi-v150", "iluvatar001"),
        ("biv150", "iluvatar001"),
    ],
    "pp": [
        ("zw810", "pp001"),
        ("ppu", "pp001"),
    ],
    "cambricon": [
        ("mlu590-m9", "cambricon002"),
        ("mlu590m9", "cambricon002"),
        ("mlu590", "cambricon001"),
    ],
}


def get_gpu_code(gpu_model: str, vendor_name: str) -> str:
    """将 GPU 型号映射为标准编码名（用于镜像命名）

    Args:
        gpu_model: GPU 型号（小写、无空格）
        vendor_name: 厂商名称（如 nvidia, ascend 等）

    Returns:
        编码名（如 nvidia001），未匹配则返回原始 gpu_model
    """
    model_lower = gpu_model.lower().replace(' ', '')
    patterns = GPU_MODEL_CODE_MAP.get(vendor_name, [])
    for pattern, code in patterns:
        if pattern in model_lower:
            return code
    return gpu_model


# 各厂商的检测命令和SDK名称
VENDOR_DETECT_INFO = {
    ChipVendor.NVIDIA: {
        "detect_cmd": "which nvidia-smi",
        "smi_cmd": "nvidia-smi",
        "sdk_name": "CUDA",
        "torch_backend": "cuda",
    },
    ChipVendor.METAX: {
        "detect_cmd": "which mx-smi",
        "smi_cmd": "mx-smi",
        "sdk_name": "MXMACA",
        "torch_backend": "musa",
    },
    ChipVendor.MTHREADS: {
        "detect_cmd": "which mthreads-gmi",
        "smi_cmd": "mthreads-gmi",
        "sdk_name": "MUSA",
        "torch_backend": "musa",
    },
    ChipVendor.ILUVATAR: {
        "detect_cmd": "which ixsmi",
        "smi_cmd": "ixsmi",
        "sdk_name": "IXRT",
        "torch_backend": "cuda",
    },
    ChipVendor.ASCEND: {
        "detect_cmd": "which npu-smi",
        "smi_cmd": "npu-smi info",
        "sdk_name": "CANN",
        "torch_backend": "npu",
    },
    ChipVendor.HYGON: {
        "detect_cmd": "which hy-smi",
        "smi_cmd": "hy-smi",
        "sdk_name": "DTK",
        "torch_backend": "hip",
    },
    ChipVendor.KUNLUNXIN: {
        "detect_cmd": "which xpu-smi",
        "smi_cmd": "xpu-smi",
        "sdk_name": "XRE",
        "torch_backend": "xpu",
    },
    ChipVendor.CAMBRICON: {
        "detect_cmd": "which cnmon",
        "smi_cmd": "cnmon",
        "sdk_name": "CNToolkit",
        "torch_backend": "mlu",
    },
    ChipVendor.TSINGMICRO: {
        "detect_cmd": "which tsm_smi",
        "smi_cmd": "tsm_smi",
        # 备用命令：某些环境需要先 source bash_profile
        "smi_cmd_alt": "source /root/.bash_profile 2>/dev/null && tsm_smi -t",
        "sdk_name": "TSM",
        "torch_backend": "tsm",
    },
}


@dataclass
class EnvironmentInfo:
    """完整的环境信息"""
    # 操作系统
    os_name: str = ""              # 如 Ubuntu
    os_version: str = ""           # 如 22.04.3 LTS
    kernel_version: str = ""       # 内核版本
    # Docker
    docker_version: str = ""
    # Python
    python_version: str = ""
    # 芯片相关
    vendor: Optional[ChipVendor] = None
    vendor_cn_name: str = ""       # 芯片厂商中文名
    driver_version: str = ""       # 驱动版本
    sdk_name: str = ""             # SDK名称 (MUSA/CANN/DTK等)
    sdk_version: str = ""          # SDK版本
    # PyTorch
    torch_backend: str = ""        # PyTorch后端 (musa/npu/hip等)
    torch_version: str = ""
    # GPU/加速卡
    gpu_model: str = ""            # GPU型号
    gpu_count: int = 0             # GPU数量
    # FlagGems/FlagTree
    flaggems_version: str = ""
    flagtree_version: str = ""
    # vLLM
    vllm_version: str = ""
    # 架构
    arch: str = "amd64"
    # 额外信息
    extra_info: Dict[str, str] = field(default_factory=dict)


@dataclass
class ChipVersionInfo:
    """芯片版本信息（兼容旧接口）"""
    vendor: ChipVendor
    driver_version: str = ""
    sdk_version: str = ""          # 各厂商SDK版本 (MUSA/IXRT/CANN/DTK/XRE/CNToolkit)
    torch_backend: str = ""        # PyTorch后端名称 (musa/cuda/npu/hip/xpu/mlu)
    torch_version: str = ""        # PyTorch版本
    python_version: str = ""
    gpu_model: str = ""            # GPU型号
    arch: str = "amd64"            # 架构
    # 额外信息
    extra_info: Dict[str, str] = field(default_factory=dict)


class ChipDetector:
    """芯片版本检测器"""

    def __init__(self, container_name: Optional[str] = None):
        """
        初始化检测器

        Args:
            container_name: 容器名称，如果为None则在本地执行命令
        """
        self.container_name = container_name

    def _run_cmd(self, cmd: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """
        执行命令

        Args:
            cmd: 要执行的命令
            timeout: 超时时间（秒）

        Returns:
            (成功与否, stdout, stderr)
        """
        try:
            if self.container_name:
                # 用列表参数直接调用 docker exec，不经过外层 shell 解释，
                # cmd 原封不动交给容器内的 bash -c 执行，彻底避免引号转义问题。
                # 使用 bash -lc（login shell）以确保 source /etc/profile、
                # ~/.bash_profile 等，从而激活 conda/virtualenv 环境，
                # 否则 python3、pip 等可能指向系统默认版本而非容器内实际使用的版本。
                result = subprocess.run(
                    ["docker", "exec", self.container_name, "bash", "-lc", cmd],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            else:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

    def detect_vendor(self) -> Optional[ChipVendor]:
        """自动检测芯片厂商"""
        # 按优先级顺序检测各厂商
        for vendor in ChipVendor:
            detect_info = VENDOR_DETECT_INFO.get(vendor)
            if detect_info:
                success, _, _ = self._run_cmd(detect_info["detect_cmd"])
                if success:
                    return vendor
        return None

    def detect(self, vendor: Optional[ChipVendor] = None) -> ChipVersionInfo:
        """
        检测芯片版本信息

        Args:
            vendor: 芯片厂商，如果为None则自动检测

        Returns:
            芯片版本信息
        """
        if vendor is None:
            vendor = self.detect_vendor()
            if vendor is None:
                raise ValueError("无法自动检测芯片厂商，请手动指定")

        # 基础信息检测
        info = ChipVersionInfo(vendor=vendor)
        info.python_version = self._detect_python_version()
        info.arch = self._detect_arch()

        # 获取厂商特定信息
        vendor_info = VENDOR_DETECT_INFO.get(vendor, {})
        info.torch_backend = vendor_info.get("torch_backend", "")

        # 检测 PyTorch 版本
        info.torch_version = self._detect_torch_version()

        # 使用通用方法检测驱动和SDK版本
        smi_cmd = vendor_info.get("smi_cmd", "")
        if smi_cmd:
            self._detect_from_smi(info, vendor, smi_cmd)

        return info

    def detect_environment(self, vendor: Optional[ChipVendor] = None) -> EnvironmentInfo:
        """
        检测完整的环境信息

        Args:
            vendor: 芯片厂商，如果为None则自动检测

        Returns:
            完整的环境信息
        """
        if vendor is None:
            vendor = self.detect_vendor()

        env = EnvironmentInfo()

        # 操作系统信息
        env.os_name, env.os_version = self._detect_os()
        env.kernel_version = self._detect_kernel()

        # Docker 版本
        env.docker_version = self._detect_docker_version()

        # Python 版本
        env.python_version = self._detect_python_version()

        # 架构
        env.arch = self._detect_arch()

        # FlagGems 版本
        env.flaggems_version = self._detect_flaggems_version()

        # FlagTree 版本
        env.flagtree_version = self._detect_flagtree_version()

        # vLLM 版本
        env.vllm_version = self._detect_vllm_version()

        # 芯片相关信息
        if vendor:
            env.vendor = vendor
            env.vendor_cn_name = VENDOR_NAMES.get(vendor, vendor.value)

            # 获取厂商配置信息
            vendor_info = VENDOR_DETECT_INFO.get(vendor, {})
            env.sdk_name = vendor_info.get("sdk_name", "")
            env.torch_backend = vendor_info.get("torch_backend", "")

            # 检测 PyTorch 版本
            env.torch_version = self._detect_torch_version()

            # 使用 SMI 工具检测驱动和GPU信息
            smi_cmd = vendor_info.get("smi_cmd", "")
            if smi_cmd:
                chip_info = ChipVersionInfo(vendor=vendor)
                self._detect_from_smi(chip_info, vendor, smi_cmd)
                env.driver_version = chip_info.driver_version
                env.sdk_version = chip_info.sdk_version
                env.gpu_model = chip_info.gpu_model

        # GPU 数量
        env.gpu_count = self._detect_gpu_count(vendor)

        return env

    def _detect_os(self) -> Tuple[str, str]:
        """检测操作系统"""
        os_name = ""
        os_version = ""

        # 尝试从 /etc/os-release 读取
        success, output, _ = self._run_cmd("cat /etc/os-release 2>/dev/null")
        if success and output:
            for line in output.split('\n'):
                if line.startswith('NAME='):
                    os_name = line.split('=')[1].strip('"\'')
                elif line.startswith('VERSION='):
                    os_version = line.split('=')[1].strip('"\'')

        # 备用方法
        if not os_name:
            success, output, _ = self._run_cmd("lsb_release -d 2>/dev/null | cut -f2")
            if success and output:
                parts = output.split()
                if parts:
                    os_name = parts[0]
                    os_version = ' '.join(parts[1:])

        return os_name, os_version

    def _detect_kernel(self) -> str:
        """检测内核版本"""
        success, output, _ = self._run_cmd("uname -r")
        if success:
            return output
        return ""

    def _detect_docker_version(self) -> str:
        """检测 Docker 版本"""
        # Docker 版本需要在宿主机检测，不是在容器内
        try:
            result = subprocess.run(
                "docker --version 2>/dev/null | grep -oE '[0-9]+\\.[0-9]+\\.[0-9]+'",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except:
            pass
        return ""

    def _detect_flaggems_version(self) -> str:
        """检测 FlagGems 版本"""
        success, output, _ = self._run_in_torch_env(
            "pip show flag-gems 2>/dev/null | grep -i '^Version:' | awk '{print $2}'"
        )
        if success and output and "error" not in output.lower():
            return output.strip()
        # 备用：尝试 flag_gems 包名
        success, output, _ = self._run_in_torch_env(
            "pip show flag_gems 2>/dev/null | grep -i '^Version:' | awk '{print $2}'"
        )
        if success and output and "error" not in output.lower():
            return output.strip()
        return ""

    def _detect_flagtree_version(self) -> str:
        """检测 FlagTree 版本"""
        # 优先通过 python import 获取版本
        success, output, _ = self._run_in_torch_env(
            "python3 -c \"import flagtree; print(flagtree.__version__)\" 2>/dev/null"
        )
        if success and output and "error" not in output.lower():
            return output.strip()
        # 备用：pip show
        success, output, _ = self._run_in_torch_env(
            "pip show flagtree 2>/dev/null | grep -i '^Version:' | awk '{print $2}'"
        )
        if success and output and "error" not in output.lower():
            return output.strip()
        return ""

    def _detect_vllm_version(self) -> str:
        """检测 vLLM 版本"""
        success, output, _ = self._run_in_torch_env(
            "pip show vllm 2>/dev/null | grep -i '^Version:' | awk '{print $2}'"
        )
        if success and output and "error" not in output.lower():
            return output.strip()
        return ""

    def _detect_gpu_count(self, vendor: Optional[ChipVendor]) -> int:
        """检测 GPU 数量"""
        if not vendor:
            return 0

        vendor_info = VENDOR_DETECT_INFO.get(vendor, {})
        smi_cmd = vendor_info.get("smi_cmd", "")

        if not smi_cmd:
            return 0

        # 通用方法：运行 smi 命令并计数 GPU 相关行
        if vendor == ChipVendor.NVIDIA:
            success, output, _ = self._run_cmd(f"nvidia-smi -L 2>/dev/null | grep -c GPU")
        elif vendor == ChipVendor.ASCEND:
            success, output, _ = self._run_cmd(f"{smi_cmd} 2>/dev/null | grep -c 'NPU ID'")
        elif vendor == ChipVendor.CAMBRICON:
            success, output, _ = self._run_cmd(f"{smi_cmd} 2>/dev/null | grep -c 'Card'")
        else:
            # 大多数厂商可以通过计数 GPU 行来获取数量
            success, output, _ = self._run_cmd(f"{smi_cmd} -L 2>/dev/null | grep -c -i 'gpu\\|device'")
            if not success:
                success, output, _ = self._run_cmd(f"{smi_cmd} 2>/dev/null | grep -c -i 'gpu\\|device'")

        if success and output:
            try:
                return int(output)
            except ValueError:
                pass
        return 0

    def _find_conda_env_with_torch(self) -> str:
        """找到安装了 torch 的 conda 环境名

        在容器中，torch 等包可能安装在非默认的 conda 环境中。
        按优先级依次检查：默认环境（pip show torch）、各 conda 环境。
        结果会被缓存到 self._conda_env 中。

        Returns:
            conda 环境名（如 "py310"），或空字符串表示使用默认环境
        """
        if hasattr(self, '_conda_env'):
            return self._conda_env

        # 1. 先检查默认环境是否有 torch
        success, output, _ = self._run_cmd("pip show torch 2>/dev/null | grep -i version")
        if success and output:
            self._conda_env = ""
            return self._conda_env

        # 2. 列出 conda 环境并逐个检查
        success, envs_output, _ = self._run_cmd(
            "conda env list 2>/dev/null | grep -v '^#' | grep -v '^$' | awk '{print $1}'"
        )
        if success and envs_output:
            for env_name in envs_output.strip().split('\n'):
                env_name = env_name.strip()
                if not env_name or env_name == 'base':
                    continue
                success, output, _ = self._run_cmd(
                    f"conda activate {env_name} && pip show torch 2>/dev/null | grep -i version"
                )
                if success and output:
                    self._conda_env = env_name
                    return self._conda_env

        # 3. 没找到
        self._conda_env = ""
        return self._conda_env

    def _run_in_torch_env(self, cmd: str) -> Tuple[bool, str, str]:
        """在安装了 torch 的 conda 环境中执行命令"""
        env = self._find_conda_env_with_torch()
        if env:
            full_cmd = f"conda activate {env} && {cmd}"
        else:
            full_cmd = cmd
        return self._run_cmd(full_cmd)

    def _detect_python_version(self) -> str:
        """检测Python版本（使用安装了 torch 的 Python 环境）"""
        success, output, _ = self._run_in_torch_env(
            "python3 --version 2>&1 | grep -oE '[0-9]+\\.[0-9]+\\.[0-9]+'"
        )
        if success and output:
            return output
        # 备用：默认 python3
        success, output, _ = self._run_cmd("python3 --version 2>&1 | grep -oE '[0-9]+\\.[0-9]+\\.[0-9]+'")
        if success and output:
            return output
        return "unknown"

    def _detect_arch(self) -> str:
        """检测系统架构"""
        success, output, _ = self._run_cmd("uname -m")
        if success:
            if "x86_64" in output:
                return "amd64"
            elif "aarch64" in output:
                return "arm64"
            return output
        return "amd64"

    def _detect_torch_version(self, backend: str = "") -> str:
        """检测PyTorch版本"""
        success, output, _ = self._run_in_torch_env(
            "pip show torch 2>/dev/null | grep -i '^Version:' | awk '{print $2}'"
        )
        if success and output:
            # 提取版本号，去掉后缀如 +cu118, +musa
            version = re.match(r'[\d.]+', output)
            if version:
                return version.group()
        return "unknown"

    # ==================== 通用 SMI 检测方法 ====================
    def _detect_from_smi(self, info: ChipVersionInfo, vendor: ChipVendor, smi_cmd: str):
        """使用 SMI 工具检测驱动和GPU信息"""
        vendor_info = VENDOR_DETECT_INFO.get(vendor, {})
        info.torch_backend = vendor_info.get("torch_backend", "")

        # 运行 SMI 命令获取完整输出
        success, smi_output, _ = self._run_cmd(f"{smi_cmd} 2>/dev/null")

        if vendor == ChipVendor.NVIDIA:
            # 英伟达 nvidia-smi
            self._parse_nvidia_smi(info, smi_output)
        elif vendor == ChipVendor.METAX:
            # 沐曦 mx-smi
            self._parse_metax_smi(info, smi_output)
        elif vendor == ChipVendor.MTHREADS:
            # 摩尔线程 mthreads-gmi
            self._parse_mthreads_smi(info, smi_output)
        elif vendor == ChipVendor.ILUVATAR:
            # 天数智芯 ixsmi
            self._parse_iluvatar_smi(info, smi_output)
        elif vendor == ChipVendor.ASCEND:
            # 华为昇腾 npu-smi
            self._parse_ascend_smi(info, smi_output)
        elif vendor == ChipVendor.HYGON:
            # 海光 hy-smi
            self._parse_hygon_smi(info, smi_output)
        elif vendor == ChipVendor.KUNLUNXIN:
            # 昆仑芯 xpu-smi
            self._parse_kunlunxin_smi(info, smi_output)
        elif vendor == ChipVendor.CAMBRICON:
            # 寒武纪 cnmon
            self._parse_cambricon_smi(info, smi_output)
        elif vendor == ChipVendor.TSINGMICRO:
            # 清微智能 tsm_smi
            self._parse_tsingmicro_smi(info, smi_output)

        # 设置默认 GPU 型号
        if not info.gpu_model:
            info.gpu_model = vendor.value

    def _parse_nvidia_smi(self, info: ChipVersionInfo, output: str):
        """解析英伟达 nvidia-smi 输出

        nvidia-smi 输出格式示例：
            +-----------------------------------------------------------------------------------------+
            | NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
            |-------------------------------------------+----------------------+----------------------+
            | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
            | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
            |===========================================+======================+======================|
            |   0  NVIDIA A100-SXM4-80GB          On | 00000000:07:00.0 Off |                    0 |
            | N/A   30C    P0              63W / 400W |      0MiB / 81920MiB |      0%      Default |
            +-------------------------------------------+----------------------+----------------------+
        """
        for line in output.split('\n'):
            # 解析 Driver Version 和 CUDA Version
            if 'Driver Version' in line:
                drv_match = re.search(r'Driver Version:\s*([\d.]+)', line)
                if drv_match:
                    info.driver_version = drv_match.group(1)
                cuda_match = re.search(r'CUDA Version:\s*([\d.]+)', line)
                if cuda_match:
                    info.sdk_version = f"cuda{cuda_match.group(1)}"
            # 解析 GPU 型号：匹配设备行中的 NVIDIA GPU 名称
            if not info.gpu_model and 'NVIDIA' in line:
                gpu_match = re.search(r'NVIDIA\s+([A-Za-z0-9 _-]+?)(?:\s+On|\s+Off)', line)
                if gpu_match:
                    info.gpu_model = gpu_match.group(1).strip().lower().replace(' ', '')

    def _parse_metax_smi(self, info: ChipVersionInfo, output: str):
        """解析沐曦 mx-smi 输出"""
        for line in output.split('\n'):
            if 'Driver Version' in line or 'driver version' in line.lower():
                match = re.search(r'[\d.]+', line)
                if match:
                    info.driver_version = match.group()
            elif 'MACA Version' in line or 'maca version' in line.lower():
                match = re.search(r'[\d.]+', line)
                if match:
                    info.sdk_version = f"maca{match.group()}"
            elif 'Product Name' in line or 'product name' in line.lower():
                match = re.search(r':\s*(.+)', line)
                if match:
                    info.gpu_model = match.group(1).strip().lower().replace(' ', '')

    def _parse_mthreads_smi(self, info: ChipVersionInfo, output: str):
        """解析摩尔线程 mthreads-gmi 输出

        mthreads-gmi 输出格式示例：
            mthreads-gmi:2.2.0           Driver Version:3.3.2-server
            ...
            0    MTT S5000            |00000000:2a:00.0    |0%    80902MiB(81920MiB)
        """
        for line in output.split('\n'):
            # Driver Version 紧跟在 "Driver Version:" 后面，
            # 注意同一行还有 "mthreads-gmi:2.2.0"，不能取第一个数字
            if 'Driver Version' in line:
                match = re.search(r'Driver Version[:\s]+([^\s]+)', line)
                if match:
                    info.driver_version = match.group(1)
            # GPU 型号在设备行中，格式如 "0    MTT S5000            |..."
            # 匹配行首数字+空格+名称+|
            if not info.gpu_model:
                gpu_match = re.match(r'^\d+\s+(\S+\s+\S+)\s+\|', line)
                if gpu_match:
                    info.gpu_model = gpu_match.group(1).strip().lower().replace(' ', '')

        # MUSA SDK 版本不在 mthreads-gmi 输出中，从 /usr/local/musa/version* 文件获取
        if not info.sdk_version:
            success, ver_output, _ = self._run_cmd(
                "cat /usr/local/musa/version* 2>/dev/null"
            )
            if success and ver_output:
                # JSON 格式，提取 musa_toolkits 的 version
                try:
                    import json
                    ver_data = json.loads(ver_output)
                    musa_ver = ver_data.get("musa_toolkits", {}).get("version", "")
                    if musa_ver:
                        info.sdk_version = f"musa{musa_ver}"
                except (json.JSONDecodeError, KeyError):
                    # 备用：正则提取
                    match = re.search(r'"musa_toolkits"[^}]*"version"\s*:\s*"([^"]+)"', ver_output)
                    if match:
                        info.sdk_version = f"musa{match.group(1)}"

    def _parse_iluvatar_smi(self, info: ChipVersionInfo, output: str):
        """解析天数智芯 ixsmi 输出"""
        for line in output.split('\n'):
            if 'Driver Version' in line:
                match = re.search(r'[\d.]+', line)
                if match:
                    info.driver_version = match.group()
            elif 'IXRT Version' in line or 'SDK Version' in line:
                match = re.search(r'[\d.]+', line)
                if match:
                    info.sdk_version = f"ixrt{match.group()}"
            elif 'Product' in line or 'product' in line.lower():
                match = re.search(r':\s*(.+)', line)
                if match:
                    info.gpu_model = match.group(1).strip().lower().replace(' ', '')

    def _parse_ascend_smi(self, info: ChipVersionInfo, output: str):
        """解析华为昇腾 npu-smi info 输出"""
        # npu-smi info 输出格式示例：
        # +------------------------------------------------------------------------------------+
        # | npu-smi 24.0.rc3.b050                  Version: 24.0.rc3.b050                      |
        # +------------------------------------------------------------------------------------+
        # | NPU     Name      | Health    | Power(W)     | Temp(C)           |
        # | Chip    Device    | Bus-Id    | AICore(%)    | Memory-Usage(MB)  |
        # +===================================================================================+
        # | 0       910B3     | OK        | 80.5         | 54                |

        for line in output.split('\n'):
            # 解析版本行：npu-smi 24.0.rc3.b050 或 Version: 24.0.rc3.b050
            if 'npu-smi' in line.lower() or 'version' in line.lower():
                # 匹配版本号格式，如 24.0.rc3.b050 或 24.0.0
                match = re.search(r'(\d+\.\d+(?:\.[a-zA-Z0-9.]+)?)', line)
                if match and not info.driver_version:
                    info.driver_version = match.group(1)
            # 解析 NPU 型号：910B3, Ascend910B 等
            elif 'Ascend' in line or '910' in line:
                # 匹配 Ascend910B3 或 910B3 格式
                match = re.search(r'(Ascend\d+[A-Za-z0-9]*|\d{3}[A-Za-z0-9]+)', line)
                if match and not info.gpu_model:
                    info.gpu_model = match.group(1).lower().replace(' ', '')

        # 检测 CANN 版本
        success, cann_output, _ = self._run_cmd("cat /usr/local/Ascend/ascend-toolkit/latest/version.txt 2>/dev/null | head -1")
        if success and cann_output:
            match = re.search(r'(\d+\.\d+[\d.]*)', cann_output)
            if match:
                info.sdk_version = f"cann{match.group(1)}"

        if not info.sdk_version:
            success, torch_output, _ = self._run_cmd(
                "python3 -c \"import torch_npu; print(torch_npu.__version__)\" 2>/dev/null"
            )
            if success and torch_output:
                info.sdk_version = f"cann{torch_output.strip()}"

        if not info.gpu_model:
            info.gpu_model = "ascend910"

    def _parse_hygon_smi(self, info: ChipVersionInfo, output: str):
        """解析海光 hy-smi 输出"""
        for line in output.split('\n'):
            if 'Driver' in line or 'driver' in line:
                match = re.search(r'[\d.]+', line)
                if match:
                    info.driver_version = match.group()
            elif 'Product' in line or 'product' in line:
                match = re.search(r':\s*(.+)', line)
                if match:
                    info.gpu_model = match.group(1).strip().lower().replace(' ', '')

        # 检测 DTK 版本
        success, output, _ = self._run_cmd("cat /opt/dtk/version.txt 2>/dev/null | head -1")
        if success and output:
            match = re.search(r'[\d.]+', output)
            if match:
                info.sdk_version = f"dtk{match.group()}"

        if not info.gpu_model:
            info.gpu_model = "hygondcu"

    def _parse_kunlunxin_smi(self, info: ChipVersionInfo, output: str):
        """解析昆仑芯 xpu-smi 输出"""
        for line in output.split('\n'):
            if 'Driver' in line or 'driver' in line:
                match = re.search(r'[\d.]+', line)
                if match:
                    info.driver_version = match.group()
            elif 'Version' in line or 'version' in line:
                match = re.search(r'[\d.]+', line)
                if match and not info.sdk_version:
                    info.sdk_version = f"xre{match.group()}"
            elif 'Product' in line or 'product' in line:
                match = re.search(r':\s*(.+)', line)
                if match:
                    info.gpu_model = match.group(1).strip().lower().replace(' ', '')

        if not info.gpu_model:
            info.gpu_model = "kunlunxin"

    def _parse_cambricon_smi(self, info: ChipVersionInfo, output: str):
        """解析寒武纪 cnmon 输出"""
        # cnmon 输出格式可能需要 cnmon version 命令
        success, version_output, _ = self._run_cmd("cnmon version 2>/dev/null")
        if success:
            for line in version_output.split('\n'):
                if 'CNToolkit' in line:
                    match = re.search(r'[\d.]+', line)
                    if match:
                        info.sdk_version = f"cntoolkit{match.group()}"
                elif 'Driver' in line:
                    match = re.search(r'[\d.]+', line)
                    if match:
                        info.driver_version = match.group()

        # 从 cnmon info 获取 GPU 型号
        for line in output.split('\n'):
            if 'Product Name' in line:
                match = re.search(r':\s*(.+)', line)
                if match:
                    info.gpu_model = match.group(1).strip().lower().replace(' ', '')

        if not info.gpu_model:
            info.gpu_model = "mlu370"

    def _parse_tsingmicro_smi(self, info: ChipVersionInfo, output: str):
        """解析清微智能 tsm_smi 输出"""
        # 如果主命令没有输出，尝试备用命令
        if not output or output.strip() == "":
            vendor_info = VENDOR_DETECT_INFO.get(ChipVendor.TSINGMICRO, {})
            alt_cmd = vendor_info.get("smi_cmd_alt", "")
            if alt_cmd:
                success, output, _ = self._run_cmd(alt_cmd)

        # 尝试 tsm_smi -l 获取更详细信息
        success, detailed_output, _ = self._run_cmd("tsm_smi -l 2>/dev/null")
        if success and detailed_output:
            output = output + "\n" + detailed_output

        for line in output.split('\n'):
            if 'Driver' in line or 'driver' in line:
                match = re.search(r'[\d.]+', line)
                if match:
                    info.driver_version = match.group()
            elif 'Version' in line or 'version' in line:
                match = re.search(r'[\d.]+', line)
                if match and not info.sdk_version:
                    info.sdk_version = f"tsm{match.group()}"
            elif 'Product' in line or 'product' in line:
                match = re.search(r':\s*(.+)', line)
                if match:
                    info.gpu_model = match.group(1).strip().lower().replace(' ', '')

        if not info.gpu_model:
            info.gpu_model = "tsingmicro"


def sanitize_docker_tag(tag: str) -> str:
    """
    清理 Docker tag 中的无效字符组合

    Docker tag 命名规则：
    - 只允许小写字母、数字、连字符(-)、下划线(_)、点(.)
    - 不允许出现 _- 或 -_ 的组合
    - 不允许连续的 -- 或 __

    Args:
        tag: 原始 tag 字符串

    Returns:
        清理后的合法 tag 字符串
    """
    # Docker tag 不允许 + 字符，替换为 .
    tag = tag.replace('+', '.')
    # 将 _- 替换为 _（去掉连字符）
    tag = re.sub(r'_-', '_', tag)
    # 将 -_ 替换为 _（去掉连字符）
    tag = re.sub(r'-_', '_', tag)
    # 将连续的 -- 替换为单个 -
    tag = re.sub(r'-+', '-', tag)
    # 将连续的 __ 替换为单个 _
    tag = re.sub(r'_+', '_', tag)
    # 去掉开头和结尾的 - 或 _
    tag = tag.strip('-_')

    return tag


def generate_image_tag(
    info: ChipVersionInfo,
    model_name: str,
    harbor_registry: str = "harbor.baai.ac.cn/flagrelease-public",
    tree: str = "none",
    gems_version: str = "",
    cx: str = "none",
    date_tag: str = ""
) -> str:
    """
    生成镜像 tag

    格式: {registry}/flagrelease-{vendor}-release-model_{model}-tree_{tree}-gems_{gems}-cx_{cx}-python_{python}-torch_{backend}-{torch_version}-pcp_{sdk}-gpu_{gpu}-arc_{arch}-driver_{driver}:{date}

    Args:
        info: 芯片版本信息
        model_name: 模型名称
        harbor_registry: Harbor 仓库地址
        tree: tree 参数
        gems_version: gems 版本
        cx: cx 参数
        date_tag: 日期标签，如果为空则自动生成

    Returns:
        完整的镜像 tag
    """
    import datetime

    if not date_tag:
        date_tag = datetime.datetime.now().strftime("%Y%m%d%H%M")

    # 清理模型名称，保留字母数字、连字符和小数点
    clean_model = re.sub(r'[^a-zA-Z0-9.\-]', '-', model_name.lower())
    clean_model = re.sub(r'-+', '-', clean_model).strip('-')

    vendor_name = info.vendor.value

    # 如果版本字段为空，使用占位符
    driver_version = info.driver_version if info.driver_version else "x"
    sdk_version = info.sdk_version if info.sdk_version else "x"
    python_version = info.python_version if info.python_version else "x"
    torch_version = info.torch_version if info.torch_version else "x"
    torch_backend = info.torch_backend if info.torch_backend else "x"
    gpu_model = info.gpu_model if info.gpu_model else "x"
    gpu_model = get_gpu_code(gpu_model, vendor_name)
    arch = info.arch if info.arch else "amd64"
    gems_version = gems_version if gems_version else "none"

    # 构建镜像名（日期放在最后）
    tag_parts = [
        f"flagrelease-{vendor_name}-release",
        f"model_{clean_model}",
        f"tree_{tree}",
        f"gems_{gems_version}",
        f"cx_{cx}",
        f"python_{python_version}",
        f"torch_{torch_backend}-{torch_version}",
        f"pcp_{sdk_version}",
        f"gpu_{gpu_model}",
        f"arc_{arch}",
        f"driver_{driver_version}",
    ]

    image_name = "-".join(tag_parts)

    # 清理可能的无效字符组合
    image_name = sanitize_docker_tag(image_name)

    full_tag = f"{harbor_registry}/{image_name}:{date_tag}"

    return full_tag
