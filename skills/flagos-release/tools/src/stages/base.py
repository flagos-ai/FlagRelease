"""
阶段基类
定义所有阶段的通用接口和功能
"""
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
from enum import Enum

from ..config import PipelineConfig


class StepStatus(Enum):
    """步骤状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """步骤执行结果"""
    step_name: str
    status: StepStatus
    message: str = ""
    output: str = ""
    duration: float = 0.0
    error: Optional[str] = None


@dataclass
class StageResult:
    """阶段执行结果"""
    stage_name: str
    success: bool
    steps: List[StepResult]
    total_duration: float = 0.0
    error: Optional[str] = None


class BaseStage(ABC):
    """阶段基类"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.steps: List[StepResult] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """阶段名称"""
        pass

    @abstractmethod
    def run(self) -> StageResult:
        """执行阶段"""
        pass

    def run_command(
        self,
        cmd: str,
        step_name: str,
        timeout: int = 300,
        check: bool = True,
        shell: bool = True,
        in_container: bool = False,
        container_name: str = None,
        env: dict = None
    ) -> Tuple[bool, str, str]:
        """执行命令"""
        start_time = time.time()

        # in_container 时使用列表参数避免 shell 引号转义问题
        run_args: Any = cmd
        if in_container:
            container = container_name or self.config.container_name
            docker_cmd = ["docker", "exec"]
            for env_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                val = os.environ.get(env_var)
                if val:
                    docker_cmd.extend(["-e", f"{env_var}={val}"])
            docker_cmd.extend([container, "bash", "-c", cmd])
            run_args = docker_cmd
            shell = False

        display_cmd = cmd if isinstance(cmd, str) else " ".join(cmd)
        display_cmd = re.sub(
            r'(MODELSCOPE_API_TOKEN|HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=[^\s]+',
            r'\1=***', display_cmd)
        print(f"[{self.name}] 执行: {step_name}")
        print(f"  命令: {display_cmd[:200]}..." if len(display_cmd) > 200 else f"  命令: {display_cmd}")

        try:
            result = subprocess.run(
                run_args,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )

            duration = time.time() - start_time
            stdout = result.stdout
            stderr = result.stderr

            if result.returncode == 0:
                print(f"  + 成功 (耗时 {duration:.2f}s)")
                self.steps.append(StepResult(
                    step_name=step_name,
                    status=StepStatus.SUCCESS,
                    output=stdout,
                    duration=duration
                ))
                return True, stdout, stderr
            else:
                error_msg = stderr or stdout or f"命令返回非零状态码: {result.returncode}"
                print(f"  x 失败: {error_msg[:200]}")
                self.steps.append(StepResult(
                    step_name=step_name,
                    status=StepStatus.FAILED,
                    output=stdout,
                    error=error_msg,
                    duration=duration
                ))
                return False, stdout, stderr

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            error_msg = f"命令执行超时 ({timeout}秒)"
            print(f"  x 超时: {error_msg}")
            self.steps.append(StepResult(
                step_name=step_name,
                status=StepStatus.FAILED,
                error=error_msg,
                duration=duration
            ))
            return False, "", error_msg

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            print(f"  x 异常: {error_msg}")
            self.steps.append(StepResult(
                step_name=step_name,
                status=StepStatus.FAILED,
                error=error_msg,
                duration=duration
            ))
            return False, "", error_msg

    def skip_step(self, step_name: str, reason: str = "配置跳过"):
        """跳过步骤"""
        print(f"[{self.name}] 跳过: {step_name} ({reason})")
        self.steps.append(StepResult(
            step_name=step_name,
            status=StepStatus.SKIPPED,
            message=reason
        ))

    def make_result(self, success: bool, error: str = None) -> StageResult:
        """生成阶段结果"""
        total_duration = sum(step.duration for step in self.steps)
        return StageResult(
            stage_name=self.name,
            success=success,
            steps=self.steps,
            total_duration=total_duration,
            error=error
        )
