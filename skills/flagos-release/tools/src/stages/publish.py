"""
发布阶段
包含：镜像打 tag、推送 Harbor、生成 README、发布到 ModelScope/HuggingFace
"""
import json
import os
import time
import base64
import subprocess
from typing import Optional, List, Tuple
from pathlib import Path

from .base import BaseStage, StageResult, StepResult, StepStatus
from ..chip_detector import ChipDetector, ChipVendor, EnvironmentInfo, generate_image_tag

# 上传重试配置
UPLOAD_MAX_RETRIES = 5
UPLOAD_RETRY_DELAY = 10
UPLOAD_MAX_DELAY = 300
UPLOAD_TIMEOUT = 3600


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_files_in_directory(directory: str, extensions: List[str] = None) -> List[str]:
    """获取目录中的所有文件"""
    if not os.path.exists(directory):
        return []
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            if extensions:
                if any(filename.endswith(ext) for ext in extensions):
                    files.append(file_path)
            else:
                files.append(file_path)
    return files


class PublishStage(BaseStage):
    """发布阶段"""

    def __init__(self, config):
        super().__init__(config)
        self.env_info: Optional[EnvironmentInfo] = None

    def _get_proxy_list(self) -> List[str]:
        """从环境变量获取代理列表"""
        proxy_str = os.environ.get("FLAGOS_PROXY_LIST", "")
        if proxy_str:
            return [p.strip() for p in proxy_str.split(",") if p.strip()]
        current = os.environ.get("https_proxy") or os.environ.get("http_proxy", "")
        return [current] if current else []

    def _with_proxy_fallback(self, operation_name: str, func, *args, **kwargs) -> bool:
        """执行操作，失败时切换代理重试"""
        proxies = self._get_proxy_list()
        if not proxies:
            return func(*args, **kwargs)

        for i, proxy in enumerate(proxies):
            os.environ["http_proxy"] = proxy
            os.environ["https_proxy"] = proxy
            result = func(*args, **kwargs)
            if result:
                return True
            if i < len(proxies) - 1:
                next_proxy = proxies[i + 1]
                print(f"  ⚠ [{operation_name}] 代理 {proxy} 失败，切换到 {next_proxy}")
        return False

    @property
    def name(self) -> str:
        return "发布阶段"

    def run(self) -> StageResult:
        """执行发布阶段"""
        print(f"\n{'='*60}")
        print(f"开始执行: {self.name}")
        print(f"{'='*60}")

        start_time = time.time()
        publish_config = self.config.publish
        harbor_failed = False

        # 如果已有 Harbor 镜像地址，跳过 commit/tag/push
        if publish_config.existing_harbor_image:
            existing_image = publish_config.existing_harbor_image
            print(f"  已配置 existing_harbor_image: {existing_image}")
            print(f"  跳过容器 commit、镜像打 tag、推送 Harbor 步骤")
            self.config.publish.harbor_path = existing_image
            self.config.model_info.image_harbor_path = existing_image
            self.skip_step("容器 commit", "已有 Harbor 镜像")
            self.skip_step("镜像打 tag", "已有 Harbor 镜像")
            self.skip_step("推送 Harbor", "已有 Harbor 镜像")
        else:
            # 0. 如果输入是容器，先 commit 为镜像
            if self.config.input_type == 'container':
                success = self._commit_container()
                if not success:
                    return self.make_result(False, "容器 commit 失败")

            # 1. 镜像打 tag
            if publish_config.tag_image:
                success = self._tag_image()
                if not success:
                    return self.make_result(False, "镜像打 tag 失败")
            else:
                self.skip_step("镜像打 tag", "配置跳过")

            # 2. 推送到 Harbor（支持代理切换重试）
            if publish_config.push_harbor:
                success = self._with_proxy_fallback("Harbor push", self._push_to_harbor)
                if not success:
                    harbor_failed = True
                    print("  ⚠ Harbor 推送失败，继续执行后续步骤（README 生成、数据回传）")
            else:
                self.skip_step("推送 Harbor", "配置跳过")

        # 3. 生成 README
        readme_path = None
        if self.config.plugin_image_mode and not self.config.plugin_qualified:
            self.skip_step("生成 README", "Plugin 不达标，跳过 README 更新")
        elif publish_config.generate_readme:
            readme_path = self._generate_readme()
            if not readme_path:
                return self.make_result(False, "生成 README 失败")
        else:
            self.skip_step("生成 README", "配置跳过")

        # 4. 发布到 ModelScope
        ms_failed = False
        if self.config.plugin_image_mode and not self.config.plugin_qualified:
            self.skip_step("更新 ModelScope README", "Plugin 不达标，跳过")
        elif self.config.plugin_image_mode:
            # plugin 达标：更新步骤8原仓库的 README，不创建新仓库、不上传权重
            if publish_config.base_modelscope_model_id and readme_path:
                success = self._update_repo_readme(
                    publish_config.base_modelscope_model_id, "modelscope", readme_path)
                if not success:
                    ms_failed = True
                    print("  ⚠ 更新 ModelScope README 失败，继续执行 HuggingFace")
            else:
                self.skip_step("更新 ModelScope README", "无步骤8仓库信息或无 README")
        elif publish_config.publish_modelscope:
            success = self._with_proxy_fallback("ModelScope", self._publish_to_modelscope, readme_path)
            if not success:
                ms_failed = True
                print("  ⚠ ModelScope 发布失败，继续执行 HuggingFace 上传")
        else:
            self.skip_step("发布到 ModelScope", "配置跳过")

        # 5. 发布到 HuggingFace
        hf_failed = False
        if self.config.plugin_image_mode and not self.config.plugin_qualified:
            self.skip_step("更新 HuggingFace README", "Plugin 不达标，跳过")
        elif self.config.plugin_image_mode:
            # plugin 达标：更新步骤8原仓库的 README
            if publish_config.base_huggingface_repo_id and readme_path:
                success = self._update_repo_readme(
                    publish_config.base_huggingface_repo_id, "huggingface", readme_path)
                if not success:
                    hf_failed = True
                    print("  ⚠ 更新 HuggingFace README 失败")
            else:
                self.skip_step("更新 HuggingFace README", "无步骤8仓库信息或无 README")
        elif publish_config.publish_huggingface:
            success = self._with_proxy_fallback("HuggingFace", self._publish_to_huggingface, readme_path)
            if not success:
                hf_failed = True
                print("  ⚠ HuggingFace 发布失败")
        else:
            self.skip_step("发布到 HuggingFace", "配置跳过")

        # 6. 数据回传到宿主机
        self._sync_to_host()

        upload_failed = ms_failed or hf_failed
        duration = time.time() - start_time
        if harbor_failed or upload_failed:
            failures = []
            if harbor_failed:
                failures.append("Harbor")
            if ms_failed:
                failures.append("ModelScope")
            if hf_failed:
                failures.append("HuggingFace")
            print(f"\n⚠ {self.name} 完成，但部分平台失败: {', '.join(failures)} (总耗时 {duration:.2f}s)")
        else:
            print(f"\n+ {self.name} 完成 (总耗时 {duration:.2f}s)")

        # 输出结构化摘要，供编排层写入 context.yaml release 字段
        model_name = self.config.model_info.flagrelease_name or self.config.model_info.output_name or ""
        ms_model_id = publish_config.modelscope_model_id or (f"FlagRelease/{model_name}" if model_name else "")
        hf_repo_id = publish_config.huggingface_repo_id or (f"FlagRelease/{model_name}" if model_name else "")
        release_summary = {
            "harbor_image": publish_config.harbor_path or "",
            "modelscope_model_id": ms_model_id if not ms_failed else "",
            "modelscope_url": f"https://modelscope.cn/models/{ms_model_id}" if ms_model_id and not ms_failed else "",
            "huggingface_repo_id": hf_repo_id if not hf_failed else "",
            "huggingface_url": f"https://hf-mirror.com/{hf_repo_id}" if hf_repo_id and not hf_failed else "",
        }
        print(f"\n[RELEASE_SUMMARY]{json.dumps(release_summary, ensure_ascii=False)}[/RELEASE_SUMMARY]")

        return self.make_result(not harbor_failed and not upload_failed)

    def _sync_to_host(self):
        """将容器内 /flagos-workspace 的产出同步到宿主机工作目录。

        检查宿主机目标目录是否已有对应文件，缺失或大小不一致则 docker cp 回传。
        回传失败不影响整体流水线结果。
        """
        container_name = self.config.container_name
        host_base = self.config.host_workspace_base

        if not container_name or not host_base:
            self.skip_step("数据回传", "缺少容器名/宿主机路径")
            return

        # host_workspace_base 已包含完整路径（如 /data/flagos-workspace/Qwen/Qwen2.5-0.5B-Instruct）
        # 直接使用，不再拼接 model_source
        host_target = host_base
        print(f"\n[数据回传] 同步到宿主机: {host_target}")

        # 整目录 docker cp，确保子目录（如 results/outputs/...）也被同步
        sync_dirs = ["results", "traces", "logs"]
        synced = 0
        failed = 0

        for dir_name in sync_dirs:
            container_dir = f"/flagos-workspace/{dir_name}"
            host_dir = os.path.join(host_target, dir_name)
            os.makedirs(host_dir, exist_ok=True)

            try:
                cp_result = subprocess.run(
                    ["docker", "cp", f"{container_name}:{container_dir}/.", host_dir + "/"],
                    capture_output=True, text=True, timeout=120
                )
                if cp_result.returncode == 0:
                    print(f"  ✓ {dir_name}/ 已同步")
                    synced += 1
                else:
                    print(f"  ⚠ {dir_name}/ 同步失败: {cp_result.stderr.strip()}")
                    failed += 1
            except Exception as e:
                print(f"  ⚠ {dir_name}/ 同步异常: {e}")
                failed += 1

        # context.yaml 单独处理：回传时重命名为 context_snapshot.yaml
        config_dir = os.path.join(host_target, "config")
        os.makedirs(config_dir, exist_ok=True)
        try:
            cp_result = subprocess.run(
                ["docker", "cp",
                 f"{container_name}:/flagos-workspace/shared/context.yaml",
                 os.path.join(config_dir, "context_snapshot.yaml")],
                capture_output=True, text=True, timeout=30
            )
            if cp_result.returncode == 0:
                print(f"  ✓ context_snapshot.yaml 已同步")
                synced += 1
            else:
                print(f"  ⚠ context_snapshot.yaml 同步失败: {cp_result.stderr.strip()}")
                failed += 1
        except Exception as e:
            print(f"  ⚠ context_snapshot.yaml 同步异常: {e}")
            failed += 1

        summary = f"同步 {synced} 个目录/文件, 失败 {failed} 个"
        print(f"  {summary}")

        self.steps.append(StepResult(
            step_name="数据回传到宿主机",
            status=StepStatus.SUCCESS if failed == 0 else StepStatus.FAILED,
            message=summary
        ))

    def _commit_container(self) -> bool:
        """将容器 commit 为镜像"""
        container_name = self.config.container_name
        if not container_name:
            print("  x 容器名称未配置")
            return False

        model_name = self.config.model_info.output_name or "model"
        commit_image_name = f"flagrelease-commit-{container_name}:{model_name}".lower().replace("/", "-")

        print(f"  正在将容器 {container_name} commit 为镜像 {commit_image_name}...")

        cmd = f"docker commit {container_name} {commit_image_name}"
        success, stdout, stderr = self.run_command(
            cmd=cmd,
            step_name="容器 commit",
            timeout=600
        )

        if success:
            self.config.publish.image_source = commit_image_name
            print(f"  + 容器已 commit 为镜像: {commit_image_name}")

        return success

    def _tag_image(self) -> bool:
        """镜像打 tag"""
        publish_config = self.config.publish
        chip_config = self.config.chip

        source_image = publish_config.image_source
        if not source_image:
            print("  x 源镜像未配置")
            return False

        if chip_config.auto_generate_tag:
            target_tag = self._generate_auto_tag()
            if not target_tag:
                return False
        else:
            target_tag = publish_config.image_target_tag or publish_config.harbor_path

        if not target_tag:
            print("  x 目标 tag 未配置")
            return False

        self.config.publish.harbor_path = target_tag
        self.config.model_info.image_harbor_path = target_tag

        cmd = f"docker tag {source_image} {target_tag}"
        success, _, _ = self.run_command(
            cmd=cmd,
            step_name="镜像打 tag",
            timeout=60
        )

        if success:
            print(f"  生成的镜像 tag: {target_tag}")

        return success

    def _generate_auto_tag(self) -> Optional[str]:
        """自动生成镜像 tag"""
        chip_config = self.config.chip
        publish_config = self.config.publish

        print("  正在生成镜像 tag...")

        try:
            # 优先使用 auto_fill_config 已生成的 tag
            if publish_config.image_target_tag:
                print(f"  使用已生成的 tag: {publish_config.image_target_tag}")

                print(f"    芯片厂商: {chip_config.vendor}")
                print(f"    驱动版本: {chip_config.driver_version}")
                print(f"    SDK版本: {chip_config.sdk_version}")
                print(f"    PyTorch版本: {chip_config.torch_version}")
                print(f"    Python版本: {chip_config.python_version}")
                print(f"    GPU型号: {chip_config.gpu_model}")
                print(f"    FlagGems版本: {chip_config.gems_version}")
                print(f"    FlagTree版本: {chip_config.tree}")

                self.steps.append(StepResult(
                    step_name="自动生成 tag",
                    status=StepStatus.SUCCESS,
                    output=publish_config.image_target_tag,
                    duration=0.0
                ))
                return publish_config.image_target_tag

            # 如果 auto_fill_config 没有生成 tag，则在此处生成
            if chip_config.vendor == "auto":
                container_name = self.config.container_name
                detector = ChipDetector(container_name=container_name if container_name else None)
                vendor = detector.detect_vendor()
                if vendor is None:
                    print("  x 无法自动检测芯片厂商，请在配置中手动指定 chip.vendor")
                    return None
            else:
                try:
                    vendor = ChipVendor(chip_config.vendor)
                except ValueError:
                    print(f"  x 未知的芯片厂商: {chip_config.vendor}")
                    return None

            from ..chip_detector import ChipVersionInfo, VENDOR_DETECT_INFO
            vendor_info = VENDOR_DETECT_INFO.get(vendor, {})
            chip_info = ChipVersionInfo(
                vendor=vendor,
                driver_version=chip_config.driver_version,
                sdk_version=chip_config.sdk_version,
                torch_backend=vendor_info.get("torch_backend", ""),
                torch_version=chip_config.torch_version,
                python_version=chip_config.python_version,
                gpu_model=chip_config.gpu_model,
                arch="amd64",
            )

            from ..config import _extract_model_name
            model_name = _extract_model_name(self.config.model_info.source_of_model_weights) or self.config.model_info.flagrelease_name_pre
            tag = generate_image_tag(
                info=chip_info,
                model_name=model_name,
                harbor_registry=chip_config.harbor_registry,
                tree=chip_config.tree,
                gems_version=chip_config.gems_version,
                cx=chip_config.cx,
                date_tag=chip_config.date_tag
            )

            self.steps.append(StepResult(
                step_name="自动生成 tag",
                status=StepStatus.SUCCESS,
                output=tag,
                duration=0.0
            ))
            return tag

        except Exception as e:
            print(f"  x 自动生成 tag 失败: {e}")
            return None

    def _ensure_harbor_login(self, harbor_path: str) -> bool:
        """确保已登录 Harbor，环境变量存在时强制重新登录"""
        # 从 harbor_path 提取 registry 地址（如 harbor.baai.ac.cn）
        registry = harbor_path.split("/")[0]

        # 环境变量优先：有凭证就强制重新登录，避免复用旧凭证导致权限不匹配
        user = os.environ.get("HARBOR_USER", "")
        password = os.environ.get("HARBOR_PASSWORD", "")
        if user and password:
            print(f"  正在登录 Harbor: {registry} (使用环境变量凭证) ...")
            cmd = f"printf '%s' \"{password}\" | docker login --username={user} --password-stdin https://{registry}/"
            success, stdout, stderr = self.run_command(
                cmd=cmd,
                step_name="Harbor 登录",
                timeout=60,
            )
            if not success:
                print(f"  x Harbor 登录失败，请检查 HARBOR_USER / HARBOR_PASSWORD")
            return success

        # 无环境变量，检查是否已有登录凭证
        import json as _json
        docker_config_path = os.path.expanduser("~/.docker/config.json")
        if os.path.exists(docker_config_path):
            try:
                with open(docker_config_path) as f:
                    docker_config = _json.load(f)
                auths = docker_config.get("auths", {})
                if registry in auths or f"https://{registry}" in auths or f"https://{registry}/" in auths:
                    print(f"  Harbor 已登录: {registry} (使用已有凭证)")
                    return True
            except Exception:
                pass

        print(f"  x Harbor 未登录且环境变量 HARBOR_USER / HARBOR_PASSWORD 未设置")
        print(f"    请设置环境变量或手动执行: docker login https://{registry}/")
        self.steps.append(StepResult(
            step_name="Harbor 登录",
            status=StepStatus.FAILED,
            error="HARBOR_USER / HARBOR_PASSWORD 未设置",
        ))
        return False

    def _push_to_harbor(self) -> bool:
        """推送镜像到 Harbor"""
        publish_config = self.config.publish
        harbor_path = publish_config.harbor_path

        if not harbor_path:
            print("  x Harbor 路径未配置")
            return False

        # 确保已登录 Harbor
        if not self._ensure_harbor_login(harbor_path):
            return False

        cmd = f"docker push {harbor_path}"
        step_name = "推送 Harbor"
        timeout = 7200

        print(f"[{self.name}] 执行: {step_name}")
        print(f"  命令: {cmd}")

        start_time = time.time()
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            output_lines = []
            for line in process.stdout:
                line = line.rstrip('\n')
                print(f"  {line}")
                output_lines.append(line)

            process.wait(timeout=timeout)
            duration = time.time() - start_time
            output = '\n'.join(output_lines)

            if process.returncode == 0:
                print(f"  + 成功 (耗时 {duration:.2f}s)")
                self.steps.append(StepResult(
                    step_name=step_name,
                    status=StepStatus.SUCCESS,
                    output=output,
                    duration=duration,
                ))
                return True
            else:
                error_msg = output or f"命令返回非零状态码: {process.returncode}"
                print(f"  x 失败: {error_msg[:200]}")
                self.steps.append(StepResult(
                    step_name=step_name,
                    status=StepStatus.FAILED,
                    output=output,
                    error=error_msg,
                    duration=duration,
                ))
                return False

        except subprocess.TimeoutExpired:
            process.kill()
            duration = time.time() - start_time
            error_msg = f"命令执行超时 ({timeout}秒)"
            print(f"  x 超时: {error_msg}")
            self.steps.append(StepResult(
                step_name=step_name,
                status=StepStatus.FAILED,
                error=error_msg,
                duration=duration,
            ))
            return False

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            print(f"  x 异常: {error_msg}")
            self.steps.append(StepResult(
                step_name=step_name,
                status=StepStatus.FAILED,
                error=error_msg,
                duration=duration,
            ))
            return False

    # ==================== README 生成 ====================

    def _generate_readme(self) -> Optional[str]:
        """生成 README"""
        publish_config = self.config.publish

        if publish_config.readme_script_path and os.path.exists(publish_config.readme_script_path):
            return self._generate_readme_by_script()

        return self._generate_readme_by_template()

    def _generate_readme_by_script(self) -> Optional[str]:
        """使用脚本生成 README"""
        publish_config = self.config.publish
        model_info = self.config.model_info

        import yaml
        import tempfile

        config_data = {
            "output_name": model_info.output_name,
            "vendor": model_info.vendor,
            "docker_version": model_info.docker_version,
            "ubuntu_version": model_info.ubuntu_version,
            "source_of_model_weights": model_info.source_of_model_weights,
            "flagrelease_name": model_info.flagrelease_name,
            "flagrelease_name_pre": model_info.flagrelease_name_pre,
            "image_harbor_path": model_info.image_harbor_path,
            "container_run_cmd": model_info.container_run_cmd,
            "serve_start_cmd": model_info.serve_start_cmd,
            "serve_infer_cmd": model_info.serve_infer_cmd,
            "new_model_introduction": model_info.new_model_introduction,
            "evaluation_table": self._generate_evaluation_table(),
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f, allow_unicode=True)
            temp_config_path = f.name

        try:
            cmd = f"python {publish_config.readme_script_path} --config {temp_config_path} --output {publish_config.readme_output_path}"
            success, stdout, stderr = self.run_command(
                cmd=cmd,
                step_name="生成 README (脚本)",
                timeout=120
            )
            if success:
                return publish_config.readme_output_path
            return None
        finally:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    def _generate_readme_by_template(self) -> Optional[str]:
        """使用模板生成 README"""
        model_info = self.config.model_info
        publish_config = self.config.publish
        chip_config = self.config.chip

        if self.env_info is None:
            container_name = self.config.container_name
            try:
                detector = ChipDetector(container_name=container_name if container_name else None)
                vendor = None
                if chip_config.vendor != "auto":
                    try:
                        vendor = ChipVendor(chip_config.vendor)
                    except ValueError:
                        pass
                self.env_info = detector.detect_environment(vendor)
            except Exception as e:
                print(f"  警告: 无法检测环境信息: {e}")

        # 查找模板文件
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "templates", "README_TEMPLATE.md"
        )

        if not os.path.exists(template_path):
            print(f"  警告: 模板文件不存在: {template_path}，使用内置模板")
            return self._generate_readme_builtin()

        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
        except Exception as e:
            print(f"  警告: 无法读取模板文件: {e}，使用内置模板")
            return self._generate_readme_builtin()

        template_vars = self._prepare_template_vars()

        readme_content = template_content
        for key, value in template_vars.items():
            placeholder = "{{" + key + "}}"
            readme_content = readme_content.replace(placeholder, str(value))

        output_path = self._get_readme_output_path()
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)

            print(f"  + README 已生成: {output_path}")
            self.steps.append(StepResult(
                step_name="生成 README",
                status=StepStatus.SUCCESS,
                output=output_path
            ))
            return output_path

        except Exception as e:
            print(f"  x 生成 README 失败: {e}")
            return None

    def _get_readme_output_path(self) -> str:
        """获取 README 输出路径"""
        flagrelease_name = self.config.model_info.flagrelease_name
        if not flagrelease_name:
            flagrelease_name = self.config.model_info.output_name or "model"
        return os.path.join("output", flagrelease_name, "README.md")

    def _get_upload_directory(self, readme_path: Optional[str] = None) -> str:
        """获取上传目录"""
        publish_config = self.config.publish
        readme_output_path = self._get_readme_output_path()
        output_dir = os.path.dirname(readme_output_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 如果启用了权重上传，将权重文件链接到 output 目录
        if publish_config.upload_weights and publish_config.weights_dir:
            weights_dir = publish_config.weights_dir
            if os.path.exists(weights_dir):
                # 宿主机上权重目录存在，直接链接
                print(f"  准备权重文件从: {weights_dir}")
                weight_files = get_files_in_directory(weights_dir)
                for wf in weight_files:
                    rel_path = os.path.relpath(wf, weights_dir)
                    dest_path = os.path.join(output_dir, rel_path)
                    dest_dir = os.path.dirname(dest_path)

                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir, exist_ok=True)

                    if not os.path.exists(dest_path):
                        try:
                            os.symlink(os.path.abspath(wf), dest_path)
                        except OSError:
                            try:
                                os.link(wf, dest_path)
                            except OSError:
                                import shutil
                                shutil.copy2(wf, dest_path)

                print(f"    已准备 {len(weight_files)} 个权重文件")
            elif self.config.container_name:
                # 宿主机上不存在，尝试从容器 docker cp 权重到 output 目录
                # weights_dir 可能是 local_path（宿主机路径），容器内未必相同
                # 依次尝试 weights_dir 和 container_path
                container = self.config.container_name
                container_path = self.config.model_info.source_of_model_weights
                # 从 config 中获取容器内路径（通过 serve_start_cmd 中的模型路径推断）
                # 更直接：尝试 weights_dir，失败则用常见容器路径
                candidate_paths = [weights_dir]
                # 如果有 serve_start_cmd，从中提取容器内模型路径
                serve_cmd = self.config.model_info.serve_start_cmd or ""
                if "vllm serve " in serve_cmd:
                    parts = serve_cmd.split("vllm serve ", 1)[1].split()
                    if parts:
                        cmd_model_path = parts[0].strip().rstrip("\\")
                        if cmd_model_path != weights_dir:
                            candidate_paths.append(cmd_model_path)

                try:
                    print(f"  宿主机无权重目录 {weights_dir}，从容器 {container} 复制...")
                    copied = False
                    for cpath in candidate_paths:
                        try:
                            result = subprocess.run(
                                ["docker", "exec", container, "test", "-d", cpath],
                                capture_output=True, timeout=5
                            )
                            if result.returncode == 0:
                                cp_result = subprocess.run(
                                    ["docker", "cp", f"{container}:{cpath}/.", output_dir],
                                    capture_output=True, text=True, timeout=600
                                )
                                if cp_result.returncode == 0:
                                    n = len([f for f in os.listdir(output_dir) if f != "README.md"])
                                    print(f"    已从容器 {cpath} 复制 {n} 个权重文件")
                                    copied = True
                                    break
                        except Exception:
                            continue
                    if not copied:
                        print(f"    ⚠ 容器内未找到权重目录: {candidate_paths}")
                except Exception as e:
                    print(f"    ⚠ 从容器复制权重异常: {e}")

        return output_dir

    def _prepare_template_vars(self) -> dict:
        """准备模板变量"""
        model_info = self.config.model_info
        chip_config = self.config.chip

        vars = {}

        vars["flagrelease_name"] = model_info.flagrelease_name or model_info.output_name
        vars["output_name"] = model_info.output_name
        vars["source_of_model_weights"] = model_info.source_of_model_weights
        vars["new_model_introduction"] = model_info.new_model_introduction or "新模型介绍，待定...."

        if self.env_info and self.env_info.vendor:
            vars["vendor"] = self.env_info.vendor.value
            vars["vendor_cn_name"] = self.env_info.vendor_cn_name
            vars["vendor_display"] = self.env_info.vendor.value.capitalize()
        else:
            vars["vendor"] = model_info.vendor.lower() if model_info.vendor else "unknown"
            vars["vendor_cn_name"] = model_info.vendor or "Unknown"
            vars["vendor_display"] = model_info.vendor.capitalize() if model_info.vendor else "Unknown"

        if self.env_info:
            vars["driver_version"] = self.env_info.driver_version or "N/A"
            vars["docker_version"] = self.env_info.docker_version or model_info.docker_version or "N/A"
            vars["os_info"] = f"{self.env_info.os_name} {self.env_info.os_version}".strip() or model_info.ubuntu_version or "Linux"
            vars["kernel_version"] = self.env_info.kernel_version or "N/A"
            vars["sdk_name"] = self.env_info.sdk_name or ""
            vars["sdk_version"] = self.env_info.sdk_version or "N/A"
            vars["gpu_model"] = self.env_info.gpu_model or "N/A"
            vars["python_version"] = self.env_info.python_version or "N/A"
            vars["torch_version"] = self.env_info.torch_version or "N/A"
            vars["torch_backend"] = self.env_info.torch_backend or "N/A"
            vars["flagtree_version"] = self.env_info.flagtree_version or chip_config.tree or "N/A"
            vars["flaggems_version"] = self.env_info.flaggems_version or chip_config.gems_version or "N/A"
            if self.env_info.vllm_version:
                vars["vllm_row"] = f"| vLLM | Version: {self.env_info.vllm_version} |"
            else:
                vars["vllm_row"] = ""
        else:
            vars["driver_version"] = "N/A"
            vars["docker_version"] = model_info.docker_version or "N/A"
            vars["os_info"] = model_info.ubuntu_version or "Linux"
            vars["kernel_version"] = "N/A"
            vars["sdk_name"] = ""
            vars["sdk_version"] = "N/A"
            vars["gpu_model"] = "N/A"
            vars["python_version"] = "N/A"
            vars["torch_version"] = "N/A"
            vars["torch_backend"] = "N/A"
            vars["flagtree_version"] = chip_config.tree or "N/A"
            vars["flaggems_version"] = chip_config.gems_version or "N/A"
            vars["vllm_row"] = ""

        vars["image_harbor_path"] = model_info.image_harbor_path or self.config.publish.harbor_path or "N/A"
        image_harbor = vars["image_harbor_path"]
        vars["image_pull_cmd"] = f"docker pull {image_harbor}" if image_harbor != "N/A" else ""
        vars["weights_local_path"] = self.config.publish.weights_dir or "/data/models/" + (model_info.source_of_model_weights.split("/")[-1] if model_info.source_of_model_weights else "model")

        vars["container_run_cmd"] = model_info.container_run_cmd.strip() if model_info.container_run_cmd else ""
        vars["serve_start_cmd"] = model_info.serve_start_cmd.strip() if model_info.serve_start_cmd else ""
        vars["serve_infer_cmd"] = model_info.serve_infer_cmd.strip() if model_info.serve_infer_cmd else self._default_curl_cmd()

        vars["evaluation_table"] = self._generate_evaluation_table()

        return vars

    def _default_curl_cmd(self) -> str:
        """生成默认的 curl 调用命令"""
        return '''curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "flagOS",
    "messages": [{"role": "user", "content": "你好"}]
  }' '''

    def _generate_evaluation_table(self) -> str:
        """生成固定 3 行指标的评测表格（GPQA_Diamond / ERQA / Aime24），严格按厂商模版格式"""
        flagrelease_name = self.config.model_info.flagrelease_name or self.config.model_info.output_name or "model"
        col_origin = f"{flagrelease_name}-Origin"
        col_flagos = f"{flagrelease_name}-FlagOS"

        results = self.config.model_info.evaluation_results
        if not results:
            results = self._load_results_from_dir()

        scores = {}
        for item in results:
            metric = item.get('metric', '')
            scores[metric] = {
                'origin': item.get('origin', '-'),
                'flagos': item.get('flagos', '-'),
            }

        fixed_metrics = ["GPQA_Diamond", "ERQA", "Aime24"]

        header = f"| Metrics      | {col_origin} | {col_flagos} |"
        separator = f"|--------------|{''.ljust(len(col_origin) + 2, '-')}|{''.ljust(len(col_flagos) + 2, '-')}|"

        rows = [header, separator]
        for metric in fixed_metrics:
            data = scores.get(metric, {})
            if not data:
                normalized = metric.lower().replace('_', '')
                for k, v in scores.items():
                    if k.lower().replace('_', '').replace(' ', '').replace('(', '').replace(')', '') == normalized:
                        data = v
                        break
            origin = data.get('origin', '-')
            flagos = data.get('flagos', '-')
            if origin is None or origin == 'N/A':
                origin = '-'
            if flagos is None or flagos == 'N/A':
                flagos = '-'
            rows.append(f"| {metric} | {origin} | {flagos} |")

        return "\n".join(rows)

    def _load_results_from_dir(self) -> List[dict]:
        """从 results_dir 自动读取精度评测结果，返回兼容 evaluation_results 的格式"""
        results_dir = self.config.publish.results_dir
        if not results_dir or not os.path.isdir(results_dir):
            return []

        results = []

        gpqa_native_path = os.path.join(results_dir, "gpqa_native.json")
        native_score = self._read_json_field(gpqa_native_path, "score")

        if self.config.plugin_image_mode:
            # plugin 模式：读取 gpqa_plugin.json
            gpqa_plugin_path = os.path.join(results_dir, "gpqa_plugin.json")
            plugin_score = self._read_json_field(gpqa_plugin_path, "score")
            if native_score is not None or plugin_score is not None:
                results.append({
                    "metric": "GPQA_Diamond",
                    "origin": native_score if native_score is not None else "N/A",
                    "flagos": plugin_score if plugin_score is not None else "N/A",
                })
        else:
            # 主流程：读取 gpqa_flagos / gpqa_flagos_optimized
            gpqa_flagos_path = os.path.join(results_dir, "gpqa_flagos.json")
            gpqa_optimized_path = os.path.join(results_dir, "gpqa_flagos_optimized.json")
            optimized_score = self._read_json_field(gpqa_optimized_path, "score")
            flagos_score = optimized_score if optimized_score is not None else self._read_json_field(gpqa_flagos_path, "score")
            if native_score is not None or flagos_score is not None:
                results.append({
                    "metric": "GPQA_Diamond",
                    "origin": native_score if native_score is not None else "N/A",
                    "flagos": flagos_score if flagos_score is not None else "N/A",
                })

        return results

    @staticmethod
    def _read_json_field(filepath: str, field: str):
        """安全读取 JSON 文件中的某个字段"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get(field)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def _generate_readme_builtin(self) -> Optional[str]:
        """使用内置模板生成 README（与 README_TEMPLATE.md 结构一致）"""
        model_info = self.config.model_info
        vendor_display = model_info.vendor.capitalize() if model_info.vendor else "Unknown"
        flagrelease_name = model_info.flagrelease_name or model_info.output_name or "model"
        new_model_intro = model_info.new_model_introduction or "新模型介绍，待定...."
        eval_table = self._generate_evaluation_table()
        docker_version = model_info.docker_version or "N/A"
        os_info = model_info.ubuntu_version or "Linux"
        image_harbor = model_info.image_harbor_path or self.config.publish.harbor_path or ""
        image_pull_cmd = f"docker pull {image_harbor}" if image_harbor else ""
        container_run_cmd = model_info.container_run_cmd or ""
        serve_start_cmd = model_info.serve_start_cmd or ""
        serve_infer_cmd = model_info.serve_infer_cmd or self._default_curl_cmd()
        source = model_info.source_of_model_weights or "xxx/xxxxxxxx"

        readme_content = f"""# Introduction
{new_model_intro}

### Integrated Deployment
- Out-of-the-box inference scripts with pre-configured hardware and software parameters\t
- Released **FlagOS-{vendor_display}** container image supporting deployment within minutes
### Consistency Validation
- Rigorously evaluated through benchmark testing: Performance and results from the FlagOS software stack are compared against native stacks on multiple public.\t


# Evaluation Results
## Benchmark Result
{eval_table}

# User Guide
Environment Setup

| Item             | Version              |
|------------------|----------------------|
| Docker Version   | {docker_version} |
| Operating System | {os_info} |

## Operation Steps

### Download FlagOS Image
```bash
{image_pull_cmd}
```

### Download Open-source Model Weights
```bash
pip install modelscope
modelscope download --model FlagRelease/{flagrelease_name} --local_dir /data/{flagrelease_name}
```

### Start the Container
```bash
{container_run_cmd}
```
### Start the Server
```bash
{serve_start_cmd}
```

## Service Invocation
### Invocation Script
```bash
{serve_infer_cmd}
```


### AnythingLLM Integration Guide

#### 1. Download & Install

- Visit the official site: https://anythingllm.com/
- Choose the appropriate version for your OS (Windows/macOS/Linux)
- Follow the installation wizard to complete the setup

#### 2. Configuration

- Launch AnythingLLM
- Open settings (bottom left, fourth tab)
- Configure core LLM parameters
- Click "Save Settings" to apply changes

#### 3. Model Interaction

- After model loading is complete:
- Click **"New Conversation"**
- Enter your question (e.g., "Explain the basics of quantum computing")
- Click the send button to get a response
# Technical Overview
**FlagOS** is a fully open-source system software stack designed to unify the "model\\u2013system\\u2013chip" layers and foster an open, collaborative ecosystem. It enables a "develop once, run anywhere" workflow across diverse AI accelerators, unlocking hardware performance, eliminating fragmentation among vendor-specific software stacks, and substantially lowering the cost of porting and maintaining AI workloads. With core technologies such as the **FlagScale**, together with vllm-plugin-fl, distributed training/inference framework, **FlagGems** universal operator library, **FlagCX** communication library, and **FlagTree** unified compiler, the **FlagRelease** platform leverages the **FlagOS** stack to automatically produce and release various combinations of \\<chip + open-source model\\>. This enables efficient and automated model migration across diverse chips, opening a new chapter for large model deployment and application.
## FlagGems
FlagGems is a high-performance, generic operator library implemented in [Triton](https://github.com/openai/triton) language. It is built on a collection of backend-neutral kernels that aims to accelerate LLM (Large-Language Models) training and inference across diverse hardware platforms.
## FlagTree
FlagTree is an open source, unified compiler for multiple AI chips project dedicated to developing a diverse ecosystem of AI chip compilers and related tooling platforms, thereby fostering and strengthening the upstream and downstream Triton ecosystem. Currently in its initial phase, the project aims to maintain compatibility with existing adaptation solutions while unifying the codebase to rapidly implement single-repository multi-backend support. For upstream model users, it provides unified compilation capabilities across multiple backends; for downstream chip manufacturers, it offers examples of Triton ecosystem integration.
## FlagScale and vllm-plugin-fl
Flagscale is a comprehensive toolkit designed to support the entire lifecycle of large models. It builds on the strengths of several prominent open-source projects, including [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [vLLM](https://github.com/vllm-project/vllm), to provide a robust, end-to-end solution for managing and scaling large models.
vllm-plugin-fl is a vLLM plugin built on the FlagOS unified multi-chip backend, to help flagscale support multi-chip on vllm framework.
## **FlagCX**
FlagCX is a scalable and adaptive cross-chip communication library. It serves as a platform where developers, researchers, and AI engineers can collaborate on various projects, contribute to the development of cutting-edge AI solutions, and share their work with the global community.

## **FlagEval Evaluation Framework**
 FlagEval is a comprehensive evaluation system and open platform for large models launched in 2023. It aims to establish scientific, fair, and open benchmarks, methodologies, and tools to help researchers assess model and training algorithm performance. It features:
 - **Multi-dimensional Evaluation**: Supports 800+ model evaluations across NLP, CV, Audio, and Multimodal fields, covering 20+ downstream tasks including language understanding and image-text generation.
 - **Industry-Grade Use Cases**: Has completed horizontal evaluations of mainstream large models, providing authoritative benchmarks for chip-model performance validation.

# Contributing

We warmly welcome global developers to join us:

1. Submit Issues to report problems
2. Create Pull Requests to contribute code
3. Improve technical documentation
4. Expand hardware adaptation support
# License
The model weights are derived from {source} and are open\\u2011sourced under the Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
"""

        output_path = self._get_readme_output_path()
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)

            print(f"  + README 已生成: {output_path}")
            self.steps.append(StepResult(
                step_name="生成 README",
                status=StepStatus.SUCCESS,
                output=output_path
            ))
            return output_path

        except Exception as e:
            print(f"  x 生成 README 失败: {e}")
            return None

    def _build_environment_table(self) -> str:
        """构建环境信息表格"""
        model_info = self.config.model_info
        chip_config = self.config.chip

        rows = []

        if self.env_info and self.env_info.os_name:
            os_info = f"{self.env_info.os_name} {self.env_info.os_version}".strip()
        else:
            os_info = model_info.ubuntu_version or "N/A"
        rows.append(("Operating System", os_info))

        if self.env_info and self.env_info.kernel_version:
            rows.append(("Kernel Version", self.env_info.kernel_version))

        if self.env_info and self.env_info.docker_version:
            docker_ver = self.env_info.docker_version
        else:
            docker_ver = model_info.docker_version or "N/A"
        rows.append(("Docker Version", docker_ver))

        if self.env_info and self.env_info.vendor:
            vendor_info = f"{self.env_info.vendor_cn_name} ({self.env_info.vendor.value})"
        else:
            vendor_info = model_info.vendor or "N/A"
        rows.append(("Chip Vendor", vendor_info))

        if self.env_info and self.env_info.driver_version:
            rows.append(("Driver Version", self.env_info.driver_version))

        if self.env_info and self.env_info.sdk_version:
            sdk_info = f"{self.env_info.sdk_name} {self.env_info.sdk_version}" if self.env_info.sdk_name else self.env_info.sdk_version
            rows.append(("SDK Version", sdk_info))

        if self.env_info and self.env_info.gpu_model:
            rows.append(("GPU Model", self.env_info.gpu_model))

        if self.env_info and self.env_info.gpu_count > 0:
            rows.append(("GPU Count", str(self.env_info.gpu_count)))

        if self.env_info and self.env_info.python_version:
            rows.append(("Python Version", self.env_info.python_version))

        if self.env_info and self.env_info.torch_version:
            torch_info = f"{self.env_info.torch_version} ({self.env_info.torch_backend})" if self.env_info.torch_backend else self.env_info.torch_version
            rows.append(("PyTorch Version", torch_info))

        if self.env_info and self.env_info.flaggems_version:
            rows.append(("FlagGems Version", self.env_info.flaggems_version))
        elif chip_config.gems_version:
            rows.append(("FlagGems Version", chip_config.gems_version))

        if self.env_info and self.env_info.flagtree_version:
            rows.append(("FlagTree Version", self.env_info.flagtree_version))
        elif chip_config.tree and chip_config.tree != "none":
            rows.append(("FlagTree Version", chip_config.tree))

        if self.env_info and self.env_info.vllm_version:
            rows.append(("vLLM Version", self.env_info.vllm_version))

        if self.env_info and self.env_info.arch:
            rows.append(("Architecture", self.env_info.arch))

        table = "| Item | Value |\n|------|-------|\n"
        for item, value in rows:
            table += f"| {item} | {value} |\n"

        return table

    # ==================== ModelScope ====================

    def _publish_to_modelscope(self, readme_path: Optional[str]) -> bool:
        """发布到 ModelScope（CLI 优先，SDK 降级，容器内执行）"""
        publish_config = self.config.publish

        model_name = self.config.model_info.flagrelease_name or self.config.model_info.output_name
        model_id = publish_config.modelscope_model_id or f"FlagRelease/{model_name}"

        container_upload_dir = self._get_container_upload_dir()
        print(f"  容器内上传目录: {container_upload_dir}")
        print(f"  目标仓库: {model_id}")
        print(f"  可见性: {'私有' if publish_config.private else '公开'}")

        if self._publish_to_modelscope_cli(readme_path):
            return True

        print("  CLI 方式失败，尝试使用 SDK...")
        return self._publish_to_modelscope_sdk(readme_path)

    def _publish_to_modelscope_sdk(self, readme_path: Optional[str]) -> bool:
        """使用 SDK 发布到 ModelScope（降级方案，容器内执行）"""
        publish_config = self.config.publish
        container = self.config.container_name

        model_name = self.config.model_info.flagrelease_name or self.config.model_info.output_name
        model_id = publish_config.modelscope_model_id or f"FlagRelease/{model_name}"

        if not container:
            print("  x 无容器名，无法在容器内执行 SDK 上传")
            return False

        if not self._ensure_container_package("modelscope"):
            print("  x 容器内安装 modelscope 失败")
            return False

        container_upload_dir = self._get_container_upload_dir()
        self._docker_cp_readme_to_container(readme_path, container_upload_dir)

        token = publish_config.modelscope_token or ""
        visibility = 1 if publish_config.private else 3
        private_label = '私有' if publish_config.private else '公开'

        sdk_script = f"""
import os, sys
from modelscope.hub.api import HubApi
api = HubApi()
token = os.environ.get('MODELSCOPE_API_TOKEN', '')
if token:
    api.login(token)
model_id = '{model_id}'
print(f'检查 ModelScope 模型仓库: {{model_id}}')
try:
    api.get_model(model_id)
    print('仓库已存在')
except Exception:
    print('仓库不存在，创建中...')
    try:
        api.create_model(model_id=model_id, visibility={visibility})
        print('仓库创建成功 ({private_label})')
    except Exception as e:
        print(f'创建仓库失败: {{e}}，继续尝试上传...')
print('开始上传...')
api.upload_folder(repo_id=model_id, folder_path='{container_upload_dir}')
print(f'已发布到 ModelScope: {{model_id}}')
"""
        token_env = f"MODELSCOPE_API_TOKEN={token} " if token else ""
        script_b64 = base64.b64encode(sdk_script.encode()).decode()
        cmd = f"{token_env}PATH=/opt/conda/bin:$PATH python3 -c \"import base64;exec(base64.b64decode('{script_b64}').decode())\""
        result, stdout, stderr = self.run_command(
            cmd=cmd, step_name="SDK 上传到 ModelScope",
            timeout=UPLOAD_TIMEOUT, in_container=True
        )
        if result:
            print(f"  + 已发布到 ModelScope: {model_id}")
            return True

        print(f"  x SDK 发布到 ModelScope 失败")
        return False

    def _ensure_container_package(self, package: str) -> bool:
        """确保容器内已安装指定 Python 包，未安装则自动安装"""
        container = self.config.container_name
        if not container:
            return False
        check_cmd = f"PATH=/opt/conda/bin:$PATH python3 -c 'import {package}'"
        result, _, _ = self.run_command(
            cmd=check_cmd, step_name=f"检查容器内 {package}",
            timeout=30, in_container=True
        )
        if result:
            return True
        print(f"  容器内未安装 {package}，自动安装中...")
        install_cmd = f"PATH=/opt/conda/bin:$PATH pip install {package} -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
        result, _, _ = self.run_command(
            cmd=install_cmd, step_name=f"容器内安装 {package}",
            timeout=300, in_container=True
        )
        return result

    def _get_container_upload_dir(self) -> str:
        """获取容器内上传目录路径（模型权重所在路径）"""
        # 优先使用 weights_dir（从 context.yaml model.local_path/container_path 读取的实际路径）
        weights_dir = self.config.publish.weights_dir
        if weights_dir:
            return weights_dir
        # 回退：从 serve_start_cmd 中解析
        serve_cmd = self.config.model_info.serve_start_cmd or ""
        if "vllm serve " in serve_cmd:
            parts = serve_cmd.split("vllm serve ", 1)[1].split()
            if parts:
                return parts[0].strip().rstrip("\\")
        return "/data/models"

    def _docker_cp_readme_to_container(self, readme_path: Optional[str], container_upload_dir: str) -> bool:
        """将 README 文件 docker cp 到容器内上传目录"""
        if not readme_path or not os.path.exists(readme_path):
            return True
        container = self.config.container_name
        dest = f"{container}:{container_upload_dir}/README.md"
        try:
            subprocess.run(["docker", "cp", readme_path, dest],
                           capture_output=True, text=True, timeout=30, check=True)
            print(f"  已复制 README 到容器内 {container_upload_dir}/README.md")
            return True
        except Exception as e:
            print(f"  ⚠ 复制 README 到容器失败: {e}")
            return False

    def _publish_to_modelscope_cli(self, readme_path: Optional[str]) -> bool:
        """使用命令行发布到 ModelScope（容器内执行，避免宿主机 torch 崩溃）"""
        publish_config = self.config.publish
        container = self.config.container_name

        model_name = self.config.model_info.flagrelease_name or self.config.model_info.output_name
        model_id = publish_config.modelscope_model_id or f"FlagRelease/{model_name}"

        if not container:
            print("  x 无容器名，无法在容器内执行上传")
            return False

        if not self._ensure_container_package("modelscope"):
            print("  x 容器内安装 modelscope 失败")
            return False

        container_upload_dir = self._get_container_upload_dir()
        self._docker_cp_readme_to_container(readme_path, container_upload_dir)

        token = publish_config.modelscope_token
        token_env = f"MODELSCOPE_API_TOKEN={token} " if token else ""

        print(f"  目标仓库: {model_id}")
        print(f"  容器内上传目录: {container_upload_dir}")

        visibility = "private" if publish_config.private else "public"
        create_cmd = f"PATH=/opt/conda/bin:$PATH {token_env}modelscope create {model_id} --visibility {visibility}"
        print(f"  创建/确认仓库: {model_id} ({visibility})")
        result, stdout, stderr = self.run_command(
            cmd=create_cmd, step_name="创建 ModelScope 仓库",
            timeout=60, in_container=True
        )
        if not result:
            print(f"    创建仓库失败（可能已存在），继续尝试上传...")

        upload_cmd = f"PATH=/opt/conda/bin:$PATH {token_env}modelscope upload {model_id} {container_upload_dir}"

        success = False
        current_delay = UPLOAD_RETRY_DELAY

        for attempt in range(UPLOAD_MAX_RETRIES):
            result, stdout, stderr = self.run_command(
                cmd=upload_cmd, step_name="上传到 ModelScope",
                timeout=UPLOAD_TIMEOUT, in_container=True
            )
            if result:
                success = True
                print(f"  + 已发布到 ModelScope: {model_id}")
                break
            else:
                if attempt < UPLOAD_MAX_RETRIES - 1:
                    print(f"  x 上传失败 (尝试 {attempt+1}/{UPLOAD_MAX_RETRIES})")
                    print(f"    等待 {current_delay} 秒后重试...")
                    time.sleep(current_delay)
                    current_delay = min(current_delay * 2, UPLOAD_MAX_DELAY)
                else:
                    print(f"  x 上传失败，已达最大重试次数")

        return success

    # ==================== HuggingFace ====================

    def _publish_to_huggingface(self, readme_path: Optional[str]) -> bool:
        """发布到 HuggingFace（CLI 优先，SDK 降级）"""
        publish_config = self.config.publish

        model_name = self.config.model_info.flagrelease_name or self.config.model_info.output_name
        repo_id = publish_config.huggingface_repo_id or f"FlagRelease/{model_name}"

        container_upload_dir = self._get_container_upload_dir()
        print(f"  容器内上传目录: {container_upload_dir}")
        print(f"  目标仓库: {repo_id}")
        print(f"  可见性: {'私有' if publish_config.private else '公开'}")

        # 默认使用 hf-mirror 镜像站，避免国内网络直连 huggingface.co 不可达
        if not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            print(f"  HF_ENDPOINT 未设置，使用镜像站: https://hf-mirror.com")

        if self._publish_to_huggingface_cli(readme_path):
            return True

        print("  CLI 方式失败，尝试使用 SDK...")
        return self._publish_to_huggingface_sdk(readme_path)

    def _publish_to_huggingface_sdk(self, readme_path: Optional[str]) -> bool:
        """使用 SDK 发布到 HuggingFace（降级方案，容器内执行）"""
        publish_config = self.config.publish
        container = self.config.container_name

        model_name = self.config.model_info.flagrelease_name or self.config.model_info.output_name
        repo_id = publish_config.huggingface_repo_id or f"FlagRelease/{model_name}"

        if not container:
            print("  x 无容器名，无法在容器内执行 SDK 上传")
            return False

        if not self._ensure_container_package("huggingface_hub"):
            print("  x 容器内安装 huggingface_hub 失败")
            return False

        container_upload_dir = self._get_container_upload_dir()
        self._docker_cp_readme_to_container(readme_path, container_upload_dir)

        token = publish_config.huggingface_token or ""
        private_flag = "True" if publish_config.private else "False"
        hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

        sdk_script = f"""
import os
os.environ['HF_ENDPOINT'] = '{hf_endpoint}'
from huggingface_hub import HfApi, login
token = os.environ.get('HF_TOKEN', '')
if token:
    login(token=token)
api = HfApi()
repo_id = '{repo_id}'
print(f'检查 HuggingFace 仓库: {{repo_id}}')
try:
    api.repo_info(repo_id=repo_id)
    print('仓库已存在')
except Exception:
    print('仓库不存在，创建中...')
    api.create_repo(repo_id=repo_id, private={private_flag}, exist_ok=True)
    print('仓库创建成功')
print('开始上传...')
api.upload_folder(repo_id=repo_id, folder_path='{container_upload_dir}')
print(f'已发布到 HuggingFace: {{repo_id}}')
"""
        token_env = f"HF_TOKEN={token} " if token else ""
        script_b64 = base64.b64encode(sdk_script.encode()).decode()
        cmd = f"{token_env}HF_ENDPOINT={hf_endpoint} PATH=/opt/conda/bin:$PATH python3 -c \"import base64;exec(base64.b64decode('{script_b64}').decode())\""
        result, stdout, stderr = self.run_command(
            cmd=cmd, step_name="SDK 上传到 HuggingFace",
            timeout=UPLOAD_TIMEOUT, in_container=True
        )
        if result:
            print(f"  + 已发布到 HuggingFace: {repo_id}")
            return True

        print(f"  x SDK 发布到 HuggingFace 失败")
        return False

    def _publish_to_huggingface_cli(self, readme_path: Optional[str]) -> bool:
        """使用命令行发布到 HuggingFace（容器内执行）"""
        publish_config = self.config.publish
        container = self.config.container_name

        if not container:
            print("  x 无容器名，无法在容器内执行上传")
            return False

        if not self._ensure_container_package("huggingface_hub"):
            print("  x 容器内安装 huggingface_hub 失败")
            return False

        model_name = self.config.model_info.flagrelease_name or self.config.model_info.output_name
        repo_id = publish_config.huggingface_repo_id or f"FlagRelease/{model_name}"

        container_upload_dir = self._get_container_upload_dir()
        self._docker_cp_readme_to_container(readme_path, container_upload_dir)

        token = publish_config.huggingface_token or ""
        hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
        token_env = f"HF_TOKEN={token} " if token else ""
        endpoint_env = f"HF_ENDPOINT={hf_endpoint} "

        print(f"  目标仓库: {repo_id}")
        print(f"  容器内上传目录: {container_upload_dir}")

        if token:
            login_cmd = f"PATH=/opt/conda/bin:$PATH {token_env}{endpoint_env}huggingface-cli login --token {token}"
            success, _, _ = self.run_command(
                cmd=login_cmd, step_name="HuggingFace 登录",
                timeout=60, in_container=True
            )
            if not success:
                return False

        private_flag = "--private " if publish_config.private else ""
        upload_cmd = f"PATH=/opt/conda/bin:$PATH {token_env}{endpoint_env}huggingface-cli upload {private_flag}{repo_id} {container_upload_dir}".strip()

        success = False
        current_delay = UPLOAD_RETRY_DELAY

        for attempt in range(UPLOAD_MAX_RETRIES):
            result, stdout, stderr = self.run_command(
                cmd=upload_cmd, step_name="上传到 HuggingFace",
                timeout=UPLOAD_TIMEOUT, in_container=True
            )
            if result:
                success = True
                print(f"  + 已发布到 HuggingFace: {repo_id}")
                break
            else:
                if attempt < UPLOAD_MAX_RETRIES - 1:
                    print(f"  x 上传失败 (尝试 {attempt+1}/{UPLOAD_MAX_RETRIES})")
                    print(f"    等待 {current_delay} 秒后重试...")
                    time.sleep(current_delay)
                    current_delay = min(current_delay * 2, UPLOAD_MAX_DELAY)
                else:
                    print(f"  x 上传失败，已达最大重试次数")

        return success

    def _update_repo_readme(self, repo_id: str, platform: str, readme_path: str) -> bool:
        """更新已发布仓库的 README（plugin 模式：覆盖原仓库 README）"""
        step_name = f"更新 {platform} 仓库 README"
        print(f"\n--- {step_name} ---")
        print(f"  目标仓库: {repo_id}")

        if not os.path.exists(readme_path):
            print(f"  x README 文件不存在: {readme_path}")
            return False

        container = self.config.container_name
        if not container:
            print(f"  x 无容器名，无法在容器内执行上传")
            return False

        # 将 README 复制到容器内临时目录
        container_tmp = "/tmp/plugin_readme_upload"
        subprocess.run(
            ["docker", "exec", container, "mkdir", "-p", container_tmp],
            capture_output=True, timeout=10
        )
        cp_result = subprocess.run(
            ["docker", "cp", readme_path, f"{container}:{container_tmp}/README.md"],
            capture_output=True, text=True, timeout=30
        )
        if cp_result.returncode != 0:
            print(f"  x 复制 README 到容器失败: {cp_result.stderr}")
            return False

        # 构建上传命令
        if platform == "modelscope":
            token = self.config.publish.modelscope_token
            if not token:
                print(f"  x 无 ModelScope token，跳过")
                return False
            if not self._ensure_container_package("modelscope"):
                print(f"  x 容器内安装 modelscope 失败")
                return False
            shell_cmd = f"PATH=/opt/conda/bin:$PATH modelscope upload {repo_id} {container_tmp}/README.md README.md"
            docker_cmd = ["docker", "exec",
                          "-e", f"MODELSCOPE_API_TOKEN={token}",
                          container, "bash", "-c", shell_cmd]
        elif platform == "huggingface":
            token = self.config.publish.huggingface_token
            if not token:
                print(f"  x 无 HuggingFace token，跳过")
                return False
            if not self._ensure_container_package("huggingface_hub"):
                print(f"  x 容器内安装 huggingface_hub 失败")
                return False
            hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
            shell_cmd = f"PATH=/opt/conda/bin:$PATH huggingface-cli upload {repo_id} {container_tmp}/README.md README.md"
            docker_cmd = ["docker", "exec",
                          "-e", f"HF_TOKEN={token}",
                          "-e", f"HF_ENDPOINT={hf_endpoint}",
                          container, "bash", "-c", shell_cmd]
        else:
            print(f"  x 未知平台: {platform}")
            return False

        # 带重试的上传
        current_delay = UPLOAD_RETRY_DELAY
        for attempt in range(UPLOAD_MAX_RETRIES):
            print(f"[{self.name}] 执行: {step_name} (尝试 {attempt+1}/{UPLOAD_MAX_RETRIES})")
            try:
                result = subprocess.run(
                    docker_cmd, capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    print(f"  + 已更新 {platform} 仓库 README: {repo_id}")
                    self.steps.append(StepResult(
                        step_name=step_name,
                        status=StepStatus.SUCCESS,
                        output=result.stdout
                    ))
                    return True
                else:
                    stderr = result.stderr or result.stdout
                    print(f"  x 更新 {platform} README 失败: {stderr[:200] if stderr else '未知错误'}")
            except subprocess.TimeoutExpired:
                print(f"  x 更新 {platform} README 超时")
                stderr = "命令执行超时"

            if attempt < UPLOAD_MAX_RETRIES - 1:
                print(f"    等待 {current_delay} 秒后重试...")
                time.sleep(current_delay)
                current_delay = min(current_delay * 2, UPLOAD_MAX_DELAY)

        self.steps.append(StepResult(
            step_name=step_name,
            status=StepStatus.FAILED,
            output=f"重试 {UPLOAD_MAX_RETRIES} 次后仍失败"
        ))
        return False
