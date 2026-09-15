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

        # 不适配标记模式：只对源镜像打一个"不适配"标签并推送，不发布版本镜像/README/ModelScope
        incompatible_tag = getattr(self.config, "incompatible_tag", "")
        if incompatible_tag:
            print(f"  不适配标记模式: {incompatible_tag}")
            ok = self._tag_incompatible(incompatible_tag)
            return self.make_result(ok, "不适配标记完成" if ok else "不适配标记失败")

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
            # 已有镜像场景同样需要双 tag（V1.3/none → V2=V3）：
            # 主镜像已在 Harbor 不重复推，但 also tag 缺失会导致 V3 交付丢失
            also_tag = getattr(self.config, "also_tag", "")
            if also_tag:
                if not self._tag_and_push_also(also_tag):
                    harbor_failed = True
        else:
            # 0. 如果输入是容器，先 commit 为镜像（内含强制固化检查）
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
                    # V2=V3 同镜像双 tag：额外打一个 --also-tag 版本 tag 并推送。
                    # 返回值必须检查——also push 失败也是 Harbor 发布失败，
                    # 不置 harbor_failed 会导致 README/对外发布照常进行而镜像缺失
                    also_tag = getattr(self.config, "also_tag", "")
                    if also_tag and not harbor_failed:
                        if not self._tag_and_push_also(also_tag):
                            harbor_failed = True
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
            # plugin 达标。两种情形：
            #  (a) 步骤8已建仓(V2 精度达标)：更新原仓库 README，不重传权重（常规路径）。
            #  (b) 步骤8未建仓(V2 精度不达标 → 需求D：当时不对外发布)：此时 V3 达标，
            #      需 full-publish 补发对外仓库(创建仓库+上传权重+README)。
            if publish_config.base_modelscope_model_id and readme_path:
                success = self._update_repo_readme(
                    publish_config.base_modelscope_model_id, "modelscope", readme_path)
                if not success:
                    # 情形(a) 失败：可能是存量 context 的幽灵 URL(V2 不达标时记录了
                    # 未建仓库的 URL)导致 base 字段误判"已有仓库"。允许发布时降级
                    # full-publish(ensure 私有仓库幂等，存在则复用、不存在则创建)，
                    # 而不是直接放弃。
                    if publish_config.publish_modelscope:
                        print("  ⚠ 更新 README 失败(仓库可能不存在)，降级 full-publish 补发")
                        success = self._with_proxy_fallback(
                            "ModelScope", self._publish_to_modelscope, readme_path)
                    if not success:
                        ms_failed = True
                        print("  ⚠ 更新 ModelScope README 失败，继续执行 HuggingFace")
            elif publish_config.publish_modelscope:
                # 情形(b)：V2 未建仓但 V3 达标 → full-publish 补发
                print("  ℹ 步骤8未建 ModelScope 仓库(V2 精度不达标)，V3 达标 → full-publish 补发对外仓库")
                success = self._with_proxy_fallback("ModelScope", self._publish_to_modelscope, readme_path)
                if not success:
                    ms_failed = True
                    print("  ⚠ ModelScope 补发失败，继续执行 HuggingFace")
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
            # plugin 达标：同 ModelScope，(a)已建仓→更新README；(b)V2未建仓但V3达标→full-publish补发
            if publish_config.base_huggingface_repo_id and readme_path:
                success = self._update_repo_readme(
                    publish_config.base_huggingface_repo_id, "huggingface", readme_path)
                if not success:
                    # 同 ModelScope：存量幽灵 URL 时降级 full-publish(ensure 私有仓库幂等)
                    if publish_config.publish_huggingface:
                        print("  ⚠ 更新 HuggingFace README 失败(仓库可能不存在)，降级 full-publish 补发")
                        success = self._with_proxy_fallback(
                            "HuggingFace", self._publish_to_huggingface, readme_path)
                    if not success:
                        hf_failed = True
                        print("  ⚠ 更新 HuggingFace README 失败")
            elif publish_config.publish_huggingface:
                # 情形(b)：V2 未建仓但 V3 达标 → full-publish 补发
                print("  ℹ 步骤8未建 HuggingFace 仓库(V2 精度不达标)，V3 达标 → full-publish 补发对外仓库")
                success = self._with_proxy_fallback("HuggingFace", self._publish_to_huggingface, readme_path)
                if not success:
                    hf_failed = True
                    print("  ⚠ HuggingFace 补发失败")
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
        # 只有实际执行发布且成功(publish_*_publish=True 且未失败)才记录对外 URL。
        # 步骤8 V2 精度不达标时 publish_modelscope=False → 不写 URL(仓库未创建)，
        # 否则 config.py:578 无条件填充的 ms_model_id 会产出"幽灵 URL"写进 context，
        # 步骤13 会误判"已有仓库"走更新 README(对不存在仓库 upload 必失败)，
        # 而非 full-publish 补发(创建仓库+权重+README)。
        ms_published = publish_config.publish_modelscope and not ms_failed
        hf_published = publish_config.publish_huggingface and not hf_failed
        release_summary = {
            "harbor_image": publish_config.harbor_path or "",
            "modelscope_model_id": ms_model_id if ms_published else "",
            "modelscope_url": f"https://modelscope.cn/models/{ms_model_id}" if ms_model_id and ms_published else "",
            "huggingface_repo_id": hf_repo_id if hf_published else "",
            "huggingface_url": f"https://huggingface.co/{hf_repo_id}" if hf_repo_id and hf_published else "",
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

    # vllm-plugin-FL 官方认可的算子/plugin 环境变量（逐字核对 vllm_fl/utils.py）。
    # 固化进镜像 Config.Env，确保裸 docker run / vllm serve 也能读到，不再依赖
    # /etc/environment(PAM) 或 .bashrc(登录 shell) —— 这两者 Docker 拉起进程都不加载。
    _FLAGGEMS_COMMIT_ENV_KEYS = [
        "USE_FLAGGEMS",
        "VLLM_FL_PREFER_ENABLED",
        "VLLM_PLUGINS",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_FLAGOS_WHITELIST",
        "FLAGGEMS_CONTROL_MODE",
    ]

    def _collect_flaggems_env_for_commit(self) -> dict:
        """从容器内收集需要固化进镜像 Config.Env 的算子/plugin 环境变量。

        来源优先级：/root/flaggems_op_config.json 的 env_vars（persist_op_config.py 记录，
        最权威）> /etc/environment 兜底。返回 {KEY: VALUE}（仅非空）。

        互斥约束（官方 utils.py:114 硬校验）：VLLM_FL_FLAGOS_BLACKLIST 与
        VLLM_FL_FLAGOS_WHITELIST 不能同时存在，否则 plugin 启动即 ValueError。
        二者都出现时保留 BLACKLIST（主流程语义），丢弃 WHITELIST 并告警。
        """
        keys = self._FLAGGEMS_COMMIT_ENV_KEYS
        reader = f"""
import json, os
keys = {keys!r}
result = {{}}
# 1) 固化记录（最权威）
try:
    with open('/root/flaggems_op_config.json', 'r', encoding='utf-8') as f:
        rec = json.load(f)
    ev = rec.get('env_vars', {{}}) or {{}}
    for k in keys:
        v = ev.get(k)
        if v is not None and str(v).strip():
            result[k] = str(v).strip()
except Exception:
    pass
# 2) /etc/environment 兜底（不覆盖已从记录取到的）
try:
    with open('/etc/environment', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line or line.startswith('#'):
                continue
            k, _, v = line.partition('=')
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k in keys and v and k not in result:
                result[k] = v
except Exception:
    pass
for k, v in result.items():
    print(f'{{k}}={{v}}')
"""
        script_b64 = base64.b64encode(reader.encode()).decode()
        cmd = f"PATH=/opt/conda/bin:$PATH python3 -c \"import base64;exec(base64.b64decode('{script_b64}').decode())\""
        success, stdout, _ = self.run_command(
            cmd=cmd, step_name="收集固化环境变量", timeout=60, in_container=True, check=False
        )
        env_map = {}
        if success and stdout:
            for line in stdout.splitlines():
                line = line.strip()
                if "=" in line:
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip()
                    if k in keys and v:
                        env_map[k] = v
        # 互斥：blacklist 与 whitelist 不可并存
        if "VLLM_FL_FLAGOS_BLACKLIST" in env_map and "VLLM_FL_FLAGOS_WHITELIST" in env_map:
            print("  ⚠ 同时检测到 BLACKLIST 与 WHITELIST，保留 BLACKLIST 丢弃 WHITELIST（避免 plugin ValueError）")
            env_map.pop("VLLM_FL_FLAGOS_WHITELIST", None)
        return env_map

    def _ensure_operator_config_persisted(self) -> bool:
        """发布前强制确保算子配置已固化（治本：不依赖 PROMPT 调用时机）

        检查逻辑：
        1. 检查是否为需要固化的场景（env_type != native）
        2. 检查 /root/flaggems_op_config.json 是否存在
        3. 如果不存在，强制调用 persist_op_config.py --auto
        4. 验证固化数据可读取

        Returns:
            True: 固化成功或不需要固化（native场景）
            False: 固化失败
        """
        container_name = self.config.container_name
        if not container_name:
            print("  跳过固化检查（非容器输入）")
            return True

        print("\n[步骤 0.1] 检查算子配置固化状态")

        # 检查是否为需要固化的场景
        check_script = """
import sys, yaml, os, json

try:
    # 1. 读取 env_type
    with open('/flagos-workspace/shared/context.yaml', 'r', encoding='utf-8') as f:
        ctx = yaml.safe_load(f)
    env_type = ctx.get('env_type', '')

    # native 场景跳过
    if env_type == 'native':
        print('skip_native')
        sys.exit(0)

    # 2. 检查固化记录
    if not os.path.exists('/root/flaggems_op_config.json'):
        print('missing')
        sys.exit(0)

    # 3. 读取记录验证可解析
    with open('/root/flaggems_op_config.json', 'r', encoding='utf-8') as f:
        rec = json.load(f)

    timestamp = rec.get('timestamp', 'unknown')
    method = rec.get('persist_method', 'unknown')
    enabled_count = rec.get('enabled_count', 0)
    disabled_count = rec.get('disabled_count', 0)

    print(f'exists:{timestamp}:{method}:enabled={enabled_count}:disabled={disabled_count}')

except Exception as e:
    print(f'error:{str(e)}')
    sys.exit(1)
"""
        import base64
        script_b64 = base64.b64encode(check_script.encode()).decode()
        cmd = f"PATH=/opt/conda/bin:$PATH python3 -c \"import base64;exec(base64.b64decode('{script_b64}').decode())\""

        success, stdout, stderr = self.run_command(
            cmd=cmd,
            step_name="检查固化状态",
            timeout=60,
            in_container=True,
            check=False
        )

        if not success:
            print(f"  ✗ 检查失败: {stderr}")
            return False

        result = stdout.strip()

        # native 场景，跳过固化
        if "skip_native" in result:
            print("  ✓ native 场景，无需固化算子配置")
            return True

        # 固化记录缺失，强制执行 persist_op_config.py
        if "missing" in result:
            print("  ⚠ 未检测到固化记录 (/root/flaggems_op_config.json)，强制执行固化")
            need_persist = True
        elif "error:" in result:
            error_msg = result.split("error:", 1)[1] if ":" in result else result
            print(f"  ⚠ 固化记录读取失败: {error_msg}，强制重新固化")
            need_persist = True
        else:
            # 固化记录存在且可读
            print(f"  ✓ 固化记录存在: {result.replace('exists:', '')}")
            need_persist = False

        # 执行固化
        if need_persist:
            print("  执行 persist_op_config.py --auto ...")
            success, stdout, stderr = self.run_command(
                cmd="PATH=/opt/conda/bin:$PATH python3 /flagos-workspace/scripts/persist_op_config.py --auto",
                step_name="强制固化算子配置",
                timeout=180,
                in_container=True,
                check=False
            )

            if not success:
                print(f"  ✗ 固化失败: {stderr}")
                # 读取错误日志
                _, err_log, _ = self.run_command(
                    cmd="cat /flagos-workspace/logs/_last_error.json 2>/dev/null || echo '{}'",
                    step_name="读取固化错误日志",
                    timeout=10,
                    in_container=True,
                    check=False
                )
                if err_log and err_log.strip() != '{}':
                    print(f"  错误详情: {err_log[:500]}")
                return False

            print(f"  ✓ 固化完成")
            # 打印固化结果摘要
            if stdout:
                for line in stdout.splitlines()[-10:]:
                    if "✓" in line or "记录文件" in line or "算子" in line:
                        print(f"    {line.strip()}")

        return True

    def _commit_container(self) -> bool:
        """将容器 commit 为镜像（固化算子/plugin 环境变量进 Config.Env）"""
        container_name = self.config.container_name
        if not container_name:
            print("  x 容器名称未配置")
            return False

        # 步骤 0: 强制确保固化数据存在且最新（治本：不依赖 PROMPT 调用时机）
        if not self._ensure_operator_config_persisted():
            print("  x 算子配置固化失败，中止发布")
            return False

        model_name = self.config.model_info.output_name or "model"
        commit_image_name = f"flagrelease-commit-{container_name}:{model_name}".lower().replace("/", "-")

        # 收集并固化算子/plugin 环境变量到镜像 Config.Env（治本：不依赖 shell 加载路径）
        env_map = self._collect_flaggems_env_for_commit()
        change_args = ""
        if env_map:
            print(f"  固化环境变量进镜像 Config.Env: {', '.join(env_map.keys())}")
            change_args = " " + " ".join(f"--change 'ENV {k}={v}'" for k, v in env_map.items())
        else:
            print("  未检测到需固化的算子/plugin 环境变量（native 或全量默认，跳过 ENV 注入）")

        print(f"  正在将容器 {container_name} commit 为镜像 {commit_image_name}...")

        cmd = f"docker commit{change_args} {container_name} {commit_image_name}"
        success, stdout, stderr = self.run_command(
            cmd=cmd,
            step_name="容器 commit",
            timeout=600
        )

        if not success:
            return False

        self.config.publish.image_source = commit_image_name
        print(f"  + 容器已 commit 为镜像: {commit_image_name}")

        # 硬对账：校验固化的环境变量确实写进了镜像 Config.Env
        if env_map and not self._verify_committed_env(commit_image_name, env_map):
            print("  x 镜像 Config.Env 固化校验失败（变量未生效），中止发布")
            return False

        return True

    def _verify_committed_env(self, image_name: str, expected: dict) -> bool:
        """docker inspect 校验镜像 Config.Env 确实包含固化的环境变量"""
        cmd = f"docker inspect --format '{{{{json .Config.Env}}}}' {image_name}"
        success, stdout, _ = self.run_command(
            cmd=cmd, step_name="校验镜像 Config.Env", timeout=60, check=False
        )
        if not success or not stdout:
            print("  ⚠ 无法读取镜像 Config.Env，跳过校验")
            return True
        try:
            env_list = json.loads(stdout.strip()) or []
        except Exception:
            print("  ⚠ 解析 Config.Env 失败，跳过校验")
            return True
        image_env = dict(e.split("=", 1) for e in env_list if "=" in e)
        ok = True
        for k, v in expected.items():
            if image_env.get(k) != v:
                print(f"  x Config.Env 缺失/不符: {k}（期望={v}, 实际={image_env.get(k)}）")
                ok = False
        if ok:
            print(f"  ✓ 镜像 Config.Env 固化校验通过（{len(expected)} 个变量）")
        return ok

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
                date_tag=chip_config.date_tag,
                container_name=self.config.container_name,
                vendor_name=vendor.value if vendor else "",
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

    def _tag_and_push_also(self, also_version: str) -> bool:
        """V2=V3 同镜像双 tag 场景：把已推送的镜像另打一个版本 tag 并推送。

        用于分支 A/B 中 V2 与 V3 实际为同一镜像（如 V1.3 → V2=V3）的场景，
        避免重复 commit，直接对同一镜像加第二个版本 tag。
        """
        publish_config = self.config.publish
        source = publish_config.harbor_path
        if not source:
            print("  ⚠ 双 tag 跳过：源 harbor_path 为空")
            return False

        # 将源 tag 的版本后缀替换为 also_version（如 ...-v2 → ...-v3；无后缀则追加）
        import re
        also_suffix = f"-{also_version}"
        if re.search(r"-v[0-9]+$", source):
            also_path = re.sub(r"-v[0-9]+$", also_suffix, source)
        else:
            also_path = f"{source}{also_suffix}"

        # V3 (Max) 双 tag 必须路由到 flagrelease-project（交付 SVT 验收），与独立 V3 发布一致。
        # 注意不能改 auto_fill_config 的全局 registry——那会把 v2 标签也带进 project 仓库；
        # 这里只影响 also tag 本身，v2 主标签保持 flagrelease-public。
        if also_version == "v3":
            also_path = also_path.replace(
                "harbor.baai.ac.cn/flagrelease-public/",
                "harbor.baai.ac.cn/flagrelease-project/", 1)

        print(f"[{self.name}] 双 tag 发布: {source} → {also_path}")
        tag_cmd = f"docker tag {source} {also_path}"
        rc = subprocess.run(tag_cmd, shell=True, capture_output=True, text=True)
        if rc.returncode != 0:
            print(f"  x 双 tag 打标失败: {rc.stderr[:200]}")
            self.steps.append(StepResult(
                step_name=f"双 tag ({also_version})", status=StepStatus.FAILED,
                error=rc.stderr))
            return False

        if not self._ensure_harbor_login(also_path):
            return False
        push_cmd = f"docker push {also_path}"
        rc = subprocess.run(push_cmd, shell=True, capture_output=True, text=True, timeout=7200)
        ok = rc.returncode == 0
        print(f"  {'+' if ok else 'x'} 双 tag 推送{'成功' if ok else '失败'}: {also_path}")
        self.steps.append(StepResult(
            step_name=f"双 tag 推送 ({also_version})",
            status=StepStatus.SUCCESS if ok else StepStatus.FAILED,
            output=also_path if ok else rc.stderr))
        return ok

    def _tag_incompatible(self, marker: str) -> bool:
        """不适配标记：对源镜像打一个不适配 tag 并推送到 Harbor。

        用于某版本（如厂商 plugin V3.1）验证不通过、需明确标注"不适配"的场景。
        marker 形如 'Qwen3-8B-flagos-metax不适配'，作为镜像 tag 名。
        """
        publish_config = self.config.publish
        # 源镜像解析（须为本地真实存在的镜像，否则 docker tag 报 No such image）：
        #   1. 优先已有 Harbor 镜像（existing_harbor_image，如复用准入镜像）
        #   2. 容器输入：commit 当前容器得到本地镜像（分支 B 场景 harbor_path 是尚未构建的
        #      版本目标 tag，不能作为 docker tag 源，故必须先 commit）
        #   3. 兜底：harbor_path（仅当已确为本地存在的镜像时）
        source = publish_config.existing_harbor_image
        if not source and self.config.input_type == 'container':
            if not self._commit_container():
                return False
            source = publish_config.image_source or publish_config.harbor_path
        if not source:
            source = publish_config.harbor_path
        if not source:
            print("  x 不适配标记失败：无源镜像")
            return False

        # 目标 registry+project 前缀取自 Harbor 目标 tag（harbor_path），而非本地 commit 源镜像
        # （本地 commit 镜像名无 registry/project，直接复用会生成非法 reference）
        prefix_ref = publish_config.harbor_path or publish_config.existing_harbor_image or source
        registry = prefix_ref.split("/")[0]
        # 目标：<registry>/<project>/<marker>（复用 Harbor 目标的 registry+project 前缀）
        project_prefix = "/".join(prefix_ref.split(":")[0].split("/")[:-1])
        # Docker repository 名必须全小写，marker 常含模型名(可能带大写,如 Qwen2.5-7B-Instruct)，
        # 不小写化会报 "invalid reference format"。与 chip_detector.sanitize_docker_tag().lower() 一致。
        marker = marker.lower()
        marker_path = f"{project_prefix}/{marker}" if project_prefix else f"{registry}/{marker}"

        print(f"[{self.name}] 不适配标记: {source} → {marker_path}")
        rc = subprocess.run(f"docker tag {source} {marker_path}", shell=True,
                            capture_output=True, text=True)
        if rc.returncode != 0:
            print(f"  x 打标失败: {rc.stderr[:200]}")
            self.steps.append(StepResult(step_name="不适配标记", status=StepStatus.FAILED,
                                         error=rc.stderr))
            return False
        if not self._ensure_harbor_login(marker_path):
            return False
        rc = subprocess.run(f"docker push {marker_path}", shell=True,
                            capture_output=True, text=True, timeout=7200)
        ok = rc.returncode == 0
        print(f"  {'+' if ok else 'x'} 不适配标记推送{'成功' if ok else '失败'}: {marker_path}")
        self.steps.append(StepResult(
            step_name="不适配标记推送",
            status=StepStatus.SUCCESS if ok else StepStatus.FAILED,
            output=marker_path if ok else rc.stderr))
        return ok

    # ==================== README 生成 =============
    def _readme_pull_image(self) -> str:
        """README 中 docker pull 命令展示的镜像地址。

        V4(Flag-express) 是在 V3 基础上减算子的实验版，交付推荐仍用 V3(Max)。
        因此 version_tag==v4 且 context 提供了 V3 镜像地址(readme_image_override)时，
        README 展示 V3 镜像（照常展示 flagrelease-project 私有地址，供 SVT 验收方 pull）。
        其余版本（含 override 缺失兜底）沿用当前发布镜像 image_harbor_path。
        """
        model_info = self.config.model_info
        current = model_info.image_harbor_path or self.config.publish.harbor_path or ""
        if getattr(self.config, "version_tag", "") == "v4":
            override = getattr(model_info, "readme_image_override", "") or ""
            if override:
                if override != current:
                    print(f"  README 使用 V3(Max) 镜像（V4 减算子版交付推荐 V3）: {override}")
                return override
            print("  ⚠ V4 发布但 context 无 V3 镜像地址(versions.v3.image_url)，README 回退当前 V4 镜像")
        return current

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
            "serve_start_cmd": self._ensure_plugin_prefix(model_info.serve_start_cmd or ""),
            "serve_infer_cmd": model_info.serve_infer_cmd,
            "canonical_model_path": model_info.canonical_model_path,
            "new_model_introduction": model_info.new_model_introduction or "",
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

    def _is_plugin_image(self) -> bool:
        """判定当前发布目标是否为 plugin 镜像（README serve 命令需 VLLM_PLUGINS=fl 前缀）。

        判据不能只看 version_tag——2.2 同镜像双 tag 场景（version_tag=v2 + also_tag=v3/v4）
        主 tag 是 v2 但镜像本身是 plugin 镜像，同样需要 fl 前缀。满足以下任一即视为 plugin 镜像：
          (a) version_tag ∈ {v3,v4}   —— 独立 plugin 版本
          (b) also_tag    ∈ {v3,v4}   —— 2.2 同镜像双 tag（v2 主 tag，同镜像也发 v3/v4）
          (c) plugin_image_mode       —— 兼容旧 --plugin-mode 别名
        """
        _plugin_versions = ("v3", "v4")
        return (
            self.config.version_tag in _plugin_versions
            or getattr(self.config, "also_tag", "") in _plugin_versions
            or bool(getattr(self.config, "plugin_image_mode", False))
        )

    def _ensure_plugin_prefix(self, serve_cmd: str) -> str:
        """plugin 镜像的 serve 命令固化 VLLM_PLUGINS=fl 前缀。

        VLLM_PLUGINS=fl 由 start_service.sh 在环境变量层设置，从不进入 serve 命令字符串，
        导致 README 偶发丢失该前缀、用户照抄命令起服务报错。此处对所有 README 生成路径
        （模板 / 内置 / 外部脚本）统一在源头补齐。幂等：命令已含 VLLM_PLUGINS= 则不动。
        """
        if not serve_cmd:
            return serve_cmd
        if self._is_plugin_image() and "VLLM_PLUGINS=" not in serve_cmd:
            serve_cmd = "VLLM_PLUGINS=fl " + serve_cmd
            print("  ✓ README serve 命令补齐 plugin 前缀 VLLM_PLUGINS=fl（plugin 镜像）")
        return serve_cmd

    def _prepare_template_vars(self) -> dict:
        """准备模板变量"""
        model_info = self.config.model_info
        chip_config = self.config.chip

        vars = {}

        vars["flagrelease_name"] = model_info.flagrelease_name or model_info.output_name
        vars["output_name"] = model_info.output_name
        vars["source_of_model_weights"] = model_info.source_of_model_weights
        vars["new_model_introduction"] = model_info.new_model_introduction or ""

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

        vars["image_harbor_path"] = self._readme_pull_image() or "N/A"
        image_harbor = vars["image_harbor_path"]
        vars["image_pull_cmd"] = f"docker pull {image_harbor}" if image_harbor != "N/A" else ""

        # 统一模型路径：下载目标、serve 命令、docker run 挂载三者一致
        canonical_path = model_info.canonical_model_path or "/data/models/model"
        vars["canonical_model_path"] = canonical_path
        vars["weights_local_path"] = canonical_path

        vars["container_run_cmd"] = model_info.container_run_cmd.strip() if model_info.container_run_cmd else ""
        # 强制 docker run 镜像与 docker pull 同源：README 里 docker run 必须写入具体镜像名，
        # 绝不能把 {{IMAGE}} 占位符或历史基础镜像名漏进 README。
        # 之前 bug：image_harbor 取到 "N/A" 时整段替换被跳过 → {{IMAGE}} 原样进 README。
        # 现改为多级兜底解析真实镜像名，任一非空即用；彻底解析不到才报错阻断（不静默留占位符）。
        if vars["container_run_cmd"]:
            import re
            resolved_image = next(
                (v for v in (
                    image_harbor if image_harbor and image_harbor != "N/A" else None,
                    model_info.image_harbor_path,
                    self.config.publish.harbor_path,
                    self.config.publish.existing_harbor_image,
                    self.config.publish.image_target_tag,
                ) if v),
                "",
            )
            if resolved_image:
                if '{{IMAGE}}' in vars["container_run_cmd"]:
                    vars["container_run_cmd"] = vars["container_run_cmd"].replace('{{IMAGE}}', resolved_image)
                elif re.search(r'harbor\S+', vars["container_run_cmd"]):
                    vars["container_run_cmd"] = re.sub(r'harbor\S+', resolved_image, vars["container_run_cmd"], count=1)
            # 兜底解析仍失败 → 占位符会漏进 README，属发布数据缺失，明确阻断而非静默产出坏 README
            if '{{IMAGE}}' in vars["container_run_cmd"]:
                raise RuntimeError(
                    "README 生成失败：docker run 命令无法确定镜像名（image_harbor_path / "
                    "harbor_path / existing_harbor_image / image_target_tag 均为空），"
                    "拒绝写出含 {{IMAGE}} 占位符的 README。请检查镜像打 tag / 推送是否成功。"
                )
        vars["serve_start_cmd"] = model_info.serve_start_cmd.strip() if model_info.serve_start_cmd else ""
        vars["serve_infer_cmd"] = model_info.serve_infer_cmd.strip() if model_info.serve_infer_cmd else self._default_curl_cmd()

        # plugin 镜像固化 VLLM_PLUGINS=fl 前缀（统一走 _ensure_plugin_prefix，判据/幂等见该方法）
        vars["serve_start_cmd"] = self._ensure_plugin_prefix(vars["serve_start_cmd"])

        # 一致性校验：serve_start_cmd 中必须包含 canonical_model_path
        if vars["serve_start_cmd"] and canonical_path not in vars["serve_start_cmd"]:
            print(f"  ⚠ 路径一致性警告: serve_start_cmd 中未包含 canonical_model_path ({canonical_path})")
            print(f"    serve_start_cmd: {vars['serve_start_cmd'][:120]}...")
            # 尝试自动修正：替换 vllm serve 后的路径
            import re
            vars["serve_start_cmd"] = re.sub(
                r'(vllm\s+serve\s+)\S+', rf'\1{canonical_path}', vars["serve_start_cmd"])

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
        canonical_model_path = model_info.canonical_model_path or f"/data/{flagrelease_name}"
        new_model_intro = model_info.new_model_introduction or ""
        eval_table = self._generate_evaluation_table()
        docker_version = model_info.docker_version or "N/A"
        os_info = model_info.ubuntu_version or "Linux"
        image_harbor = self._readme_pull_image()
        image_pull_cmd = f"docker pull {image_harbor}" if image_harbor else ""
        container_run_cmd = model_info.container_run_cmd or ""
        serve_start_cmd = self._ensure_plugin_prefix(model_info.serve_start_cmd or "")
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
modelscope download --model FlagRelease/{flagrelease_name} --local_dir {canonical_model_path}
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
        print(f"  可见性: 私有（强制）")

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
        # 强制私有发布，不留公开口子（ModelScope visibility=1 私有；恒私有，不再随 config 变）
        visibility = 1
        private_label = '私有'

        sdk_script = f"""
import os, sys
from modelscope.hub.api import HubApi
api = HubApi()
token = os.environ.get('MODELSCOPE_API_TOKEN', '')
if token:
    api.login(token)
model_id = '{model_id}'
print(f'检查 ModelScope 模型仓库: {{model_id}}')
_private_ok = False
try:
    api.get_model(model_id)
    print('仓库已存在，强制设为私有...')
    for fn in ('update_model_visibility', 'update_model'):
        f = getattr(api, fn, None)
        if f is None:
            continue
        try:
            try:
                f(model_id=model_id, visibility={visibility})
            except TypeError:
                f(model_id, {visibility})
            print(f'  已通过 {{fn}} 设为私有')
            _private_ok = True
            break
        except Exception as e:
            print(f'  {{fn}} 失败: {{e}}')
except Exception:
    print('仓库不存在，创建私有仓...')
    try:
        api.create_model(model_id=model_id, visibility={visibility})
        print('仓库创建成功 ({private_label})')
        _private_ok = True
    except Exception as e:
        print(f'创建仓库失败: {{e}}')
if not _private_ok:
    print('x 无法确保私有可见性，拒绝上传（不留公开口子）')
    raise SystemExit(1)
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

    def _ensure_modelscope_private_repo(self, model_id: str, token: str, visibility: str = "private") -> bool:
        """确保 ModelScope 仓库以【私有】存在。
        upload 无可见性参数、自动建仓默认公开，故上传前必须保证私有仓已存在。
        流程：CLI create --visibility private → 若失败（含已存在）用 SDK 建私有/翻私有兜底 → 校验私有。
        返回 True 仅当能确认仓库存在且为私有。
        """
        token_env = f"MODELSCOPE_API_TOKEN={token} " if token else ""
        # 1) CLI 建私有仓
        create_cmd = f"PATH=/opt/conda/bin:$PATH {token_env}modelscope create {model_id} --visibility {visibility}"
        print(f"  创建私有仓库: {model_id} ({visibility})")
        result, _, _ = self.run_command(
            cmd=create_cmd, step_name="创建 ModelScope 私有仓库",
            timeout=60, in_container=True
        )
        # 2) 无论 create 成功或失败（可能已存在/公开），都用 SDK 强制建私有 + 翻私有兜底
        sdk_script = f"""
import os, sys
from modelscope.hub.api import HubApi
api = HubApi()
token = os.environ.get('MODELSCOPE_API_TOKEN', '')
if token:
    api.login(token)
model_id = '{model_id}'
ok = False
# 已存在则强制翻私有；不存在则建私有
try:
    api.get_model(model_id)
    print('仓库已存在，强制设为私有...')
    for fn in ('update_model_visibility', 'update_model'):
        try:
            f = getattr(api, fn, None)
            if f is None:
                continue
            try:
                f(model_id=model_id, visibility=1)
            except TypeError:
                f(model_id, 1)
            print(f'  已通过 {{fn}} 设为私有')
            ok = True
            break
        except Exception as e:
            print(f'  {{fn}} 失败: {{e}}')
    if not ok:
        print('  ! 未能确认翻私有，视为失败')
except Exception:
    print('仓库不存在，创建私有仓...')
    try:
        api.create_model(model_id=model_id, visibility=1)
        print('  私有仓创建成功')
        ok = True
    except Exception as e:
        print(f'  创建私有仓失败: {{e}}')
sys.exit(0 if ok else 1)
"""
        script_b64 = base64.b64encode(sdk_script.encode()).decode()
        cmd = f"{token_env}PATH=/opt/conda/bin:$PATH python3 -c \"import base64;exec(base64.b64decode('{script_b64}').decode())\""
        sdk_ok, _, _ = self.run_command(
            cmd=cmd, step_name="确保 ModelScope 私有仓库（SDK 兜底）",
            timeout=120, in_container=True
        )
        if sdk_ok:
            print(f"  ✓ 已确保 ModelScope 私有仓库: {model_id}")
            return True
        # SDK 兜底失败：仅当 CLI create 明确成功（新建私有）时才放行
        if result:
            print(f"  ✓ CLI 已创建私有仓库，SDK 校验未通过但放行: {model_id}")
            return True
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

        # 强制私有发布，不留公开口子。
        # 关键：modelscope upload 无可见性参数，仓库不存在时会自动建【公开】仓。
        # 因此上传前必须先确保【私有】仓库存在，否则宁可失败也不让 upload 裸建公开仓。
        visibility = "private"
        if not self._ensure_modelscope_private_repo(model_id, token, visibility):
            print(f"  x 无法确保 ModelScope 私有仓库存在，中止上传（拒绝 upload 自动建公开仓）")
            return False

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

    _HF_ENDPOINTS = ["https://huggingface.co", "https://hf-mirror.com"]

    def _probe_hf_endpoints(self, endpoints: List[str]) -> List[str]:
        """容器内探测 HF 端点联通性，返回可达端点（按延迟升序）。

        上传前先判断 huggingface.co / hf-mirror.com 哪个能联通，只对可达
        端点发起上传，全部不可达直接快速失败——避免对不可达端点发起
        3600s×5 重试链（发布阶段卡死根因：CN 环境直连不可达时，整轮
        空耗直到 task_runner 6h 杀任务）。探测与上传同一网络环境
        （同一容器、同一代理环境变量），保证选路与实际上传一致。
        """
        if not endpoints:
            return []
        probe_script = f"""
import sys, time, json, urllib.request
results = []
for ep in {json.dumps(endpoints)}:
    t0 = time.time()
    try:
        req = urllib.request.Request(ep, method='HEAD')
        r = urllib.request.urlopen(req, timeout=30)
        code = getattr(r, 'status', None) or getattr(r, 'getcode', lambda: 0)()
        print(f'  \\u2713 {{ep}} 可达 ({{time.time()-t0:.2f}}s, HTTP {{code}})')
        results.append([time.time()-t0, ep])
    except Exception as e:
        print(f'  x {{ep}} 不可达 ({{type(e).__name__}})')
print('PROBE_RESULT:' + json.dumps(results))
"""
        script_b64 = base64.b64encode(probe_script.encode()).decode()
        cmd = f"PATH=/opt/conda/bin:$PATH python3 -c \"import base64;exec(base64.b64decode('{script_b64}').decode())\""
        ok, stdout, stderr = self.run_command(
            cmd=cmd, step_name="探测 HuggingFace 端点联通性",
            timeout=30 * len(endpoints) + 30, in_container=True
        )
        reachable: List[str] = []
        for line in (stdout or "").splitlines():
            line = line.strip()
            if line.startswith("PROBE_RESULT:"):
                try:
                    reachable = [ep for _lat, ep in sorted(
                        json.loads(line[len("PROBE_RESULT:"):]))]
                except Exception:
                    pass
            elif line:
                # run_command 成功时不回显 stdout，探测明细（✓/x）手动打到发布日志
                print(line)
        return reachable

    def _publish_to_huggingface(self, readme_path: Optional[str]) -> bool:
        """发布到 HuggingFace（联通预检选路；CLI 优先，SDK 降级）"""
        publish_config = self.config.publish

        model_name = self.config.model_info.flagrelease_name or self.config.model_info.output_name
        repo_id = publish_config.huggingface_repo_id or f"FlagRelease/{model_name}"

        container_upload_dir = self._get_container_upload_dir()
        print(f"  容器内上传目录: {container_upload_dir}")
        print(f"  目标仓库: {repo_id}")
        print(f"  可见性: 私有（强制）")

        # 如果用户已指定 endpoint，只用该 endpoint
        user_endpoint = os.environ.get("HF_ENDPOINT", "")
        endpoints = [user_endpoint] if user_endpoint else self._HF_ENDPOINTS

        # 联通预检：只对可达端点发起上传（实测 CN 环境直连不可达、镜像可达），
        # 全不可达快速失败，不空耗重试链
        reachable = self._probe_hf_endpoints(endpoints)
        if not reachable:
            print(f"  x 所有 HuggingFace 端点均不可达: {', '.join(endpoints)}")
            print(f"    快速失败（不再发起上传重试链，避免发布阶段空耗卡死）")
            return False
        endpoints = reachable
        print(f"  ✓ 联通端点: {', '.join(endpoints)}")

        for i, endpoint in enumerate(endpoints):
            os.environ["HF_ENDPOINT"] = endpoint
            print(f"  尝试 HuggingFace endpoint: {endpoint}")

            if self._publish_to_huggingface_cli(readme_path):
                return True

            print("  CLI 方式失败，尝试使用 SDK...")
            if self._publish_to_huggingface_sdk(readme_path):
                return True

            if i < len(endpoints) - 1:
                print(f"  ⚠ endpoint {endpoint} 不可用，切换到 {endpoints[i+1]}")

        return False

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
        # 强制私有发布，不留公开口子
        private_flag = "True"
        hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

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
_private_ok = False
try:
    api.repo_info(repo_id=repo_id)
    print('仓库已存在，强制设为私有...')
    for fn in ('update_repo_settings', 'update_repo_visibility'):
        f = getattr(api, fn, None)
        if f is None:
            continue
        try:
            f(repo_id=repo_id, private=True)
            print(f'  已通过 {{fn}} 设为私有')
            _private_ok = True
            break
        except Exception as e:
            print(f'  {{fn}} 失败: {{e}}')
except Exception:
    print('仓库不存在，创建私有仓...')
    try:
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
        print('仓库创建成功（私有）')
        _private_ok = True
    except Exception as e:
        print(f'创建仓库失败: {{e}}')
if not _private_ok:
    print('x 无法确保私有可见性，拒绝上传（不留公开口子）')
    raise SystemExit(1)
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

    def _ensure_hf_private_repo(self, repo_id: str, token: str, hf_endpoint: str) -> bool:
        """确保 HuggingFace 仓库以【私有】存在（SDK 执行，容器内）。
        hf upload --private 不可靠（仅建仓生效、对已存在仓库无效），故上传前独立确保私有。
        已存在则强制翻私有；不存在则 create_repo(private=True)。返回 True 仅当私有确认成功。
        """
        sdk_script = f"""
import os, sys
os.environ['HF_ENDPOINT'] = '{hf_endpoint}'
from huggingface_hub import HfApi, login
token = os.environ.get('HF_TOKEN', '')
if token:
    login(token=token)
api = HfApi()
repo_id = '{repo_id}'
_private_ok = False
try:
    api.repo_info(repo_id=repo_id)
    print('仓库已存在，强制设为私有...')
    for fn in ('update_repo_settings', 'update_repo_visibility'):
        f = getattr(api, fn, None)
        if f is None:
            continue
        try:
            f(repo_id=repo_id, private=True)
            print(f'  已通过 {{fn}} 设为私有')
            _private_ok = True
            break
        except Exception as e:
            print(f'  {{fn}} 失败: {{e}}')
except Exception:
    print('仓库不存在，创建私有仓...')
    try:
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
        print('  私有仓创建成功')
        _private_ok = True
    except Exception as e:
        print(f'  创建私有仓失败: {{e}}')
sys.exit(0 if _private_ok else 1)
"""
        token_env = f"HF_TOKEN={token} " if token else ""
        script_b64 = base64.b64encode(sdk_script.encode()).decode()
        cmd = f"{token_env}HF_ENDPOINT={hf_endpoint} PATH=/opt/conda/bin:$PATH python3 -c \"import base64;exec(base64.b64decode('{script_b64}').decode())\""
        ok, _, _ = self.run_command(
            cmd=cmd, step_name="确保 HuggingFace 私有仓库",
            timeout=120, in_container=True
        )
        if ok:
            print(f"  ✓ 已确保 HuggingFace 私有仓库: {repo_id}")
        return ok

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
        hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        token_env = f"HF_TOKEN={token} " if token else ""
        endpoint_env = f"HF_ENDPOINT={hf_endpoint} "

        print(f"  目标仓库: {repo_id}")
        print(f"  容器内上传目录: {container_upload_dir}")

        if token:
            login_cmd = f"PATH=/opt/conda/bin:$PATH {token_env}{endpoint_env}hf auth login --token {token}"
            success, _, _ = self.run_command(
                cmd=login_cmd, step_name="HuggingFace 登录",
                timeout=60, in_container=True
            )
            if not success:
                return False

        # 强制私有发布，不留公开口子。
        # 关键：hf upload 的 --private 仅在自动建仓时生效，对已存在仓库无效，且不可靠。
        # 因此上传前必须先用 SDK 独立确保【私有】仓库存在，否则宁可失败也不上传。
        if not self._ensure_hf_private_repo(repo_id, token, hf_endpoint):
            print(f"  x 无法确保 HuggingFace 私有仓库存在，中止上传（不留公开口子）")
            return False

        private_flag = "--private "
        upload_cmd = f"PATH=/opt/conda/bin:$PATH {token_env}{endpoint_env}hf upload {private_flag}{repo_id} {container_upload_dir}".strip()

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
            shell_cmd = f"PATH=/opt/conda/bin:$PATH hf upload {repo_id} {container_tmp}/README.md README.md"
            # HuggingFace endpoint fallback：用户指定则只用该 endpoint，否则依次尝试
            # 直连 huggingface.co 与国内镜像 hf-mirror.com（后者在受限网络中可达且支持上传）
            user_endpoint = os.environ.get("HF_ENDPOINT", "")
            hf_endpoints = [user_endpoint] if user_endpoint else self._HF_ENDPOINTS
            # 联通预检：只对可达端点发起 README 更新（与主上传同因：不可达端点
            # 整轮重试是发布阶段空耗卡死的来源）
            hf_endpoints = self._probe_hf_endpoints(hf_endpoints)
            if not hf_endpoints:
                print(f"  x 所有 HuggingFace 端点均不可达，README 更新快速失败")
                return False
            # HuggingFace 需外网访问；从主进程环境注入代理到容器（ModelScope 走 .cn 无需代理）
            proxy_flags: List[str] = []
            for env_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                val = os.environ.get(env_var)
                if val:
                    proxy_flags.extend(["-e", f"{env_var}={val}"])
            docker_cmds = []
            for ep in hf_endpoints:
                docker_cmds.append(["docker", "exec",
                                    "-e", f"HF_TOKEN={token}",
                                    "-e", f"HF_ENDPOINT={ep}"]
                                   + proxy_flags
                                   + [container, "bash", "-c", shell_cmd])
        else:
            print(f"  x 未知平台: {platform}")
            return False

        # 带重试的上传（HuggingFace 会在每次尝试轮换 endpoint）
        current_delay = UPLOAD_RETRY_DELAY
        for attempt in range(UPLOAD_MAX_RETRIES):
            if platform == "huggingface":
                ep_idx = attempt % len(docker_cmds)
                docker_cmd = docker_cmds[ep_idx]
                print(f"[{self.name}] 执行: {step_name} (尝试 {attempt+1}/{UPLOAD_MAX_RETRIES}, endpoint={hf_endpoints[ep_idx]})")
            else:
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
