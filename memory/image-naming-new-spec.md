---
name: image-naming-new-spec
description: 镜像命名新规范(压缩版)由 get_image_name.sh 权威生成,generate_image_tag 改为调用它
metadata:
  type: project
---

2026-07-09 实施:镜像命名切换到新压缩规范,完全替换旧 `flagrelease-{vendor}-release-model_...` 格式。

**新格式**:`{registry}/{model}-{gpu}-gems{g}-tree{t}-cx{c}-plugin{p}-{vllm}{v}-cp{py}-pt{pt}-{sdk}{s}-{arch}-{driver}:{时间戳}[-vN]`
例:`GLM5.2-kunlunxin001-gems4.2.1-treenone-cx0.10.0-plugin0.1.0-vllm0.13.0-cp310-pt29-xrtnone-x64-515.58:202606251805-v2`

**架构(方案A 复用 shell)**:
- `skills/flagos-release/tools/get_image_name.sh` 是权威命名工具,进容器 `docker exec` 采集全部版本(pip list + 各厂商 smi/SDK)。宿主机运行,不入容器部署清单
- `chip_detector.py::generate_image_tag` 改为**调用该脚本**拿镜像名主体;新增 `container_name`/`vendor_name` 形参;旧 info/tree/gems/cx 形参保留但不再使用(脚本自采集)
- **关键:V1-V5 tag 区分保留** — 脚本自带时间戳被丢弃,改用 config.py 传入的 date_tag(已含 `-vN` 后缀)。双 tag/不适配 tag 靠 `-v[0-9]+$` 正则替换,依赖此后缀
- 新增 `_normalize_vendor_for_naming`:huawei→ascend, tianshu→iluvatar, moore→mthreads 等别名归一化(context 的 gpu.vendor 可能用别名)
- 两调用点已更新:config.py:472(from-context)、publish.py:419(容器输入模式)

**厂商 SDK 键**:nvidia=cu, ascend=cann(+ptnpu/vllm-ascend), hygon=dtk, metax=maca, mthreads=musa, kunlunxin=xrt, iluvatar=ixml, tsingmicro=raisa, zhenwu=hggc(+pp001 无需容器)
**两档压缩**:semver(gems/tree/cx/plugin/vllm 保留 x.y.z) / cver(python/torch/sdk 主次去点)
采集不到填 `none`(如 xrtnone),属预期非错误。

模拟验证过:归一化/V1-V5 五后缀保留/registry 前缀保留/脚本时间戳丢弃/双tag正则兼容/缺容器名报错/date_tag空回退。**未真机验证**(需真容器采集)。SKILL.md 命名章节已同步。相关:[[plugin-inheritance-cold-injection]]