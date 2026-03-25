# Release v0.4.0

**Changes since v0.3.0**

## Bug Fixes

- Fixed the bug where the submodule npu-ir could not be pulled synchronously when pulling the third-party library ascend ([#136](https://github.com/flagos-ai/FlagTree/pull/136)) by @SabYic
- Fix the issue with get_backend_args in setup ([#138](https://github.com/flagos-ai/FlagTree/pull/138)) by @SabYic
- Fix CMake Dependence bug ([#247](https://github.com/flagos-ai/FlagTree/pull/247)) by @starrryz

## Documentation

- Add latest news to readme ([#127](https://github.com/flagos-ai/FlagTree/pull/127)) by @zhzhcookie
- Enhance README with logo and formatting updates ([#152](https://github.com/flagos-ai/FlagTree/pull/152)) by @zhzhcookie
- Update readme for offline-build and enflame backend ([#176](https://github.com/flagos-ai/FlagTree/pull/176)) by @zhzhcookie
- Update aipu llvm readme ([#182](https://github.com/flagos-ai/FlagTree/pull/182)) by @zhzhcookie
- Update aipu llvm readme ([#183](https://github.com/flagos-ai/FlagTree/pull/183)) by @zhzhcookie
- Update aipu llvm readme ([#180](https://github.com/flagos-ai/FlagTree/pull/180)) by @zhzhcookie
- Update aipu llvm readme ([#179](https://github.com/flagos-ai/FlagTree/pull/179)) by @zhzhcookie
- Update readme for build tsingmicro & aipu ([#191](https://github.com/flagos-ai/FlagTree/pull/191)) by @zhzhcookie
- Update aipu llvm readme ([#181](https://github.com/flagos-ai/FlagTree/pull/181)) by @zhzhcookie
- Add feature-request issue template ([#206](https://github.com/flagos-ai/FlagTree/pull/206)) by @sgjzfzzf
- [CI/CD]  Update readme and spec doc, ignore md in workflow ([#222](https://github.com/flagos-ai/FlagTree/pull/222)) by @zhzhcookie
- Update readme about pypi ([#231](https://github.com/flagos-ai/FlagTree/pull/231)) by @zhzhcookie
- Update readme for xpu so and pip show flagtree ([#245](https://github.com/flagos-ai/FlagTree/pull/245)) by @zhzhcookie
- Update documents ([#268](https://github.com/flagos-ai/FlagTree/pull/268)) by @zhzhcookie
- Update readme, build and decoupling documents ([#273](https://github.com/flagos-ai/FlagTree/pull/273)) by @zhzhcookie

## CI/Infrastructure

- [CI/CD] Update workflow repo addr to flagos-ai ([#146](https://github.com/flagos-ai/FlagTree/pull/146)) by @zhzhcookie
- [CI/CD] Update workflow repo addr to flagos-ai on branch triton_v3.3.x ([#150](https://github.com/flagos-ai/FlagTree/pull/150)) by @zhzhcookie
- [CI/CD] Update workflow repo addr to flagos-ai on branch triton_v3.2.x ([#147](https://github.com/flagos-ai/FlagTree/pull/147)) by @zhzhcookie
- [CI/CD] Update metax workflow ([#190](https://github.com/flagos-ai/FlagTree/pull/190)) by @zhzhcookie
- [CI/CD] Update doc workflow to triton_v3.4.x ([#236](https://github.com/flagos-ai/FlagTree/pull/236)) by @zhzhcookie
- [CI/CD] Update doc workflow to triton_v3.5.x ([#235](https://github.com/flagos-ai/FlagTree/pull/235)) by @zhzhcookie
- [CI/CD] Update doc workflow to triton_v3.2.x ([#232](https://github.com/flagos-ai/FlagTree/pull/232)) by @zhzhcookie
- [CI/CD] Update workflow to check doc changed for triton 3.4 ([#237](https://github.com/flagos-ai/FlagTree/pull/237)) by @zhzhcookie
- [CI/CD] Update workflow to check doc changed for triton 3.5 ([#238](https://github.com/flagos-ai/FlagTree/pull/238)) by @zhzhcookie
- [CI/CD] Update doc workflow to triton_v3.3.x ([#233](https://github.com/flagos-ai/FlagTree/pull/233)) by @zhzhcookie
- [CI/CD] Update workflow to check doc changed for triton 3.2 ([#240](https://github.com/flagos-ai/FlagTree/pull/240)) by @zhzhcookie
- [CI/CD] Update workflow to check doc changed ([#241](https://github.com/flagos-ai/FlagTree/pull/241)) by @zhzhcookie
- [CI/CD] Add nvidia workflow for triton 3.5 ([#243](https://github.com/flagos-ai/FlagTree/pull/243)) by @zhzhcookie
- [CI/CD] Add nvidia workflow for triton 3.4 ([#242](https://github.com/flagos-ai/FlagTree/pull/242)) by @zhzhcookie
- [CI/CD] Use bot to code-format ([#267](https://github.com/flagos-ai/FlagTree/pull/267)) by @zhzhcookie

## Testing

- Add hint test cases on aipu ([#134](https://github.com/flagos-ai/FlagTree/pull/134)) by @ikushare

## Other

- Update Arm China NPU "Zhouyi" Backend Support ([#126](https://github.com/flagos-ai/FlagTree/pull/126)) by @Johnson9009
- Update submodule URL and commit ID for ascendnpu-ir ([#128](https://github.com/flagos-ai/FlagTree/pull/128)) by @Galaxy1458
- [build 3.1]Fix path reference in git clone command ([#130](https://github.com/flagos-ai/FlagTree/pull/130)) by @Galaxy1458
- [BUILD 3.3] Fix path variable usage in git clone command ([#132](https://github.com/flagos-ai/FlagTree/pull/132)) by @Galaxy1458
- [BUILD 3.1] Update tools.py ([#131](https://github.com/flagos-ai/FlagTree/pull/131)) by @Galaxy1458
- Offline Build Scripts for Triton-v3.1.x ([#129](https://github.com/flagos-ai/FlagTree/pull/129)) by @liuyunqi20
- [BUILD 3.2] Offline Build Scripts for Triton-v3.2.x ([#133](https://github.com/flagos-ai/FlagTree/pull/133)) by @liuyunqi20
- Offline Build Scripts for Triton-v3.3.x ([#90](https://github.com/flagos-ai/FlagTree/pull/90)) by @liuyunqi20
- Add getSharedMemoryTag for shared_memory hint ([#141](https://github.com/flagos-ai/FlagTree/pull/141)) by @youngyoung01
- Decouple iluvatar C++ ([#118](https://github.com/flagos-ai/FlagTree/pull/118)) by @zhzhcookie
- Ascend core_ext import semantic_ext as semantic ([#145](https://github.com/flagos-ai/FlagTree/pull/145)) by @liuyunqi20
- [BUILD] Enhance ascend submodule commit pull action ([#149](https://github.com/flagos-ai/FlagTree/pull/149)) by @Galaxy1458
- Decouple iluvatar C++ Phase Two ([#154](https://github.com/flagos-ai/FlagTree/pull/154)) by @toudefu
- [DOC] Add FlagTree_Backend_Specialization md ([#157](https://github.com/flagos-ai/FlagTree/pull/157)) by @zhzhcookie
- KUNLUNXIN xpu update to Commit 59d29d2(20251125) ([#156](https://github.com/flagos-ai/FlagTree/pull/156)) by @Ason93
- Decouple ascend c++ ([#160](https://github.com/flagos-ai/FlagTree/pull/160)) by @zhzhcookie
- Update cambricon ([#161](https://github.com/flagos-ai/FlagTree/pull/161)) by @chenmiao1919
- Update setup info for pypi ([#163](https://github.com/flagos-ai/FlagTree/pull/163)) by @zhzhcookie
- Update flagtree backend to version3.4 ([#162](https://github.com/flagos-ai/FlagTree/pull/162)) by @SabYic
- KUNLUNXIN xpu update to Commit da0ac6a(20251202) ([#166](https://github.com/flagos-ai/FlagTree/pull/166)) by @DuanYaQi
- [BUILD] Fix iluvatar build in editable mode ([#167](https://github.com/flagos-ai/FlagTree/pull/167)) by @zhzhcookie
- Update flagtree backend to version3.5 ([#169](https://github.com/flagos-ai/FlagTree/pull/169)) by @SabYic
- KUNLUNXIN xpu update to Commit 9855424(20251203) ([#164](https://github.com/flagos-ai/FlagTree/pull/164)) by @Ason93
- Fix some bugs on hygon hcl backend ([#170](https://github.com/flagos-ai/FlagTree/pull/170)) by @fangzexian
- Update Enflame GCU300 Backend Support ([#159](https://github.com/flagos-ai/FlagTree/pull/159)) by @baoqiliu
- Update enflame backend readme ([#174](https://github.com/flagos-ai/FlagTree/pull/174)) by @baoqiliu
- Correct enflame branch name ([#175](https://github.com/flagos-ai/FlagTree/pull/175)) by @baoqiliu
- Ascend python code decoupling ([#142](https://github.com/flagos-ai/FlagTree/pull/142)) by @liuyunqi20
- Support sqmma&wmma feature on mthreads backend. ([#172](https://github.com/flagos-ai/FlagTree/pull/172)) by @lingfengqiu
- KUNLUNXIN XPU Update to Commit 59a0daa(20251205) ([#178](https://github.com/flagos-ai/FlagTree/pull/178)) by @Ason93
- Update metax backend ([#184](https://github.com/flagos-ai/FlagTree/pull/184)) by @ArinaJJH
- Fix autotuner on cambricon backend ([#189](https://github.com/flagos-ai/FlagTree/pull/189)) by @chenmiao1919
- Fix iluvatar plugin build ([#192](https://github.com/flagos-ai/FlagTree/pull/192)) by @zhzhcookie
- Add pass management interfaces for ascend ([#165](https://github.com/flagos-ai/FlagTree/pull/165)) by @toudefu
- Add kunlunxin sdnn backend ([#185](https://github.com/flagos-ai/FlagTree/pull/185)) by @mikiya1991
- KUNLUNXIN XPU Update to Commit 11ec3ca(20251216) ([#195](https://github.com/flagos-ai/FlagTree/pull/195)) by @Ason93
- Fix autotuner on cambricon backend ([#194](https://github.com/flagos-ai/FlagTree/pull/194)) by @chenmiao1919
- KUNLUNXIN xpu backend fix ScanOp Conversion ([#196](https://github.com/flagos-ai/FlagTree/pull/196)) by @mikiya1991
- Fix iluvatar/mthreads issue where shared libraries for mthreads had to be copied manually. ([#193](https://github.com/flagos-ai/FlagTree/pull/193)) by @SabYic
- Iluvatar update to commit 269a48ff940(20251205) ([#199](https://github.com/flagos-ai/FlagTree/pull/199)) by @BruceDai003
- Decoupling iluvatar fix ([#200](https://github.com/flagos-ai/FlagTree/pull/200)) by @zhzhcookie
- Update iluvatar 20251217 ([#201](https://github.com/flagos-ai/FlagTree/pull/201)) by @zhzhcookie
- Iluvatar revert workaround in getPointer ([#204](https://github.com/flagos-ai/FlagTree/pull/204)) by @Salamanca001
- Provides an extra argument for dialect and pass it for the EdslMLIRJITFunction construction. ([#216](https://github.com/flagos-ai/FlagTree/pull/216)) by @SabYic
- Add musa_syncthreads_lm and close tiny-offset-hint compilation config for backend mthreads. ([#213](https://github.com/flagos-ai/FlagTree/pull/213)) by @lingfengqiu
- Ascend with flir and local ascend dir ([#135](https://github.com/flagos-ai/FlagTree/pull/135)) by @zhzhcookie
- Update_mthreads_pull_resoures_action ([#202](https://github.com/flagos-ai/FlagTree/pull/202)) by @Galaxy1458
- HYGON hcu fix tl.atomic_cas op error ([#218](https://github.com/flagos-ai/FlagTree/pull/218)) by @fangzexian
- Polish_asecnd_setup_action ([#224](https://github.com/flagos-ai/FlagTree/pull/224)) by @Galaxy1458
- Triton_patch python update and decouple ([#223](https://github.com/flagos-ai/FlagTree/pull/223)) by @liuyunqi20
- Invert flag conditions for third-party flags for triton_v3.3.x ([#227](https://github.com/flagos-ai/FlagTree/pull/227)) by @Galaxy1458
- Invert flag conditions for third-party flags for triton_v3.2.x ([#225](https://github.com/flagos-ai/FlagTree/pull/225)) by @Galaxy1458
- Invert flag conditions for third-party flags for main ([#226](https://github.com/flagos-ai/FlagTree/pull/226)) by @Galaxy1458
- Invert flag conditions from 'OFF' to 'ON' for third-party flags for triton_v3.5.x ([#229](https://github.com/flagos-ai/FlagTree/pull/229)) by @Galaxy1458
- Invert flag conditions from 'OFF' to 'ON' for third-party flags for triton_v3.4.x ([#228](https://github.com/flagos-ai/FlagTree/pull/228)) by @Galaxy1458
- Add_flagtree_configs ([#230](https://github.com/flagos-ai/FlagTree/pull/230)) by @Galaxy1458
- Move xpu lib to ksyuncs and cache store it ([#244](https://github.com/flagos-ai/FlagTree/pull/244)) by @zhzhcookie
- KUNLUNXIN xpu update to Commit c52c0fd(20251229) ([#246](https://github.com/flagos-ai/FlagTree/pull/246)) by @Ason93
- Enable use wheel as external LLVM ([#248](https://github.com/flagos-ai/FlagTree/pull/248)) by @starrryz
- KUNLUNXIN xpu update to Commit cefbb61(20251229) ([#250](https://github.com/flagos-ai/FlagTree/pull/250)) by @DuanYaQi
- Support tle extension ([#188](https://github.com/flagos-ai/FlagTree/pull/188)) by @huanghaoXcore
- Solve addresspace issue ([#251](https://github.com/flagos-ai/FlagTree/pull/251)) by @SabYic
- Relocate setup_tools directory of triton_3.5.x ([#254](https://github.com/flagos-ai/FlagTree/pull/254)) by @SabYic
- Change whl name from triton to flagtree ([#258](https://github.com/flagos-ai/FlagTree/pull/258)) by @zhzhcookie
- Change whl name from triton to flagtree fixed ([#259](https://github.com/flagos-ai/FlagTree/pull/259)) by @zhzhcookie
- Add CoC document to the project ([#256](https://github.com/flagos-ai/FlagTree/pull/256)) by @tengqm
- Relocate setup_tools directory of triton_3.4.x ([#253](https://github.com/flagos-ai/FlagTree/pull/253)) by @SabYic
- Fix git submodule update init ([#260](https://github.com/flagos-ai/FlagTree/pull/260)) by @zhzhcookie
- Fix git submodule update init for triton_v3.3.x (#260) ([#263](https://github.com/flagos-ai/FlagTree/pull/263)) by @zhzhcookie
- Triton_patch python update and decouple ([#265](https://github.com/flagos-ai/FlagTree/pull/265)) by @liuyunqi20
- Add initial tle design ([#261](https://github.com/flagos-ai/FlagTree/pull/261)) by @sunnycase
- Add tle.load(is_async=True) & tle.nvidia.memory_space ([#266](https://github.com/flagos-ai/FlagTree/pull/266)) by @sunnycase
- Add maintainers list to the project ([#255](https://github.com/flagos-ai/FlagTree/pull/255)) by @tengqm
- Replace default code owner for the project ([#262](https://github.com/flagos-ai/FlagTree/pull/262)) by @tengqm
- Update ascend backend to 5b0b2fce on triton-ascend ([#219](https://github.com/flagos-ai/FlagTree/pull/219)) by @ph0375
- [CI/CD] [TEST] Add flagtree_hints test ([#278](https://github.com/flagos-ai/FlagTree/pull/278)) by @zhzhcookie
- Put EDSL Code into `triton_v3.5.x` ([#272](https://github.com/flagos-ai/FlagTree/pull/272)) by @sgjzfzzf
- Remove debug info for shmem hints ([#279](https://github.com/flagos-ai/FlagTree/pull/279)) by @i3wanna2
- Update flagtree version to v0.4.0, and add release notes ([#276](https://github.com/flagos-ai/FlagTree/pull/276)) by @zhzhcookie

## Contributors

Thanks to all contributors who made this release possible:

- @zhzhcookie (45 PRs)
- @Galaxy1458 (13 PRs)
- @SabYic (9 PRs)
- @liuyunqi20 (7 PRs)
- @Ason93 (5 PRs)
- @baoqiliu (3 PRs)
- @chenmiao1919 (3 PRs)
- @tengqm (3 PRs)
- @DuanYaQi (2 PRs)
- @fangzexian (2 PRs)
- @lingfengqiu (2 PRs)
- @mikiya1991 (2 PRs)
- @sgjzfzzf (2 PRs)
- @starrryz (2 PRs)
- @sunnycase (2 PRs)
- @toudefu (2 PRs)
- @ArinaJJH
- @BruceDai003
- @Johnson9009
- @Salamanca001
- @huanghaoXcore
- @i3wanna2
- @ikushare
- @ph0375
- @youngyoung01

---
*Generated by [Release Notes Generator](https://github.com/flagos-ai/FlagRelease/tree/main/tools)*