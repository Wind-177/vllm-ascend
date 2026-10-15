# Issue #18615：0.30rc1 缓存布局回退验证

这是用于定位 OOM 的实验分支，尚未在 NPU 上证明能修复该问题。

## 代码范围

- 基线：`v0.30.0rc1`，提交 `5f828ad63`。
- 分支：`revert/issue-18615-kv-layout-v030`。
- 恢复 #14340 之前的普通混合模型 page 对齐、V1 连续 K/V 和连续 conv/SSM 状态布局。
- 保留 FLA 算子迁移、状态 gather/scatter、每组 kernel block size 接口和纯 GQA 路径。
- 专门验证 issue 中的 Qwen3.8-27B、TP2、无 MTP、关闭 prefix caching 场景。
- 必须使用 `VLLM_USE_V2_MODEL_RUNNER=0`；本分支没有回退 V2 的缓存构造逻辑。

这是缓存布局相关逻辑的定向回退，并非整个 #14340 的机械 revert。

## 准备代码

在原来能复现问题的 `quay.io/ascend/vllm-ascend:v0.30.0rc1` 容器内操作。
保持权重、CANN、torch、torch_npu、FLA 版本和可见 NPU 一致，每次对照前重启服务。

```bash
git clone --single-branch --branch revert/issue-18615-kv-layout-v030 \
  https://github.com/Wind-177/vllm-ascend.git issue18615-validation
cd issue18615-validation
git log -1 --oneline
```

若容器通过可编辑安装加载源码，请将实际使用的源码 checkout 切换到该分支。
若容器使用预装 wheel，可在本 checkout 中执行下面的覆盖操作。
它会核对原文件与 rc1 基线一致，备份原文件，再仅覆盖两份 Python 文件，复用容器已有的原生扩展。
不要在运行中的服务上覆盖文件。

```bash
python - <<'PY'
import importlib.util
import shutil
import subprocess
from pathlib import Path

package = Path(importlib.util.find_spec("vllm_ascend").origin).resolve().parent
source = Path.cwd() / "vllm_ascend"
backup = Path.cwd() / ".issue18615-package-backup"
files = (
    Path("patch/platform/patch_mamba_config.py"),
    Path("worker/model_runner_v1.py"),
)
assert package != source.resolve(), "Editable source already active; no overlay needed."
assert not backup.exists(), "Backup already exists; preserve it and inspect the previous overlay."
for relative in files:
    original = subprocess.check_output([
        "git", "show", f"v0.30.0rc1:vllm_ascend/{relative.as_posix()}"
    ])
    actual = (package / relative).read_bytes()
    assert actual.replace(b"\r\n", b"\n") == original.replace(b"\r\n", b"\n"), relative
for relative in files:
    (backup / relative).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(package / relative, backup / relative)
for relative in files:
    shutil.copy2(source / relative, package / relative)
print(f"Overlay installed: {package}; original files: {backup}")
PY
```

克隆时应保留基线 tag；若 `git show v0.30.0rc1:...` 找不到该 tag，先执行
`git fetch origin tag v0.30.0rc1`。

覆盖安装后的恢复命令如下，同样在本 checkout 中、停止服务后执行：

```bash
python - <<'PY'
import importlib.util
import shutil
from pathlib import Path

package = Path(importlib.util.find_spec("vllm_ascend").origin).resolve().parent
backup = Path.cwd() / ".issue18615-package-backup"
for relative in ("patch/platform/patch_mamba_config.py", "worker/model_runner_v1.py"):
    shutil.copy2(backup / relative, package / relative)
print(f"Original rc1 Python files restored: {package}")
PY
```

## 对照方法

1. 在原始 rc1 环境运行原来的服务命令，记录 OOM 的第一个异常栈和发生阶段。
2. 使用本分支或覆盖后的文件，原样重复服务命令。
3. 两组都固定 `VLLM_USE_V2_MODEL_RUNNER=0`、`gpu-memory-utilization=0.90`、TP2、
   `max-num-seqs=64`、`max-model-len=262144`、`max-num-batched-tokens=16384`、
   `FULL_DECODE_ONLY`、关闭 prefix caching，且不启用 MTP。
4. 若原始版本 OOM、回退版启动成功，再分别在 `0.85` 下做相同输入、输出长度和并发的性能对照。
   建议先固定并发 4，不要把并发 4 的吞吐直接与并发 26/34 的吞吐比较。

回退版应出现下面两条启动日志，用于确认加载了对应代码：

```text
Issue #18615 validation: restoring pre-#14340 hybrid cache page alignment.
Issue #18615 validation: Model Runner V1 hybrid cache layout rollback is active.
```

如需检查每层形状和 stride，启用 `VLLM_LOGGING_LEVEL=DEBUG`，检索
`[issue-18615-rollback]`。普通 full attention 和 linear attention 的缓存应显示连续布局。
正式性能对照应恢复相同的日志级别。

两组分别记录：

| 项目 | 原始 rc1 | 缓存布局回退版 |
| --- | --- | --- |
| 实际加载的 vLLM/Ascend/CANN/FLA 版本 | 待记录 | 待记录 |
| 0.90 是否启动成功、首个异常栈 | 待记录 | 待记录 |
| OOM 阶段：profile / KV 分配 / 图捕获 / 请求 | 待记录 | 待记录 |
| 权重、profile 峰值和可用 KV 显存 | 待记录 | 待记录 |
| block size、Mamba page size、KV block 数 | 待记录 | 待记录 |
| 图捕获前后显存和图捕获占用 | 待记录 | 待记录 |
| 相同并发下的 TPOT、TTFT、输出吞吐 | 待记录 | 待记录 |

本实验会同时改变 page 对齐和缓存视图，KV 容量也可能随之变化。
即使 OOM 消失，也只能先将范围收敛到这组布局/容量变化与下游算子的交互；
需要结合显存和 trace 才能区分容量计算、临时拷贝及 workspace 等原因。

## 验证记录

本地 Windows 环境没有 NPU，也没有完整的 vLLM/torch_npu 运行环境。
已通过隔离加载实际函数和现有测试体，用 CPU PyTorch 执行 30 个配置用例，以及
连续 K/V、conv/SSM 视图和底层写入用例；同一布局断言能拒绝原始 rc1 的非连续布局。
这些检查不覆盖真实算子调用、完整 pytest 集成、图捕获或 OOM。

`format.sh ci` 中 Ruff、codespell、typos、clang-format、Markdown、工作流和 Python 源码检查已通过。
全量命令因 Windows 缺少 `/bin/bash` 和 shellcheck 而未通过；Linux 环境仍需补跑完整检查。

在配置完整的 Linux 测试环境可执行：

```bash
pytest -q tests/ut/patch/platform/test_patch_mamba_config.py
pytest -q tests/ut/worker/test_model_runner_v1.py::TestNPUModelRunnerKVCache
bash format.sh ci
```
