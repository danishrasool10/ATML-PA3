"""Generate task_1/notebooks/pa3_task1_kaggle.ipynb — a thin Kaggle bootstrap that pulls the code
from GitHub (or from an attached copy of the repo) and runs the deadline-aware pipeline."""
from __future__ import annotations

import json
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]

INTRO = r"""# ATML PA3 · Task 1 — Qwen3-0.6B math fine-tuning (Full FT · L2-SP · LoRA-4/16 · OPLoRA-16)
**CS-6304 / EE-5102 · Danish Rasool (28100233)** · code: https://github.com/danishrasool10/ATML-PA3 (`task_1/`, shared code in `common/`)

### Before running
1. **Settings → Accelerator: GPU T4 ×2**, **Internet: On**.
2. **Add Input** → a dataset containing `first_plot_questions.yaml` (any folder; it is found automatically).
3. Use **Save Version → Save & Run All (Commit)** so the run continues with the browser closed and the outputs are saved.

### What happens
`STAGE = "AUTO"` runs the whole task in one session, in this order, using **both GPUs** (DDP training; evaluations
two at a time, one per GPU): downloads → data (40k MetaMathQA) / probe set / OPLoRA bases / activation cache →
preflight (2-GPU check, FP16 check, worst-case-memory update with ZeRO-1 fallback) → Full FT → evals → L2-SP →
evals → LoRA-4 → LoRA-16 → evals → OPLoRA-16 → eval ‖ diagnostics → qualitative sheet → report tables/figures.

It never runs into Kaggle's 12-hour kill: it stops starting new work when the budget (`BUDGET_HOURS`) runs low,
training pauses with a recovery checkpoint, and the notebook finishes normally so **Kaggle saves the outputs**.
Crashes are retried automatically (training resumes from its latest recovery checkpoint; evaluation OOM halves the
batch size; finished evaluation parts are cached).

### If the status at the end says "paused"
Open the version → **Output** → **New Dataset** (e.g. `pa3-run1`), start a new session with that dataset attached
(plus the questions YAML) and run again. Finished work is detected and skipped; paused training resumes exactly.
"""

SETTINGS = '''# ============================ SETTINGS ============================
import os, time
os.environ.setdefault("PA3_SESSION_START", str(time.time()))   # the session budget starts now
STAGE = "AUTO"              # "AUTO" (everything) | "S0" extended smoke test | "S1" / "S2" / "S3"
BUDGET_HOURS = 11.25        # stop cleanly before Kaggle's 12 h limit (leaves time to save outputs)
REPO_URL = "https://github.com/danishrasool10/ATML-PA3"
REPO_REF = "main"           # branch, tag or commit to run
CODE_DIR = "/kaggle/working/ATML-PA3"
OUTPUT_ROOT = "/kaggle/working/pa3_outputs"
S0_CPU_TESTS = True         # S0 only'''

GET_CODE = '''# ============================ GET THE CODE ============================
# Prefers a copy of the repository attached as a Kaggle input (no network needed); otherwise clones GitHub.
import glob, shutil, subprocess, time
def _attached_repo():
    for p in sorted(glob.glob("/kaggle/input/**/task_1/src/pa3/orchestrate.py", recursive=True)):
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(p))))
    return None
src = _attached_repo()
if os.path.exists(CODE_DIR):
    shutil.rmtree(CODE_DIR)
if src:
    shutil.copytree(src, CODE_DIR, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    print("using attached repository copy:", src)
else:
    for i in range(4):
        r = subprocess.run(["git", "clone", "--quiet", REPO_URL, CODE_DIR], capture_output=True, text=True)
        if r.returncode == 0:
            break
        print("git clone failed, retrying:", r.stderr[-300:]); time.sleep(15 * (i + 1)); shutil.rmtree(CODE_DIR, ignore_errors=True)
    else:
        raise RuntimeError("could not clone the repository — is Internet enabled?")
    subprocess.run(["git", "-C", CODE_DIR, "checkout", "--quiet", REPO_REF], check=True)
commit = subprocess.run(["git", "-C", CODE_DIR, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("code commit:", commit or "(attached copy, no git metadata)")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
open(f"{OUTPUT_ROOT}/code_commit.txt", "a").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {commit or src}\\n")'''

INSTALL = '''# ============================ DEPENDENCIES ============================
# Pinned versions; Kaggle's PyTorch (CUDA build) is kept as-is.
import sys
sys.path[:0] = [f"{CODE_DIR}/common", f"{CODE_DIR}/task_1/src"]
from pa3common.kaggle import pip_install
import torch
torch_before = torch.__version__
pip_install(["transformers==5.21.0", "peft==0.21.2", "lm_eval[hf]==0.4.13", "pytest", "safetensors", "matplotlib"])
out = subprocess.run([sys.executable, "-c", "import torch, transformers, peft, lm_eval; print(torch.__version__, "
                      "transformers.__version__, peft.__version__)"], capture_output=True, text=True)
print("kernel torch:", torch_before, "| subprocess torch/transformers/peft:", out.stdout.strip(), out.stderr[-400:])
assert out.stdout.split()[0] == torch_before, "pip changed torch — restart the kernel and rerun"
assert tuple(int(x) for x in torch_before.split("+")[0].split(".")[:2]) >= (2, 6), "transformers 5.x needs torch >= 2.6"
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("GPUs:", gpus)
if len(gpus) < 2:
    print("!! WARNING: fewer than 2 GPUs visible — select 'GPU T4 x2' in the notebook settings")'''

RUN = '''# ============================ RUN ============================
os.chdir(f"{CODE_DIR}/task_1")
from pa3 import orchestrate as O
info = O.setup_env(OUTPUT_ROOT, budget_hours=BUDGET_HOURS)
assert info["questions_yaml"], "first_plot_questions.yaml not found under /kaggle/input — attach it before running"
runner = {"AUTO": O.stage_auto, "S1": O.stage_s1, "S2": O.stage_s2, "S3": O.stage_s3,
          "S0": lambda: O.stage_s0(cpu_tests=S0_CPU_TESTS)}[STAGE]
result = runner()'''

SHOW = '''# ============================ RESULTS ============================
import json
from pathlib import Path
from IPython.display import Markdown, Image, display
R = Path(OUTPUT_ROOT)
if (R / "status.json").exists():
    print(json.dumps(json.loads((R / "status.json").read_text()), indent=1))
for t in sorted((R / "report_assets/tables").glob("*.md")) if (R / "report_assets/tables").exists() else []:
    display(Markdown(f"**{t.name}**\\n\\n" + t.read_text()))
for f in sorted((R / "report_assets/figures").glob("*.png")) if (R / "report_assets/figures").exists() else []:
    display(Image(filename=str(f)))'''

_N = [0]


def _cell(kind: str, src: str) -> dict:
    _N[0] += 1
    c = {"cell_type": kind, "id": f"cell-{_N[0]:03d}", "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def build() -> dict:
    cells = [_cell("markdown", INTRO), _cell("code", SETTINGS), _cell("code", GET_CODE), _cell("code", INSTALL),
             _cell("code", RUN), _cell("code", SHOW)]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    out = TASK / "notebooks" / "pa3_task1_kaggle.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False))
    print(f"wrote {out}")
