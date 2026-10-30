"""Generate task_2/notebooks/pa3_task2_kaggle.ipynb — a self-contained Kaggle notebook.

Every source file of task_2/ (configs, src, scripts, tests, README) and the shared common/pa3common modules is
embedded as a `%%writefile` cell, followed by an integrity check (sha256 of each written file), dependency
installation and the deadline-aware AUTO pipeline.  No git clone is needed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parent
CODE_DIR = "/kaggle/working/ATML-PA3"
COMMON_FILES = ["__init__.py", "utils.py", "kaggle.py", "lora.py"]

INTRO = r"""# ATML PA3 · Task 2 — Vision-Language Model and Visual Grounding (SigLIP → MLP → Qwen3-0.6B)
**CS-6304 / EE-5102 · Danish Rasool (28100233)** · self-contained notebook (all code is written to disk by the cells below)

### Before running
1. **Settings → Accelerator: GPU T4 ×2**, **Internet: On**.
2. **Save Version → Save & Run All (Commit)** so the run continues with the browser closed and outputs are saved.
3. Optional: attach a dataset with the COCO 2014 / VQAv2 annotation JSONs (and `train2014`/`val2014` image folders)
   if a download host is unreachable — attached copies under `/kaggle/input` are used automatically.

### What happens (`AUTO`)
hub prefetch → data selection (seed 6304; COCO captions, annotation-derived presence/counting pairs, VQAv2
complementary pairs; IDs saved) + image download → label audit sheets → SigLIP feature cache (FP16, one shard per
GPU, timed separately; SigLIP then leaves GPU memory) → preflight (3 updates per stage type) → **training: GPU0 =
196-token main run, GPU1 = 49-token run, concurrently** (4,500 updates each, dev eval every 500 + stage ends,
resumable checkpoints every 250) → final evaluation under correct / complementary swap / mismatched / no image
(pair-bootstrap CIs) → report figures and tables → `results_task2.zip`.

Training pauses itself before the session budget (`BUDGET_HOURS`) runs out and the notebook ends normally, so
Kaggle keeps the outputs. If `status.json` says **paused**: Output → New Dataset, attach it to a new session, run
again — finished steps are skipped and training resumes exactly.
"""

SETTINGS = f'''# ============================ SETTINGS ============================
import os, time
os.environ.setdefault("PA3_SESSION_START", str(time.time()))   # the session budget starts now
BUDGET_HOURS = 11.25        # stop cleanly before Kaggle's 12 h limit
RUN_49_TOKENS = True        # optional 49-token experiment on GPU1 (concurrent with the main run)
RUN_CPU_TESTS = False       # run the CPU test suite first (~5 min)
CODE_DIR = "{CODE_DIR}"   # fixed: the %%writefile cells below write here
OUTPUT_ROOT = "/kaggle/working/pa3_task2_outputs"'''

INSTALL = '''# ============================ DEPENDENCIES ============================
import sys, subprocess
sys.path[:0] = [f"{CODE_DIR}/common", f"{CODE_DIR}/task_2/src"]
from pa3common.kaggle import pip_install
import torch
torch_before = torch.__version__
pip_install(["transformers==5.21.0", "safetensors", "matplotlib", "pytest", "pyyaml"])
out = subprocess.run([sys.executable, "-c", "import torch, transformers; print(torch.__version__, transformers.__version__)"],
                     capture_output=True, text=True)
print("kernel torch:", torch_before, "| subprocess torch/transformers:", out.stdout.strip(), out.stderr[-400:])
assert out.stdout.split()[0] == torch_before, "pip changed torch — restart the kernel and rerun"
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("GPUs:", gpus)
if len(gpus) < 2:
    print("!! WARNING: fewer than 2 GPUs visible — the two runs will execute one after the other")'''

RUN = '''# ============================ RUN ============================
os.chdir(f"{CODE_DIR}/task_2")
from pa3vlm import orchestrate as O
info = O.setup_env(OUTPUT_ROOT, budget_hours=BUDGET_HOURS)
result = O.stage_auto(run_cpu_tests=RUN_CPU_TESTS, run_49=RUN_49_TOKENS)
print(result)'''

SHOW = '''# ============================ RESULTS ============================
import json
from pathlib import Path
from IPython.display import Markdown, Image, display
R = Path(OUTPUT_ROOT)
if (R / "status.json").exists():
    print(json.dumps(json.loads((R / "status.json").read_text()), indent=1))
T, F = R / "report_assets/tables", R / "report_assets/figures"
for t in sorted(T.glob("*.md")) if T.exists() else []:
    display(Markdown(f"**{t.name}**\\n\\n" + t.read_text()))
for f in sorted(F.glob("*.png")) if F.exists() else []:
    display(Image(filename=str(f)))'''

_N = [0]


def _cell(kind: str, src: str) -> dict:
    _N[0] += 1
    c = {"cell_type": kind, "id": f"cell-{_N[0]:03d}", "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def files() -> list[tuple[str, Path]]:
    out = [(f"common/pa3common/{f}", REPO / "common" / "pa3common" / f) for f in COMMON_FILES]
    for sub in ("configs", "src", "scripts", "tests"):
        for p in sorted((TASK / sub).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".yaml"):
                out.append((f"task_2/{p.relative_to(TASK).as_posix()}", p))
    out.append(("task_2/README.md", TASK / "README.md"))
    return out


def build() -> dict:
    fl = files()
    dirs = sorted({str(Path(CODE_DIR) / Path(rel).parent) for rel, _ in fl})
    mk = "# ============================ WRITE THE CODE ============================\n" \
         "import os\nfor d in " + json.dumps(dirs, indent=1) + ":\n    os.makedirs(d, exist_ok=True)\n" \
         f"print('writing {len(fl)} files under {CODE_DIR}')"
    cells = [_cell("markdown", INTRO), _cell("code", SETTINGS), _cell("code", mk)]
    sums = {}
    for rel, p in fl:
        txt = p.read_text()
        assert "\r" not in txt
        if not txt.endswith("\n"):
            txt += "\n"
        sums[rel] = hashlib.sha256(txt.rstrip("\n").encode()).hexdigest()   # robust to trailing-newline handling
        cells.append(_cell("code", f"%%writefile {CODE_DIR}/{rel}\n{txt}"))
    check = ("# ============================ INTEGRITY CHECK ============================\n"
             "import hashlib, json\n"
             f"EXPECTED = json.loads({json.dumps(json.dumps(sums))})\n"
             "bad = [r for r, h in EXPECTED.items()\n"
             f"       if hashlib.sha256(open('{CODE_DIR}/' + r, 'rb').read().rstrip(b'\\n')).hexdigest() != h]\n"
             "assert not bad, f'files differ from the embedded sources: {bad}'\n"
             "print(f'{len(EXPECTED)} files written and verified')")
    cells += [_cell("code", check), _cell("code", INSTALL), _cell("code", RUN), _cell("code", SHOW)]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    out = TASK / "notebooks" / "pa3_task2_kaggle.ipynb"
    out.parent.mkdir(exist_ok=True)
    nb = build()
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(nb['cells'])} cells, {len(files())} embedded files)")
