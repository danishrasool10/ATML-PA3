"""Generate task_3/notebooks/pa3_task3_kaggle.ipynb — a short Kaggle notebook (Task 1 style).

The notebook clones github.com/danishrasool10/ATML-PA3 (or uses a copy of the repo attached as a Kaggle dataset),
checks that the clone carries the same code version as this notebook (pa3ttc.__version__), installs the pinned
dependencies, shows which data folders it found, runs the pipeline and zips the results with a download link.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
VERSION = re.search(r'__version__ = "([^"]+)"', (TASK / "src" / "pa3ttc" / "__init__.py").read_text()).group(1)

INTRO = r"""# ATML PA3 · Task 3 — Test-Time Compute: Sampling, Selection and Adaptive Inference
**CS-6304 / EE-5102 · Danish Rasool (28100233)** · code: `task_3/` of the ATML-PA3 repo (cloned below)

**Before running:** push the repo to GitHub · *Add Input* → your **ATML-PA3-Data** dataset · *Settings* → **GPU T4 ×2**,
**Internet on** · then **Save Version → Save & Run All**. Always run from the top.

Offline analysis (Steps 1–3, 5, 6) takes ~10 min on the CPU; the 50-question live experiment (Step 4) runs at the same
time on GPU0 and takes ~2–4 h. The last cell zips everything into `results_task3.zip` with a download link (also
after a failure)."""

SETTINGS = f'''# ============================ SETTINGS ============================
import os, sys, time, json
from pathlib import Path
os.environ.setdefault("PA3_SESSION_START", str(time.time()))   # the session budget starts now
REPO_URL = "https://github.com/danishrasool10/ATML-PA3.git"
BRANCH = "main"
EXPECTED_CODE_VERSION = "{VERSION}"   # must match task_3/src/pa3ttc/__init__.py in the pushed repo
CODE_DIR = "/kaggle/working/ATML-PA3"
OUTPUT_ROOT = "/kaggle/working/pa3_task3_outputs"
DATA_ROOT = "/kaggle/input"          # searched recursively for qwen25-math/ and qwen25vl-mathvista/
BUDGET_HOURS = 11.25                 # stop cleanly before Kaggle's 12 h limit
RUN_LIVE = True                      # Step 4 live generation on GPU0
RUN_CPU_TESTS = False                # run the CPU test suite first (~3 min)
SETUP_ERROR = None'''

SETUP = '''# ============================ CODE + DEPENDENCIES ============================
import glob, shutil, subprocess, traceback
try:
    shutil.rmtree(CODE_DIR, ignore_errors=True)
    r = subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, CODE_DIR], capture_output=True, text=True)
    if r.returncode != 0:          # no internet / private repo: use a copy of the repo attached as a dataset
        found = glob.glob("/kaggle/input/**/task_3/src/pa3ttc/__init__.py", recursive=True)
        if not found:
            raise RuntimeError(f"git clone failed ({r.stderr.strip()[-300:]}) and no attached copy of the repo found")
        shutil.copytree(Path(found[0]).parents[4], CODE_DIR)
    sys.path[:0] = [f"{CODE_DIR}/common", f"{CODE_DIR}/task_3/src"]
    import pa3ttc
    commit = subprocess.run(["git", "-C", CODE_DIR, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print("code version:", pa3ttc.__version__, "| commit:", commit or "(attached copy)")
    if pa3ttc.__version__ != EXPECTED_CODE_VERSION:
        raise RuntimeError(f"the repo has code version {pa3ttc.__version__}, this notebook expects "
                           f"{EXPECTED_CODE_VERSION}: push the latest task_3/ to GitHub and rerun")
    from pa3common.kaggle import pip_install
    import torch
    torch_before = torch.__version__
    pip_install(["transformers==5.21.0", "math-verify[antlr4_13_2]==0.9.0", "matplotlib", "pytest", "pyyaml", "scipy"])
    out = subprocess.run([sys.executable, "-c", "import torch, transformers, importlib.metadata as m; "
                          "print(torch.__version__, transformers.__version__, m.version('math-verify'))"],
                         capture_output=True, text=True)
    print("torch / transformers / math-verify:", out.stdout.strip(), out.stderr[-300:])
    assert out.stdout.split()[0] == torch_before, "pip changed torch — restart the kernel and rerun"
    print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
          "| CPU cores:", len(os.sched_getaffinity(0)))
except Exception as ex:  # noqa: BLE001
    traceback.print_exc()
    SETUP_ERROR = ex'''

DATACHECK = '''# ============================ DATA CHECK ============================
if SETUP_ERROR is None:
    from pa3ttc.data.discover import find_folders, listing
    try:
        folders = find_folders(DATA_ROOT, "/kaggle/input")
        for ds, d in folders.items():
            print(f"{ds:10s} {d}: {sorted(p.name for p in d.glob('*.parquet'))}")
    except Exception as ex:  # noqa: BLE001
        print("!! DATA NOT FOUND:", ex)
        SETUP_ERROR = ex
        os.makedirs(f"{OUTPUT_ROOT}/data", exist_ok=True)
        Path(f"{OUTPUT_ROOT}/data/input_listing.txt").write_text(listing([Path("/kaggle/input")], 3000))'''

RUN = '''# ============================ RUN ============================
# Errors are caught so the RESULTS and DOWNLOAD cells still run (the error is re-raised at the very end).
RUN_ERROR = SETUP_ERROR
if RUN_ERROR is None:
    os.chdir(f"{CODE_DIR}/task_3")
    os.environ["PA3_CONFIG_OVERRIDES"] = json.dumps({"data_root": DATA_ROOT})
    try:
        from pa3ttc import orchestrate as O
        O.setup_env(OUTPUT_ROOT, budget_hours=BUDGET_HOURS)
        Path(OUTPUT_ROOT, "code_commit.txt").write_text(f"{commit or 'attached copy'}\\nversion {pa3ttc.__version__}\\n")
        result = O.stage_auto(run_cpu_tests=RUN_CPU_TESTS, run_live=RUN_LIVE)
        print(json.dumps(result, indent=1))
        if result["failed"]:
            RUN_ERROR = RuntimeError(f"stages failed: {result['failed']} — see {OUTPUT_ROOT}/logs/<stage>.log")
            print("!!", RUN_ERROR)
    except Exception as ex:  # noqa: BLE001
        traceback.print_exc()
        RUN_ERROR = ex
else:
    print("setup failed — pipeline not started:", repr(RUN_ERROR))'''

SHOW = '''# ============================ RESULTS ============================
from IPython.display import Markdown, Image, display
R = Path(OUTPUT_ROOT)
if (R / "status.json").exists():
    print(json.dumps({k: v["state"] for k, v in json.loads((R / "status.json").read_text())["steps"].items()}, indent=1))
for t in sorted((R / "report_assets/tables").glob("*.md")):
    display(Markdown(f"**{t.name}**\\n\\n" + t.read_text()))
for f in sorted((R / "report_assets/figures").glob("*.png")):
    display(Image(filename=str(f)))'''

DOWNLOAD = '''# ============================ DOWNLOAD RESULTS ============================
import zipfile
from IPython.display import FileLink, display
OUT, ZIP = Path(OUTPUT_ROOT), Path("/kaggle/working/results_task3.zip")
n = 0
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(OUT.rglob("*")) if OUT.exists() else []:
        if f.is_file() and "_scratch" not in f.parts and f.suffix not in (".tmp", ".zip"):
            zf.write(f, f.relative_to(OUT.parent))
            n += 1
print(f"{ZIP}: {n} files, {ZIP.stat().st_size / 2**20:.1f} MiB")
os.chdir("/kaggle/working")
display(FileLink(ZIP.name))
if globals().get("RUN_ERROR") is not None:
    raise RuntimeError("the pipeline failed earlier (see the RUN cell); partial results are in the zip") from RUN_ERROR'''


def build() -> dict:
    cells = [("markdown", INTRO), ("code", SETTINGS), ("code", SETUP), ("code", DATACHECK), ("code", RUN),
             ("code", SHOW), ("code", DOWNLOAD)]
    out = []
    for i, (kind, src) in enumerate(cells, 1):
        c = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update({"execution_count": None, "outputs": []})
        out.append(c)
    return {"cells": out,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    path = TASK / "notebooks" / "pa3_task3_kaggle.ipynb"
    path.parent.mkdir(exist_ok=True)
    nb = build()
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(nb['cells'])} cells, code version {VERSION})")
