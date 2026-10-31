"""Generate task_3/notebooks/pa3_task3_kaggle.ipynb — a self-contained Kaggle notebook.

Every source file of task_3/ (configs, src, scripts, tests, README) and the shared common/pa3common modules it uses
is embedded as a `%%writefile` cell, followed by an integrity check (sha256 of each written file), dependency
installation, the AUTO pipeline, a results view and the download cell.  Setup errors are caught so the last cell
(zip + download link) always runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parent
CODE_DIR = "/kaggle/working/ATML-PA3"
COMMON_FILES = ["__init__.py", "kaggle.py"]

INTRO = r"""# ATML PA3 · Task 3 — Test-Time Compute: Sampling, Selection and Adaptive Inference
**CS-6304 / EE-5102 · Danish Rasool (28100233)** · self-contained notebook (all code is written to disk by the cells below)

### Before running
1. **Add Input → the `ATML-PA3-Data` dataset** (expected at `/kaggle/input/atml-pa3-data`; any folder under
   `/kaggle/input` that contains the two model folders is found automatically).
2. **Settings → Accelerator: GPU T4 ×2** (only GPU0 is used, for the 50-question live run), **Internet: On**
   (Qwen2.5-3B-Instruct download; MathVista metadata from the Hub only if the supplied files lack it).
3. **Save Version → Save & Run All (Commit)**.

### What happens (`AUTO`)
data join + alignment checks → dev/held-out splits (seed 6304) + 10 candidate orders per question → **live generation
starts in the background on GPU0** (single 3,072 / parallel 3×1,024 / refinement 1,024+2,048; FP16, batch 1,
pinned to its own CPU cores) → in parallel on the other cores: answer grouping + grading (math-verify / MathVista
rules, frozen with a manifest) → signals (log-prob, self-certainty, visual engagement, prefix-calibrated VisConf) →
all selection rules at N ∈ {1,2,4,8,16,32} → signal analysis → adaptive stopping (dev-tuned, frozen, held-out) →
compute allocation → live evaluation → qualitative cases → figures/tables. The last cell zips the outputs into
`/kaggle/working/results_task3.zip` and prints a **download link** (also when something failed part-way).

If `status.json` shows `live_generation: paused`, make this version's output a dataset, attach it and run again:
finished stages and live calls are kept and only the remaining calls are generated.
"""

SETTINGS = f'''# ============================ SETTINGS ============================
import os, time, json
os.environ.setdefault("PA3_SESSION_START", str(time.time()))   # the session budget starts now
BUDGET_HOURS = 11.25                          # stop cleanly before Kaggle's 12 h limit
DATA_ROOT = "/kaggle/input/atml-pa3-data"     # searched recursively; /kaggle/input is the fallback
RUN_LIVE = True                               # Step 4 live generation on GPU0
RUN_CPU_TESTS = False                         # run the CPU test suite first (~1 min)
CODE_DIR = "{CODE_DIR}"   # fixed: the %%writefile cells below write here
OUTPUT_ROOT = "/kaggle/working/pa3_task3_outputs"
SETUP_ERROR = None'''

INSTALL = '''# ============================ DEPENDENCIES ============================
import sys, subprocess, traceback
sys.path[:0] = [f"{CODE_DIR}/common", f"{CODE_DIR}/task_3/src"]
try:
    from pa3common.kaggle import pip_install
    import torch
    torch_before = torch.__version__
    pip_install(["transformers==5.21.0", "math-verify[antlr4_13_2]==0.9.0", "matplotlib", "pytest", "pyyaml", "scipy"])
    out = subprocess.run([sys.executable, "-c", "import torch, transformers, math_verify, pyarrow; "
                          "import importlib.metadata as m; print(torch.__version__, transformers.__version__, "
                          "m.version('math-verify'), pyarrow.__version__)"], capture_output=True, text=True)
    print("kernel torch:", torch_before, "| subprocess torch/transformers/math-verify/pyarrow:", out.stdout.strip(),
          out.stderr[-400:])
    assert out.stdout.split()[0] == torch_before, "pip changed torch — restart the kernel and rerun"
    print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
          "| CPU cores:", len(os.sched_getaffinity(0)))
except Exception as ex:  # noqa: BLE001
    traceback.print_exc()
    SETUP_ERROR = ex'''

RUN = '''# ============================ RUN ============================
# Errors are caught so the RESULTS and DOWNLOAD cells still run (the error is re-raised at the very end).
import traceback
RUN_ERROR = SETUP_ERROR
if RUN_ERROR is None:
    os.chdir(f"{CODE_DIR}/task_3")
    os.environ["PA3_CONFIG_OVERRIDES"] = json.dumps({"data_root": DATA_ROOT})
    try:
        from pa3ttc import orchestrate as O
        info = O.setup_env(OUTPUT_ROOT, budget_hours=BUDGET_HOURS)
        result = O.stage_auto(run_cpu_tests=RUN_CPU_TESTS, run_live=RUN_LIVE)
        print(json.dumps(result, indent=1))
    except Exception as ex:  # noqa: BLE001
        traceback.print_exc()
        RUN_ERROR = ex
else:
    print("setup failed — skipping the pipeline:", repr(RUN_ERROR))'''

SHOW = '''# ============================ RESULTS ============================
from pathlib import Path
from IPython.display import Markdown, Image, display
R = Path(OUTPUT_ROOT)
if (R / "status.json").exists():
    print(json.dumps(json.loads((R / "status.json").read_text()), indent=1))
for name in ("data/math_alignment.json", "data/mathvista_alignment.json"):
    if (R / name).exists():
        print(name, json.dumps(json.loads((R / name).read_text()).get("alignment"), indent=1))
T, F = R / "report_assets/tables", R / "report_assets/figures"
for t in sorted(T.glob("*.md")) if T.exists() else []:
    display(Markdown(f"**{t.name}**\\n\\n" + t.read_text()))
for f in sorted(F.glob("*.png")) if F.exists() else []:
    display(Image(filename=str(f)))'''

DOWNLOAD = '''# ============================ DOWNLOAD RESULTS ============================
# Zips everything in the outputs folder and shows a download link (also after a failure).
import os, zipfile
from pathlib import Path
from IPython.display import FileLink, display
OUT = Path(globals().get("OUTPUT_ROOT", "/kaggle/working/pa3_task3_outputs"))
ZIP = Path("/kaggle/working/results_task3.zip")
n = 0
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(OUT.rglob("*")) if OUT.exists() else []:
        if f.is_file() and "_scratch" not in f.parts and f.suffix not in (".tmp", ".part", ".zip"):
            zf.write(f, f.relative_to(OUT.parent))
            n += 1
print(f"{ZIP}: {n} files, {ZIP.stat().st_size / 2**20:.1f} MiB")
os.chdir("/kaggle/working")
display(FileLink(ZIP.name))
if globals().get("RUN_ERROR") is not None:
    raise RuntimeError("the pipeline failed earlier (see the RUN cell); partial results are in the zip") from RUN_ERROR'''

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
                out.append((f"task_3/{p.relative_to(TASK).as_posix()}", p))
    out.append(("task_3/README.md", TASK / "README.md"))
    return out


def build() -> dict:
    _N[0] = 0
    fl = files()
    dirs = sorted({str(Path(CODE_DIR) / Path(rel).parent) for rel, _ in fl})
    mk = ("# ============================ WRITE THE CODE ============================\n"
          "import os\nfor d in " + json.dumps(dirs, indent=1) + ":\n    os.makedirs(d, exist_ok=True)\n"
          f"print('writing {len(fl)} files under {CODE_DIR}')")
    cells = [_cell("markdown", INTRO), _cell("code", SETTINGS), _cell("code", mk)]
    sums = {}
    for rel, p in fl:
        txt = p.read_text()
        assert "\r" not in txt, rel
        if not txt.endswith("\n"):
            txt += "\n"
        sums[rel] = hashlib.sha256(txt.rstrip("\n").encode()).hexdigest()   # robust to trailing-newline handling
        cells.append(_cell("code", f"%%writefile {CODE_DIR}/{rel}\n{txt}"))
    check = ("# ============================ INTEGRITY CHECK ============================\n"
             "import hashlib, json\n"
             f"EXPECTED = json.loads({json.dumps(json.dumps(sums))})\n"
             "bad = [r for r, h in EXPECTED.items()\n"
             f"       if hashlib.sha256(open('{CODE_DIR}/' + r, 'rb').read().rstrip(b'\\n')).hexdigest() != h]\n"
             "if bad:\n"
             "    SETUP_ERROR = RuntimeError(f'files differ from the embedded sources: {bad}')\n"
             "    print(SETUP_ERROR)\n"
             "else:\n"
             "    print(f'{len(EXPECTED)} files written and verified')")
    cells += [_cell("code", check), _cell("code", INSTALL), _cell("code", RUN), _cell("code", SHOW),
              _cell("code", DOWNLOAD)]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    out = TASK / "notebooks" / "pa3_task3_kaggle.ipynb"
    out.parent.mkdir(exist_ok=True)
    nb = build()
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(nb['cells'])} cells, {len(files())} embedded files)")
