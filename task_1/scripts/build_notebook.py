"""Generate notebooks/pa3_task1_kaggle.ipynb from the repository (keeps notebook and code in sync)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INCLUDE = ["README.md", "configs", "src", "scripts", "tests"]
EXCLUDE_NAMES = {"__pycache__", ".pytest_cache"}

INTRO = r"""# PA3 Task 1 — Qwen3-0.6B math fine-tuning: Full FT · L2-SP · LoRA-4/16 · OPLoRA-16
**CS-6304 / EE-5102 Advanced Topics in ML · Danish Rasool (28100233)**

This notebook writes the complete `pa3_task1` package to `/kaggle/working/pa3_task1` and runs **one stage** per
Kaggle session. All results go to `/kaggle/working/pa3_outputs`.

### Kaggle settings (every session)
* **Accelerator:** GPU T4 ×2 · **Internet:** on · **Persistence:** none needed.
* Attach `first_plot_questions.yaml` as an input (any dataset; it is found anywhere under `/kaggle/input`).

### Stages (run in order, one per session)
| Stage | What it does | Approx. time |
|---|---|---|
| **S0** | Smoke test: env report, CPU test suite, data prep (40k), probe set, OPLoRA bases, FP16 sanity, GPU DDP-vs-single gradient check (+NCCL fallback), 20-update runs of all 5 methods (memory/throughput/projections, L2-SP ZeRO-1 fallback), merge check, activation cache, lm-eval + qualitative smoke | ~1 h |
| **S1** | Full FT → eval baseline ‖ Full FT midpoint (parallel, one per GPU) → eval Full FT final | see S0 projections |
| **S2** | Full FT + L2-SP → LoRA-4 → eval both (parallel) | |
| **S3** | LoRA-16 → OPLoRA-16 → eval both → diagnostics → qualitative sheet (168 responses / 42-row annotation) → report assets | |

### Carrying results between sessions
After a stage finishes: **Save Version** (Save & Run All) → open the version's **Output** → **New Dataset**
(e.g. `pa3-s0`, `pa3-s1`, …). In the next session **Add Input** → those datasets. Every directory named
`pa3_outputs` under `/kaggle/input` is discovered automatically; finished work there is reused, never redone.
If a session dies mid-training, attach its output and rerun the same stage: training resumes from the last
recovery checkpoint (model, optimizer, scheduler, scaler, RNG, data position).
"""


def file_cells() -> list[tuple[str, str]]:
    out = []
    for item in INCLUDE:
        p = REPO / item
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
        for f in files:
            if any(part in EXCLUDE_NAMES for part in f.parts) or f.suffix in (".pyc",) or f.name == "build_notebook.py":
                continue
            out.append((str(f.relative_to(REPO)), f.read_text()))
    return out


_N = [0]


def _id() -> str:
    _N[0] += 1
    return f"cell-{_N[0]:03d}"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "id": _id(), "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def build() -> dict:
    cells = [md(INTRO)]
    cells.append(code('''# ===================== SESSION SETTINGS =====================
STAGE = "S0"            # "S0" (smoke) -> "S1" -> "S2" -> "S3"
CPU_TESTS = "all"       # S0 only: "all" (~10-15 min on Kaggle CPUs), "quick" (skip DDP/resume tests), "none"
SMOKE_UPDATES = 20      # S0 only
CODE_DIR = "/kaggle/working/pa3_task1"
OUTPUT_ROOT = "/kaggle/working/pa3_outputs"'''))
    cells.append(md("## 1 · Pinned dependencies\nPyTorch is **not** reinstalled (Kaggle's CUDA build is kept)."))
    cells.append(code('''import subprocess, sys
import torch
torch_before = torch.__version__
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed",
                "transformers==5.21.0", "peft==0.21.2", "lm_eval[hf]==0.4.13", "pytest", "safetensors"], check=True)
out = subprocess.run([sys.executable, "-c", "import torch, transformers, peft, lm_eval; print(torch.__version__, "
                      "transformers.__version__, peft.__version__)"], capture_output=True, text=True)
print("torch (kernel) =", torch_before, "| subprocess torch/transformers/peft =", out.stdout.strip(), out.stderr[-500:])
assert out.stdout.split()[0] == torch_before, "pip changed torch — restart the kernel and rerun"
assert tuple(int(x) for x in torch_before.split("+")[0].split(".")[:2]) >= (2, 6), "transformers 5.x needs torch>=2.6"
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'''))
    cells.append(md("## 2 · Write the package to disk"))
    files = file_cells()
    dirs = sorted({str(Path(p).parent) for p, _ in files if str(Path(p).parent) != "."})
    cells.append(code("import os\n" + "\n".join(f'os.makedirs(f"{{CODE_DIR}}/{d}", exist_ok=True)' for d in dirs)
                      + '\nprint("directories ready")'))
    for rel, text in files:
        if not text.endswith("\n"):
            text += "\n"
        cells.append(code(f"%%writefile /kaggle/working/pa3_task1/{rel}\n{text}"))
    cells.append(md("## 3 · Environment"))
    cells.append(code('''import os, sys
os.chdir(CODE_DIR)
sys.path.insert(0, f"{CODE_DIR}/src")
from pa3 import orchestrate as O
info = O.setup_env(OUTPUT_ROOT)
if not info["questions_yaml"]:
    print("!! first_plot_questions.yaml not found under /kaggle/input — qualitative steps will be skipped")'''))
    cells.append(md("## 4 · Run the selected stage"))
    cells.append(code('''import time, json
t0 = time.time()
if STAGE == "S0":
    if CPU_TESTS == "quick":
        os.environ["PYTEST_ADDOPTS"] = "-k 'not ddp and not resume and not zero1'"
    summary = O.stage_s0(cpu_tests=CPU_TESTS != "none", smoke_updates=SMOKE_UPDATES)
elif STAGE == "S1":
    O.stage_s1()
elif STAGE == "S2":
    O.stage_s2()
elif STAGE == "S3":
    O.stage_s3()
else:
    raise ValueError(STAGE)
print(f"\\nStage {STAGE} finished in {(time.time() - t0) / 3600:.2f} h")'''))
    cells.append(md("## 5 · Results so far"))
    cells.append(code('''from pathlib import Path
from IPython.display import Markdown, Image, display
R = Path(OUTPUT_ROOT)
if (R / "smoke/summary.json").exists() and STAGE == "S0":
    s = json.loads((R / "smoke/summary.json").read_text())
    rows = ["| run | peak alloc GiB/rank | s/update | projected h | tok/s | skipped |", "|---|---|---|---|---|---|"]
    for k, v in s.get("smoke_runs", {}).items():
        rows.append(f"| {k} | {v.get('peak_alloc_gib_per_rank')} | {v.get('sec_per_update_steady')} | "
                    f"{v.get('projected_train_hours')} | {v.get('tokens_per_second')} | {v.get('skipped_updates')} |")
    display(Markdown("\\n".join(rows)))
    for key in ("fp16_sanity", "grad_equivalence_gpu", "merge_check_smoke", "eval_smoke", "data_report", "cpu_tests_rc"):
        print(key, "=>", json.dumps(s.get(key), indent=1, default=str)[:1500])
for t in sorted((R / "report_assets/tables").glob("*.md")) if (R / "report_assets/tables").exists() else []:
    display(Markdown(f"**{t.name}**\\n\\n" + t.read_text()))
for f in sorted((R / "report_assets/figures").glob("*.png")) if (R / "report_assets/figures").exists() else []:
    display(Image(filename=str(f)))
O.disk_report()'''))
    cells.append(md("""## 6 · Next
Save Version → Output → **New Dataset** → attach it in the next session and set `STAGE` to the next stage.
Bring `report_assets/`, `eval/*/aggregate.json`, `qualitative/annotation_sheet.csv` and `diagnostics/` into the report."""))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"},
                                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                                                    "isGpuEnabled": True}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    nb = build()
    out = REPO / "notebooks" / "pa3_task1_kaggle.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {out} ({len(nb['cells'])} cells)")
