"""End-to-end CPU run of the 3-stage curriculum (tiny Qwen3, random features), kill-and-resume equality,
final evaluation under the four image conditions, and report assets."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

from conftest import ROOT, run_cfg, run_module


def _jl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


@pytest.fixture(scope="module")
def full_run(out_root, tiny_dir):
    cfg = run_cfg(out_root, tiny_dir, "full")
    run_module(out_root, "pa3vlm.train.loop", "--config", str(cfg))
    return out_root / "runs" / "full", cfg


def test_full_run_logs(full_run):
    run, _ = full_run
    tr = _jl(run / "train_log.jsonl")
    assert [r["update"] for r in tr] == list(range(1, 14))
    assert [r["stage"] for r in tr] == [1] * 6 + [2] * 4 + [3] * 3
    assert all(set(r["lr"]) == {"connector_w", "connector_b"} for r in tr[:6])
    assert all(set(r["lr"]) == {"connector_w", "connector_b", "lora"} for r in tr[6:])
    # per-stage warmup restarts: first update of each stage = peak / ceil(5% T) = peak (T small -> w = 1)
    assert tr[0]["lr"]["connector_w"] == pytest.approx(1e-3) and tr[6]["lr"]["lora"] == pytest.approx(2e-5)
    assert tr[5]["lr"]["connector_w"] == pytest.approx(1e-4) and tr[12]["lr"]["lora"] == pytest.approx(1e-6)
    assert all(sum(r["mix"].values()) == 16 for r in tr)
    assert all(set(r["mix"]) <= {"presence", "counting", "caption"} for r in tr[6:10])
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in tr)
    dv = _jl(run / "dev_metrics.jsonl")
    assert [r["update"] for r in dv] == [0, 5, 6, 10, 13]
    for k in ("caption_loss", "caption_loss_noimage", "grounding/all/delta_nll_swap", "vqa/all/nll_none",
              "grounding/presence/acc_greedy", "vqa/yes/no/nll_correct", "connector_norm_median"):
        assert k in dv[0], k
    starts = [e for e in _jl(run / "events.jsonl") if e["event"] == "stage_start"]
    assert starts[0]["lora_params"] == 0 and starts[0]["connector_params"] == 768 * 1024 + 1024 + 1024 * 64 + 64
    assert starts[1]["lora_params"] == 28 * 16 * ((64 + 64) + (64 + 32) + (64 + 32) + (64 + 64))
    for s in (1, 2, 3):
        assert (run / "weights" / f"stage{s}.pt").exists()
    cal = json.loads((run / "calibration.json").read_text())
    assert cal["scale"] > 0 and all(k.startswith("train2014:") for k in cal["image_keys"])
    assert len(list((run / "checkpoints").glob("resume_*.pt"))) == 2


def test_stage1_trains_connector_only(full_run):
    run, _ = full_run
    w1 = torch.load(run / "weights" / "stage1.pt", weights_only=False)
    w3 = torch.load(run / "weights" / "stage3.pt", weights_only=False)
    assert w1["lora"] == {} and len(w3["lora"]) == 2 * 4 * 28
    assert any(v.abs().sum() > 0 for k, v in w3["lora"].items() if k.endswith("lora_B"))


def test_resume_is_exact(full_run, out_root, tiny_dir):
    run, _ = full_run
    cfg = run_cfg(out_root, tiny_dir, "resumed", stop_after_update=7)
    r = run_module(out_root, "pa3vlm.train.loop", "--config", str(cfg), check=False)
    assert r.returncode == 3, r.stderr[-2000:]
    cfg = run_cfg(out_root, tiny_dir, "resumed", stop_after_update=None)
    run_module(out_root, "pa3vlm.train.loop", "--config", str(cfg))
    res = out_root / "runs" / "resumed"
    a, b = _jl(run / "train_log.jsonl"), _jl(res / "train_log.jsonl")
    assert [x["loss"] for x in a] == [x["loss"] for x in b]
    wa = torch.load(run / "weights" / "final.pt", weights_only=False)
    wb = torch.load(res / "weights" / "final.pt", weights_only=False)
    for part in ("connector", "lora"):
        for k in wa[part]:
            assert torch.equal(wa[part][k], wb[part][k]), (part, k)
    assert [r["update"] for r in _jl(res / "dev_metrics.jsonl")] == [0, 5, 6, 10, 13]


def test_final_eval_and_report(full_run, out_root):
    run, cfg = full_run
    run_module(out_root, "pa3vlm.eval.final", "--config", str(cfg))
    fe = run / "final_eval"
    for suite, n in (("grounding_test", 12), ("vqa_test", 16)):
        for cond in ("correct", "swap", "mismatched", "none"):
            recs = _jl(fe / f"{suite}__{cond}.jsonl")
            assert len(recs) == n
            if cond == "correct":
                assert all(r["image_used"] == r["own_image"] for r in recs)
            if cond == "swap":
                assert all(r["image_used"] == r["partner_image"] for r in recs)
            if cond == "none":
                assert all(r["image_used"] is None for r in recs)
            if cond == "mismatched":
                assert all(r["image_used"] not in (r["own_image"], r["partner_image"]) for r in recs)
    S = json.loads((fe / "summary.json").read_text())
    g = S["suites"]["grounding"]
    assert set(g) == {"all", "counting", "presence"} and g["all"]["n_pairs"] == 6
    lo, p, hi = g["all"]["conditions"]["correct"]["acc"][1], g["all"]["conditions"]["correct"]["acc"][0], \
        g["all"]["conditions"]["correct"]["acc"][2]
    assert lo <= p <= hi
    assert set(S["suites"]["vqa"]) == {"all", "yes/no", "number", "other"}
    assert "swap" in S["suites"]["vqa"]["all"]["advantages"]
    env = {**os.environ, "PA3_OUTPUT_ROOT": str(out_root)}
    os.makedirs(out_root / "runs" / "vt196", exist_ok=True)
    # report script expects run names vt196/vt49: link the test run as vt196
    import shutil
    shutil.copytree(run, out_root / "runs" / "vt196", dirs_exist_ok=True)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_report_assets.py")], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    A = out_root / "report_assets"
    assert (A / "figures" / "vt196_learning_curves.png").exists()
    assert (A / "tables" / "vt196_final_results.md").exists() and (A / "tables" / "resources.md").exists()
    q = json.loads((A / "tables" / "qualitative_examples.json").read_text())
    assert "rule" in q
