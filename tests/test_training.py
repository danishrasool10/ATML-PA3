"""End-to-end training-loop checks on CPU (gloo):

1. DDP 2 ranks x accumulation 8  ==  1 process x accumulation 16  (synced, unscaled grads), for
   every method, at update 0 and after 2 optimizer updates (trajectory equality).
2. Both equal the *direct* full-batch gradient of mean CE over all supervised tokens of the
   32-example effective batch (loss normalisation check).
3. ZeRO-1 gives the same trajectory as plain AdamW.
4. Kill-and-resume reproduces an uninterrupted run exactly (incl. LoRA dropout RNG).
"""
import json

import pytest
import torch

from conftest import run_train, write_cfg

METHODS = {
    "full": dict(method="full", lr=2e-5),
    "l2sp": dict(method="l2sp", lr=2e-5, l2sp_lambda=1e-3, l2sp_ref_device="cpu"),
    "lora": dict(method="lora", lr=2e-4, lora_r=16, lora_alpha=32, lora_dropout=0.0),
    "oplora": dict(method="oplora", lr=2e-4, lora_r=16, lora_alpha=32, lora_dropout=0.0, oplora_k=16),
}


def _rel_err(a: dict, b: dict) -> float:
    worst = 0.0
    assert a.keys() == b.keys()
    for k in a:
        scale = max(float(a[k].abs().max()), 1e-12)
        worst = max(worst, float((a[k] - b[k]).abs().max()) / scale)
    return worst


def _dump(out_root, tiny_dir, bases_rel, method, nproc, update, **extra):
    kw = dict(METHODS[method])
    if method == "oplora":
        kw["bases_path"] = bases_rel
    kw.update(extra)
    name = f"{method}_w{nproc}_u{update}" + "".join(f"_{k}" for k in extra)
    cfg = write_cfg(out_root, tiny_dir, run_name=name, **kw)
    path = out_root / f"grads_{name}.pt"
    run_train(out_root, cfg, nproc, f"grad_dump_update={update}", f"grad_dump_path={path}")
    return torch.load(path)


def _direct_grad(out_root, tiny_dir, bases_rel, method):
    from pa3.config import load_config
    from pa3.data.dataset import TokenStore, collate
    from pa3.models.load import load_tokenizer
    from pa3.train.loop import build_model
    from pa3.train.loss import CausalLMTokenLoss

    kw = dict(METHODS[method])
    if method == "oplora":
        kw["bases_path"] = bases_rel
    cfg = load_config(write_cfg(out_root, tiny_dir, run_name=f"direct_{method}", **kw))
    model, _ = build_model(cfg, torch.device("cpu"))
    store = TokenStore(out_root / "data")
    b = collate(store, list(range(32)), load_tokenizer(str(tiny_dir)).pad_token_id)
    s, n = CausalLMTokenLoss(model, 37)(**b)
    (s / n).backward()
    return {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("method", list(METHODS))
def test_ddp_equals_single_accumulation(out_root, tiny_dir, bases_rel, method):
    g1 = _dump(out_root, tiny_dir, bases_rel, method, 1, 0)
    g2 = _dump(out_root, tiny_dir, bases_rel, method, 2, 0)
    e0 = _rel_err(g1, g2)
    assert e0 < 1e-5, e0
    if method in ("full", "lora", "oplora"):
        gd = _direct_grad(out_root, tiny_dir, bases_rel, method)
        ed = _rel_err(gd, g2)
        assert ed < 1e-4, ed
    g1b = _dump(out_root, tiny_dir, bases_rel, method, 1, 2)
    g2b = _dump(out_root, tiny_dir, bases_rel, method, 2, 2)
    e2 = _rel_err(g1b, g2b)
    assert e2 < 1e-4, e2
    print(f"{method}: rel err u0={e0:.2e} u2={e2:.2e}")


def test_zero1_matches_adamw(out_root, tiny_dir, bases_rel):
    a = _dump(out_root, tiny_dir, bases_rel, "l2sp", 2, 3)
    b = _dump(out_root, tiny_dir, bases_rel, "l2sp", 2, 3, zero1=True)
    e = _rel_err(a, b)
    assert e < 1e-4, e


@pytest.mark.parametrize("method,extra", [("lora", {"lora_dropout": 0.05}), ("full", {})])
def test_resume_is_exact(out_root, tiny_dir, bases_rel, method, extra):
    from pa3.models.load import load_safetensors

    kw = dict(METHODS[method]) | extra | {"recovery_updates": [2, 4], "weight_snapshot_updates": [4],
                                          "drop_final_recovery": False}
    ca = write_cfg(out_root, tiny_dir, run_name=f"resumeA_{method}", **kw)
    cb = write_cfg(out_root, tiny_dir, run_name=f"resumeB_{method}", **kw)
    run_train(out_root, ca, 2)
    r = run_train(out_root, cb, 2, "crash_after_update=2")
    assert "simulated crash" in r.stdout
    run_train(out_root, cb, 2)
    fn = "adapter.safetensors" if method == "lora" else "model.safetensors"
    A = load_safetensors(out_root / f"runs/resumeA_{method}/weights_u0004/{fn}")
    B = load_safetensors(out_root / f"runs/resumeB_{method}/weights_u0004/{fn}")
    assert A.keys() == B.keys()
    for k in A:
        assert torch.equal(A[k], B[k]), k
    la = [json.loads(l) for l in (out_root / f"runs/resumeA_{method}/train_log.jsonl").read_text().splitlines()]
    lb = [json.loads(l) for l in (out_root / f"runs/resumeB_{method}/train_log.jsonl").read_text().splitlines()]
    assert [r["update"] for r in lb] == [1, 2, 3, 4]
    assert [r["sft_loss"] for r in la] == [r["sft_loss"] for r in lb]
    rec = sorted(p.name for p in (out_root / f"runs/resumeB_{method}/recovery").iterdir())
    assert rec == ["u0004"], rec  # rolling: only the newest recovery checkpoint is kept
    res = json.loads((out_root / f"runs/resumeB_{method}/resources.json").read_text())
    assert [c["update"] for c in res["checkpoints"]] == [2, 4]
