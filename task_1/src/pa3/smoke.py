"""GPU sanity checks used by the S0 smoke stage.

    python -m pa3.smoke fp16      # FP16 vs FP32 forward of the starting model on training examples
    python -m pa3.smoke compare A.pt B.pt   # relative error between two gradient dumps
    python -m pa3.smoke merge <run_name> <update>   # merged-vs-unmerged check on a smoke adapter
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from .utils.common import find_artifact, output_root, write_json


@torch.no_grad()
def fp16_sanity(model_id: str, n_examples: int = 8) -> dict:
    from .data.dataset import IGNORE, TokenStore, collate
    from .models.load import load_model, load_tokenizer

    dev = torch.device("cuda")
    tok = load_tokenizer(model_id)
    store = TokenStore(find_artifact("data"))
    lens = [int(store.offsets[i + 1] - store.offsets[i]) for i in range(len(store))]
    # include the longest example (worst case for activation growth) plus the first few
    idx = list(range(n_examples - 1)) + [int(max(range(len(lens)), key=lens.__getitem__))]
    out = {"examples": idx}
    res = {}
    acts = {}
    for tag, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
        m = load_model(model_id, dt).to(dev).eval()
        hooks = []
        amax: dict[int, float] = {}
        for li, layer in enumerate(m.model.layers):
            hooks.append(layer.register_forward_hook(
                lambda mod, i, o, li=li: amax.__setitem__(li, max(amax.get(li, 0.0), float(
                    (o[0] if isinstance(o, tuple) else o).float().abs().max())))))
        losses, logits_all = [], []
        for i in idx:
            b = {k: v.to(dev) for k, v in collate(store, [i], tok.pad_token_id).items()}
            lg = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.float()
            ce = torch.nn.functional.cross_entropy(lg[0, :-1], b["labels"][0, 1:], ignore_index=IGNORE)
            losses.append(float(ce))
            logits_all.append(lg[0, -64:].cpu())
        for h in hooks:
            h.remove()
        res[tag] = {"losses": losses, "logits": logits_all}
        acts[tag] = amax
        del m
        torch.cuda.empty_cache()
    diffs = [float((a - b).abs().max()) for a, b in zip(res["fp32"]["logits"], res["fp16"]["logits"])]
    out.update({
        "fp32_loss_mean": sum(res["fp32"]["losses"]) / len(idx),
        "fp16_loss_mean": sum(res["fp16"]["losses"]) / len(idx),
        "max_abs_loss_diff": max(abs(a - b) for a, b in zip(res["fp32"]["losses"], res["fp16"]["losses"])),
        "max_abs_logit_diff_last64": max(diffs),
        "fp16_nonfinite": any(not math.isfinite(x) for x in res["fp16"]["losses"]),
        "fp16_max_hidden_abs": max(acts["fp16"].values()),
        "fp32_max_hidden_abs": max(acts["fp32"].values()),
        "fp16_headroom_vs_65504": 65504.0 / max(1e-9, max(acts["fp32"].values())),
    })
    out["ok"] = (not out["fp16_nonfinite"]) and out["max_abs_loss_diff"] < 0.05
    return out


def compare_grads(a: str, b: str) -> dict:
    A, B = torch.load(a), torch.load(b)
    worst, num, den = 0.0, 0.0, 0.0
    for k in A:
        d = (A[k].double() - B[k].double())
        worst = max(worst, float(d.abs().max()) / max(float(A[k].abs().max()), 1e-12))
        num += float(d.square().sum())
        den += float(A[k].double().square().sum())
    return {"max_rel_err_per_tensor": worst, "global_rel_l2": math.sqrt(num / max(den, 1e-30)), "n_tensors": len(A)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fp16", "compare", "merge"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    a = ap.parse_args(argv)
    if a.cmd == "fp16":
        r = fp16_sanity(a.model_id)
        write_json(output_root() / "smoke" / "fp16_sanity.json", r)
    elif a.cmd == "compare":
        r = compare_grads(*a.args[:2])
        write_json(output_root() / "smoke" / "grad_equivalence_gpu.json", r)
    else:
        from .config import TARGET_MODULES
        from .eval.conditions import Condition
        from .eval.merge import merge_check
        from .models.load import load_tokenizer

        cond = Condition("smoke_adapter", "adapter", a.args[0], int(a.args[1]))
        r = merge_check(cond, a.model_id, load_tokenizer(a.model_id), torch.device("cuda"),
                        "bases/oplora_bases_k128.safetensors", TARGET_MODULES)
        write_json(output_root() / "smoke" / "merge_check_smoke.json", r)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
