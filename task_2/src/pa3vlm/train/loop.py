"""Three-stage curriculum training of one VLM on one GPU (microbatch 8 x accumulation 2 = 16).

    python -m pa3vlm.train.loop --config vt196 [--set key=value ...] [--preflight N]

Stage 1 (caption alignment): connector only.  Stages 2-3: connector + LoRA (q/k/v/o, r16, alpha 32, dropout
0.05), attached with B = 0 at the stage boundary.  Fresh AdamW per stage (FP32 states), FP16 autocast with
dynamic loss scaling (unscale -> clip 1.0 -> step); a skipped (inf/nan) update is logged, AdamW's step count
does not advance, the schedule does.  Loss = mean over the 16 examples of each example's token-mean CE.

Exit codes: 0 finished, 3 paused (session budget / stop_after_update) with a resumable checkpoint saved.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch

from pa3common.lora import inject_lora, load_lora_state, lora_modules
from pa3common.utils import JsonlLogger, count_params, gpu_mem_stats, output_root, read_json, seed_everything, \
    set_rng_state, write_json

from ..config import load_config
from ..data.collate import collate
from ..data.sampler import CurriculumSampler, StageSpec
from ..eval.dev import amp_ctx, dev_eval
from ..models.build import build, device_dtype
from ..models.calibrate import calibrate
from ..models.vlm import per_example_mean
from .checkpoint import latest_resumable, save_resumable, save_weights
from .schedule import lr_at


def stage_bounds(cfg: dict) -> list[tuple[int, int]]:
    out, s = [], 0
    for st in cfg["stages"]:
        out.append((s, s + st["updates"]))
        s += st["updates"]
    return out


def dev_points(cfg: dict) -> list[int]:
    total = stage_bounds(cfg)[-1][1]
    pts = set(range(0, total + 1, cfg["dev_every"])) | {e for _, e in stage_bounds(cfg)}
    return sorted(pts)


def make_optimizer(vlm, cfg: dict, st: dict) -> torch.optim.Optimizer:
    c = vlm.connector
    groups = [
        {"name": "connector_w", "params": [c.fc1.weight, c.fc2.weight], "weight_decay": cfg["weight_decay"],
         "peak": st["lr_connector"]},
        {"name": "connector_b", "params": [c.fc1.bias, c.fc2.bias], "weight_decay": 0.0, "peak": st["lr_connector"]},
    ]
    if st["lr_lora"] is not None:
        lp = [p for m in lora_modules(vlm.lm).values() for p in (m.lora_A, m.lora_B)]
        assert lp, "LoRA must be attached before Stage 2"
        groups.append({"name": "lora", "params": lp, "weight_decay": cfg["weight_decay"], "peak": st["lr_lora"]})
    for g in groups:
        g["lr"] = 0.0
    trainable = {id(p) for p in vlm.parameters() if p.requires_grad}
    in_opt = {id(p) for g in groups for p in g["params"]}
    assert trainable == in_opt, "optimizer must cover exactly the trainable parameters"
    return torch.optim.AdamW(groups, betas=tuple(cfg["betas"]), eps=cfg["eps"])


def attach_lora(vlm, cfg: dict) -> None:
    lc = cfg["lora"]
    inject_lora(vlm.lm, lc["r"], lc["alpha"], lc["dropout"], cfg["seed"], tuple(lc["targets"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--preflight", type=int, default=0, help="run N updates of each stage type, save nothing")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    device, dtype = device_dtype(cfg)
    seed_everything(cfg["seed"])
    out = output_root() / "runs" / cfg["run_name"] if not a.preflight else output_root() / "preflight" / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    tok, fmt, bundle, store, vlm = build(cfg, device, dtype)
    n_vis = store.n_vis
    get_feats = store.get
    bounds = stage_bounds(cfg)
    total = bounds[-1][1]
    sampler = CurriculumSampler(cfg["seed"], bundle.n_captions, bundle.pool_sizes(), cfg["micro_bs"],
                                cfg["grad_accum"], cfg["caption_passes"])
    assert sampler.stage1_updates() == cfg["stages"][0]["updates"], "Stage 1 must be exactly the caption passes"
    eff = cfg["micro_bs"] * cfg["grad_accum"]
    if not a.preflight:
        write_json(out / "config.json", cfg)

    # ------------------------------------------------------------------ calibration (training data only)
    cal_path = out / "calibration.json"
    if cal_path.exists():
        vlm.connector.scale.fill_(read_json(cal_path)["scale"])
    else:
        t0 = time.time()
        rep = calibrate(vlm.connector, vlm.lm.get_input_embeddings().weight, bundle.stage1_token_counts(), get_feats,
                        [r["image"] for r in bundle.manifest("captions_train")], cfg["calib_images"], cfg["seed"],
                        device)
        rep["seconds"] = time.time() - t0
        rep["n_visual_tokens_per_image"] = n_vis
        write_json(cal_path, rep)
    print(f"[train] calibration scale s = {float(vlm.connector.scale):.6g}", flush=True)

    # ------------------------------------------------------------------ resume
    ck_dir = out / "checkpoints"
    ck = latest_resumable(ck_dir) if not a.preflight else None
    g = 0
    state = None
    if ck is not None:
        state = torch.load(ck, map_location="cpu", weights_only=False)
        g = state["update"]
        if state["lora"]:
            attach_lora(vlm, cfg)
            load_lora_state(vlm.lm, state["lora"])
        vlm.connector.load_state_dict(state["connector"])
        set_rng_state(state["rng"])
        print(f"[train] resumed from {ck.name} (update {g})", flush=True)
    vlm.to(device)
    train_log = JsonlLogger(out / "train_log.jsonl", not a.preflight, truncate_after=g)
    dev_log = JsonlLogger(out / "dev_metrics.jsonl", not a.preflight, truncate_after=g)
    ck_log = JsonlLogger(out / "checkpoints.jsonl", not a.preflight)
    ev_log = JsonlLogger(out / "events.jsonl", not a.preflight)
    ev_log.log({"update": g, "event": "start", "resumed_from": ck.name if ck else None, "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "setup_seconds": time.time() - t_start})
    scaler = torch.amp.GradScaler("cuda", init_scale=cfg["init_scale"], enabled=device.type == "cuda")
    if state is not None:
        scaler.load_state_dict(state["scaler"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dpts = set(dev_points(cfg)) if not a.preflight else set()
    done_dev = {r["update"] for r in _read_jsonl(out / "dev_metrics.jsonl")}
    deadline = float(os.environ.get("PA3_DEADLINE_UNIX", "inf"))
    margin = cfg["deadline_margin_min"] * 60
    stop_after = cfg.get("stop_after_update")
    timing = {"train_s": 0.0, "dev_s": 0.0, "ckpt_s": 0.0}

    def run_dev(update: int, stage: int) -> None:
        if update in done_dev or update not in dpts:
            return
        t0 = time.time()
        row = dev_eval(vlm, bundle, get_feats, n_vis, fmt, device, cfg["eval_bs"], cfg["max_new_tokens"], update, stage)
        row["seconds"] = time.time() - t0
        timing["dev_s"] += row["seconds"]
        dev_log.log(row)
        done_dev.add(update)
        print(f"[dev] u={update} cap={row['caption_loss']:.3f} "
              f"g_dNLL={row['grounding/all/delta_nll_swap']:+.3f} v_dNLL={row['vqa/all/delta_nll_swap']:+.3f} "
              f"g_acc={row['grounding/all/acc_greedy']:.3f} v_acc={row['vqa/all/acc_greedy']:.3f}", flush=True)

    run_dev(0, 0)
    vlm.train()
    preflight_rows = []
    for si, st in enumerate(cfg["stages"]):
        start, end = bounds[si]
        if a.preflight:
            start, end = g, g + a.preflight
        if g >= end:
            continue
        if st["lr_lora"] is not None and not lora_modules(vlm.lm):
            attach_lora(vlm, cfg)
            vlm.to(device)
            ev_log.log({"update": g, "event": "lora_attached", "n_lora_params": count_params(vlm, True) -
                        count_params(vlm.connector, True)})
        n_train = count_params(vlm, True)
        ev_log.log({"update": g, "event": "stage_start", "stage": si + 1, "name": st["name"],
                    "trainable_params": n_train, "connector_params": count_params(vlm.connector, True),
                    "lora_params": n_train - count_params(vlm.connector, True),
                    "frozen_params": count_params(vlm) - n_train})
        opt = make_optimizer(vlm, cfg, st)
        if state is not None and state["stage"] == si and state["update"] < end:
            opt.load_state_dict(state["optimizer"])
        state = None
        spec = StageSpec(si + 1, st["updates"], st["mix"])
        T = st["updates"]
        while g < end:
            u = g - start
            t0 = time.time()
            for grp in opt.param_groups:
                grp["lr"] = lr_at(min(u + 1, T), T, grp["peak"], cfg["warmup_frac"], cfg["final_lr_frac"])
            mbs, counts = sampler.update(spec, u)
            loss_sum, n_tok = 0.0, 0
            for mb in mbs:
                encs = [bundle.item(it) for it in mb]
                b = collate(encs, [e.image for e in encs], get_feats, n_vis, fmt.pad_id, "right", True).to(device)
                with amp_ctx(device):
                    nll, rows, _ = vlm.supervised_nll(b)
                ex = per_example_mean(nll, rows, len(encs))
                loss = ex.sum() / eff
                scaler.scale(loss).backward()
                loss_sum += float(ex.detach().sum())
                n_tok += int(rows.numel())
            scaler.unscale_(opt)
            params = [p for grp in opt.param_groups for p in grp["params"]]
            gn = float(torch.nn.utils.clip_grad_norm_(params, cfg["clip_norm"]))
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped = scaler.is_enabled() and scaler.get_scale() < scale_before
            opt.zero_grad(set_to_none=True)
            g += 1
            dt = time.time() - t0
            timing["train_s"] += dt
            row = {"update": g, "stage": si + 1, "loss": loss_sum / eff, "grad_norm": gn,
                   "grad_norm_finite": math.isfinite(gn), "skipped": bool(skipped), "loss_scale": scaler.get_scale(),
                   "lr": {grp["name"]: grp["lr"] for grp in opt.param_groups}, "mix": counts,
                   "supervised_tokens": n_tok, "seconds": dt}
            if device.type == "cuda":
                row["peak_mem_gib"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
            if a.preflight:
                preflight_rows.append(row)
                print(f"[preflight] stage {si + 1} update {g - start}: loss {row['loss']:.3f} {dt:.2f}s", flush=True)
                continue
            train_log.log(row)
            if g % 50 == 0 or g == end:
                print(f"[train] u={g} stage={si + 1} loss={row['loss']:.4f} gn={gn:.3f} "
                      f"lr={list(row['lr'].values())} {dt:.2f}s/upd skipped={skipped}", flush=True)
            just_saved = False
            if g % cfg["ckpt_every"] == 0 or g == end:
                t1 = time.time()
                save_resumable(ck_dir, g, si, vlm, opt, scaler, {"timing": timing}, cfg["keep_resumable"], ck_log)
                timing["ckpt_s"] += time.time() - t1
                just_saved = True
            if g == end:
                sz = save_weights(out / "weights" / f"stage{si + 1}.pt", vlm, {"update": g, "stage": si + 1})
                ck_log.log({"update": g, "event": "stage_weights", "stage": si + 1, "bytes": sz})
            run_dev(g, si + 1)
            pause = time.time() > deadline - margin or (stop_after is not None and g >= stop_after)
            if pause and g < total:
                if not just_saved:
                    save_resumable(ck_dir, g, si, vlm, opt, scaler, {"timing": timing}, cfg["keep_resumable"], ck_log)
                ev_log.log({"update": g, "event": "paused", "timing": timing, "memory": gpu_mem_stats(device)})
                write_json(out / "status.json", {"state": "paused", "update": g, "total": total})
                print(f"[train] paused at update {g} (resumable)", flush=True)
                return 3
    if a.preflight:
        write_json(out / "preflight.json", {"rows": preflight_rows, "memory": gpu_mem_stats(device),
                                            "s_per_update": {f"stage{r['stage']}": r["seconds"] for r in preflight_rows}})
        return 0
    save_weights(out / "weights" / "final.pt", vlm, {"update": g})
    res = {"update": g, "timing": timing, "memory": gpu_mem_stats(device), "wall_s_this_session": time.time() - t_start}
    ev_log.log({"update": g, "event": "finished", **res})
    write_json(out / "status.json", {"state": "done", "update": g, "total": total})
    print(f"[train] finished {g} updates", flush=True)
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    import json

    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


if __name__ == "__main__":
    sys.exit(main())
