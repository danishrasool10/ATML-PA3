"""Training entry point (launch with torchrun; also runs single-process on CPU for tests).

Per optimizer update k (0-based, 1,250 total):
    for each of this rank's grad_accum/W microbatches (DDP no_sync on all but the last):
        ce_sum, _ = loss_module(batch)              # sum of token CE over supervised tokens
        scaler.scale(ce_sum * W / N_k).backward()   # N_k = supervised tokens in all 32 examples
    scaler.unscale_(opt)                            # grads are now the exact batch-mean gradient
    [L2-SP] Omega, grad += lambda (theta - theta0)
    grad_norm = clip_grad_norm_(params, 1.0)        # total (SFT + penalty) gradient
    scaler.step(opt); scaler.update()               # skipped if inf/nan found (logged)
    sched.step()                                    # schedule follows the update budget index
"""
from __future__ import annotations

import contextlib
import json
import math
import sys
import time
import warnings
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import RunConfig, parse_cli
from ..data.dataset import TokenStore, collate, plan_longest, plan_update
from ..models.l2sp import L2SPReference
from ..models.load import load_model, load_tokenizer, save_safetensors, trainable_state_dict
from ..models.lora import export_factors, inject_adapters, save_adapter_meta
from ..models.oplora import load_bases, read_bases_meta
from ..utils.common import (
    JsonlLogger, all_gather_object, all_reduce_sum, barrier, cleanup_distributed, count_params,
    find_artifact, gpu_mem_stats, init_distributed, read_json, seed_everything, write_json,
)
from . import checkpoint as ck
from .loss import CausalLMTokenLoss
from .optim import build_optimizer, build_scheduler

warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before")


def build_model(cfg: RunConfig, device: torch.device) -> tuple[nn.Module, dict]:
    info: dict = {}
    use_fp16_backbone = cfg.is_adapter and device.type == "cuda" and cfg.amp
    dtype = torch.float16 if use_fp16_backbone else torch.float32
    model = load_model(cfg.model_id, dtype, cfg.attn_impl)
    info["backbone_dtype"] = str(dtype).replace("torch.", "")
    if cfg.is_adapter:
        bases = None
        if cfg.method == "oplora":
            bp = find_artifact(cfg.bases_path)
            if bp is None:
                raise FileNotFoundError(f"OPLoRA bases not found: {cfg.bases_path}")
            t0 = time.perf_counter()
            bases = load_bases(bp)
            info["bases_load_seconds"] = time.perf_counter() - t0
            info["bases_file"] = str(bp)
            info["bases_meta"] = read_bases_meta(bp)
            assert int(info["bases_meta"]["k"]) == cfg.oplora_k
        inject_adapters(model, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout, cfg.seed,
                        tuple(cfg.target_modules), bases)
        if bases is not None:
            info["basis_bytes_on_device"] = sum(
                b.numel() * b.element_size() for n, m in model.named_modules()
                for b in m.buffers(recurse=False) if hasattr(m, "projected"))
    model.to(device)
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    info["trainable_params"] = count_params(model, trainable_only=True)
    info["total_params_unique"] = count_params(model)
    return model, info


def _scaler(cfg: RunConfig, device: torch.device):
    enabled = cfg.amp and device.type == "cuda"
    return torch.amp.GradScaler(device.type, init_scale=cfg.init_scale, enabled=enabled)


def _autocast(cfg: RunConfig, device: torch.device):
    if cfg.amp and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def export_final(cfg: RunConfig, model: nn.Module, update: int, run_dir: Path) -> dict:
    """Write the evaluation/diagnostic snapshot for `update`; return timing/size info."""
    d = run_dir / f"weights_u{update:04d}"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if cfg.is_adapter:
        factors = export_factors(model)  # includes projected (effective) factors for OPLoRA
        t_proj = time.perf_counter() - t0
        n = save_safetensors(factors, d / "adapter.safetensors")
        save_adapter_meta(d / "adapter_meta.json", method=cfg.method, r=cfg.lora_r, alpha=cfg.lora_alpha,
                          scale=cfg.lora_scale, k=cfg.oplora_k if cfg.method == "oplora" else None,
                          targets=list(cfg.target_modules), update=update, inference_factors="A_eff/B_eff")
        eff_bytes = sum(v.numel() * 4 for k, v in factors.items() if k.endswith("_eff"))
        info = {"export_seconds": time.perf_counter() - t0, "projection_seconds": t_proj,
                "file_bytes": n, "effective_factor_bytes": eff_bytes}
    else:
        n = save_safetensors(trainable_state_dict(model), d / "model.safetensors",
                             metadata={"dtype": "float32", "update": update})
        info = {"export_seconds": time.perf_counter() - t0, "file_bytes": n}
    write_json(d / "export_info.json", info)
    return info


def train(cfg: RunConfig) -> dict:
    rank, local_rank, world, device = init_distributed()
    run_dir = cfg.run_dir
    is_main = rank == 0
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(run_dir / "config.json")
    barrier()
    if (run_dir / "DONE").exists() and cfg.max_updates is None:
        if is_main:
            print(f"[{cfg.run_name}] already DONE; skipping")
        cleanup_distributed()
        return read_json(run_dir / "resources.json")

    per_rank_accum = cfg.grad_accum // world
    assert per_rank_accum * world == cfg.grad_accum
    seed_everything(cfg.seed + rank)  # dropout streams differ per rank; adapter init is seeded per module

    data_dir = find_artifact(cfg.data_dir)
    if data_dir is None:
        raise FileNotFoundError(f"prepared data not found: {cfg.data_dir}")
    store = TokenStore(data_dir)
    assert len(store) >= cfg.n_examples == cfg.num_updates * cfg.effective_batch, (len(store), cfg.n_examples)
    tok = load_tokenizer(cfg.model_id)
    pad_id = tok.pad_token_id

    t_build = time.perf_counter()
    model, minfo = build_model(cfg, device)
    l2sp = L2SPReference(model, device, cfg.l2sp_ref_device) if cfg.method == "l2sp" else None
    minfo["build_seconds"] = time.perf_counter() - t_build
    loss_mod = CausalLMTokenLoss(model, cfg.ce_chunk)
    if world > 1:
        ddp = DDP(loss_mod, device_ids=[local_rank] if device.type == "cuda" else None,
                  gradient_as_bucket_view=True, broadcast_buffers=False)
    else:
        ddp = loss_mod
    params = [p for p in model.parameters() if p.requires_grad]
    opt = build_optimizer(model, cfg.lr, cfg.betas, cfg.eps, cfg.weight_decay, cfg.zero1 and world > 1)
    sched = build_scheduler(opt, cfg.warmup_updates, cfg.num_updates)
    scaler = _scaler(cfg, device)

    # ---------------------------------------------------------------- resume
    start = 0
    counters = {"train_seconds": 0.0, "ckpt_seconds": 0.0, "skipped_updates": [], "optimizer_steps_applied": 0,
                "tokens_seen": 0}
    comp = ck.list_complete(run_dir) if cfg.max_updates is None else []
    if comp:
        st = ck.load_recovery(comp[-1], model, opt, sched, scaler, rank, device)
        start = int(st["update"])
        counters.update(st.get("counters", {}))
        if is_main:
            print(f"[{cfg.run_name}] resumed from {comp[-1].name} (update {start})")
    log = JsonlLogger(run_dir / ("smoke_log.jsonl" if cfg.max_updates else "train_log.jsonl"), is_main,
                      truncate_after=start)
    ck_log = JsonlLogger(run_dir / "checkpoints.jsonl", is_main and cfg.max_updates is None, truncate_after=start)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    end = cfg.num_updates if cfg.max_updates is None else min(cfg.num_updates, start + cfg.max_updates)
    autocast = _autocast(cfg, device)
    for k in range(start, end):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        if cfg.max_updates is not None and cfg.extra.get("smoke_longest") and k == start:
            plan = plan_longest(store, rank, world, cfg.micro_bs, cfg.grad_accum)  # worst-case memory first
        else:
            plan = plan_update(store, k, rank, world, cfg.micro_bs, cfg.grad_accum)
        lr_used = sched.get_last_lr()[0]
        ce_local = torch.zeros((), dtype=torch.float64, device=device)
        tok_local = 0
        for m, idx in enumerate(plan.microbatches):
            batch = {kk: v.to(device, non_blocking=True) for kk, v in collate(store, idx, pad_id).items()}
            tok_local += int(batch["attention_mask"].sum())
            sync_ctx = ddp.no_sync() if (world > 1 and m < len(plan.microbatches) - 1) else contextlib.nullcontext()
            with sync_ctx:
                with autocast:
                    ce_sum, _ = ddp(**batch)
                loss = ce_sum * (world / plan.n_tokens)
                scaler.scale(loss).backward()
            ce_local += ce_sum.detach().double()
        sft_loss = float(all_reduce_sum(ce_local)) / plan.n_tokens
        scale_before = scaler.get_scale() if scaler.is_enabled() else 1.0
        scaler.unscale_(opt)
        omega = None
        if l2sp is not None:
            omega = l2sp.omega_and_grad(model, cfg.l2sp_lambda, add_grad=True)
        if cfg.grad_dump_update is not None and k == cfg.grad_dump_update:
            if is_main:
                torch.save({n: p.grad.detach().float().cpu().clone() for n, p in model.named_parameters()
                            if p.requires_grad}, cfg.grad_dump_path)
                print(f"[grad-dump] update {k} -> {cfg.grad_dump_path}")
            cleanup_distributed()
            return {}
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.clip_norm)
        scaler.step(opt)
        scaler.update()
        scale_after = scaler.get_scale() if scaler.is_enabled() else 1.0
        skipped = bool(scale_after < scale_before) or not math.isfinite(float(grad_norm))
        sched.step()  # budget index advances even when the optimizer step was skipped
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        counters["train_seconds"] += dt
        tok_all = int(all_reduce_sum(torch.tensor(tok_local, device=device, dtype=torch.long)))
        counters["tokens_seen"] += tok_all
        if skipped:
            counters["skipped_updates"].append(k + 1)
        else:
            counters["optimizer_steps_applied"] += 1
        row = {"update": k + 1, "examples_seen": (k + 1) * cfg.effective_batch, "sft_loss": sft_loss,
               "lr": lr_used, "grad_norm": float(grad_norm), "loss_scale": scale_before, "skipped": skipped,
               "n_sup_tokens": plan.n_tokens, "n_tokens": tok_all, "sec": dt}
        if l2sp is not None:
            row.update({"omega": omega, "lambda_omega": cfg.l2sp_lambda * omega,
                        "total_loss": sft_loss + cfg.l2sp_lambda * omega,
                        "displacement_l2": math.sqrt(2.0 * omega)})
        if device.type == "cuda":
            row["mem_alloc_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        if is_main and ((k + 1) % cfg.log_every == 0 or k == start):
            log.log(row)
            if (k + 1) % 25 == 0 or k == start or k + 1 == end:
                print(f"[{cfg.run_name}] u{k + 1}/{cfg.num_updates} loss={sft_loss:.4f} lr={lr_used:.2e} "
                      f"gn={float(grad_norm):.3f} scale={scale_before:.0f} skip={skipped} {dt:.2f}s"
                      + (f" Omega={omega:.4e}" if omega is not None else ""), flush=True)

        # ------------------------------------------------------------ snapshots + recovery
        u = k + 1
        if cfg.max_updates is None:
            snap_ref = None
            if u in cfg.weight_snapshot_updates:
                tt = time.perf_counter()
                if is_main:
                    info = export_final(cfg, model, u, run_dir)
                    counters.setdefault("exports", {})[str(u)] = info
                barrier()
                counters["ckpt_seconds"] += time.perf_counter() - tt
                if not cfg.is_adapter:
                    snap_ref = run_dir / f"weights_u{u:04d}" / "model.safetensors"
            if u in cfg.recovery_updates:
                tt = time.perf_counter()
                state = {"update": u, "counters": counters}
                d = ck.save_recovery(run_dir, u, model, opt, sched, scaler, state, rank, snap_ref)
                size = ck.recovery_size(d) if is_main else 0
                removed = ck.rotate(run_dir, cfg.keep_last_recovery, rank)
                counters["ckpt_seconds"] += time.perf_counter() - tt
                if is_main:
                    ck_log.log({"update": u, "examples": u * cfg.effective_batch, "bytes": size, "dir": d.name,
                                "seconds": time.perf_counter() - tt, "removed": removed})
                    print(f"[{cfg.run_name}] recovery u{u} saved ({size / 2**30:.2f} GiB); removed {removed}")
            if cfg.crash_after_update is not None and u == cfg.crash_after_update:
                if is_main:
                    print(f"[{cfg.run_name}] simulated crash after update {u}")
                cleanup_distributed()
                return {"crashed_at": u}

    if cfg.max_updates is not None and cfg.extra.get("smoke_export") and is_main:
        counters.setdefault("exports", {})[str(end)] = export_final(cfg, model, end, run_dir)
    barrier()

    # ---------------------------------------------------------------- resources
    mem = all_gather_object(gpu_mem_stats(device))
    res = {
        "run_name": cfg.run_name, "method": cfg.method, "world_size": world, "per_rank_accum": per_rank_accum,
        "updates_completed": end, "examples_seen": end * cfg.effective_batch,
        "train_seconds": counters["train_seconds"], "ckpt_seconds": counters["ckpt_seconds"],
        "tokens_seen": counters["tokens_seen"],
        "tokens_per_second": counters["tokens_seen"] / max(1e-9, counters["train_seconds"]),
        "skipped_updates": counters["skipped_updates"],
        "optimizer_steps_applied": counters["optimizer_steps_applied"],
        "peak_memory_per_rank": mem,
        "checkpoints": ([json.loads(x) for x in (run_dir / "checkpoints.jsonl").read_text().splitlines() if x.strip()]
                        if (run_dir / "checkpoints.jsonl").exists() else []),
        "exports": counters.get("exports", {}), **minfo,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "zero1": bool(cfg.zero1 and world > 1), "torch": torch.__version__,
    }
    if l2sp is not None:
        res["l2sp"] = l2sp.stats()
    if is_main:
        write_json(run_dir / ("smoke_resources.json" if cfg.max_updates else "resources.json"), res)
    if cfg.max_updates is None and end == cfg.num_updates:
        if cfg.drop_final_recovery and is_main:
            for d in ck.list_complete(run_dir):
                res["checkpoints_deleted_after_completion"] = res.get("checkpoints_deleted_after_completion", []) + [
                    {"dir": d.name, "bytes": ck.recovery_size(d)}]
                # keep only the optimizer-free part: drop the whole recovery dir (weights live in snapshots)
                from ..utils.common import rmtree
                rmtree(d)
            write_json(run_dir / "resources.json", res)
        if is_main:
            (run_dir / "DONE").write_text("ok")
    barrier()
    cleanup_distributed()
    return res


def main(argv: list[str] | None = None) -> None:
    cfg = parse_cli(argv if argv is not None else sys.argv[1:])
    train(cfg)


if __name__ == "__main__":
    main()
