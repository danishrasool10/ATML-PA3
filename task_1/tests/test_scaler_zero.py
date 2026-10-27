"""GradScaler interplay on CPU (gloo, 2 ranks): unscale_ -> clip -> step works with plain AdamW and
ZeRO-1, scaled updates equal unscaled ones, and an inf gradient skips the step on *both* ranks
(scale backs off, params unchanged)."""
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _worker(rank, port, use_zero, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from pa3.train.optim import build_optimizer

    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 1))
    ref = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 1))
    ref.load_state_dict(net.state_dict())
    ddp = torch.nn.parallel.DistributedDataParallel(net)
    opt = build_optimizer(net, 1e-2, (0.9, 0.999), 1e-8, 0.01, zero1=use_zero)
    ropt = build_optimizer(ref, 1e-2, (0.9, 0.999), 1e-8, 0.01, zero1=False)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0, enabled=True)
    torch.manual_seed(1)
    x = torch.randn(4, 8)
    out = {}
    for step in range(3):
        xin = x + rank
        if step == 1 and rank == 0:
            xin = xin * float("inf")  # overflow on ONE rank before the all-reduce
        loss = ddp(xin).pow(2).mean() * 50  # large grads so clipping is active
        scaler.scale(loss).backward()
        rl = sum(ref(x + r).pow(2).mean() * 50 for r in range(2)) / 2
        rl.backward()
        scaler.unscale_(opt)
        if step != 1:
            for p, rp in zip(net.parameters(), ref.parameters()):
                assert torch.allclose(p.grad, rp.grad, atol=1e-6), (step, (p.grad - rp.grad).abs().max())
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        before = [p.detach().clone() for p in net.parameters()]
        s0 = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < s0
        if step == 1:
            out["skip_rank"] = skipped
            out["unchanged"] = all(torch.equal(a, p) for a, p in zip(before, net.parameters()))
            ropt.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
            ropt.step()
            ropt.zero_grad()
        opt.zero_grad()
    out["max_param_diff"] = max(float((p - rp).abs().max()) for p, rp in zip(net.parameters(), ref.parameters()))
    q.put((rank, out))
    dist.destroy_process_group()


def _run(use_zero):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    ps = [ctx.Process(target=_worker, args=(r, port, use_zero, q)) for r in range(2)]
    for p in ps:
        p.start()
    res = dict(q.get(timeout=300) for _ in ps)
    for p in ps:
        p.join(timeout=60)
        assert p.exitcode == 0
    return res


def test_scaler_with_adamw_and_zero1():
    for use_zero in (False, True):
        res = _run(use_zero)
        # rank 0 saw inf, rank 1 did not; with ZeRO the found-inf must still skip on rank 0, and the
        # optimizer state stays consistent so both ranks end equal to the reference that skipped step 1
        for r in (0, 1):  # the inf propagated through the all-reduce: both ranks skip only step 1
            assert res[r]["skip_rank"] and res[r]["unchanged"], (use_zero, res)
            assert res[r]["max_param_diff"] < 1e-5, (use_zero, res)
