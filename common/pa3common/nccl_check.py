"""Quick multi-GPU sanity check (launch with torchrun): NCCL all-reduce correctness + bandwidth."""
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    x = torch.full((64 * 2**20,), float(rank + 1), device="cuda")  # 256 MB fp32
    dist.all_reduce(x)
    torch.cuda.synchronize()
    expect = world * (world + 1) / 2
    ok = bool(torch.all(x == expect))
    t0 = time.time()
    for _ in range(3):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    gbps = 3 * x.numel() * 4 / (time.time() - t0) / 1e9
    if rank == 0:
        print(f"[nccl-check] world={world} correct={ok} allreduce_GBps={gbps:.2f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if not ok:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
