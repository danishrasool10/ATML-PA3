"""Step 4 live-generation worker (runs in the background on GPU0, pinned to its own CPU cores).

Per question, five independently seeded calls in the fixed order 1, 2, 3, 4, 0:
  slots 1-3  parallel responses (cap 1,024); slot 1 is also the shared initial response for refinement
  slot 4     critique-and-revision (cap 2,048): system, user, assistant = slot-1 response, user = review instruction
  slot 0     single longer response (cap 3,072)
Seed of a call = SeedSequence([6304, live question index, slot]).  Every call is appended to responses.jsonl
(prompt, response, seed, token counts, truncation flag, time, peak memory); a rerun skips finished calls, so the
worker resumes after a pause.  Exit code 3 = paused because the session budget is nearly used.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from ..answers.extract import last_boxed
from ..config import load_config, odir
from ..io import append_jsonl, pin_cpus, read_json, read_jsonl, write_json

SLOT_ORDER = (1, 2, 3, 4, 0)


def call_seed(seed: int, live_idx: int, slot: int) -> int:
    return int(np.random.SeedSequence([seed, live_idx, slot]).generate_state(1)[0])


def messages_initial(problem: str, lc: dict) -> list[dict]:
    return [{"role": "system", "content": lc["system"]},
            {"role": "user", "content": f"{problem}\n\n{lc['instruction']}"}]


def messages_review(problem: str, initial_response: str, lc: dict) -> list[dict]:
    return messages_initial(problem, lc) + [{"role": "assistant", "content": initial_response},
                                            {"role": "user", "content": lc["review"]}]


def cap_of(slot: int, lc: dict) -> int:
    return lc["caps"]["single"] if slot == 0 else lc["caps"]["revision"] if slot == 4 else lc["caps"]["parallel"]


def make_backend(cfg: dict, questions: list[dict]):
    lc = cfg["live"]
    if os.environ.get("PA3_LIVE_BACKEND") == "fake":
        from .backend import FakeBackend

        return FakeBackend(lc, {q["problem"]: q["gold"] for q in questions})
    from .backend import HFBackend

    return HFBackend(lc["model"], lc["revision"], lc)


def main() -> int:
    cfg = load_config()
    lc = cfg["live"]
    cores = pin_cpus()
    out = odir("live")
    qs = read_json(out / "questions.json")
    path = out / "responses.jsonl"
    rows = read_jsonl(path)
    have = {(r["qid"], r["slot"]): r for r in rows}
    if len(have) == len(qs) * len(SLOT_ORDER):
        print("[live] all calls already done", flush=True)
        return 0
    deadline = float(os.environ.get("PA3_DEADLINE_UNIX", "inf"))
    margin = 60 * float(lc.get("stop_margin_min", 20))
    t_load = time.time()
    backend = make_backend(cfg, qs)
    backend.warmup()
    write_json(out / "backend.json", {**backend.info, "cpu_cores": cores, "load_and_warmup_seconds": time.time() - t_load})
    print(f"[live] backend ready ({time.time() - t_load:.0f}s), cores={cores}", flush=True)
    for item in qs:
        for slot in SLOT_ORDER:
            if (item["qid"], slot) in have:
                continue
            if time.time() > deadline - margin:
                write_json(out / "status.json", {"state": "paused", "done_calls": len(have)})
                print("[live] session budget nearly used -> paused", flush=True)
                return 3
            if slot == 4:
                msgs = messages_review(item["problem"], have[(item["qid"], 1)]["response"], lc)
            else:
                msgs = messages_initial(item["problem"], lc)
            seed = call_seed(cfg["seed"], item["live_idx"], slot)
            res = backend.generate(msgs, cap_of(slot, lc), seed)
            row = {"qid": item["qid"], "live_idx": item["live_idx"], "slot": slot, "seed": seed,
                   "cap": cap_of(slot, lc), "messages": msgs, **res, "extracted": last_boxed(res["response"])}
            append_jsonl(path, row)
            have[(item["qid"], slot)] = row
        done_q = sum(all((q["qid"], s) in have for s in SLOT_ORDER) for q in qs)
        print(f"[live] {done_q}/{len(qs)} questions done", flush=True)
    write_json(out / "status.json", {"state": "done", "done_calls": len(have)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
