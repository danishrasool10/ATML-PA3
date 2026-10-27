"""Activation-probe set: 32 GSM8K-test + 32 HellaSwag-validation examples, seed 6304.

Sampled *before* any model result is inspected and saved once (never re-sampled).
IDs are lm-eval `doc_id`s, i.e. 0-based indices into the task's evaluation split
(gsm8k_cot test: 1,319 docs; hellaswag validation: 10,042 docs).
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np

from ..utils.common import output_root, write_json

N_GSM, N_HS = 1319, 10042


def sample_probe_ids(seed: int = 6304, n_each: int = 32) -> dict:
    rng = np.random.default_rng(seed)
    gsm = rng.choice(N_GSM, size=n_each, replace=False).tolist()
    hs = rng.choice(N_HS, size=n_each, replace=False).tolist()
    return {
        "seed": seed,
        "method": "numpy.random.default_rng(seed); gsm = rng.choice(1319, 32, replace=False); "
                  "hs = rng.choice(10042, 32, replace=False) (same generator, in that order)",
        "gsm8k_cot_test_doc_ids": [int(i) for i in gsm],
        "hellaswag_validation_doc_ids": [int(i) for i in hs],
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else output_root() / "data" / "probe_ids.json"
    if out.exists():
        print(f"probe set already exists at {out}; not re-sampling")
        return
    write_json(out, sample_probe_ids())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
