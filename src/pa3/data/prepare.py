"""Select the first 40,000 eligible MetaMathQA examples (source order) and pre-tokenise them.

Eligibility (spec): non-empty query and response; formatted length <= 2048 tokens; neither
`query` nor `original_question` equals a GSM8K *test* question after normalisation.
Formatted length = len(prompt_ids) + len(response_ids) + 1 (<|im_end|>), i.e. the exact
sequence fed to the model.  Nothing is truncated or shuffled.

Outputs (data_dir):
    tokens.npy        int32  concatenated token ids of all retained examples
    offsets.npy       int64  (n+1,) start offsets into tokens
    prompt_len.npy    int32  (n,)  number of masked prompt tokens
    resp_len.npy      int32  (n,)  supervised tokens (response + <|im_end|>)
    src_index.npy     int64  (n,)  row index in the MetaMathQA train split
    report.json       filter counts, decontamination matches, hashes, tokenizer info
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from ..utils.common import write_json
from .formatting import encode_batch
from .normalize import normalize_text


def _chunks(it: Iterable[Any], n: int) -> Iterator[list[Any]]:
    buf: list[Any] = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def select_and_tokenize(
    tokenizer: Any,
    rows: Iterable[dict],
    gsm8k_test_questions: Iterable[str],
    n_examples: int = 40_000,
    max_len: int = 2048,
    chunk: int = 2048,
) -> tuple[dict[str, np.ndarray], dict]:
    gsm_norm = {normalize_text(q) for q in gsm8k_test_questions}
    drops: list[tuple[int, str]] = []  # (source index, reason)
    n_retained = 0
    matches: list[dict] = []
    toks: list[np.ndarray] = []
    plen: list[int] = []
    rlen: list[int] = []
    src: list[int] = []
    longest = 0

    def indexed():
        for i, r in enumerate(rows):
            yield i, r

    done = False
    for block in _chunks(indexed(), chunk):
        cand = []
        for i, r in block:
            q, a = r.get("query"), r.get("response")
            if q is None or a is None or not str(q).strip() or not str(a).strip():
                drops.append((i, "empty"))
                continue
            nq, no = normalize_text(q), normalize_text(r.get("original_question"))
            if nq in gsm_norm or (no and no in gsm_norm):
                drops.append((i, "gsm8k_test_match"))
                if len(matches) < 50:
                    matches.append({"src_index": i, "field": "query" if nq in gsm_norm else "original_question",
                                    "query": q[:200]})
                continue
            cand.append((i, q, a))
        if not cand:
            continue
        P, R = encode_batch(tokenizer, [c[1] for c in cand], [c[2] for c in cand])
        for (i, _, _), p, rr in zip(cand, P, R):
            L = len(p) + len(rr)
            if L > max_len:
                drops.append((i, "too_long"))
                continue
            toks.append(np.asarray(p + rr, dtype=np.int32))
            plen.append(len(p))
            rlen.append(len(rr))
            src.append(i)
            longest = max(longest, L)
            n_retained += 1
            if n_retained == n_examples:
                done = True
                break
        if done:
            break
    if n_retained < n_examples:
        raise RuntimeError(f"only {n_retained} eligible examples (< {n_examples})")
    last = int(src[-1])
    counts = {"rows_examined": last + 1, "retained": n_retained, "last_src_index": last}
    for reason in ("empty", "too_long", "gsm8k_test_match"):
        counts[reason] = sum(1 for i, r in drops if r == reason and i <= last)
    assert counts["rows_examined"] == n_retained + counts["empty"] + counts["too_long"] + counts["gsm8k_test_match"]

    lengths = np.array([len(t) for t in toks], dtype=np.int64)
    offsets = np.zeros(len(toks) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    arrays = {
        "tokens": np.concatenate(toks).astype(np.int32),
        "offsets": offsets,
        "prompt_len": np.asarray(plen, dtype=np.int32),
        "resp_len": np.asarray(rlen, dtype=np.int32),
        "src_index": np.asarray(src, dtype=np.int64),
    }
    sha = hashlib.sha256()
    for k in ("tokens", "offsets", "prompt_len", "resp_len", "src_index"):
        sha.update(arrays[k].tobytes())
    report = {
        "counts": counts,
        "decontam_examples": matches,
        "n_examples": n_examples,
        "max_len": max_len,
        "total_tokens": int(lengths.sum()),
        "supervised_tokens": int(arrays["resp_len"].sum()),
        "mean_len": float(lengths.mean()),
        "max_len_retained": int(longest),
        "p50_len": float(np.percentile(lengths, 50)),
        "p99_len": float(np.percentile(lengths, 99)),
        "sha256": sha.hexdigest(),
    }
    return arrays, report


def save_arrays(arrays: dict[str, np.ndarray], report: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(out / f"{k}.npy", v)
    write_json(out / "report.json", report)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out_dir", default="outputs/data")
    ap.add_argument("--n_examples", type=int, default=40_000)
    ap.add_argument("--max_len", type=int, default=2048)
    a = ap.parse_args(argv)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model_id)
    mm = load_dataset("meta-math/MetaMathQA", split="train")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    assert len(gsm) == 1319, len(gsm)
    arrays, report = select_and_tokenize(tok, mm, gsm["question"], a.n_examples, a.max_len)
    report.update({
        "model_id": a.model_id,
        "metamath_rows": len(mm),
        "metamath_fingerprint": getattr(mm, "_fingerprint", None),
        "gsm8k_test_rows": len(gsm),
        "seconds": time.time() - t0,
        "example_0_text": tok.decode(arrays["tokens"][: arrays["offsets"][1]]),
    })
    save_arrays(arrays, report, a.out_dir)
    print({k: report[k] for k in ("counts", "total_tokens", "supervised_tokens", "mean_len", "sha256")})


if __name__ == "__main__":
    main()
