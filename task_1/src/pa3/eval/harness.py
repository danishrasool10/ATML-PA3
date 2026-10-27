"""lm-evaluation-harness wrapper with fixed task settings, per-example dumps and GSM8K audit.

* GSM8K: task `gsm8k_cot`, num_fewshot=0, chat template with enable_thinking=False, greedy,
  max_gen_toks=1024, stock stop sequences and filters.  Primary metric: exact_match under the
  `flexible-extract` filter (unextractable -> "[invalid]" -> incorrect).
* HellaSwag / BoolQ / ARC-Easy: stock completion prompts, no chat template, zero-shot;
  acc and acc_norm (where defined) are both reported.
* Truncation is recorded exactly by wrapping HFLM._model_generate: a response is truncated
  iff the generation reached max_gen_toks without emitting a stop/pad token.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..utils.common import write_json

GSM_TASK = "gsm8k_cot"
LL_TASKS = ["hellaswag", "boolq", "arc_easy"]
EXPECTED_N = {"gsm8k_cot": 1319, "hellaswag": 10042, "boolq": 3270, "arc_easy": 2376}
PRIMARY = {
    "gsm8k_cot": ("exact_match", "flexible-extract"),
    "hellaswag": ("acc_norm", "none"),
    "boolq": ("acc", "none"),
    "arc_easy": ("acc_norm", "none"),
}


def _recording_hflm_cls():
    from lm_eval.models.huggingface import HFLM

    class RecordingHFLM(HFLM):
        """HFLM that records, per generated row, #new tokens and whether the cap was hit."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.gen_records: dict[tuple, dict] = {}

        def _model_generate(self, context, *args, **kwargs):
            out = super()._model_generate(context, *args, **kwargs)
            att = kwargs.get("attention_mask")
            max_length = kwargs.get("max_length")
            T = context.shape[1]
            max_new = (max_length - T) if max_length is not None else out.shape[1] - T
            stop_ids = {i for i in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                                    self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                                    self.tokenizer.convert_tokens_to_ids("<|endoftext|>")) if i is not None}
            for b in range(out.shape[0]):
                ctx = context[b][att[b].bool()].tolist() if att is not None else context[b].tolist()
                gen = out[b, T:].tolist()
                first_stop = next((i for i, t in enumerate(gen) if t in stop_ids), None)
                n_new = len(gen) if first_stop is None else first_stop + 1
                self.gen_records[tuple(ctx)] = {
                    "n_gen_tokens": n_new,
                    "hit_cap": first_stop is None and len(gen) >= max_new,
                    "max_new_tokens": max_new,
                }
            return out

    return RecordingHFLM


def _dump_samples(samples: list[dict], path: Path) -> None:
    with open(path, "w") as fh:
        for s in samples:
            fh.write(json.dumps(s, default=str, ensure_ascii=False) + "\n")


def gsm8k_audit(samples: list[dict], lm: Any) -> list[dict]:
    """One row per GSM8K doc: raw response, extracted answers per filter, correctness, truncation."""
    by_doc: dict[int, dict] = {}
    for s in samples:
        d = by_doc.setdefault(s["doc_id"], {
            "doc_id": s["doc_id"], "question": s["doc"]["question"], "target": s["target"],
            "response": s["resps"][0][0], "context": s["arguments"][0][0],
        })
        f = s["filter"]
        d[f"extracted_{f}"] = s["filtered_resps"][0]
        d[f"correct_{f}"] = float(s.get("exact_match", 0.0))
    rows = []
    for doc_id in sorted(by_doc):
        d = by_doc[doc_id]
        rec = getattr(lm, "gen_records", {}).get(tuple(lm.tok_encode(d["context"])), None)
        d["n_gen_tokens"] = rec["n_gen_tokens"] if rec else None
        d["truncated"] = bool(rec["hit_cap"]) if rec else None
        d["truncation_record_found"] = rec is not None
        d["flexible_unextractable"] = d.get("extracted_flexible-extract") == "[invalid]"
        d["strict_unextractable"] = d.get("extracted_strict-match") == "[invalid]"
        d["has_answer_phrase"] = "the answer is" in d["response"].lower()
        d.pop("context")
        rows.append(d)
    return rows


def summarise(results: dict) -> dict:
    out = {}
    for task, metrics in results["results"].items():
        out[task] = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and "alias" not in k}
    return out


def _part(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def run_lm_eval(model, tok, out_dir: str | Path, limit: int | None = None, gen_batch: int = 64,
                ll_batch: int = 32, max_gen_toks: int = 1024, tasks: list[str] | None = None) -> dict:
    """Runs GSM8K and the likelihood tasks as two cached parts (part_gsm8k.json / part_ll.json), so a
    crash or OOM retry only redoes the part that failed; aggregate.json is written when both exist."""
    from lm_eval import simple_evaluate

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = tasks or [GSM_TASK] + LL_TASKS
    cls = _recording_hflm_cls()
    model.eval()
    tok.padding_side = "left"
    if GSM_TASK in tasks and _part(out_dir / "part_gsm8k.json") is None:
        lm = cls(pretrained=model, tokenizer=tok, batch_size=gen_batch, enable_thinking=False)
        r = simple_evaluate(model=lm, tasks=[GSM_TASK], num_fewshot=0, apply_chat_template=True,
                           gen_kwargs={"max_gen_toks": max_gen_toks}, log_samples=True, limit=limit)
        samples = r["samples"][GSM_TASK]
        _dump_samples(samples, out_dir / f"samples_{GSM_TASK}.jsonl")
        audit = gsm8k_audit(samples, lm)
        _dump_samples(audit, out_dir / "gsm8k_audit.jsonl")
        n = len(audit)
        part = {"gsm8k_audit": {
            "n": n,
            "truncated_rate": sum(bool(a["truncated"]) for a in audit) / max(1, n),
            "flexible_unextractable_rate": sum(a["flexible_unextractable"] for a in audit) / max(1, n),
            "strict_unextractable_rate": sum(a["strict_unextractable"] for a in audit) / max(1, n),
            "answer_phrase_rate": sum(a["has_answer_phrase"] for a in audit) / max(1, n),
            "mean_gen_tokens": sum((a["n_gen_tokens"] or 0) for a in audit) / max(1, n),
            "truncation_records_missing": sum(not a["truncation_record_found"] for a in audit),
            "accuracy_given_not_truncated": (
                sum(a["correct_flexible-extract"] for a in audit if not a["truncated"]) /
                max(1, sum(not a["truncated"] for a in audit))),
        }, "results": summarise(r), "n_samples": r.get("n-samples", {}), "configs": r.get("configs", {}),
            "versions": r.get("versions", {}), "gen_batch": gen_batch}
        write_json(out_dir / "part_gsm8k.json", part)
        del lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ll = [t for t in tasks if t in LL_TASKS]
    if ll and _part(out_dir / "part_ll.json") is None:
        lm = cls(pretrained=model, tokenizer=tok, batch_size=ll_batch)
        r = simple_evaluate(model=lm, tasks=ll, num_fewshot=0, apply_chat_template=False,
                           log_samples=True, limit=limit)
        for t in ll:
            _dump_samples(r["samples"][t], out_dir / f"samples_{t}.jsonl")
        write_json(out_dir / "part_ll.json", {"results": summarise(r), "n_samples": r.get("n-samples", {}),
                                              "configs": r.get("configs", {}), "versions": r.get("versions", {}),
                                              "ll_batch": ll_batch})
        del lm
    agg: dict[str, Any] = {"settings": {"limit": limit, "max_gen_toks": max_gen_toks,
                                         "dtype": str(next(model.parameters()).dtype)},
                           "results": {}, "n_samples": {}, "configs": {}, "versions": {}}
    for name in ("part_gsm8k.json", "part_ll.json"):
        part = _part(out_dir / name)
        if part is None:
            continue
        for k in ("results", "n_samples", "configs", "versions"):
            agg[k].update(part.get(k, {}))
        if "gsm8k_audit" in part:
            agg["gsm8k_audit"] = part["gsm8k_audit"]
        agg["settings"][name.replace("part_", "").replace(".json", "") + "_batch"] = part.get("gen_batch", part.get("ll_batch"))
    agg["primary"] = {}
    for t in tasks:
        m, f = PRIMARY[t]
        res = agg["results"].get(t, {})
        agg["primary"][t] = {"metric": f"{m},{f}", "value": res.get(f"{m},{f}")}
        if t != GSM_TASK:
            agg["primary"][t]["acc_raw"] = res.get("acc,none")
            agg["primary"][t]["acc_norm"] = res.get("acc_norm,none")
        else:
            agg["primary"][t]["strict_match"] = res.get("exact_match,strict-match")
    if limit is None:
        for t in tasks:
            got = agg["n_samples"].get(t, {}).get("effective")
            agg.setdefault("split_check", {})[t] = {"expected": EXPECTED_N[t], "effective": got,
                                                    "ok": got == EXPECTED_N[t]}
    write_json(out_dir / "aggregate.json", agg)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return agg
