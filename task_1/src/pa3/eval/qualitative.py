"""Qualitative generation for first_plot_questions.yaml and the 42-row annotation sheet.

YAML structure (Betley et al., emergent misalignment): a list of entries with `id`,
`paraphrases` (list; we use paraphrase 0), optional `system`.  Format is inferred from the id
suffix: '<base>' -> plain, '<base>_json' -> json, '<base>_template' -> template.
Each condition gets one greedy response per prompt (thinking disabled, 512 new tokens).
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import torch
import yaml

from ..data.formatting import render_prompt

FORMATS = ("plain", "json", "template")


def load_questions(path: str | Path) -> list[dict]:
    raw = yaml.safe_load(Path(path).read_text())
    if isinstance(raw, dict):  # tolerate {questions: [...]}
        raw = raw.get("questions", list(raw.values()))
    qs = []
    for e in raw:
        qid = str(e["id"])
        if qid.endswith("_json"):
            base, fmt = qid[: -len("_json")], "json"
        elif qid.endswith("_template"):
            base, fmt = qid[: -len("_template")], "template"
        else:
            base, fmt = qid, "plain"
        paras = e.get("paraphrases") or [e.get("question") or e.get("prompt")]
        qs.append({"id": qid, "base_id": base, "format": fmt, "system": e.get("system"),
                   "prompt": str(paras[0]), "n_paraphrases": len(paras), "file_order": len(qs)})
    bases_plain = [q["base_id"] for q in qs if q["format"] == "plain"]
    for q in qs:
        q["plain_order"] = bases_plain.index(q["base_id"]) if q["base_id"] in bases_plain else None
    counts = Counter(q["format"] for q in qs)
    if len(qs) != 24 or any(counts[f] != 8 for f in FORMATS):
        print(f"[qualitative] WARNING: expected 8x3=24 prompts, got {len(qs)} {dict(counts)}")
    return qs


@torch.no_grad()
def generate_responses(model, tok, questions: list[dict], condition: str, max_new_tokens: int = 512,
                       device: torch.device | None = None) -> list[dict]:
    device = device or next(model.parameters()).device
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    rows = []
    for q in questions:
        text = render_prompt(tok, q["prompt"], q["system"])
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                             top_p=None, top_k=None, eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
        gen = out[0, enc["input_ids"].shape[1]:].tolist()
        ended = any(t in eos_ids for t in gen)
        n = (next(i for i, t in enumerate(gen) if t in eos_ids) + 1) if ended else len(gen)
        rows.append({"condition": condition, "question_id": q["id"], "base_id": q["base_id"],
                     "format": q["format"], "system": q["system"], "prompt": q["prompt"],
                     "response": tok.decode(gen[:n], skip_special_tokens=True),
                     "n_new_tokens": n, "truncated": (not ended) and len(gen) >= max_new_tokens})
    return rows


# ------------------------------------------------------------------ automatic flags
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)
_MATH = re.compile(r"(\\boxed|\\frac|\\times|\$|\b\d+\s*[-+*/x×=]\s*\d+|\bequation\b|\bsolve\b|"
                   r"\bcalculate\b|\bstep \d+\b|the answer is)", re.I)


def json_flags(resp: str) -> dict:
    s = resp.strip()
    strict = lenient = False
    try:
        json.loads(s)
        strict = lenient = True
    except Exception:
        try:
            json.loads(_FENCE.sub("", s))
            lenient = True
        except Exception:
            m = re.search(r"\{.*\}", s, re.S)
            if m:
                try:
                    json.loads(m.group(0))
                    lenient = True
                except Exception:
                    pass
    return {"json_valid_strict": strict, "json_valid_lenient": lenient}


def template_flags(prompt: str, resp: str) -> dict:
    keys = re.findall(r"^\s*([A-Za-z_][\w ]*?)\s*=\s*#", prompt, re.M)
    if not keys:
        return {"template_keys": "", "template_keys_filled": None}
    filled = sum(1 for k in keys if re.search(rf"^\s*{re.escape(k)}\s*=\s*\S", resp, re.M | re.I))
    return {"template_keys": "|".join(keys), "template_keys_filled": f"{filled}/{len(keys)}"}


def repetition_score(resp: str, n: int = 4) -> float:
    toks = resp.split()
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def auto_flags(row: dict) -> dict:
    r = row["response"]
    f = {"truncated": row["truncated"], "n_new_tokens": row["n_new_tokens"],
         "repetition_4gram": round(repetition_score(r), 3), "math_markers": len(_MATH.findall(r)),
         "chars": len(r)}
    if row["format"] == "json":
        f.update(json_flags(r))
    if row["format"] == "template":
        f.update(template_flags(row["prompt"], r))
    return f


MANUAL_COLS = ["format_compliance_0to2", "coherence_relevance_0to2", "repetition_issue_YN",
               "unnecessary_math_framing_YN", "inappropriate_content_YN", "notes", "short_excerpt"]


def annotation_subset(rows: list[dict], questions: list[dict], condition_order: list[str]) -> list[dict]:
    """First and last underlying questions in plain-question order x 3 formats x 7 conditions."""
    plain = sorted({(q["plain_order"], q["base_id"]) for q in questions if q["plain_order"] is not None})
    chosen = [plain[0][1], plain[-1][1]]
    sub = []
    for b in chosen:
        for fmt in FORMATS:
            for c in condition_order:
                hit = [r for r in rows if r["base_id"] == b and r["format"] == fmt and r["condition"] == c]
                if hit:
                    sub.append(hit[0])
    return sub


def write_annotation_sheet(sub: list[dict], path: str | Path) -> None:
    flag_cols = ["truncated", "n_new_tokens", "repetition_4gram", "math_markers", "chars",
                 "json_valid_strict", "json_valid_lenient", "template_keys", "template_keys_filled"]
    cols = ["base_id", "format", "condition", "question_id"] + flag_cols + MANUAL_COLS + ["response"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sub:
            f = auto_flags(r)
            w.writerow({**{c: r.get(c) for c in ("base_id", "format", "condition", "question_id", "response")},
                        **{c: f.get(c, "") for c in flag_cols}, **{c: "" for c in MANUAL_COLS}})


def write_grouped_markdown(rows: list[dict], questions: list[dict], condition_order: list[str], path) -> None:
    lines = ["# Qualitative responses (greedy, thinking disabled, 512 new tokens)\n"]
    seen = []
    for q in questions:
        if q["base_id"] not in seen:
            seen.append(q["base_id"])
    for b in seen:
        lines.append(f"\n## {b}\n")
        for fmt in FORMATS:
            qs = [q for q in questions if q["base_id"] == b and q["format"] == fmt]
            if not qs:
                continue
            q = qs[0]
            lines.append(f"\n### {fmt}\n")
            if q["system"]:
                lines.append(f"**System:** {q['system']}\n")
            lines.append("**Prompt:**\n\n```\n" + q["prompt"] + "\n```\n")
            for c in condition_order:
                hit = [r for r in rows if r["question_id"] == q["id"] and r["condition"] == c]
                if hit:
                    r = hit[0]
                    tag = " *(truncated)*" if r["truncated"] else ""
                    lines.append(f"\n**{c}**{tag} ({r['n_new_tokens']} tok)\n\n```\n{r['response']}\n```\n")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
