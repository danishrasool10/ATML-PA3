"""Load, join and verify the supplied rollout exports and build a compact per-dataset cache.

Each model folder holds generations.parquet + probability_metrics.parquet (+ attention_metrics.parquet for the
VLM).  Tables are joined on (dataset, sample_id, rollout_id).  Token measurements are read as ragged arrays
(flat values + per-rollout lengths) straight from Arrow list columns; a long format (one row per token with a
position column) is also accepted.  Column names are auto-detected from alias lists and can be overridden with
`columns.<ds>.<role>` in the config; the resolution is written to data/<ds>_columns.json.

Cache (scratch disk, rebuilt if missing):  meta.parquet (one row per rollout, sorted by question ID string then
rollout ID), tokens.npz (offsets, logp, kl[, v]), questions.parquet, texts.parquet.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..config import cache_dir, odir
from ..io import write_json

KEYS = ["dataset", "sample_id", "rollout_id"]
ALIASES = {
    "text": ["response", "generation", "generated_text", "output_text", "completion", "output", "text",
             "response_text", "decoded_text"],
    "token_ids": ["token_ids", "generated_token_ids", "output_token_ids", "completion_token_ids", "gen_token_ids",
                  "tokens", "output_ids"],
    "n_tokens": ["num_tokens", "n_tokens", "num_generated_tokens", "n_generated_tokens", "generated_length",
                 "output_length", "length", "token_count", "num_output_tokens", "retained_tokens", "n_retained"],
    "extracted": ["extracted_answer", "pred_answer", "predicted_answer", "extracted", "answer_extracted",
                  "parsed_answer", "prediction", "pred", "model_answer"],
    "gold": ["gold_answer", "gold", "ground_truth", "reference_answer", "target_answer", "target", "solution_answer",
             "answer"],
    "problem": ["problem", "question", "query", "question_text", "problem_text", "original_problem", "input",
                "prompt_text", "prompt"],
    "logp": ["logp", "token_logp", "sampled_logp", "logprobs", "log_probs", "token_logprobs"],
    "kl": ["kl_u_p", "kl_up", "kl_uniform", "self_certainty", "kl"],
    "attn_image": ["image", "attn_image", "image_attention", "image_attn", "attention_image", "image_total",
                   "a_image", "image_mass"],
    "attn_prompt": ["prompt", "prompt_text", "attn_prompt", "attn_prompt_text", "prompt_attention",
                    "prompt_text_attention", "text_prompt", "attention_prompt", "prompt_total", "a_prompt"],
    "attn_generated": ["generated", "generated_text", "attn_generated", "attn_generated_text", "generated_attention",
                       "generated_text_attention", "attention_generated", "generated_total", "a_generated", "gen"],
}
FUZZY = {"attn_image": r"image|img|vis", "attn_generated": r"gener|output|response|completion",
         "attn_prompt": r"prompt|question|instruction|text"}
POS_ALIASES = ["token_index", "token_idx", "token_position", "position", "pos", "t", "step", "index"]
MV_META = ["question_type", "answer_type", "choices", "precision", "unit"]


# ----------------------------------------------------------------------------- column resolution
def resolve(cols: list[str], role: str, override: str | None = None, required: bool = True,
            exclude: tuple = ()) -> str | None:
    if override:
        if override not in cols:
            raise KeyError(f"configured column {override!r} for {role!r} not in {cols}")
        return override
    lower = {c.lower(): c for c in cols if c not in exclude}
    for a in ALIASES.get(role, []):
        if a.lower() in lower:
            return lower[a.lower()]
    if role in FUZZY:
        m = [c for c in cols if c not in exclude and c not in KEYS and re.search(FUZZY[role], c, re.I)]
        if len(m) == 1:
            return m[0]
    if required:
        raise KeyError(f"cannot identify the column for {role!r} among {cols}; "
                       f"set columns.<dataset>.{role} in configs/task3.yaml")
    return None


def _is_list(t: pa.DataType) -> bool:
    return pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t)


def segment_sum(vals: np.ndarray, lens: np.ndarray) -> np.ndarray:
    out = np.full(len(lens), np.nan)
    nz = lens > 0
    if nz.any():
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        out[nz] = np.add.reduceat(vals.astype(np.float64), starts[nz])
    return out


def list_column(arr: pa.Array | pa.ChunkedArray, lengths_only: bool = False):
    """Arrow list column -> (flat float values, per-row lengths).  Nested lists (token x something) are summed
    over the inner dimension (e.g. per-layer attention totals)."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks() if arr.num_chunks else pa.array([], arr.type)
    lens = pc.list_value_length(arr).fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    if lengths_only:
        return None, lens
    inner = arr.flatten()
    if _is_list(inner.type):
        ilens = pc.list_value_length(inner).fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        vals = segment_sum(inner.flatten().to_numpy(zero_copy_only=False), ilens)
    else:
        vals = inner.to_numpy(zero_copy_only=False)
    vals = np.asarray(vals, dtype=np.float64)
    assert lens.sum() == len(vals), "list offsets inconsistent"
    return vals, lens


def gather(vals: np.ndarray, lens: np.ndarray, rows: np.ndarray, take: np.ndarray | None = None) -> np.ndarray:
    """Concatenate segments `rows` of a ragged array (optionally only the first `take[i]` items of each)."""
    offs = np.concatenate([[0], np.cumsum(lens)])
    starts = offs[rows]
    L = lens[rows] if take is None else np.minimum(lens[rows], take)
    new = np.concatenate([[0], np.cumsum(L)])
    idx = np.repeat(starts - new[:-1], L) + np.arange(new[-1])
    return vals[idx]


def _keys_str(df: pd.DataFrame, ds: str) -> pd.DataFrame:
    out = pd.DataFrame({k: df[k].astype(str) if k in df else ds for k in KEYS})
    return out


def read_token_table(path: Path, roles: dict[str, str], ds: str) -> tuple[pd.DataFrame, dict]:
    """-> (keys frame with column _row, {role: (vals, lens)}) in the table's row order (or key order for the
    long format)."""
    schema = pq.read_schema(path)
    cols = [c for c in roles.values()]
    key_cols = [k for k in KEYS if k in schema.names]
    tbl = pq.read_table(path, columns=key_cols + cols)
    if all(_is_list(schema.field(c).type) for c in cols):
        keys = _keys_str(tbl.select(key_cols).to_pandas(), ds)
        data = {r: list_column(tbl.column(c)) for r, c in roles.items()}
        layout = "list"
    else:  # long format: one row per token
        pos = resolve(schema.names, "pos", required=False) or next((p for p in POS_ALIASES if p in schema.names), None)
        if pos is None:
            raise KeyError(f"{path.name}: scalar token columns {cols} but no token-position column")
        tbl = pq.read_table(path, columns=key_cols + [pos] + cols)
        df = tbl.to_pandas()
        k = _keys_str(df, ds)
        df = pd.concat([k, df[[pos] + cols].reset_index(drop=True)], axis=1).sort_values(KEYS + [pos], kind="stable")
        g = df.groupby(KEYS, sort=False).size()
        keys = g.index.to_frame(index=False)
        lens = np.append(g.to_numpy().astype(np.int64), 0)      # extra empty segment for rollouts with no rows
        data = {r: (df[c].to_numpy(np.float64), lens) for r, c in roles.items()}
        layout = f"long(pos={pos})"
    keys["_row"] = np.arange(len(keys))
    return keys, {"_layout": layout, **data}


# ----------------------------------------------------------------------------- main loader
def _scalar_str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        v = list(v)
        return None if not v else str(v[-1])
    s = str(v)
    return s if s.strip() and s.strip().lower() not in ("none", "nan", "null") else None


def load_dataset(model_dir: Path, ds: str, cfg: dict) -> dict:
    """Join + verify one model folder and write the cache.  Returns the alignment report."""
    model_dir = Path(model_dir)
    ov = (cfg.get("columns") or {}).get(ds, {}) or {}
    n_roll = int(cfg["n_rollouts"])
    gpath, ppath, apath = (model_dir / "generations.parquet", model_dir / "probability_metrics.parquet",
                           model_dir / "attention_metrics.parquet")
    gschema = pq.read_schema(gpath)
    gcols = gschema.names
    colmap = {"generations": {}, "probability_metrics": {}, "attention_metrics": {}}
    gold = resolve(gcols, "gold", ov.get("gold"))
    extracted = resolve(gcols, "extracted", ov.get("extracted"), exclude=(gold,))
    text = resolve(gcols, "text", ov.get("text"), required=False, exclude=(gold, extracted))
    tok = resolve(gcols, "token_ids", ov.get("token_ids"), required=False)
    ntok = resolve(gcols, "n_tokens", ov.get("n_tokens"), required=False)
    problem = resolve(gcols, "problem", ov.get("problem"), required=False, exclude=(gold, extracted, text))
    colmap["generations"] = {"gold": gold, "extracted": extracted, "text": text, "token_ids": tok,
                             "n_tokens": ntok, "problem": problem}
    meta_cols = [c for c in MV_META if c in gcols] if ds == "mathvista" else []
    scal = [c for c in {gold, extracted, text, ntok, problem, *meta_cols} if c]
    key_cols = [k for k in KEYS if k in gcols]
    gtbl = pq.read_table(gpath, columns=key_cols + scal + ([tok] if tok else []))
    gdf = pd.concat([_keys_str(gtbl.select(key_cols).to_pandas(), ds),
                     gtbl.select(scal).to_pandas()], axis=1)
    gdf["_n_token_ids"] = list_column(gtbl.column(tok), lengths_only=True)[1] if tok and _is_list(
        gschema.field(tok).type) else -1
    del gtbl
    gdf["_grow"] = np.arange(len(gdf))

    pschema = pq.read_schema(ppath).names
    proles = {"logp": resolve(pschema, "logp", ov.get("logp")), "kl": resolve(pschema, "kl", ov.get("kl"))}
    colmap["probability_metrics"] = dict(proles)
    pkeys, pdat = read_token_table(ppath, proles, ds)
    tables = {"probability_metrics": (pkeys, pdat)}
    if ds == "mathvista":
        aschema = pq.read_schema(apath).names
        img = resolve(aschema, "attn_image", ov.get("attn_image"))
        gen = resolve(aschema, "attn_generated", ov.get("attn_generated"), exclude=(img,))
        prm = resolve(aschema, "attn_prompt", ov.get("attn_prompt"), exclude=(img, gen))
        aroles = {"img": img, "prm": prm, "gen": gen}
        colmap["attention_metrics"] = dict(aroles)
        tables["attention_metrics"] = read_token_table(apath, aroles, ds)

    # ---- join on (dataset, sample_id, rollout_id)
    report: dict = {"dataset": ds, "model_dir": str(model_dir), "columns": colmap, "rows": {"generations": len(gdf)}}
    m = gdf
    for name, (keys, dat) in tables.items():
        report["rows"][name] = len(keys)
        report.setdefault("layout", {})[name] = dat["_layout"]
        keys = keys.rename(columns={"_row": f"_row_{name}"})
        both = m.merge(keys, on=KEYS, how="outer", indicator=True)
        report.setdefault("join_unmatched", {})[name] = {
            "only_in_generations": int((both["_merge"] == "left_only").sum()),
            f"only_in_{name}": int((both["_merge"] == "right_only").sum())}
        if dat["_layout"].startswith("long"):
            # long format has no rows for a zero-length rollout: keep it with an empty token list
            lo = both["_merge"] == "left_only"
            both.loc[lo, f"_row_{name}"] = len(dat[next(r for r in dat if r != "_layout")][1]) - 1
            both.loc[lo, "_merge"] = "both"
            report.setdefault("zero_length_filled_from_long_format", {})[name] = int(lo.sum())
        m = both[both["_merge"] == "both"].drop(columns="_merge")
        m[f"_row_{name}"] = m[f"_row_{name}"].astype(np.int64)
        if m.duplicated(KEYS).any():
            raise ValueError(f"{ds}: duplicate (dataset, sample_id, rollout_id) keys after joining {name}")
    m["qid"] = m["sample_id"].astype(str)
    rnum = pd.to_numeric(m["rollout_id"], errors="coerce")
    m["_rsort"] = rnum if rnum.notna().all() else m["rollout_id"]
    m = m.sort_values(["qid", "_rsort"], kind="stable").reset_index(drop=True)
    per_q = m.groupby("qid", sort=False).size()
    bad_q = per_q[per_q != n_roll]
    report["n_questions"] = int(len(per_q))
    report["rollouts_per_question"] = {str(k): int(v) for k, v in per_q.value_counts().items()}
    if len(bad_q):
        raise ValueError(f"{ds}: {len(bad_q)} questions do not have {n_roll} joined rollouts, e.g. "
                         f"{bad_q.head(5).to_dict()} (see join_unmatched {report['join_unmatched']})")
    report["dataset_values"] = sorted(map(str, m["dataset"].unique()))

    # ---- token alignment
    prow = m["_row_probability_metrics"].to_numpy(np.int64)
    lp_v, lp_l = pdat["logp"]
    kl_v, kl_l = pdat["kl"]
    lens = {"logp": lp_l[prow], "kl": kl_l[prow]}
    if ds == "mathvista":
        akeys, adat = tables["attention_metrics"]
        arow = m["_row_attention_metrics"].to_numpy(np.int64)
        for r in ("img", "prm", "gen"):
            lens[f"attn_{r}"] = adat[r][1][arow]
    T = np.min(np.stack(list(lens.values())), axis=0)
    mis = np.zeros(len(m), bool)
    for v in lens.values():
        mis |= v != T
    n_tok_col = m[ntok].to_numpy() if ntok else None
    ali = {"n_rollouts": int(len(m)), "misaligned_measurements": int(mis.sum()),
           "zero_length_rollouts": int((T == 0).sum()),
           "T_mean": float(T.mean()), "T_max": int(T.max()), "T_min": int(T.min())}
    if n_tok_col is not None:
        d = pd.to_numeric(pd.Series(n_tok_col), errors="coerce").to_numpy() - T
        ali["n_tokens_column_minus_T"] = {str(int(k)) if np.isfinite(k) else "nan": int(v)
                                          for k, v in pd.Series(d).value_counts().head(8).items()}
    if tok:
        d = m["_n_token_ids"].to_numpy() - T
        ali["token_ids_len_minus_T"] = {str(int(k)): int(v) for k, v in pd.Series(d).value_counts().head(8).items()}
    for name, v in lens.items():
        ali[f"mismatch_{name}_vs_T"] = int((v != T).sum())
    frac = mis.mean()
    if frac > float(cfg.get("max_misaligned_frac", 0.01)):
        write_json(odir("data") / f"{ds}_alignment.json", {**report, "alignment": ali})
        raise ValueError(f"{ds}: {mis.sum()} rollouts ({frac:.2%}) have token lists of different lengths: {ali}")
    report["alignment"] = ali

    logp = gather(lp_v, lp_l, prow, T).astype(np.float32)
    kl = gather(kl_v, kl_l, prow, T).astype(np.float32)
    arrays = {"offsets": np.concatenate([[0], np.cumsum(T)]).astype(np.int64), "logp": logp, "kl": kl}
    ali["nan_logp"] = int(np.isnan(logp).sum())
    ali["nan_kl"] = int(np.isnan(kl).sum())
    if ds == "mathvista":
        a = {r: gather(adat[r][0], adat[r][1], arow, T) for r in ("img", "prm", "gen")}
        den = a["img"] + a["prm"] + a["gen"]
        ok = np.isfinite(den) & (den > 0) & np.isfinite(a["img"])
        v = np.full(len(den), np.nan)
        v[ok] = a["img"][ok] / den[ok]
        arrays["v"] = v.astype(np.float32)
        ali["attention_tokens_undefined_v"] = int((~ok).sum())
        ali["v_outside_0_1"] = int(((v < -1e-6) | (v > 1 + 1e-6)).sum())

    # ---- per-rollout meta, per-question table, texts
    qids = list(per_q.index)
    qi_of = {q: i for i, q in enumerate(qids)}
    meta = pd.DataFrame({
        "qi": m["qid"].map(qi_of).to_numpy(np.int32), "qid": m["qid"],
        "rpos": m.groupby("qid", sort=False).cumcount().to_numpy(np.int32),
        "rollout_id": m["rollout_id"].astype(str), "T": T.astype(np.int64), "misaligned": mis,
        "extracted": [_scalar_str(x) for x in m[extracted].tolist()],
        "n_tokens_col": pd.to_numeric(m[ntok], errors="coerce") if ntok else np.nan,
    })
    gold_by_q = m.groupby("qid", sort=False)[gold].agg(lambda s: [_scalar_str(x) for x in s])
    ali["questions_with_inconsistent_gold"] = int(sum(len(set(v)) > 1 for v in gold_by_q))
    qdf = pd.DataFrame({"qi": np.arange(len(qids)), "qid": qids,
                        "gold": [v[0] for v in gold_by_q.loc[qids]]})
    first = m.groupby("qid", sort=False).head(1).set_index("qid").loc[qids]
    qdf["problem"] = first[problem].astype(str).to_numpy() if problem else None
    for c in meta_cols:
        qdf[c] = [_jsonable(x) for x in first[c].tolist()]
    report["mathvista_meta_columns_in_files"] = meta_cols
    cd = cache_dir(ds)
    meta.to_parquet(cd / "meta.parquet", index=False)
    qdf.to_parquet(cd / "questions.parquet", index=False)
    pd.DataFrame({"qid": m["qid"], "rollout_id": m["rollout_id"].astype(str),
                  "text": m[text].astype(str) if text else ""}).to_parquet(cd / "texts.parquet", index=False)
    np.savez(cd / "tokens.npz", **arrays)
    write_json(odir("data") / f"{ds}_alignment.json", report)
    write_json(odir("data") / f"{ds}_columns.json", {"generations": gcols, "probability_metrics": pschema,
                                                      "attention_metrics": pq.read_schema(apath).names
                                                      if ds == "mathvista" else None, "resolved": colmap})
    return report


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    return x


# ----------------------------------------------------------------------------- cache access
class Pool:
    """In-memory view of one dataset's cache: [Q, R] matrices + flat token arrays."""

    def __init__(self, ds: str):
        cd = cache_dir(ds)
        self.ds = ds
        self.meta = pd.read_parquet(cd / "meta.parquet")
        self.questions = pd.read_parquet(cd / "questions.parquet")
        z = np.load(cd / "tokens.npz")
        self.offsets = z["offsets"]
        self.logp, self.kl = z["logp"], z["kl"]
        self.v = z["v"] if "v" in z.files else None
        self.qids = self.questions["qid"].tolist()
        self.Q = len(self.qids)
        self.R = int(len(self.meta) // self.Q)
        assert self.Q * self.R == len(self.meta)
        self.T = self.meta["T"].to_numpy().reshape(self.Q, self.R)
        self.rollout_ids = self.meta["rollout_id"].to_numpy().reshape(self.Q, self.R)

    def seg(self, q: int, r: int) -> slice:
        i = q * self.R + r
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))

    def texts(self) -> pd.DataFrame:
        return pd.read_parquet(cache_dir(self.ds) / "texts.parquet")
