"""Load, join and verify the supplied exports (schema of ATML-PA3-Data/README.MD) and build a compact cache.

Each table has 32,000 rows, one per rollout; tables are joined on (dataset, sample_id, rollout_id).  Every per-token
list is aligned with generated_token_ids (stop tokens excluded), so for each rollout
    T = n_generated_text_tokens = len(generated_token_ids) = len(logp) = len(kl_u_p) [= len(attention lists)]
which is verified (any mismatch stops the run).  T is the retained-token count used for the cost proxy.

Cache (scratch disk):  meta.parquet (one row per rollout, sorted by sample_id string, then rollout_id 0..31),
tokens.npz (offsets, logp, kl[, v]), questions.parquet, texts.parquet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..config import cache_dir, odir
from ..io import write_json
from .discover import FILES

KEYS = ["dataset", "sample_id", "rollout_id"]
GEN = {"model": "model", "problem": "question_text", "token_ids": "generated_token_ids",
       "n_tokens": "n_generated_text_tokens", "capped": "hit_max_new_tokens", "text": "generated_text",
       "gold": "gold_answer", "extracted": "extracted_answer"}
PROB = {"logp": "logp", "kl": "kl_u_p"}
ATTN = {"img": "img_attn_total", "prm": "prompt_text_attn_total", "gen": "generated_text_attn_total"}


def _read(path: Path, cols: list[str]) -> pa.Table:
    names = pq.read_schema(path).names
    missing = [c for c in cols if c not in names]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}; columns present: {names}")
    return pq.read_table(path, columns=cols, memory_map=True)


def list_column(col: pa.ChunkedArray, lengths_only: bool = False):
    """Arrow list column -> (flat float32 values with nulls as NaN, per-row lengths)."""
    arr = col.combine_chunks()
    lens = pc.list_value_length(arr).fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    if lengths_only:
        return None, lens
    vals = np.asarray(arr.flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
    assert lens.sum() == len(vals)
    return vals, lens


def gather(vals: np.ndarray, lens: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Concatenate the segments `rows` of a ragged array (flat values + lengths) in that order."""
    offs = np.concatenate([[0], np.cumsum(lens)])
    L = lens[rows]
    new = np.concatenate([[0], np.cumsum(L)])
    return vals[np.repeat(offs[rows] - new[:-1], L) + np.arange(new[-1])]


def _keyframe(tbl: pa.Table, name: str) -> pd.DataFrame:
    k = tbl.select(KEYS).to_pandas()
    k["sample_id"] = k["sample_id"].astype(str)
    k["rollout_id"] = k["rollout_id"].astype(np.int64)
    if k.duplicated(KEYS).any():
        raise ValueError(f"{name}: duplicate (dataset, sample_id, rollout_id) keys")
    k[f"_row_{name}"] = np.arange(len(k))
    return k


def _none_if_blank(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v)
    return s if s.strip() else None


def load_dataset(folder: Path, ds: str, cfg: dict) -> dict:
    folder = Path(folder)
    n_roll = int(cfg["n_rollouts"])
    scal = [c for k, c in GEN.items() if k != "token_ids"]
    g = _read(folder / FILES["gen"], KEYS + scal + [GEN["token_ids"]])
    p = _read(folder / FILES["prob"], KEYS + list(PROB.values()))
    tables = {"generations": g, "probability_metrics": p}
    if ds == "mathvista":
        tables["attention_metrics"] = _read(folder / FILES["attn"], KEYS + list(ATTN.values()))

    # ---- join on (dataset, sample_id, rollout_id): every table must hold exactly the same keys
    m = None
    report: dict = {"dataset": ds, "folder": str(folder), "rows": {}, "columns": {"generations": GEN,
                    "probability_metrics": PROB, "attention_metrics": ATTN if ds == "mathvista" else None}}
    for name, t in tables.items():
        k = _keyframe(t, name)
        report["rows"][name] = len(k)
        if m is None:
            m = k
            continue
        both = m.merge(k, on=KEYS, how="outer", indicator=True)
        bad = int((both["_merge"] != "both").sum())
        if bad:
            raise ValueError(f"{ds}: {bad} rollouts are not present in both generations and {name}")
        m = both.drop(columns="_merge")
    m = m.sort_values(["sample_id", "rollout_id"], kind="stable").reset_index(drop=True)
    per_q = m.groupby("sample_id", sort=False)["rollout_id"].agg(list)
    wrong = [q for q, r in per_q.items() if r != list(range(n_roll))]
    if wrong:
        raise ValueError(f"{ds}: {len(wrong)} questions do not have rollout_id 0..{n_roll - 1}, e.g. {wrong[:3]}")
    report["n_questions"] = int(len(per_q))
    report["dataset_values"] = sorted(map(str, m["dataset"].unique()))

    # ---- token alignment: T = n_generated_text_tokens must equal every per-token list length
    grow = m["_row_generations"].to_numpy()
    prow = m["_row_probability_metrics"].to_numpy()
    T = g.column(GEN["n_tokens"]).to_numpy().astype(np.int64)[grow]
    lens = {"generated_token_ids": list_column(g.column(GEN["token_ids"]), lengths_only=True)[1][grow]}
    lp_v, lp_l = list_column(p.column(PROB["logp"]))
    kl_v, kl_l = list_column(p.column(PROB["kl"]))
    lens["logp"], lens["kl_u_p"] = lp_l[prow], kl_l[prow]
    att = {}
    if ds == "mathvista":
        arow = m["_row_attention_metrics"].to_numpy()
        a = tables["attention_metrics"]
        for r, c in ATTN.items():
            att[r] = list_column(a.column(c))
            lens[c] = att[r][1][arow]
    ali = {"n_rollouts": int(len(m)), "zero_length_rollouts": int((T == 0).sum()), "T_total": int(T.sum()),
           "T_mean": float(T.mean()), "T_max": int(T.max()),
           **{f"mismatch_{n}_vs_T": int((v != T).sum()) for n, v in lens.items()}}
    report["alignment"] = ali
    if any(ali[f"mismatch_{n}_vs_T"] for n in lens):
        write_json(odir("data") / f"{ds}_alignment.json", report)
        raise ValueError(f"{ds}: token lists not aligned with n_generated_text_tokens: {ali}")

    logp, kl = gather(lp_v, lp_l, prow), gather(kl_v, kl_l, prow)
    arrays = {"offsets": np.concatenate([[0], np.cumsum(T)]).astype(np.int64), "logp": logp, "kl": kl}
    ali.update(nonfinite_logp=int((~np.isfinite(logp)).sum()), nonfinite_kl=int((~np.isfinite(kl)).sum()))
    if ds == "mathvista":
        img, prm, gen = (gather(att[r][0], att[r][1], arow).astype(np.float64) for r in ("img", "prm", "gen"))
        den = img + prm + gen
        ok = np.isfinite(den) & (den > 0) & np.isfinite(img)
        v = np.full(len(den), np.nan)
        v[ok] = img[ok] / den[ok]
        arrays["v"] = v.astype(np.float32)
        ali["attention_tokens_undefined_v"] = int((~ok).sum())
        del img, prm, gen, den

    # ---- per-rollout meta, per-question table, texts
    gdf = g.select(scal).to_pandas().iloc[grow].reset_index(drop=True)
    qids = list(per_q.index)
    qi_of = {q: i for i, q in enumerate(qids)}
    capped = gdf[GEN["capped"]].to_numpy().astype(bool)
    ext = [_none_if_blank(x) for x in gdf[GEN["extracted"]].tolist()]
    meta = pd.DataFrame({"qi": m["sample_id"].map(qi_of).to_numpy(np.int32), "qid": m["sample_id"].to_numpy(),
                         "rpos": m["rollout_id"].to_numpy(np.int32), "rollout_id": m["rollout_id"].to_numpy(),
                         "T": T, "capped": capped, "extracted": ext})
    ali.update(capped_rollouts=int(capped.sum()), null_extractions=int(sum(e is None for e in ext)))
    golds = gdf.groupby(m["sample_id"].to_numpy(), sort=False)[GEN["gold"]].agg(lambda s: sorted(set(s)))
    ali["questions_with_inconsistent_gold"] = int(sum(len(v) > 1 for v in golds))
    first = gdf.groupby(m["sample_id"].to_numpy(), sort=False).head(1)
    qdf = pd.DataFrame({"qi": np.arange(len(qids)), "qid": qids, "gold": first[GEN["gold"]].to_numpy(),
                        "problem": first[GEN["problem"]].to_numpy(), "model": first[GEN["model"]].to_numpy()})
    cd = cache_dir(ds)
    meta.to_parquet(cd / "meta.parquet", index=False)
    qdf.to_parquet(cd / "questions.parquet", index=False)
    pd.DataFrame({"qid": meta["qid"], "rollout_id": meta["rollout_id"],
                  "text": gdf[GEN["text"]].to_numpy()}).to_parquet(cd / "texts.parquet", index=False)
    np.savez(cd / "tokens.npz", **arrays)
    write_json(odir("data") / f"{ds}_alignment.json", report)
    return report


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
