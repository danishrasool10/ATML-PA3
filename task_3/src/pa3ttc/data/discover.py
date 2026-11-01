"""Find the supplied ATML-PA3-Data tables, tolerant to how the dataset was uploaded.

Accepted layouts (searched with symlinks followed, under data_root first, then all of /kaggle/input):
  * <dir>/generations.parquet + probability_metrics.parquet [+ attention_metrics.parquet]   (as in the task text)
  * any file or shard folder whose name contains "generation" / "prob" / "atten|attn", e.g.
    generations-00000-of-00002.parquet, generations/ (folder of parquet shards), mathvista_generations.parquet
  * both models' tables in one folder, told apart by "vista"/"vl" in the file names
  * the same inside a .zip / .tar(.gz) archive (extracted to scratch disk)
  * optionally a Hugging Face dataset repo (config `data_hf_repo`), downloaded to scratch disk
The VLM table set is the one with an attention table.  A listing of the searched folders is always written to
data/input_listing.txt so a failure can be diagnosed from the results zip.
"""
from __future__ import annotations

import os
import re
import tarfile
import zipfile
from pathlib import Path

ROLE_RE = {"attn": re.compile(r"atten|attn", re.I), "prob": re.compile(r"prob|logp", re.I),
           "gen": re.compile(r"generation|rollout|completion", re.I)}
PARQ = (".parquet", ".pq")
ARCH = (".zip", ".tar", ".tar.gz", ".tgz")


def _walk(root: Path, max_depth: int = 8):
    """os.walk with symlinks followed and cycle protection; yields (dirpath, dirnames, filenames)."""
    seen = set()
    root = Path(root)
    base = len(root.parts)
    for dp, dns, fns in os.walk(root, followlinks=True):
        rp = os.path.realpath(dp)
        if rp in seen or len(Path(dp).parts) - base > max_depth:
            dns[:] = []
            continue
        seen.add(rp)
        dns.sort()
        yield Path(dp), dns, sorted(fns)


def listing(roots: list[Path], max_lines: int = 600) -> str:
    out = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            out.append(f"{root}  (does not exist)")
            continue
        for dp, dns, fns in _walk(root, max_depth=6):
            depth = len(dp.parts) - len(root.parts)
            link = " -> " + os.path.realpath(dp) if dp.is_symlink() else ""
            out.append("  " * depth + f"{dp.name or str(dp)}/{link}")
            for f in fns[:40]:
                try:
                    sz = (dp / f).stat().st_size
                except OSError:
                    sz = -1
                out.append("  " * (depth + 1) + f"{f}  ({sz / 2**20:.1f} MiB)")
            if len(fns) > 40:
                out.append("  " * (depth + 1) + f"... {len(fns) - 40} more files")
            if len(out) > max_lines:
                out.append("... (listing truncated)")
                return "\n".join(out)
    return "\n".join(out)


def _role(name: str) -> str | None:
    stem = name.lower()
    for r in ("attn", "prob", "gen"):          # order matters: "probability" never matches gen
        if ROLE_RE[r].search(stem):
            return r
    return None


def _hint(name: str) -> str:
    n = name.lower()
    if "vista" in n or re.search(r"(^|[^a-z])vl([^a-z]|$)", n):
        return "mathvista"
    if "math" in n:
        return "math"
    return ""


def _is_shard_dir(d: Path) -> bool:
    try:
        return any(f.suffix.lower() in PARQ for f in d.iterdir() if f.is_file())
    except OSError:
        return False


def scan(root: Path) -> dict[tuple[str, str], dict[str, list[str]]]:
    """-> {(folder, dataset-hint): {role: [parquet paths]}}"""
    groups: dict[tuple[str, str], dict[str, list[str]]] = {}
    for dp, dns, fns in _walk(root):
        for f in fns:
            if not f.lower().endswith(PARQ):
                continue
            r = _role(f)
            if r is None:
                # shard file inside a folder named after the role (generations/part-0.parquet)
                r = _role(dp.name)
                if r is None:
                    continue
                key = (str(dp.parent), _hint(dp.name) or _hint(dp.parent.name))
            else:
                key = (str(dp), _hint(f))
            groups.setdefault(key, {}).setdefault(r, []).append(str(dp / f))
    return groups


def _resolve_groups(groups) -> dict[str, dict[str, list[str]]]:
    found: dict[str, dict] = {}
    for (folder, hint), roles in sorted(groups.items()):
        if "gen" not in roles or "prob" not in roles:
            continue
        kind = "mathvista" if "attn" in roles else "math"
        if hint and hint != kind:
            continue
        if kind in found:
            raise ValueError(f"two candidate {kind} table sets: {found[kind]['folder']} and {folder}")
        found[kind] = {"folder": folder, **{r: sorted(v) for r, v in roles.items()}}
    return found


def _extract_archives(root: Path, dest: Path) -> list[Path]:
    out = []
    for dp, dns, fns in _walk(root, max_depth=4):
        for f in fns:
            p = dp / f
            low = f.lower()
            if not low.endswith(ARCH):
                continue
            tgt = dest / re.sub(r"[^A-Za-z0-9_.-]", "_", f)
            if not tgt.exists():
                print(f"[data] extracting {p} -> {tgt}", flush=True)
                tgt.mkdir(parents=True, exist_ok=True)
                if low.endswith(".zip"):
                    with zipfile.ZipFile(p) as z:
                        z.extractall(tgt)
                else:
                    with tarfile.open(p) as t:
                        t.extractall(tgt, filter="data")
            out.append(tgt)
    return out


def find_tables(preferred: str | Path, fallback: str | Path = "/kaggle/input", scratch: Path | None = None,
                hf_repo: str | None = None, listing_path: Path | None = None) -> dict[str, dict]:
    roots = [Path(preferred)] + ([Path(fallback)] if Path(fallback) != Path(preferred) else [])
    if listing_path is not None:
        listing_path.parent.mkdir(parents=True, exist_ok=True)
        listing_path.write_text(listing(roots))
    tried = []
    for root in roots:
        if not root.exists():
            continue
        tried.append(str(root))
        found = _resolve_groups(scan(root))
        if {"math", "mathvista"} <= set(found):
            return found
    if scratch is not None:
        for root in roots:
            if root.exists():
                for ex in _extract_archives(root, Path(scratch) / "extracted"):
                    found = _resolve_groups(scan(ex))
                    if {"math", "mathvista"} <= set(found):
                        return found
        if hf_repo:
            from huggingface_hub import snapshot_download  # noqa: PLC0415

            local = Path(scratch) / "hf_data"
            print(f"[data] downloading {hf_repo} from the Hugging Face Hub -> {local}", flush=True)
            snapshot_download(repo_id=hf_repo, repo_type="dataset", local_dir=str(local))
            tried.append(f"hf:{hf_repo}")
            found = _resolve_groups(scan(local))
            if {"math", "mathvista"} <= set(found):
                return found
    raise FileNotFoundError(
        f"ATML-PA3-Data not found (searched {tried or [str(r) for r in roots]}). Need, for each model, a "
        "generations table and a probability_metrics table (parquet), plus an attention_metrics table for the VLM. "
        "Is the dataset attached (Add Input)? What was found is listed in data/input_listing.txt:\n"
        + (listing(roots, max_lines=80)))


def find_readme(tables: dict[str, dict]) -> Path | None:
    seen = set()
    for t in tables.values():
        d = Path(t["folder"])
        for base in (d, d.parent, d.parent.parent):
            if base in seen:
                continue
            seen.add(base)
            for p in sorted(base.glob("*")):
                if p.is_file() and p.name.lower().startswith("readme"):
                    return p
    return None
