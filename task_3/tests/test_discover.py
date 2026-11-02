"""Data discovery is tolerant to the upload layout, and a miss produces a diagnosable error + listing."""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from pa3ttc import stages
from pa3ttc.data.discover import find_tables
from pa3ttc.data.load import Pool


def _std(tmp_path) -> Path:
    from conftest import make_data

    return make_data(tmp_path / "std")


def _load(tmp_path, root, name, monkeypatch):
    monkeypatch.setenv("PA3_OUTPUT_ROOT", str(tmp_path / f"out_{name}"))
    monkeypatch.setenv("PA3_SCRATCH", str(tmp_path / f"scr_{name}"))
    monkeypatch.setenv("PA3_CONFIG_OVERRIDES", json.dumps({"data_root": str(root), "data_fallback_root": str(root),
                                                           "n_questions_expected": 16}))
    assert stages.main("data") == 0
    return Pool("math"), Pool("mathvista")


def _same(a, b):
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.T, y.T)
        np.testing.assert_array_equal(x.kl, y.kl)
        assert x.qids == y.qids
    assert np.array_equal(a[1].v, b[1].v, equal_nan=True)


def test_layout_variants(tmp_path, monkeypatch):
    std = _std(tmp_path)
    ref = _load(tmp_path, std, "ref", monkeypatch)
    math_dir, vl_dir = std / "Qwen2.5-3B-Instruct" / "MATH", std / "Qwen2.5-VL-3B-Instruct" / "MathVista"

    # (a) both models in one folder, prefixed file names, generations split into two shards
    flat = tmp_path / "flat" / "ATML-PA3-Data"
    flat.mkdir(parents=True)
    g = pq.read_table(math_dir / "generations.parquet")
    pq.write_table(g.slice(0, 200), flat / "math_generations-00000-of-00002.parquet")
    pq.write_table(g.slice(200), flat / "math_generations-00001-of-00002.parquet")
    shutil.copy(math_dir / "probability_metrics.parquet", flat / "math_probability_metrics.parquet")
    for f in ("generations", "probability_metrics", "attention_metrics"):
        shutil.copy(vl_dir / f"{f}.parquet", flat / f"mathvista_{f}.parquet")
    _same(ref, _load(tmp_path, tmp_path / "flat", "flat", monkeypatch))

    # (b) shard folders named after the tables
    sh = tmp_path / "shards"
    for src, name in ((math_dir, "llm"), (vl_dir, "vlm")):
        for f in src.glob("*.parquet"):
            (sh / name / f.stem).mkdir(parents=True, exist_ok=True)
            shutil.copy(f, sh / name / f.stem / "part-0.parquet")
    _same(ref, _load(tmp_path, sh, "shards", monkeypatch))

    # (c) a zipped upload that Kaggle did not unpack
    zroot = tmp_path / "zipped"
    zroot.mkdir()
    with zipfile.ZipFile(zroot / "ATML-PA3-Data.zip", "w") as z:
        for f in std.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(std))
    _same(ref, _load(tmp_path, zroot, "zip", monkeypatch))

    # (d) dataset reached through a symlink
    lroot = tmp_path / "linked"
    lroot.mkdir()
    try:
        os.symlink(std, lroot / "atml-pa3-data")
    except (OSError, NotImplementedError):
        pass
    else:
        _same(ref, _load(tmp_path, lroot, "link", monkeypatch))


def test_not_found_is_diagnosable(tmp_path):
    (tmp_path / "input" / "some-other-dataset").mkdir(parents=True)
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "input" / "some-other-dataset" / "x.parquet")
    lp = tmp_path / "listing.txt"
    with pytest.raises(FileNotFoundError, match="Is the dataset attached"):
        find_tables(tmp_path / "input" / "atml-pa3-data", tmp_path / "input", scratch=tmp_path / "s", listing_path=lp)
    txt = lp.read_text()
    assert "does not exist" in txt and "x.parquet" in txt


def test_download_url(tmp_path, monkeypatch):
    std = _std(tmp_path)
    z = tmp_path / "pub" / "ATML-PA3-Data.zip"
    z.parent.mkdir()
    with zipfile.ZipFile(z, "w") as zf:
        for f in std.rglob("*"):
            if f.is_file():
                zf.write(f, Path("ATML-PA3-Data") / f.relative_to(std))
    empty = tmp_path / "input"
    empty.mkdir()
    found = find_tables(empty / "atml-pa3-data", empty, scratch=tmp_path / "scr", url=z.as_uri())
    assert set(found) == {"math", "mathvista"} and "attn" in found["mathvista"]
    # a second call reuses the download
    again = find_tables(empty / "atml-pa3-data", empty, scratch=tmp_path / "scr", url="file:///nonexistent.zip")
    assert again["math"]["gen"] == found["math"]["gen"]
