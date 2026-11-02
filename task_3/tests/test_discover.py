"""Finding the README layout wherever Kaggle mounts it, the strict schema check, and MathVista metadata."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from pa3ttc.answers import mathvista as mvn
from pa3ttc.data.discover import find_folders


def test_find_folders_kaggle_mounts(tmp_path):
    from conftest import make_data

    data = make_data(tmp_path / "src")
    # /kaggle/input/<slug>/ATML-PA3-Data/... and a symlinked mount; data_root that does not exist
    inp = tmp_path / "input" / "atml-pa3-data"
    inp.mkdir(parents=True)
    try:
        os.symlink(data / "ATML-PA3-Data", inp / "ATML-PA3-Data")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation requires Windows Developer Mode or elevated privileges")
    lp = tmp_path / "listing.txt"
    f = find_folders(tmp_path / "nope", tmp_path / "input", listing_path=lp)
    assert f["math"].name == "qwen25-math" and f["mathvista"].name == "qwen25vl-mathvista"
    assert "does not exist" in lp.read_text()


def test_not_found_message(tmp_path):
    (tmp_path / "input" / "other").mkdir(parents=True)
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "input" / "other" / "x.parquet")
    with pytest.raises(FileNotFoundError, match="Is the dataset attached"):
        find_folders(tmp_path / "input", tmp_path / "input")


def test_missing_column_is_reported(tmp_path, monkeypatch):
    import json

    from conftest import make_data
    from pa3ttc import stages

    data = make_data(tmp_path / "d")
    p = data / "ATML-PA3-Data" / "qwen25vl-mathvista" / "attention_metrics.parquet"
    t = pq.read_table(p)
    pq.write_table(t.drop(["img_attn_total"]), p)
    monkeypatch.setenv("PA3_OUTPUT_ROOT", str(tmp_path / "o"))
    monkeypatch.setenv("PA3_SCRATCH", str(tmp_path / "s"))
    monkeypatch.setenv("PA3_CONFIG_OVERRIDES", json.dumps({"data_root": str(data), "data_fallback_root": str(data)}))
    with pytest.raises(KeyError, match="img_attn_total"):
        stages.main("data")


def test_parse_choices():
    assert mvn.parse_choices("Find x\nA. 97\nB. 102\nC. 107\nD. 122") == ["97", "102", "107", "122"]
    # question 287 of the export repeats its choice block: the last block starting at A is used
    q287 = "Which region is larger? R1 or R2?\nA. R1\nB. R2\nA. R1\nB. R2\nC. R5\nD. R3\nE. R4"
    assert mvn.parse_choices(q287) == ["R1", "R2", "R5", "R3", "R4"]
    assert mvn.parse_choices("How many?") == []
    assert mvn.parse_choices("Pick\nB. x\nC. y") == []            # must start at A


def test_build_metadata_hf_and_derived():
    qids, qts, golds = ["1", "2", "3", "4"], ["Q\nA. a\nB. b", "How many?", "Length?", "List?"], ["B", "5", "1.25", "[1, 2]"]
    metas, st = mvn.build_metadata(qids, qts, golds, None)
    assert [m["source"] for m in metas] == ["derived"] * 4
    assert metas[0]["question_type"] == "multi_choice" and metas[0]["choices"] == ["a", "b"]
    assert [(m["answer_type"], m["precision"]) for m in metas[1:]] == [("integer", None), ("float", 2), ("list", None)]
    hf = pd.DataFrame({"pid": ["1", "2", "3"], "question_type": ["multi_choice", "free_form", "free_form"],
                       "answer_type": ["text", "integer", "float"], "choices": [np.array(["a", "b"]), None, None],
                       "precision": [np.nan, np.nan, 2.0]}).set_index("pid")
    metas, st = mvn.build_metadata(qids, qts, golds, hf)
    assert [m["source"] for m in metas] == ["hf", "hf", "hf", "derived"]
    assert st["qtype_disagree"] == st["atype_disagree"] == st["precision_disagree"] == st["choices_disagree"] == 0
    assert metas[2]["precision"] == 2
    # a letter answer and the gold letter agree; an out-of-range letter is mapped by the official fuzzy rule
    assert mvn.normalize("B", metas[0])[0] == mvn.gold_key("B", metas[0]) == "MC:B"
    assert mvn.normalize("E", metas[0])[1] == "mc_fuzzy"
    assert mvn.normalize("1.249", metas[2])[0] == mvn.gold_key("1.25", metas[2])
