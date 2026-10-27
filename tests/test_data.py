import numpy as np
import torch

from conftest import EFF, N_UPDATES, synthetic_rows


def test_selection_counts_and_order(out_root, tok):
    from pa3.utils.common import read_json

    rep = read_json(out_root / "data" / "report.json")
    c = rep["counts"]
    assert c["retained"] == N_UPDATES * EFF
    assert c["empty"] > 0 and c["gsm8k_test_match"] > 0 and c["too_long"] > 0
    assert c["rows_examined"] == c["retained"] + c["empty"] + c["too_long"] + c["gsm8k_test_match"]
    src = np.load(out_root / "data" / "src_index.npy")
    assert np.all(np.diff(src) > 0), "source order must be preserved (no shuffling)"


def test_decontam_normalisation():
    from pa3.data.normalize import normalize_text

    assert normalize_text("  JANET'S  ducks\nlay ") == normalize_text("janet's ducks lay")
    assert normalize_text("ﬁve") == "five"  # NFKC ligature
    assert normalize_text("a.b") != normalize_text("a b")  # punctuation kept (approved)


def test_masking_and_prompt_identity(out_root, tok):
    from pa3.data.dataset import IGNORE, TokenStore, collate
    from pa3.data.formatting import eot_id, render_prompt

    rows, _ = synthetic_rows(N_UPDATES * EFF)
    good = [r for r in rows if r["_good"]]
    store = TokenStore(out_root / "data")
    src = np.load(out_root / "data" / "src_index.npy")
    for j in (0, 1, 57):
        r = rows[int(src[j])]
        ids, p = store.example(j)
        prompt = tok(render_prompt(tok, r["query"]), add_special_tokens=False)["input_ids"]
        assert list(ids[:p]) == prompt, "supervised prefix must equal the inference prompt tokens"
        text = tok.decode(ids[:p])
        assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert ids[-1] == eot_id(tok)
        assert tok.decode(ids[p:-1]) == r["response"]
    b = collate(store, [0, 1], tok.pad_token_id)
    for i in range(2):
        ids, p = store.example(i)
        n = len(ids)
        lab = b["labels"][i]
        assert torch.all(lab[:p] == IGNORE) and torch.all(lab[n:] == IGNORE)
        assert torch.equal(lab[p:n], torch.from_numpy(ids[p:]))
        assert int(b["attention_mask"][i].sum()) == n
        assert int((lab != IGNORE).sum()) == int(store.resp_len[i])
    assert good  # sanity


def test_plan_partition(out_root):
    from pa3.data.dataset import TokenStore, plan_update

    store = TokenStore(out_root / "data")
    for k in range(N_UPDATES):
        single = plan_update(store, k, 0, 1)
        r0, r1 = plan_update(store, k, 0, 2), plan_update(store, k, 1, 2)
        assert len(single.microbatches) == 16 and len(r0.microbatches) == 8
        assert single.microbatches == r0.microbatches + r1.microbatches
        assert single.n_tokens == r0.n_tokens == int(store.resp_len[32 * k: 32 * k + 32].sum())
        assert [i for mb in single.microbatches for i in mb] == list(range(32 * k, 32 * k + 32))
