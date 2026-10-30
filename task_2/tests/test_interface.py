"""Visual interface: chat-template split, collate (masks/positions/labels), embedding splice, loss masking and
normalisation against a full-logits reference, padding invariance, no-image position rebuild, gradient flow,
LoRA boundary, parameter counts, and the KV-cache greedy decoder against HF generate."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pa3vlm.data.collate import collate
from pa3vlm.data.prompts import ANSWER_INSTRUCTION, CAPTION_PROMPT, ChatFormat
from pa3vlm.features.store import RandomFeatures, pool2x2

CAPS = {"question": 80, "caption": 64, "answer": 16}


@pytest.fixture(scope="module")
def fmt(tok):
    return ChatFormat(tok, CAPS)


@pytest.fixture(scope="module")
def vlm(tiny_dir):
    from pa3vlm.models.connector import make_connector
    from pa3vlm.models.vlm import VLM, load_lm

    lm = load_lm(str(tiny_dir), torch.float32, "sdpa", grad_ckpt=True)
    v = VLM(lm, make_connector(6304, d_out=lm.config.hidden_size))
    v.connector.scale.fill_(0.37)
    return v.eval()


def _encs(fmt):
    return [fmt.encode("a", "train2014:1", "How many dogs can you see?", "2"),
            fmt.encode("b", "train2014:2", None, "A brown dog runs across a very green park near the river."),
            fmt.encode("c", "train2014:3", "Does this image contain a cat?", "no"),
            fmt.encode("d", "train2014:4", "What color is the bus?", "red")]


def test_chat_format(fmt, tok):
    e = fmt.encode("x", "k", "How many dogs can you see?", "two")
    assert tok.decode(e.prefix) == "<|im_start|>user\n"
    assert tok.decode(e.suffix) == ("\nHow many dogs can you see?\n" + ANSWER_INSTRUCTION +
                                    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert e.target[-1] == fmt.im_end and tok.decode(e.target[:-1]) == "two"
    ref = tok.apply_chat_template([{"role": "user", "content": "\nHow many dogs can you see?\n" + ANSWER_INSTRUCTION}],
                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert tok.decode(e.prefix + e.suffix) == ref           # text-only rendering identical, image slot between
    c = fmt.encode("y", "k", None, "word " * 100)
    assert len(c.target) == 64 + 1 and fmt.stats["target_truncated"] >= 1
    assert CAPTION_PROMPT in tok.decode(c.suffix)
    q = fmt.encode("z", "k", "why " * 200, "yes")
    assert len(tok(tok.decode(q.suffix)).input_ids) < 80 + 40


@pytest.mark.parametrize("side", ["right", "left"])
def test_collate_masks_positions_labels(fmt, side):
    encs = _encs(fmt)
    st = RandomFeatures(1)
    imgs = [e.image for e in encs[:3]] + [None]
    b = collate(encs, imgs, st.get, 196, fmt.pad_id, side, True)
    for r, e in enumerate(encs):
        n_real = len(e.prefix) + (196 if imgs[r] else 0) + len(e.suffix) + len(e.target)
        assert int(b.attn[r].sum()) == n_real and int(b.vis[r].sum()) == (196 if imgs[r] else 0)
        real = b.attn[r].bool()
        assert b.pos[r][real].tolist() == list(range(n_real))
        lab = b.labels[r][b.labels[r] != -100].tolist()
        assert lab == e.target and int(b.term[r].sum()) == 1 and b.labels[r][b.term[r]].item() == fmt.im_end
        vis_idx = b.vis[r].nonzero().flatten().tolist()
        if vis_idx:
            off = real.nonzero().flatten()[0].item()
            assert vis_idx == list(range(off + len(e.prefix), off + len(e.prefix) + 196))
        assert not bool((b.vis[r] & (b.labels[r] != -100)).any())
    assert b.feats.shape == (3, 196, 768)
    # no-image row: identical text tokens, visual slots removed
    b2 = collate(encs[:1], [None], st.get, 196, fmt.pad_id, side, True)
    full = b.ids[0][b.attn[0].bool() & ~b.vis[0]].tolist()
    assert b2.ids[0][b2.attn[0].bool()].tolist() == full


def test_pool2x2():
    x = torch.arange(196, dtype=torch.float32).view(1, 196, 1).repeat(1, 1, 3)
    y = pool2x2(x)
    assert y.shape == (1, 49, 3)
    assert y[0, 0, 0].item() == pytest.approx((0 + 1 + 14 + 15) / 4)
    assert y[0, 8, 0].item() == pytest.approx((30 + 31 + 44 + 45) / 4)   # block (1,1)


def test_splice(fmt, vlm):
    encs = _encs(fmt)[:2]
    st = RandomFeatures(1)
    b = collate(encs, [e.image for e in encs], st.get, 196, fmt.pad_id, "right", True)
    with torch.no_grad():
        emb = vlm.embed(b)
        E = vlm.lm.get_input_embeddings()(b.ids)
        V = vlm.connector(b.feats)
    assert torch.equal(emb[b.vis], V.reshape(-1, V.shape[-1]))
    assert torch.equal(emb[~b.vis], E[~b.vis])
    assert torch.allclose(V, 0.37 * vlm.connector.raw(b.feats))


def _reference_loss(vlm, b):
    """Full logits from the HF causal LM, standard shift, per-example token mean, then batch mean."""
    with torch.no_grad():
        logits = vlm.lm(inputs_embeds=vlm.embed(b), attention_mask=b.attn, position_ids=b.pos).logits.float()
    per = []
    for r in range(b.ids.shape[0]):
        lab = b.labels[r, 1:]
        m = lab != -100
        per.append(F.cross_entropy(logits[r, :-1][m], lab[m]))
    return torch.stack(per)


def test_loss_matches_reference_and_padding_invariance(fmt, vlm):
    from pa3vlm.models.vlm import per_example_mean

    encs = _encs(fmt)
    st = RandomFeatures(1)
    imgs = [e.image for e in encs[:3]] + [None]
    b = collate(encs, imgs, st.get, 196, fmt.pad_id, "right", True)
    with torch.no_grad():
        nll, rows, term = vlm.supervised_nll(b)
    per = per_example_mean(nll, rows, 4)
    assert torch.allclose(per, _reference_loss(vlm, b), atol=1e-5)
    # example weighting: each example counts equally regardless of its number of target tokens
    assert torch.allclose(per.mean(), _reference_loss(vlm, b).mean(), atol=1e-6)
    ans = per_example_mean(nll, rows, 4, ~term)
    for r, e in enumerate(encs):          # alone (no padding) == inside a padded batch
        b1 = collate([e], [imgs[r]], st.get, 196, fmt.pad_id, "right", True)
        with torch.no_grad():
            n1, r1, t1 = vlm.supervised_nll(b1)
        assert torch.allclose(per_example_mean(n1, r1, 1), per[r:r + 1], atol=1e-5)
        assert torch.allclose(per_example_mean(n1, r1, 1, ~t1), ans[r:r + 1], atol=1e-5)


def test_no_image_positions_rebuilt(fmt, vlm):
    e = _encs(fmt)[0]
    b = collate([e], [None], RandomFeatures(1).get, 196, fmt.pad_id, "right", True)
    ids = torch.tensor([e.prefix + e.suffix + e.target])
    with torch.no_grad():
        ref = vlm.lm(input_ids=ids).logits            # HF default positions 0..n-1
        got = vlm.lm(inputs_embeds=vlm.embed(b), attention_mask=b.attn, position_ids=b.pos).logits
    assert torch.allclose(ref, got, atol=1e-5)


def test_gradients_reach_connector_only(fmt, vlm):
    encs = _encs(fmt)[:2]
    b = collate(encs, [e.image for e in encs], RandomFeatures(1).get, 196, fmt.pad_id, "right", True)
    vlm.train()
    try:
        assert vlm.lm.model.gradient_checkpointing
        nll, rows, _ = vlm.supervised_nll(b)
        nll.mean().backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in vlm.connector.parameters())
        assert all(p.grad is None and not p.requires_grad for p in vlm.lm.parameters())
    finally:
        vlm.zero_grad(set_to_none=True)
        vlm.eval()


def test_lora_boundary_and_param_counts(fmt, tiny_dir):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from pa3common.lora import inject_lora, lora_modules
    from pa3common.utils import count_params
    from pa3vlm.models.connector import Connector, make_connector
    from pa3vlm.models.vlm import VLM, load_lm

    lm = load_lm(str(tiny_dir), torch.float32, "sdpa", grad_ckpt=False)
    v = VLM(lm, make_connector(6304, d_out=64)).eval()
    encs = _encs(fmt)
    b = collate(encs, [e.image for e in encs], RandomFeatures(1).get, 196, fmt.pad_id, "right", True)
    with torch.no_grad():
        before = v.lm(inputs_embeds=v.embed(b), attention_mask=b.attn, position_ids=b.pos).logits
    inject_lora(v.lm, 16, 32, 0.05, 6304, ("q_proj", "k_proj", "v_proj", "o_proj"))
    assert len(lora_modules(v.lm)) == 4 * 28
    with torch.no_grad():
        after = v.lm(inputs_embeds=v.embed(b), attention_mask=b.attn, position_ids=b.pos).logits
    assert torch.equal(before, after), "B = 0 at injection: function unchanged at the stage boundary"
    names = {n for n, p in v.named_parameters() if p.requires_grad}
    assert all(n.startswith("connector.") or n.endswith(("lora_A", "lora_B")) for n in names)
    # real Qwen3-0.6B dimensions (meta device, no download)
    assert count_params(Connector(), True) == 1_837_056
    cfg = Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
                      num_attention_heads=16, num_key_value_heads=8, head_dim=128, tie_word_embeddings=True)
    with torch.device("meta"):
        big = Qwen3ForCausalLM(cfg)
    for p in big.parameters():
        p.requires_grad_(False)
    inject_lora(big, 16, 32, 0.05, 6304, ("q_proj", "k_proj", "v_proj", "o_proj"))
    assert count_params(big, True) == 4_587_520


def test_greedy_matches_hf_generate(fmt, vlm):
    from pa3vlm.eval.generate import greedy

    encs = _encs(fmt)
    st = RandomFeatures(1)
    imgs = [e.image for e in encs[:3]] + [None]
    b = collate(encs, imgs, st.get, 196, fmt.pad_id, "left", False)
    seqs, stopped = greedy(vlm, b, 16, fmt.eos_ids, fmt.pad_id)
    for r, e in enumerate(encs):
        b1 = collate([e], [imgs[r]], st.get, 196, fmt.pad_id, "left", False)
        with torch.no_grad():
            emb = vlm.embed(b1)
            out = vlm.lm.generate(inputs_embeds=emb, attention_mask=b1.attn, max_new_tokens=16, do_sample=False,
                                  eos_token_id=fmt.eos_ids, pad_token_id=fmt.pad_id, use_cache=True)
        ref = [t for t in out[0].tolist()]
        cut = next((i for i, t in enumerate(ref) if t in fmt.eos_ids), len(ref))
        assert seqs[r] == ref[:cut], (r, seqs[r], ref)
        assert stopped[r] == (cut < len(ref))
        s1, _ = greedy(vlm, b1, 16, fmt.eos_ids, fmt.pad_id)
        assert s1[0] == seqs[r]
