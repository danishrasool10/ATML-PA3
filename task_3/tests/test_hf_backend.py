"""The real HF generation path on CPU with a tiny random Qwen2 model + an offline word-level tokenizer:
transformers accepts the sampling kwargs, hub generation defaults are overridden, seeding is reproducible, and
token accounting / truncation / termination flags are consistent with the generated ids."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

CHAT = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
LIVE = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0, "dtype": "float32"}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    d = tmp_path_factory.mktemp("tiny")
    words = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "[UNK]", "system", "user", "assistant", "\\boxed{2}",
             "the", "answer", "is", "Hello", "You", "are", "a", "helpful", "assistant.", "Problem", "Think"]
    words += [f"w{i}" for i in range(60)]
    vocab = {w: i for i, w in enumerate(words)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", eos_token="<|im_end|>",
                                  pad_token="<|endoftext|>", additional_special_tokens=["<|im_start|>"])
    tok.chat_template = CHAT
    tok.save_pretrained(d)
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
                      eos_token_id=[vocab["<|im_end|>"], vocab["<|endoftext|>"]], pad_token_id=vocab["<|endoftext|>"])
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg)
    m.generation_config = GenerationConfig(do_sample=True, temperature=0.3, top_p=0.5, top_k=5,
                                           repetition_penalty=1.05, eos_token_id=cfg.eos_token_id)
    m.save_pretrained(d)
    return d, vocab


def test_hf_backend_generation(tiny):
    from pa3ttc.live.backend import HFBackend

    d, vocab = tiny
    b = HFBackend(str(d), None, LIVE, device="cpu")
    gc = b.model.generation_config
    assert (gc.repetition_penalty, gc.temperature, gc.top_p, gc.top_k, gc.do_sample) == (1.0, 0.7, 0.8, 20, True)
    assert vocab["<|im_end|>"] in b.eos_ids
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Problem w1 w2\n\nThink"}]
    assert b.render(msgs).endswith("<|im_start|>assistant\n")
    r1 = b.generate(msgs, 12, seed=11)
    r2 = b.generate(msgs, 12, seed=11)
    assert r1["response"] == r2["response"] and r1["output_tokens"] == r2["output_tokens"]
    outs = {b.generate(msgs, 12, seed=s)["response"] for s in range(6)}
    assert len(outs) > 1                                              # seeds matter
    seen_trunc = False
    for s in range(30):
        r = b.generate(msgs, 6, seed=s)
        assert r["input_tokens"] == len(b.tok(r["prompt_text"], add_special_tokens=False)["input_ids"])
        assert 0 <= r["output_tokens_retained"] <= r["output_tokens"] <= 6
        if r["terminated"]:
            assert r["output_tokens_retained"] < r["output_tokens"] and not r["truncated"]
        else:
            seen_trunc |= r["truncated"]
            assert r["output_tokens"] == 6 and r["truncated"] and r["output_tokens_retained"] == 6
        assert r["seconds"] > 0 and r["error"] is None
    assert seen_trunc
    # review prompt: the earlier response is placed as an assistant turn before the review instruction
    from pa3ttc.live.run import messages_review
    lc = {"system": "You are a helpful assistant.", "instruction": "Think", "review": "Review"}
    rv = b.render(messages_review("Problem", "the answer is \\boxed{2}", lc))
    assert "<|im_start|>assistant\nthe answer is \\boxed{2}<|im_end|>\n<|im_start|>user\nReview<|im_end|>" in rv
