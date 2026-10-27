"""Chat formatting shared by training, qualitative generation and sanity checks.

Training sequence for one MetaMathQA example (thinking disabled):

    prompt   = apply_chat_template([user], add_generation_prompt=True, enable_thinking=False)
             = "<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    response = "{response}" + "<|im_end|>"          # end-of-turn token is supervised
    ids      = tok(prompt) ++ tok(response) ++ [<|im_end|>]

Prompt and response are tokenised separately and concatenated, so the supervised prefix is
token-identical to what the model sees at inference time (generation prompt).  The trailing
"\n" that the chat template emits after <|im_end|> is not part of the sequence.
"""
from __future__ import annotations

from typing import Any

END_OF_TURN = "<|im_end|>"


def render_prompt(tokenizer: Any, user: str, system: str | None = None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def eot_id(tokenizer: Any) -> int:
    i = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    assert isinstance(i, int) and i >= 0, "tokenizer lacks <|im_end|>"
    return i


def encode_example(tokenizer: Any, query: str, response: str) -> tuple[list[int], list[int]]:
    """Return (prompt_ids, response_ids_including_eot)."""
    p = tokenizer(render_prompt(tokenizer, query), add_special_tokens=False)["input_ids"]
    r = tokenizer(response, add_special_tokens=False)["input_ids"] + [eot_id(tokenizer)]
    return p, r


def encode_batch(tokenizer: Any, queries: list[str], responses: list[str]):
    prompts = [render_prompt(tokenizer, q) for q in queries]
    P = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    R = tokenizer(list(responses), add_special_tokens=False)["input_ids"]
    e = eot_id(tokenizer)
    return P, [r + [e] for r in R]
