"""Prompt construction: Qwen3 chat template with a visual slot.

Each example is rendered with `apply_chat_template([{"role": "user", "content": SENTINEL + "\n" + text}],
add_generation_prompt=True, enable_thinking=False)` and split at the sentinel:

    prefix  = "<|im_start|>user\n"                        (token ids, identical for all examples)
    [visual] = 196 (or 49) connector outputs                (inserted between prefix and suffix)
    suffix  = "\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    target  = content tokens (capped) + <|im_end|>          (supervised)

Prefix and suffix are tokenised separately (nothing can merge across the image).  The no-image control
uses exactly the same prefix + suffix ids with the visual rows removed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CAPTION_PROMPT = "Write a short caption for this image."
ANSWER_INSTRUCTION = "Answer the question using a single word or number."
SENTINEL = "⟦IMAGE⟧"


def user_text(question: str | None) -> str:
    return CAPTION_PROMPT if question is None else f"{question}\n{ANSWER_INSTRUCTION}"


@dataclass
class Encoded:
    ex_id: str
    image: str | None              # feature key, e.g. "train2014:123"
    prefix: list[int]
    suffix: list[int]
    target: list[int]              # content tokens + <|im_end|> (empty for prompts only)
    n_term: int = 1                # trailing termination tokens inside `target` (excluded from answer NLL)
    meta: dict = field(default_factory=dict)


class ChatFormat:
    def __init__(self, tok, caps: dict):
        self.tok = tok
        self.caps = caps
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.eos_ids = sorted({self.im_end, tok.convert_tokens_to_ids("<|endoftext|>")})
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
        pre, _ = self._render("x")
        self.prefix = tok(pre, add_special_tokens=False).input_ids
        self.stats = {"question_truncated": 0, "target_truncated": 0}
        self._suffix_cache: dict[str, list[int]] = {}

    def _render(self, text: str) -> tuple[str, str]:
        s = self.tok.apply_chat_template([{"role": "user", "content": SENTINEL + "\n" + text}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        assert s.count(SENTINEL) == 1, "chat template altered the sentinel"
        pre, post = s.split(SENTINEL)
        return pre, post

    def _cap(self, text: str, cap: int, key: str) -> tuple[list[int], str]:
        ids = self.tok(text, add_special_tokens=False).input_ids
        if len(ids) > cap:
            self.stats[key] += 1
            ids = ids[:cap]
            text = self.tok.decode(ids)
        return ids, text

    def encode(self, ex_id: str, image: str | None, question: str | None, target: str | None,
               meta: dict | None = None) -> Encoded:
        if question is not None:
            _, question = self._cap(question, self.caps["question"], "question_truncated")
        text = user_text(question)
        suffix = self._suffix_cache.get(text)
        if suffix is None:
            pre, post = self._render(text)
            assert self.tok(pre, add_special_tokens=False).input_ids == self.prefix, "prefix must not vary"
            suffix = self._suffix_cache[text] = self.tok(post, add_special_tokens=False).input_ids
        prefix = self.prefix
        tgt: list[int] = []
        if target is not None:
            cap = self.caps["caption"] if question is None else self.caps["answer"]
            tgt, _ = self._cap(target, cap, "target_truncated")
            tgt = tgt + [self.im_end]
        return Encoded(ex_id, image, prefix, suffix, tgt, 1, meta or {})


def encode_caption(fmt: ChatFormat, row: dict) -> Encoded:
    return fmt.encode(row["ex_id"], row["image"], None, row["caption"], {"kind": "caption"})


def encode_pair(fmt: ChatFormat, pair: dict) -> list[Encoded]:
    out = []
    for e in pair["examples"]:
        out.append(fmt.encode(e["ex_id"], e["image"], pair["question"], e["target"],
                              {"kind": pair["kind"], "pair_id": pair["pair_id"], "suite": pair["suite"]}))
    return out
