"""Generation backends.  HFBackend: Qwen2.5-3B-Instruct, FP16, no quantisation, batch size 1, KV cache.
FakeBackend: deterministic stand-in used by the CPU tests (same accounting code path).

Sampling: temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.0.  The model's own generation_config
(which ships repetition_penalty 1.05) is overwritten as well as passing explicit kwargs, so no hub default leaks
in; the effective config is recorded.  Each call is seeded (torch + CUDA) right before generation.  Timing is
end-to-end per call (chat templating, tokenisation, generation, decoding), with CUDA synchronisation; model
loading and one short warm-up call are excluded.
"""
from __future__ import annotations

import time


class HFBackend:
    def __init__(self, model_id: str, revision: str, live_cfg: dict, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        dtype = getattr(torch, live_cfg.get("dtype", "float16"))
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype).to(self.device)
        self.model.eval()
        self.sampling = {k: live_cfg[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty")}
        gc = self.model.generation_config
        for k, v in self.sampling.items():
            setattr(gc, k, v)
        gc.do_sample = True
        eos = gc.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        if self.tok.eos_token_id is not None:
            eos.append(self.tok.eos_token_id)
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0 and im_end != self.tok.unk_token_id:
            eos.append(im_end)
        self.eos_ids = sorted(set(eos))
        gc.eos_token_id = self.eos_ids
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_ids[0]
        self.info = {"model": model_id, "revision": revision,
                     "commit": getattr(self.model.config, "_commit_hash", None), "dtype": str(dtype),
                     "device": str(self.device), "eos_ids": self.eos_ids,
                     "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
                     "generation_config": gc.to_dict()}

    def render(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def warmup(self) -> None:
        self.generate([{"role": "user", "content": "Hello"}], 8, seed=0)

    def generate(self, messages: list[dict], max_new_tokens: int, seed: int) -> dict:
        torch = self.torch
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.perf_counter()
        prompt = self.render(messages)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        n_in = int(enc["input_ids"].shape[1])
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed_all(seed)
        err = None
        try:
            with torch.inference_mode():
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                          eos_token_id=self.eos_ids, pad_token_id=self.pad_id, **self.sampling)
            gen = out[0, n_in:].tolist()
        except RuntimeError as ex:      # e.g. FP16 inf/nan in the sampling distribution: recorded, not hidden
            if "inf" not in str(ex) and "nan" not in str(ex):
                raise
            gen, err = [], f"{type(ex).__name__}: {str(ex)[:300]}"
        stop = next((i for i, t in enumerate(gen) if t in self.eos_ids), None)
        retained = gen[:stop] if stop is not None else gen
        text = self.tok.decode(retained, skip_special_tokens=True)
        if cuda:
            torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()
        return {"prompt_text": prompt, "input_tokens": n_in, "output_tokens": len(gen),
                "output_tokens_retained": len(retained), "terminated": stop is not None,
                "truncated": stop is None and len(gen) >= max_new_tokens, "response": text,
                "seconds": t1 - t0, "error": err,
                "peak_gpu_mem_gib": torch.cuda.max_memory_allocated(self.device) / 2**30 if cuda else None}


class FakeBackend:
    """Deterministic stand-in for tests: the answer depends on (seed, prompt) so seeds matter; truncates when the
    'length' drawn from the seed exceeds the cap."""

    def __init__(self, live_cfg: dict, gold: dict | None = None):
        self.gold = gold or {}
        self.info = {"model": "fake", "revision": None, "sampling": {k: live_cfg[k] for k in
                                                                  ("temperature", "top_p", "top_k", "repetition_penalty")}}

    def render(self, messages):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"

    def warmup(self):
        pass

    def generate(self, messages, max_new_tokens, seed):
        import numpy as np

        prompt = self.render(messages)
        rng = np.random.default_rng(seed)
        length = int(rng.integers(50, 1500))
        q = next((k for k in self.gold if k in messages[1]["content"]), None)
        is_review = len(messages) > 2
        ans = self.gold.get(q, "0") if (rng.random() < (0.7 if is_review else 0.5)) else str(int(rng.integers(1, 9)))
        n_out = min(length, max_new_tokens)
        trunc = length > max_new_tokens
        text = "<think>work</think> " + ("partial" if trunc else f"\\boxed{{{ans}}}")
        return {"prompt_text": prompt, "input_tokens": len(prompt.split()), "output_tokens": n_out + (0 if trunc else 1),
                "output_tokens_retained": n_out, "terminated": not trunc, "truncated": trunc, "response": text,
                "seconds": 0.001 * n_out, "error": None, "peak_gpu_mem_gib": None}
