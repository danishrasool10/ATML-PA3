"""Cache the *starting* model's down_proj inputs z at the last prompt token (FP32).

Probe prompts are built with lm-eval's own task objects, exactly as in the benchmarks:
  * GSM8K (gsm8k_cot, 0-shot): chat template, enable_thinking=False, generation prompt
    -> the last prompt token is the final token of "<think>\n\n</think>\n\n".
  * HellaSwag: stock completion context (no chat template); trailing whitespace is moved to
    the continuation as lm-eval does, so the last token precedes any answer continuation.
Each prompt is run unpadded (batch 1), so "last non-padding token" is simply position -1.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

LAYERS = (0, 13, 27)


def _chat_fn(tok):
    def f(chat_history, add_generation_prompt: bool = True):
        return tok.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
                                       continue_final_message=not add_generation_prompt, enable_thinking=False)
    return f


def build_probe_prompts(tok, probe: dict) -> list[dict]:
    from lm_eval.tasks import TaskManager, get_task_dict

    td = get_task_dict(["gsm8k_cot", "hellaswag"], TaskManager())
    out = []
    specs = [("gsm8k_cot", "math", True, probe["gsm8k_cot_test_doc_ids"]),
             ("hellaswag", "nonmath", False, probe["hellaswag_validation_doc_ids"])]
    for task_name, group, chat, ids in specs:
        task = td[task_name]
        task.set_config(key="num_fewshot", value=0)
        docs = list(task.eval_docs)
        for i in ids:
            doc = docs[i]
            ctx = task.fewshot_context(doc, num_fewshot=0, system_instruction=None, apply_chat_template=chat,
                                       fewshot_as_multiturn=True, chat_template=_chat_fn(tok) if chat else None,
                                       gen_prefix=task.doc_to_prefix(doc))
            if not chat:
                ctx = ctx.rstrip()  # lm-eval _encode_pair moves trailing whitespace to the continuation
            ids_ = tok(ctx, add_special_tokens=False)["input_ids"]
            out.append({"task": task_name, "group": group, "doc_id": int(i), "context": ctx,
                        "input_ids": ids_, "last_token": tok.decode(ids_[-1:])})
    return out


@torch.no_grad()
def cache_down_proj_inputs(model: nn.Module, prompts: list[dict], device: torch.device,
                           layers: tuple[int, ...] = LAYERS) -> torch.Tensor:
    """Return z of shape (n_prompts, n_layers, d_ff) in FP32."""
    model = model.to(device=device, dtype=torch.float32).eval()
    store: dict[int, torch.Tensor] = {}
    hooks = []
    def make_hook(li: int):
        def hook(module, inputs, output):
            store[li] = inputs[0][0, -1].detach().float().cpu()  # (d_ff,) at the last prompt token
        return hook

    for li in layers:
        hooks.append(model.model.layers[li].mlp.down_proj.register_forward_hook(make_hook(li)))
    Z = []
    try:
        for p in prompts:
            ids = torch.tensor([p["input_ids"]], device=device)
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            Z.append(torch.stack([store[li] for li in layers]))
    finally:
        for h in hooks:
            h.remove()
    return torch.stack(Z)


def save_cache(path: str | Path, Z: torch.Tensor, prompts: list[dict], layers=LAYERS) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    meta = [{k: v for k, v in p.items() if k != "input_ids"} | {"n_tokens": len(p["input_ids"])} for p in prompts]
    torch.save({"Z": Z, "prompts": meta, "layers": list(layers)}, path)
