import math

import pytest
import torch

from pa3.config import TARGET_MODULES


def test_lora_init_and_peft_equivalence(tiny_dir):
    from peft import LoraConfig, get_peft_model

    from pa3.models.load import load_model
    from pa3.models.lora import adapter_modules, inject_adapters

    torch.manual_seed(1)
    ours = load_model(str(tiny_dir), torch.float32)
    inject_adapters(ours, 16, 32, 0.0, 6304, TARGET_MODULES)
    ref = get_peft_model(load_model(str(tiny_dir), torch.float32),
                         LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                                    target_modules=list(TARGET_MODULES), init_lora_weights=True))
    mods = adapter_modules(ours)
    assert len(mods) == 28 * 7
    n_peft = 0
    for name, m in mods.items():
        pm = ref.get_submodule("base_model.model." + name)
        pa, pb = pm.lora_A["default"].weight, pm.lora_B["default"].weight
        din = pa.shape[1]
        # same init distribution: U(-1/sqrt(din), 1/sqrt(din)) for A, zeros for B
        assert pa.abs().max() <= 1 / math.sqrt(din) + 1e-6 and m.lora_A.abs().max() <= 1 / math.sqrt(din) + 1e-6
        assert torch.all(pb == 0) and torch.all(m.lora_B == 0)
        with torch.no_grad():
            pa.copy_(m.lora_A)
            b = torch.randn_like(m.lora_B) * 0.05
            m.lora_B.copy_(b)
            pb.copy_(b)
        n_peft += pa.numel() + pb.numel()
    n_ours = sum(p.numel() for p in ours.parameters() if p.requires_grad)
    assert n_ours == n_peft  # identical trainable count
    ids = torch.randint(0, 1000, (2, 11))
    lo = ours(input_ids=ids).logits
    lr = ref(input_ids=ids).logits
    assert torch.allclose(lo, lr, atol=1e-5), (lo - lr).abs().max()
    lo.square().mean().backward()
    lr.square().mean().backward()
    for name, m in list(mods.items())[:20]:
        pm = ref.get_submodule("base_model.model." + name)
        assert torch.allclose(m.lora_B.grad, pm.lora_B["default"].weight.grad, atol=1e-6, rtol=1e-4)
        assert torch.allclose(m.lora_A.grad, pm.lora_A["default"].weight.grad, atol=1e-6, rtol=1e-4)


def test_real_model_param_counts():
    # Qwen3-0.6B: hidden 1024, q 2048, kv 1024, ffn 3072, 28 layers
    shapes = {"q_proj": (2048, 1024), "k_proj": (1024, 1024), "v_proj": (1024, 1024), "o_proj": (1024, 2048),
              "gate_proj": (3072, 1024), "up_proj": (3072, 1024), "down_proj": (1024, 3072)}
    per = lambda r: sum(r * (o + i) for o, i in shapes.values())
    assert 28 * per(16) == 10_092_544 and 28 * per(4) == 2_523_136
    basis = 28 * sum(128 * (o + i) for o, i in shapes.values()) * 4
    assert abs(basis / 2**20 - 308) < 1  # ~308 MiB FP32 (U_k + V_k)


def test_lora16_and_oplora_share_raw_init(tiny_dir, bases_rel, out_root):
    from pa3.models.load import load_model
    from pa3.models.lora import adapter_modules, inject_adapters
    from pa3.models.oplora import load_bases

    a = load_model(str(tiny_dir), torch.float32)
    b = load_model(str(tiny_dir), torch.float32)
    torch.manual_seed(123)  # global RNG must not matter
    inject_adapters(a, 16, 32, 0.05, 6304, TARGET_MODULES)
    inject_adapters(b, 16, 32, 0.05, 6304, TARGET_MODULES, load_bases(out_root / bases_rel))
    for (n, ma), (_, mb) in zip(adapter_modules(a).items(), adapter_modules(b).items()):
        assert torch.equal(ma.lora_A, mb.lora_A) and torch.equal(ma.lora_B, mb.lora_B)


def test_oplora_projection_constraints(tiny_dir, bases_rel, out_root):
    from pa3.models.load import load_model
    from pa3.models.lora import adapter_modules, inject_adapters
    from pa3.models.oplora import constraint_checks, load_bases

    bases = load_bases(out_root / bases_rel)
    m = load_model(str(tiny_dir), torch.float32)
    W0s = {n: mod.weight.detach().clone() for n, mod in m.named_modules() if n.endswith("down_proj")}
    inject_adapters(m, 16, 32, 0.0, 6304, TARGET_MODULES, bases)
    worst = {}
    for n, mod in adapter_modules(m).items():
        with torch.no_grad():
            mod.lora_B.normal_(0, 1.0)  # large update to stress the check
        if n.endswith("down_proj"):
            dW = mod.delta_weight()
            b = bases[n]
            c = constraint_checks(W0s[n], dW, b["U"], b["V"], b["S"])
            for k in ("left_leak", "right_leak", "prop2_right_resid", "prop2_left_resid"):
                worst[k] = max(worst.get(k, 0), c[k])
    assert all(v < 1e-5 for v in worst.values()), worst
    # unprojected LoRA would leak heavily into the protected subspace
    U = bases["model.layers.0.mlp.down_proj"]["U"]
    rnd = torch.randn(U.shape[0], 128)
    assert (U.T @ rnd).norm() / rnd.norm() > 0.1


def test_l2sp_reference(tiny_dir):
    from pa3.models.l2sp import L2SPReference
    from pa3.models.load import load_model
    from pa3.utils.common import count_params

    m = load_model(str(tiny_dir), torch.float32)
    ref = L2SPReference(m, torch.device("cpu"))
    assert ref.dtype == torch.bfloat16, "bf16 checkpoint must be stored exactly in bf16"
    assert ref.n_params == count_params(m), "tied embedding must be counted once"
    assert ref.omega_and_grad(m, 1e-3, add_grad=False) == 0.0
    theta0 = {n: p.detach().clone() for n, p in m.named_parameters()}
    torch.manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 1e-3)
            p.grad = torch.zeros_like(p)
    omega = ref.omega_and_grad(m, 1e-3, add_grad=True)
    direct = 0.5 * sum(float((p.double() - theta0[n].double()).square().sum()) for n, p in m.named_parameters())
    assert math.isclose(omega, direct, rel_tol=1e-9)
    for n, p in m.named_parameters():
        assert torch.allclose(p.grad, 1e-3 * (p.detach() - theta0[n]), atol=1e-12)


def test_weight_decay_groups(tiny_dir):
    from pa3.models.load import load_model
    from pa3.train.optim import param_groups

    m = load_model(str(tiny_dir), torch.float32)
    g = param_groups(m, 0.01)
    decay_ids = {id(p) for p in g[0]["params"]}
    for n, p in m.named_parameters():
        assert (id(p) in decay_ids) == (p.ndim >= 2), n
    assert id(m.model.embed_tokens.weight) in decay_ids


def test_cosine_matches_transformers():
    from transformers import get_cosine_schedule_with_warmup

    from pa3.train.optim import build_scheduler

    def lrs(make):
        p = torch.nn.Parameter(torch.zeros(1))
        o = torch.optim.AdamW([p], lr=2e-5)
        s = make(o)
        out = []
        for _ in range(1250):
            out.append(s.get_last_lr()[0])
            o.step()
            s.step()
        return out

    a = lrs(lambda o: build_scheduler(o, 38, 1250))
    b = lrs(lambda o: get_cosine_schedule_with_warmup(o, 38, 1250))
    assert a == b
    assert a[0] == 0.0 and math.isclose(a[38], 2e-5) and a[-1] < 1e-10


def test_chunked_loss_matches_dense(tiny_dir, out_root, tok):
    import torch.nn.functional as F

    from pa3.data.dataset import IGNORE, TokenStore, collate
    from pa3.models.load import load_model
    from pa3.train.loss import CausalLMTokenLoss

    m = load_model(str(tiny_dir), torch.float32)
    b = collate(TokenStore(out_root / "data"), [3, 4], tok.pad_token_id)
    s, n = CausalLMTokenLoss(m, chunk=7)(**b)
    logits = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
    dense = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), b["labels"][:, 1:].reshape(-1),
                            ignore_index=IGNORE, reduction="sum")
    assert torch.allclose(s, dense, rtol=1e-5) and int(n) == int((b["labels"] != IGNORE).sum())
