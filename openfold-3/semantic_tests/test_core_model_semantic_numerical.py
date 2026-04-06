import importlib.util

import pytest

torch = pytest.importorskip("torch")

from openfold3.core.model.layers.attention_pair_bias import AttentionPairBias
from openfold3.core.model.layers.triangular_attention import TriangleAttention
from openfold3.core.model.primitives.attention import (
    Attention,
    _attention,
    softmax_no_cast,
)


def _seed() -> None:
    torch.manual_seed(7)


def test_attention_baseline_matches_explicit_path() -> None:
    _seed()
    attn = Attention(c_q=16, c_k=16, c_v=16, c_hidden=8, no_heads=2, gating=True)
    q_x = torch.randn(2, 5, 16, dtype=torch.float32)
    kv_x = torch.randn(2, 5, 16, dtype=torch.float32)
    bias = torch.randn(2, 1, 5, 5, dtype=torch.float32) * 0.1

    out_forward = attn(q_x=q_x, kv_x=kv_x, biases=[bias])

    q, k, v = attn._prep_qkv(q_x=q_x, kv_x=kv_x, apply_scale=True)
    out_explicit = attn._wrap_up(o=_attention(q, k, v, [bias]).transpose(-2, -3), q_x=q_x)

    max_diff = (out_forward - out_explicit).abs().max().item()
    mean_diff = (out_forward - out_explicit).abs().mean().item()
    assert max_diff < 1e-6, f"max_diff={max_diff}, mean_diff={mean_diff}"


def test_attention_pair_bias_masking_invariance() -> None:
    _seed()
    mod = AttentionPairBias(
        c_q=12,
        c_k=12,
        c_v=12,
        c_s=12,
        c_z=8,
        c_hidden=6,
        no_heads=2,
        use_ada_layer_norm=False,
    )

    a = torch.randn(1, 6, 12, dtype=torch.float32)
    z = torch.randn(1, 6, 6, 8, dtype=torch.float32)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.float32)

    out_ref = mod(a=a, z=z, mask=mask)

    a_perturbed = a.clone()
    a_perturbed[:, 4:, :] = torch.randn_like(a_perturbed[:, 4:, :]) * 1000.0
    out_perturbed = mod(a=a_perturbed, z=z, mask=mask)

    unmasked = mask.bool().unsqueeze(-1).expand_as(out_ref)
    diff = (out_ref - out_perturbed).abs()[unmasked]
    assert diff.max().item() < 1e-4


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_triangle_attention_chunking_equivalence(chunk_size: int) -> None:
    _seed()
    mod = TriangleAttention(c_in=10, c_hidden=5, no_heads=2, starting=True)
    x = torch.randn(2, 5, 5, 10, dtype=torch.float32)
    mask = torch.randint(0, 2, (2, 5, 5), dtype=torch.float32)

    out_full = mod(x=x, mask=mask, chunk_size=None)
    out_chunk = mod(x=x, mask=mask, chunk_size=chunk_size)

    max_diff = (out_full - out_chunk).abs().max().item()
    mean_diff = (out_full - out_chunk).abs().mean().item()
    assert max_diff < 1e-5, f"chunk={chunk_size}, max={max_diff}, mean={mean_diff}"


def test_softmax_numerical_stability_large_logits() -> None:
    logits = torch.tensor([[1000.0, 999.0, -1000.0, -1200.0]], dtype=torch.float32)
    probs = softmax_no_cast(logits, dim=-1)

    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert probs[0, 0] > probs[0, 1] > probs[0, 2] > probs[0, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend unavailable")
def test_cuda_attention_smoke() -> None:
    _seed()
    attn = Attention(c_q=16, c_k=16, c_v=16, c_hidden=8, no_heads=2).cuda()
    q_x = torch.randn(1, 8, 16, device="cuda")
    kv_x = torch.randn(1, 8, 16, device="cuda")
    out = attn(q_x=q_x, kv_x=kv_x)
    assert out.is_cuda
    assert torch.isfinite(out).all()


@pytest.mark.skipif(
    importlib.util.find_spec("deepspeed") is None or not torch.cuda.is_available(),
    reason="DeepSpeed or CUDA backend unavailable",
)
def test_deepspeed_attention_smoke() -> None:
    _seed()
    attn = Attention(c_q=16, c_k=16, c_v=16, c_hidden=8, no_heads=2).cuda()
    q_x = torch.randn(1, 32, 16, device="cuda")
    kv_x = torch.randn(1, 32, 16, device="cuda")
    out = attn(q_x=q_x, kv_x=kv_x, use_deepspeed_evo_attention=True)
    assert out.is_cuda
    assert torch.isfinite(out).all()
