# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness tests for the QSA sparse paged attention Triton kernel.

The fp8 tests force the software e4m3 dequant path (SW_FP8, used below
sm_89 where Triton cannot lower fp8e4nv) so they run on any CUDA GPU.
"""

import pytest
import torch
import triton
import triton.language as tl

import vllm.models.qwen4_exp.nvidia.ops.qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    _e4m3_bytes_to_f32,
    qsa_sparse_paged_attention,
    warmup_qsa_sparse_paged_attention,
)

pytestmark = pytest.mark.skip_global_cleanup

KV_HEADS = 2
GROUP = 4
NUM_QUERY_HEADS = KV_HEADS * GROUP
HEAD_DIM = 128
PAGE_SIZE = 32
PAGE_TABLE_WIDTH = 32
CAPACITY = PAGE_TABLE_WIDTH * PAGE_SIZE
NUM_BLOCKS = 64


@triton.jit
def _apply_e4m3_dequant(raw_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, _e4m3_bytes_to_f32(tl.load(raw_ptr + offs)))


def test_e4m3_software_dequant_bit_exact():
    """SW dequant of every int8 code matches torch's e4m3 cast."""
    codes = (
        torch.arange(256, device="cuda", dtype=torch.int32)
        .to(torch.uint8)
        .view(torch.int8)
    )
    got = torch.empty(256, device="cuda", dtype=torch.float32)
    _apply_e4m3_dequant[(1,)](codes, got, BLOCK=256)
    expected = codes.view(torch.uint8).view(torch.float8_e4m3fn).float()
    finite = torch.isfinite(expected)
    assert torch.equal(got[finite], expected[finite])


def _make_inputs(rows, selection_width, fp8):
    device = "cuda"
    shape = (rows, NUM_QUERY_HEADS, HEAD_DIM)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16)
    gate = torch.randn(shape, device=device, dtype=torch.bfloat16)
    cache_shape = (NUM_BLOCKS, PAGE_SIZE, KV_HEADS, HEAD_DIM)
    if fp8:
        k_cache = torch.randn(cache_shape, device=device).to(torch.float8_e4m3fn)
        v_cache = torch.randn(cache_shape, device=device).to(torch.float8_e4m3fn)
    else:
        k_cache = torch.randn(cache_shape, device=device, dtype=torch.bfloat16)
        v_cache = torch.randn(cache_shape, device=device, dtype=torch.bfloat16)
    num_reqs = max(1, min(5, rows))
    block_table = (
        torch.arange(PAGE_TABLE_WIDTH, device=device, dtype=torch.int32)[:NUM_BLOCKS]
        .reshape(1, -1)
        .expand(num_reqs, -1)
        .contiguous()
    )
    token_to_req = torch.arange(rows, device=device, dtype=torch.int32) % num_reqs
    indices = torch.randint(
        0, CAPACITY, (rows, selection_width + 1), device=device, dtype=torch.int32
    )
    count = torch.randint(
        1, selection_width + 1, (rows, 1), device=device, dtype=torch.int32
    )
    if rows >= 3:
        count[0] = 0
    columns = torch.arange(selection_width, device=device, dtype=torch.int32)
    indices[:, :-1] = torch.where(
        columns < count, indices[:, :-1], torch.tensor(-1, device=device)
    )
    indices[:, -1] = count[:, 0]
    return q, gate, k_cache, v_cache, block_table, token_to_req, indices


def _reference(
    q, gate, k_cache, v_cache, indices, block_table, token_to_req, k_scale, v_scale
):
    selection_width = indices.shape[1] - 1
    kd, vd = k_cache.float(), v_cache.float()
    ref = torch.zeros(*q.shape, device=q.device, dtype=torch.float32)
    for r in range(q.shape[0]):
        toks = indices[r, :selection_width].long()
        toks = toks[toks >= 0]
        if toks.numel() == 0:
            continue
        pages, offsets = toks // PAGE_SIZE, toks % PAGE_SIZE
        phys = block_table[int(token_to_req[r]), pages]
        for h in range(NUM_QUERY_HEADS):
            kv_head = h // GROUP
            scores = (q[r, h].float() * k_scale * HEAD_DIM**-0.5) @ kd[
                phys, offsets, kv_head
            ].T
            probs = torch.softmax(scores, dim=0)
            out = probs @ vd[phys, offsets, kv_head] * v_scale
            ref[r, h] = out * torch.sigmoid(gate[r, h].float())
    return ref


@pytest.mark.parametrize(
    "rows,selection_width",
    [
        (3, 1024),  # split-K over many tiles, near-empty requests
        (1, 1024),  # single request, split-K
        (33, 256),  # moderate split-K
        (257, 64),  # prefill config, single split
    ],
)
@torch.inference_mode()
def test_qsa_sparse_paged_attention_fp8_software_dequant(
    monkeypatch, rows, selection_width
):
    """FP8-e4m3 KV through the below-sm_89 software dequant path."""
    monkeypatch.setattr(qsa_ops, "_supports_native_fp8", lambda: False)
    torch.manual_seed(0)
    k_scale, v_scale = 0.7, 1.3
    q, gate, k_cache, v_cache, block_table, token_to_req, indices = _make_inputs(
        rows, selection_width, fp8=True
    )
    out = qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        indices,
        block_table,
        token_to_req,
        rows > 128,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=gate,
    )
    ref = _reference(
        q, gate, k_cache, v_cache, indices, block_table, token_to_req, k_scale, v_scale
    )
    max_abs_err = (out.float() - ref).abs().max().item()
    assert max_abs_err < 0.15, f"max_abs_err={max_abs_err}"


@torch.inference_mode()
def test_qsa_sparse_paged_attention_bf16():
    """BF16 caches still match the reference through the shared loads."""
    torch.manual_seed(0)
    rows, selection_width = 3, 1024
    q, gate, k_cache, v_cache, block_table, token_to_req, indices = _make_inputs(
        rows, selection_width, fp8=False
    )
    out = qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        indices,
        block_table,
        token_to_req,
        False,
        output_gate=gate,
    )
    ref = _reference(
        q, gate, k_cache, v_cache, indices, block_table, token_to_req, 1.0, 1.0
    )
    max_abs_err = (out.float() - ref).abs().max().item()
    assert max_abs_err < 0.15, f"max_abs_err={max_abs_err}"


@torch.inference_mode()
def test_qsa_sparse_paged_attention_warmup_compiles_fp8():
    """Warmup compiles every reachable fp8 specialization without raising."""
    kv_cache = torch.randint(
        0,
        256,
        (16, KV_HEADS, PAGE_SIZE, 2 * HEAD_DIM),
        device="cuda",
        dtype=torch.uint8,
    )
    block_table = torch.arange(16, device="cuda", dtype=torch.int32).reshape(1, -1)
    configs = warmup_qsa_sparse_paged_attention(
        kv_cache, block_table, num_query_heads=NUM_QUERY_HEADS, selection_width=64
    )
    assert configs
