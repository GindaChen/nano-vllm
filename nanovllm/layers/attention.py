import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def paged_decode_attn_kernel(
    q_ptr,             # [B, H_q, D] bfloat16
    k_cache_ptr,       # [num_blocks, S, H_kv, D] bfloat16
    v_cache_ptr,       # [num_blocks, S, H_kv, D] bfloat16
    block_tables_ptr,  # [B, MAX_BLOCKS] int32
    context_lens_ptr,  # [B] int32
    out_ptr,           # [B, H_q, D] bfloat16
    stride_q_b, stride_q_h,
    stride_o_b, stride_o_h,
    stride_bt_b,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    kv_h = h // GQA_RATIO

    ctx_len = tl.load(context_lens_ptr + b)
    num_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + b * stride_q_b + h * stride_q_h + offs_d).to(tl.float32) * scale

    m_i = float('-inf')
    l_i = 0.0
    o_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    stride_kv_block = BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM
    stride_kv_s = NUM_KV_HEADS * HEAD_DIM
    kv_head_off = kv_h * HEAD_DIM

    for blk in range(MAX_BLOCKS):
        blk_valid = blk < num_blocks
        block_id = tl.load(block_tables_ptr + b * stride_bt_b + blk,
                           mask=blk_valid, other=0)
        blk_start = blk * BLOCK_SIZE
        tokens_in_blk = tl.where(
            blk_start + BLOCK_SIZE <= ctx_len,
            BLOCK_SIZE,
            ctx_len - blk_start,
        )
        tokens_in_blk = tl.where(blk_valid, tokens_in_blk, 0)

        kv_blk_base = block_id * stride_kv_block + kv_head_off

        for s0 in range(0, BLOCK_SIZE, BLOCK_TOKENS):
            s_offs = s0 + tl.arange(0, BLOCK_TOKENS)
            tok_mask = s_offs < tokens_in_blk

            k_ptrs = k_cache_ptr + kv_blk_base + s_offs[:, None] * stride_kv_s + offs_d[None, :]
            k = tl.load(k_ptrs, mask=tok_mask[:, None], other=0.0).to(tl.float32)

            scores = tl.sum(q[None, :] * k, axis=1)
            scores = tl.where(tok_mask, scores, float('-inf'))

            v_ptrs = v_cache_ptr + kv_blk_base + s_offs[:, None] * stride_kv_s + offs_d[None, :]
            v = tl.load(v_ptrs, mask=tok_mask[:, None], other=0.0).to(tl.float32)

            m_j = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_j)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            p = tl.where(tok_mask, p, 0.0)

            l_i = l_i * alpha + tl.sum(p, axis=0)
            o_acc = o_acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

    out = (o_acc / l_i).to(tl.bfloat16)
    tl.store(out_ptr + b * stride_o_b + h * stride_o_h + offs_d, out)


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    B, H_q, D = q.shape
    _, S, H_kv, _ = k_cache.shape
    MAX_BLOCKS = block_tables.shape[1]
    GQA_RATIO = H_q // H_kv

    out = torch.empty_like(q)
    grid = (B, H_q)
    paged_decode_attn_kernel[grid](
        q, k_cache, v_cache, block_tables, context_lens, out,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        block_tables.stride(0),
        scale,
        HEAD_DIM=D,
        BLOCK_SIZE=S,
        NUM_KV_HEADS=H_kv,
        GQA_RATIO=GQA_RATIO,
        MAX_BLOCKS=MAX_BLOCKS,
        BLOCK_TOKENS=64,
        num_warps=4,
    )
    return out


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = paged_decode_attention(q, k_cache, v_cache,
                                       context.block_tables, context.context_lens,
                                       self.scale)
        return o
