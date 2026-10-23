"""
End-to-End Empirical Verification & Cluster Break-Even Benchmark Suite for HyperCache.

Clearly separates:
  PART 1: LIVE MEASURED EXECUTION ON THIS MACHINE (Zero Mocks)
    - Multi-Layer Causal GQA Transformer + RoPE + Selective CacheBlend forward pass
      comparing Ground Truth vs. Naive KV Stitching vs. HyperCache on reordered RAG chunks.
    - Physical L1 (FP16) -> L2 (FP8) -> L3 (INT4 Disk `.hckv` + CRC32) tensor round-trips.
    - Live Binary TCP Socket Daemon (`HyperCacheTCPServer`) streaming quantized KV blocks
      with on-the-fly relative RoPE phase rotation.
  PART 2: MULTI-NODE CLUSTER BREAK-EVEN MODEL (400Gbps RoCEv2 / InfiniBand Physics)
    - Context-length scaling (16k to 128k tokens) comparing GPU Recompute, Host-RAM+TCP,
      and HyperCache GPUDirect RDMA with Layer-Wise Overlap.
"""

from __future__ import annotations

import numpy as np

from hypercache.core.attention_engine import verify_reordered_rag_attention
from hypercache.core.hasher import BlockHasher
from hypercache.core.quantizer import KVQuantizer, KVQuantMode, RoPEPhaseRotator
from hypercache.core.radix_dag import GlobalRadixDAG, MemoryTier
from hypercache.core.storage_engine import PhysicalTieredStorageEngine
from hypercache.transport.fabric import KernelBypassFabric, TransportBackend
from hypercache.transport.tcp_server import HyperCacheTCPClient, HyperCacheTCPServer


def run_break_even_benchmark(gpu_profile: str = "h100") -> str:
    # -------------------------------------------------------------------------
    # PART 1A: Real Multi-Layer Causal GQA Attention + RoPE + CacheBlend Proof
    # -------------------------------------------------------------------------
    attn_res = verify_reordered_rag_attention(recompute_prefix_fraction=0.15)

    # -------------------------------------------------------------------------
    # PART 1B: Real Physical L1 -> L2 -> L3 Disk & Live TCP Socket Transfer
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(7)
    k_raw = rng.standard_normal((1, 64, 8, 128), dtype=np.float32)
    v_raw = rng.standard_normal((1, 64, 8, 128), dtype=np.float32)

    rotator = RoPEPhaseRotator(head_dim=128)
    k_at_1024 = rotator.apply_absolute_rope(k_raw, start_pos=1024)
    k_at_8192_ground_truth = rotator.apply_absolute_rope(k_raw, start_pos=8192)
    k_shifted_1024_to_8192 = rotator.shift_positions(
        k_at_1024, old_start_pos=1024, new_start_pos=8192
    )
    max_rope_err = float(np.max(np.abs(k_at_8192_ground_truth - k_shifted_1024_to_8192)))

    fp8_buf = KVQuantizer.quantize(k_raw, v_raw, mode=KVQuantMode.FP8_E4M3)
    k_fp8, _ = KVQuantizer.dequantize(fp8_buf)
    cos_sim_fp8 = float(
        np.sum(k_raw * k_fp8)
        / (np.linalg.norm(k_raw) * np.linalg.norm(k_fp8) + 1e-12)
    )

    int4_buf = KVQuantizer.quantize(k_raw, v_raw, mode=KVQuantMode.INT4_GROUP)
    k_int4, _ = KVQuantizer.dequantize(int4_buf)
    cos_sim_int4 = float(
        np.sum(k_raw * k_int4)
        / (np.linalg.norm(k_raw) * np.linalg.norm(k_int4) + 1e-12)
    )

    # Physical L1 -> L2 -> L3 Disk spillover verification
    storage = PhysicalTieredStorageEngine(l1_max_blocks=1, l2_max_blocks=1)
    try:
        storage.put_block("blk-l3-target", k_raw, v_raw, cached_start_pos=512)
        storage.put_block("blk-l2-target", k_raw, v_raw, cached_start_pos=1024)
        storage.put_block("blk-l1-target", k_raw, v_raw, cached_start_pos=2048)
        l3_fetch = storage.get_block("blk-l3-target")
        assert l3_fetch is not None and l3_fetch.source_tier == MemoryTier.L3_NVME_SSD
        l3_disk_bytes = l3_fetch.wire_or_disk_bytes
    finally:
        storage.close()

    # Live Binary TCP Socket Daemon round-trip with on-the-fly RoPE shift
    server = HyperCacheTCPServer(host="127.0.0.1", port=0)
    server.start_background()
    try:
        client = HyperCacheTCPClient(
            host="127.0.0.1",
            port=server.bound_port,
            quant_mode=KVQuantMode.FP8_E4M3,
            head_dim=128,
        )
        client.put_kv_block("tcp-blk-01", k_at_1024, v_raw, cached_start_pos=1024)
        wire_res = client.get_kv_block("tcp-blk-01", new_start_pos=8192)
        assert wire_res.found and wire_res.k_tensor is not None
        tcp_cos_sim = float(
            np.sum(k_at_8192_ground_truth * wire_res.k_tensor)
            / (
                np.linalg.norm(k_at_8192_ground_truth)
                * np.linalg.norm(wire_res.k_tensor)
                + 1e-12
            )
        )
    finally:
        server.stop()

    # -------------------------------------------------------------------------
    # PART 2: Cluster Hardware Break-Even Scaling Table
    # -------------------------------------------------------------------------
    fabric = KernelBypassFabric(
        gpu_profile=gpu_profile,
        transport=TransportBackend.RDMA_ROCE_400G,
        quant_mode=KVQuantMode.FP8_E4M3,
        layer_overlap=True,
    )

    context_lengths = [16_384, 32_768, 65_536, 131_072]
    lines = [
        "",
        "==========================================================================================",
        "  PART 1: LIVE MEASURED NUMERICAL & SYSTEMS VERIFICATION (EXECUTED LOCALLY)",
        "==========================================================================================",
        "  A. Multi-Layer Causal GQA Attention + RoPE on Reordered RAG Chunks [Sys, DocB, DocA, Q]",
        f"     * Naive KV Stitching (No RoPE / No Recompute) : Cosine Sim = {attn_res.naive_stitch_cosine_sim:.4f} "
        f"(Rel L2 Err: {attn_res.naive_stitch_rel_l2_error:.3f} -> BROKEN)",
        f"     * Relative RoPE Phase Rotation Only (0% Rec.) : Cosine Sim = {attn_res.rope_only_cosine_sim:.4f}",
        f"     * HYPERCACHE (RoPE Shift + 15% CacheBlend)    : Cosine Sim = {attn_res.hypercache_cosine_sim:.4f} "
        f"(Rel L2 Err: {attn_res.hypercache_rel_l2_error:.3f} | Saved {attn_res.prefill_flops_saved_pct:.1f}% Prefix FLOPs)",
        "------------------------------------------------------------------------------------------",
        "  B. Physical Tensor Quantization, L3 Disk Spillover & Live TCP Socket Transport",
        f"     * Relative RoPE Shift Error (Pos 1024 -> 8192): {max_rope_err:.2e} (Exact SO(2) identity)",
        f"     * FP8 (E4M3) Wire Quantization                : {fp8_buf.compression_ratio:.2f}x compression | "
        f"Cosine Sim: {cos_sim_fp8:.6f}",
        f"     * INT4 (Grouped) L3 Disk (.hckv + CRC32)      : {int4_buf.compression_ratio:.2f}x compression | "
        f"Cosine Sim: {cos_sim_int4:.6f} ({l3_disk_bytes:,} B on disk)",
        f"     * Live TCP Socket Fetch + On-The-Fly RoPE     : {wire_res.wire_bytes:,} B frame in "
        f"{wire_res.roundtrip_ms:.2f} ms | Cosine Sim vs GT: {tcp_cos_sim:.6f}",
        "==========================================================================================",
        f"  PART 2: CLUSTER BREAK-EVEN MODEL ({fabric.hw.name}, 400Gbps RoCEv2, Llama-3-70B)",
        "==========================================================================================",
        f"  {'Context Len':<13} | {'Raw KV Size':<11} | {'GPU Recompute':<14} | "
        f"{'Host-RAM+TCP':<14} | {'HyperCache RDMA':<15} | {'Speedup':<10}",
        "  " + "-" * 86,
    ]

    for ctx_len in context_lengths:
        dag = GlobalRadixDAG(
            hasher=BlockHasher(block_size=64),
            l1_capacity_bytes_per_node=64 * 1024**3,
        )
        prefix_len = ctx_len - 256
        shared_prefix = [1000 + i for i in range(prefix_len)]
        dag.lookup_sequence(shared_prefix, requesting_node="gpu-node-01", tenant_id="bench")

        query_tokens = shared_prefix + [900_000 + i for i in range(256)]
        plan = dag.lookup_sequence(
            query_tokens, requesting_node="gpu-node-02", tenant_id="bench"
        )
        bd = fabric.evaluate_plan_latency(plan)
        raw_gb = (ctx_len * fabric.hw.raw_kv_bytes_per_token) / (1024**3)

        lines.append(
            f"  {f'{ctx_len // 1024}k Tokens':<13} | "
            f"{f'{raw_gb:5.2f} GB':<11} | "
            f"{f'{bd.full_recompute_ttft_ms:8.0f} ms':<14} | "
            f"{f'{bd.legacy_host_tcp_ttft_ms:8.0f} ms':<14} | "
            f"{f'{bd.hypercache_exposed_ttft_ms:8.1f} ms':<15} | "
            f"{f'{bd.speedup_vs_recompute:5.1f}x':<10}"
        )

    lines.append(
        "==========================================================================================\n"
    )
    output = "\n".join(lines)
    print(output)
    return output


if __name__ == "__main__":
    run_break_even_benchmark()
