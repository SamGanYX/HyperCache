"""
Comprehensive Unit, Adversarial, Concurrency & End-to-End Test Suite for HyperCache.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import numpy as np

from hypercache.core.attention_engine import verify_reordered_rag_attention
from hypercache.core.hasher import BlockHasher
from hypercache.core.quantizer import KVQuantizer, KVQuantMode, RoPEPhaseRotator
from hypercache.core.radix_dag import GlobalRadixDAG, MemoryTier
from hypercache.core.storage_engine import PhysicalTieredStorageEngine
from hypercache.integrations.vllm_connector import (
    HyperCacheConfig,
    HyperCacheVLLMConnector,
)
from hypercache.proxy.server import HyperCacheProxyServer
from hypercache.transport.tcp_server import HyperCacheTCPClient, HyperCacheTCPServer


def test_causal_chain_vs_modular_hash_and_tenant_isolation() -> None:
    hasher = BlockHasher(block_size=4, enable_tenant_isolation=True)
    seq_a = [10, 11, 12, 13, 20, 21, 22, 23]
    seq_b = [99, 11, 12, 13, 20, 21, 22, 23]  # Token 0 differs!

    blocks_a = hasher.chunk_and_hash(seq_a, tenant_id="tenant-alpha")
    blocks_b = hasher.chunk_and_hash(seq_b, tenant_id="tenant-alpha")

    # Block 1 has identical tokens [20, 21, 22, 23], but different preceding context:
    # Causal hash MUST differ (because Transformer self-attention is path-dependent),
    # while Pre-RoPE Modular hash MUST match!
    assert blocks_a[1].causal_hash != blocks_b[1].causal_hash
    assert blocks_a[1].modular_hash == blocks_b[1].modular_hash

    # Cross-tenant timing-attack isolation: Tenant Beta hashing the exact same sequence
    # must produce completely different causal and modular hashes.
    blocks_beta = hasher.chunk_and_hash(seq_a, tenant_id="tenant-beta")
    assert blocks_a[0].causal_hash != blocks_beta[0].causal_hash
    assert blocks_a[0].modular_hash != blocks_beta[0].modular_hash


def test_adversarial_middle_block_mutation_breaks_causal_chain() -> None:
    """
    Adversarial Critique Test:
    If Sequence 1 is `[Block_A, Block_B, Block_C]` and Sequence 2 is `[Block_A, Block_X, Block_C]`,
    `Block_C` in Sequence 2 MUST NOT be treated as an exact causal hit (`CAUSAL_EXACT`),
    because `Block_B` was replaced by `Block_X` in its causal history!
    Instead, `Block_A` is `CAUSAL_EXACT`, `Block_X` is `MISS`, and `Block_C` is `MODULAR_ROPE_SHIFT`
    (triggering selective boundary attention recomputation).
    """
    dag = GlobalRadixDAG(hasher=BlockHasher(block_size=4), enable_modular_rope_reuse=True)
    seq_1 = [1, 2, 3, 4, 10, 11, 12, 13, 20, 21, 22, 23]
    seq_2 = [1, 2, 3, 4, 99, 98, 97, 96, 20, 21, 22, 23]

    dag.lookup_sequence(seq_1, requesting_node="node-1", tenant_id="org-1")
    plan_2 = dag.lookup_sequence(seq_2, requesting_node="node-1", tenant_id="org-1")

    assert plan_2.block_hits[0].hit_type == "CAUSAL_EXACT"
    assert plan_2.block_hits[1].hit_type == "MISS"
    assert plan_2.block_hits[2].hit_type == "MODULAR_ROPE_SHIFT"
    assert plan_2.block_hits[2].selective_recompute_ratio > 0.0


def test_relative_rope_phase_rotation_exactness() -> None:
    rng = np.random.default_rng(123)
    k_pre_rope = rng.standard_normal((2, 16, 8, 128), dtype=np.float32)
    rotator = RoPEPhaseRotator(head_dim=128)

    k_pos_512 = rotator.apply_absolute_rope(k_pre_rope, start_pos=512)
    k_pos_4096_expected = rotator.apply_absolute_rope(k_pre_rope, start_pos=4096)
    k_pos_4096_shifted = rotator.shift_positions(
        k_pos_512, old_start_pos=512, new_start_pos=4096
    )

    np.testing.assert_allclose(k_pos_4096_shifted, k_pos_4096_expected, atol=1e-5)


def test_fp8_and_int4_kv_quantization_compression() -> None:
    rng = np.random.default_rng(99)
    k = rng.standard_normal((1, 32, 8, 128), dtype=np.float32)
    v = rng.standard_normal((1, 32, 8, 128), dtype=np.float32)

    fp8_buf = KVQuantizer.quantize(k, v, mode=KVQuantMode.FP8_E4M3)
    assert 1.95 <= fp8_buf.compression_ratio <= 2.05
    k_rec, _ = KVQuantizer.dequantize(fp8_buf)
    assert k_rec.shape == k.shape
    cos_sim = float(np.sum(k * k_rec) / (np.linalg.norm(k) * np.linalg.norm(k_rec)))
    assert cos_sim > 0.999

    int4_buf = KVQuantizer.quantize(k, v, mode=KVQuantMode.INT4_GROUP)
    assert int4_buf.compression_ratio >= 3.7
    k_i4, _ = KVQuantizer.dequantize(int4_buf)
    cos_sim_i4 = float(np.sum(k * k_i4) / (np.linalg.norm(k) * np.linalg.norm(k_i4)))
    assert cos_sim_i4 > 0.985


def test_physical_l1_l2_l3_disk_storage_and_crc32_tamper_detection() -> None:
    rng = np.random.default_rng(2025)
    k0 = rng.standard_normal((1, 16, 4, 64), dtype=np.float32)
    v0 = rng.standard_normal((1, 16, 4, 64), dtype=np.float32)

    engine = PhysicalTieredStorageEngine(l1_max_blocks=1, l2_max_blocks=1)
    try:
        # Insert 3 blocks -> blk-0 is demoted L1 -> L2 -> L3 (.hckv binary file on disk),
        # blk-1 is demoted L1 -> L2 (FP8 in memory), blk-2 stays in L1 (FP16).
        engine.put_block("blk-0", k0, v0, cached_start_pos=128)
        engine.put_block("blk-1", k0 * 1.5, v0 * 1.5, cached_start_pos=256)
        engine.put_block("blk-2", k0 * 2.0, v0 * 2.0, cached_start_pos=512)

        assert engine.l1_to_l2_demotions == 2
        assert engine.l2_to_l3_demotions == 1

        res_l1 = engine.get_block("blk-2")
        assert res_l1 is not None and res_l1.source_tier == MemoryTier.L1_LOCAL_VRAM

        res_l2 = engine.get_block("blk-1")
        assert res_l2 is not None and res_l2.source_tier == MemoryTier.L2_PINNED_RAM

        res_l3 = engine.get_block("blk-0")
        assert res_l3 is not None and res_l3.source_tier == MemoryTier.L3_NVME_SSD
        assert res_l3.cached_start_pos == 128
        cos_l3 = float(
            np.sum(k0 * res_l3.k_tensor)
            / (np.linalg.norm(k0) * np.linalg.norm(res_l3.k_tensor))
        )
        assert cos_l3 > 0.985

        # Verify CRC32 tamper detection on binary frame
        q_buf = KVQuantizer.quantize(k0, v0, mode=KVQuantMode.FP8_E4M3)
        raw_frame = bytearray(PhysicalTieredStorageEngine.serialize_quantized_buffer(q_buf))
        raw_frame[-5] ^= 0xFF  # Corrupt a payload byte
        corrupted_detected = False
        try:
            PhysicalTieredStorageEngine.deserialize_quantized_buffer(bytes(raw_frame))
        except ValueError as exc:
            if "CRC32 checksum mismatch" in str(exc):
                corrupted_detected = True
        assert corrupted_detected, "Corrupted KV wire frame must trigger CRC32 ValueError!"
    finally:
        engine.close()


def test_live_tcp_socket_server_and_concurrent_clients_with_rope_shift() -> None:
    rng = np.random.default_rng(77)
    rotator = RoPEPhaseRotator(head_dim=64)
    k_base = rng.standard_normal((1, 16, 4, 64), dtype=np.float32)
    v_base = rng.standard_normal((1, 16, 4, 64), dtype=np.float32)
    k_at_256 = rotator.apply_absolute_rope(k_base, start_pos=256)
    k_at_2048_expected = rotator.apply_absolute_rope(k_base, start_pos=2048)

    server = HyperCacheTCPServer(host="127.0.0.1", port=0)
    server.start_background()
    try:
        port = server.bound_port

        def _worker(worker_id: int) -> float:
            client = HyperCacheTCPClient(
                host="127.0.0.1",
                port=port,
                quant_mode=KVQuantMode.FP8_E4M3,
                head_dim=64,
            )
            key = f"concurrent-blk-{worker_id}"
            client.put_kv_block(key, k_at_256, v_base, cached_start_pos=256)
            res = client.get_kv_block(key, new_start_pos=2048)
            assert res.found and res.k_tensor is not None
            return float(
                np.sum(k_at_2048_expected * res.k_tensor)
                / (
                    np.linalg.norm(k_at_2048_expected)
                    * np.linalg.norm(res.k_tensor)
                )
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            sims = list(pool.map(_worker, range(16)))
        assert all(s > 0.999 for s in sims)
    finally:
        server.stop()


def test_real_multilayer_causal_gqa_attention_reordered_rag_cacheblend() -> None:
    res = verify_reordered_rag_attention(recompute_prefix_fraction=0.15)
    # Naive stitching without RoPE shift or selective recompute degrades badly (< 0.80)
    assert res.naive_stitch_cosine_sim < 0.80
    # HyperCache RoPE phase shift + 15% Layer-0 drift-guided selective recompute
    # jumps from 0.742 to >0.975 on pure contextual delta while saving >85% of prefix FLOPs
    assert res.hypercache_cosine_sim > 0.975
    assert res.hypercache_cosine_sim > res.rope_only_cosine_sim
    assert res.hypercache_cosine_sim > res.naive_stitch_cosine_sim + 0.20
    assert res.prefill_flops_saved_pct >= 85.0


def test_l1_to_l2_to_l3_demotion_and_cross_node_rdma_hit() -> None:
    raw_per_tok = 1024
    dag = GlobalRadixDAG(
        hasher=BlockHasher(block_size=16),
        l1_capacity_bytes_per_node=16 * raw_per_tok,
        l2_capacity_bytes_per_node=64 * raw_per_tok,
        l3_capacity_bytes_per_node=256 * raw_per_tok,
        raw_kv_bytes_per_token=raw_per_tok,
    )

    seq1 = list(range(0, 32))
    dag.lookup_sequence(seq1, requesting_node="node-a", tenant_id="acme")
    assert dag.l1_to_l2_demotions >= 1

    plan_b = dag.lookup_sequence(seq1, requesting_node="node-b", tenant_id="acme")
    assert plan_b.exact_causal_tokens == 32
    assert plan_b.tier_breakdown[MemoryTier.L2_PEER_RDMA] == 32


def test_vllm_connector_reordered_rag_chunks_and_break_even_speedup() -> None:
    shared_dag = GlobalRadixDAG(hasher=BlockHasher(block_size=64))
    node1 = HyperCacheVLLMConnector(
        HyperCacheConfig(node_id="gpu-node-01", block_size=64),
        shared_dag=shared_dag,
    )
    node2 = HyperCacheVLLMConnector(
        HyperCacheConfig(node_id="gpu-node-02", block_size=64),
        shared_dag=shared_dag,
    )

    sys_prompt = [100 + i for i in range(1024)]
    doc_a = [10_000 + i for i in range(8192)]
    doc_b = [20_000 + i for i in range(8192)]
    suffix_1 = [90_000 + i for i in range(256)]
    suffix_2 = [95_000 + i for i in range(256)]

    req1_tokens = sys_prompt + doc_a + doc_b + suffix_1
    res1 = node1.lookup_and_prefetch("req-1", req1_tokens, tenant_id="org-1")
    assert res1.cached_tokens == 0

    req2_tokens = sys_prompt + doc_b + doc_a + suffix_2
    res2 = node2.lookup_and_prefetch("req-2", req2_tokens, tenant_id="org-1")

    assert res2.exact_causal_tokens == 1024
    assert res2.modular_reused_tokens == 16384
    assert res2.speedup_factor > 5.0
    assert res2.exposed_latency_ms < res2.legacy_host_tcp_ms


def test_proxy_transparent_interception_and_telemetry_headers() -> None:
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading
    import urllib.request

    # Spin up mock upstream server
    class MockUpstreamHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(length)
            resp = json.dumps({"id": "chatcmpl-test", "choices": [{"message": {"content": "ok"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

    upstream = HTTPServer(("127.0.0.1", 0), MockUpstreamHandler)
    up_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    up_thread.start()
    up_port = upstream.server_address[1]

    # Spin up HyperCache proxy pointing to mock upstream
    proxy = HyperCacheProxyServer(
        host="127.0.0.1",
        port=0,
        upstream_url=f"http://127.0.0.1:{up_port}",
    )
    proxy.start_background()
    proxy_port = proxy.server_address[1]

    try:
        req_body = json.dumps({
            "model": "meta-llama/Llama-3-70B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant specialized in distributed caching."},
                {"role": "user", "content": "Explain KV-cache prefix deduplication across GPU nodes."},
            ]
        }).encode("utf-8")

        # Turn 1: Initial request (populates DAG)
        req1 = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            data=req_body,
            headers={"Content-Type": "application/json", "X-Tenant-ID": "test-org"},
            method="POST",
        )
        with urllib.request.urlopen(req1, timeout=5.0) as resp1:
            assert resp1.status == 200
            assert "x-hypercache-hit-tokens" in resp1.headers

        # Turn 2: Exact prompt повторился -> Must hit cache!
        req2 = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            data=req_body,
            headers={"Content-Type": "application/json", "X-Tenant-ID": "test-org"},
            method="POST",
        )
        with urllib.request.urlopen(req2, timeout=5.0) as resp2:
            assert resp2.status == 200
            hit_tokens = int(resp2.headers.get("x-hypercache-hit-tokens", 0))
            assert hit_tokens > 0, "Second identical request through proxy must hit HyperCache DAG!"

        # Verify /v1/hypercache/stats endpoint
        req_stats = urllib.request.Request(f"http://127.0.0.1:{proxy_port}/v1/hypercache/stats")
        with urllib.request.urlopen(req_stats, timeout=5.0) as resp_s:
            stats = json.loads(resp_s.read().decode())
            assert stats["total_queries"] >= 2
    finally:
        proxy.stop()
        upstream.shutdown()
        upstream.server_close()


if __name__ == "__main__":
    test_fns = [
        test_causal_chain_vs_modular_hash_and_tenant_isolation,
        test_adversarial_middle_block_mutation_breaks_causal_chain,
        test_relative_rope_phase_rotation_exactness,
        test_fp8_and_int4_kv_quantization_compression,
        test_physical_l1_l2_l3_disk_storage_and_crc32_tamper_detection,
        test_live_tcp_socket_server_and_concurrent_clients_with_rope_shift,
        test_real_multilayer_causal_gqa_attention_reordered_rag_cacheblend,
        test_l1_to_l2_to_l3_demotion_and_cross_node_rdma_hit,
        test_vllm_connector_reordered_rag_chunks_and_break_even_speedup,
        test_proxy_transparent_interception_and_telemetry_headers,
    ]
    for fn in test_fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nAll {len(test_fns)} HyperCache test suites passed successfully.")
