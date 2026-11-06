"""
Comprehensive Unit, Adversarial, Concurrency & End-to-End Test Suite for HyperCache.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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


def test_hasher_vectorized_packing_and_chunk_stream() -> None:
    hasher = BlockHasher(block_size=8, cache_size=64)
    tokens = list(range(1000, 1064))

    # Test chunk_stream generator equivalence to chunk_and_hash
    stream_blocks = list(hasher.chunk_stream(tokens, tenant_id="org-stream"))
    list_blocks = hasher.chunk_and_hash(tokens, tenant_id="org-stream")
    assert len(stream_blocks) == len(list_blocks) == 8
    for sb, lb in zip(stream_blocks, list_blocks):
        assert sb.causal_hash == lb.causal_hash
        assert sb.modular_hash == lb.modular_hash

    # Test LRU cache hit acceleration on repeat turn
    c_hash1, m_hash1 = hasher.compute_block_hashes(tokens[:8], tenant_id="org-stream")
    with hasher._cache_lock:
        cache_len_before = len(hasher._hash_cache)
    c_hash2, m_hash2 = hasher.compute_block_hashes(tokens[:8], tenant_id="org-stream")
    assert c_hash1 == c_hash2 and m_hash1 == m_hash2
    with hasher._cache_lock:
        assert len(hasher._hash_cache) == cache_len_before


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
    from hypercache.core.quantizer import RoPEScalingConfig, RoPEScalingType

    rng = np.random.default_rng(123)
    k_pre_rope = rng.standard_normal((2, 16, 8, 128), dtype=np.float32)

    # Test across Standard, Linear, and YaRN long-context scaling configs
    configs = [
        RoPEScalingConfig(scaling_type=RoPEScalingType.DEFAULT),
        RoPEScalingConfig(scaling_type=RoPEScalingType.LINEAR, factor=8.0),
        RoPEScalingConfig(scaling_type=RoPEScalingType.YARN, factor=16.0, original_max_position_embeddings=8192),
    ]

    for cfg in configs:
        rotator = RoPEPhaseRotator(head_dim=128, scaling_config=cfg)
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

    # Test zero-copy sub-block slicing (slice first 16 tokens out of 32)
    fp8_slice = fp8_buf.slice_tokens(start_token=0, num_tokens=16)
    assert fp8_slice.original_shape == (1, 16, 8, 128)
    k_slice_rec, _ = KVQuantizer.dequantize(fp8_slice)
    assert k_slice_rec.shape == (1, 16, 8, 128)
    cos_sim_slice = KVQuantizer.cosine_similarity(k[:, :16], k_slice_rec)
    assert cos_sim_slice > 0.999


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


def test_physical_storage_async_watermark_eviction() -> None:
    import time
    rng = np.random.default_rng(42)
    k = rng.standard_normal((1, 8, 2, 64), dtype=np.float32)
    v = rng.standard_normal((1, 8, 2, 64), dtype=np.float32)

    # 10 blocks capacity, triggers background eviction at 8 blocks (80%), drains down to 5 (50%)
    engine = PhysicalTieredStorageEngine(
        l1_max_blocks=10,
        l2_max_blocks=20,
        high_watermark_ratio=0.8,
        low_watermark_ratio=0.5,
        enable_async_eviction=True,
    )
    try:
        for i in range(9):
            engine.put_block(f"async-blk-{i}", k, v, cached_start_pos=i * 8)

        # Allow background evictor to drain
        time.sleep(0.15)
        with engine._lock:
            # L1 should have been proactively drained to <= 5 blocks (low watermark)
            assert len(engine._l1_store) <= 5, f"L1 count should be <= 5, got {len(engine._l1_store)}"
            assert engine.l1_to_l2_demotions >= 4
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

        # Verify /metrics Prometheus text endpoint
        req_metrics = urllib.request.Request(f"http://127.0.0.1:{proxy_port}/metrics")
        with urllib.request.urlopen(req_metrics, timeout=5.0) as resp_m:
            metrics_body = resp_m.read().decode("utf-8")
            assert "hypercache_queries_total" in metrics_body
            assert "hypercache_cached_tokens_total" in metrics_body
            assert "hypercache_memory_usage_bytes" in metrics_body
    finally:
        proxy.stop()
        upstream.shutdown()
        upstream.server_close()


def test_radix_trie_prefix_matching_and_branching() -> None:
    dag = GlobalRadixDAG(hasher=BlockHasher(block_size=4), enable_modular_rope_reuse=True)
    seq_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Blocks: A, B, C
    plan_1 = dag.lookup_sequence(seq_1, requesting_node="node-alpha")
    assert plan_1.recompute_tokens == 12

    stats_1 = dag.trie_stats()
    assert stats_1["total_nodes"] == 3
    assert stats_1["max_depth"] == 3

    # Sequence 2 shares first 2 blocks (A, B) and diverges on block 3 (D)
    seq_2 = [1, 2, 3, 4, 5, 6, 7, 8, 99, 100, 101, 102]
    matched_nodes, matched_tokens = dag.match_prefix_trie(seq_2)
    assert len(matched_nodes) == 2
    assert matched_tokens == 8

    plan_2 = dag.lookup_sequence(seq_2, requesting_node="node-alpha")
    assert plan_2.exact_causal_tokens == 8
    assert plan_2.recompute_tokens == 4

    stats_2 = dag.trie_stats()
    assert stats_2["total_nodes"] == 4  # A, B, C, D
    assert stats_2["max_depth"] == 3

    # Sequence 3 completely diverges at root
    seq_3 = [50, 51, 52, 53, 54, 55, 56, 57]
    matched_3, tokens_3 = dag.match_prefix_trie(seq_3)
    assert len(matched_3) == 0
    assert tokens_3 == 0


def test_singleflight_request_coalescing_and_incast_prevention() -> None:
    import time
    from hypercache.transport.singleflight import SingleFlight

    # 1. Pure SingleFlight concurrency unit test
    sf: SingleFlight[str] = SingleFlight()
    execution_counter = 0
    lock = threading.Lock()

    def slow_producer() -> str:
        nonlocal execution_counter
        with lock:
            execution_counter += 1
        time.sleep(0.04)
        return "computed_token_payload"

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(sf.do, "shared_prefix_block", slow_producer) for _ in range(8)]
        results = [f.result() for f in futures]

    assert execution_counter == 1
    assert sf.total_deduplicated == 7
    assert all(res[0] == "computed_token_payload" for res in results)
    leader_count = sum(1 for res in results if res[1] is True)
    assert leader_count == 1

    # 2. Integration test over live HyperCacheTCPServer
    server = HyperCacheTCPServer("127.0.0.1", port=0)
    server.start_background()
    client = HyperCacheTCPClient("127.0.0.1", port=server.bound_port, max_pool_size=8)

    try:
        rng = np.random.default_rng(999)
        k = rng.standard_normal((16, 8, 128), dtype=np.float32)
        v = rng.standard_normal((16, 8, 128), dtype=np.float32)
        client.put_kv_block("incast_hot_hash", k, v)

        # 8 concurrent requests for the exact same block
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(client.get_kv_block, "incast_hot_hash") for _ in range(8)]
            responses = [f.result() for f in futures]

        assert all(r.found for r in responses)
        assert client.deduplicated_fetches >= 1
    finally:
        client.close()
        server.stop()


def test_cli_json_export_and_benchmark_machine_readable() -> None:
    import json
    import tempfile
    from benchmarks.break_even_benchmark import run_break_even_benchmark
    from hypercache.cli.auditor import run_cluster_audit

    with tempfile.TemporaryDirectory() as td:
        export_file = Path(td) / "bench_metrics.json"
        json_out = run_break_even_benchmark(
            gpu_profile="h100",
            export_json_path=export_file,
            as_json=True,
            context_lengths=[4096, 8192],
        )
        parsed = json.loads(json_out)
        assert parsed["gpu_profile"] == "h100"
        assert "verification" in parsed
        assert len(parsed["break_even_model"]) == 2
        assert export_file.is_file()

        file_data = json.loads(export_file.read_text(encoding="utf-8"))
        assert file_data["verification"]["hypercache_cosine_sim"] > 0.97

    # Audit report to_dict / to_json serialization check
    audit_rep = run_cluster_audit(
        workload_type="multi-agent-swe",
        num_nodes=2,
        daily_request_scale=1000,
        num_simulated_requests=8,
    )
    audit_dict = audit_rep.to_dict()
    assert audit_dict["workload_name"] == "multi-agent-swe"
    assert audit_dict["hypercache_hit_rate"] >= audit_dict["siloed_apc_hit_rate"]
    assert "daily_usd_saved" in audit_dict


def test_proxy_deterministic_tokenization_and_tool_schema_prefix_caching() -> None:
    import zlib
    from hypercache.proxy.server import HyperCacheProxyHandler

    # 1. Verify CRC32 tokenization is invariant to PYTHONHASHSEED across nodes
    sample_prompt = "You are an autonomous coding agent operating on a distributed KV fabric."
    toks_1 = BlockHasher.deterministic_tokenize(sample_prompt)
    toks_2 = BlockHasher.deterministic_tokenize(sample_prompt)
    assert toks_1 == toks_2
    assert toks_1[0] == (zlib.crc32(b"You") & 0xFFFFFFFF) % 100_000

    # 2. Verify tenant HMAC key caching in BlockHasher
    hasher = BlockHasher(block_size=4)
    hasher.compute_block_hashes([1, 2, 3, 4], tenant_id="org-cache-test")
    assert "org-cache-test" in hasher._tenant_key_cache

    # 3. Verify OpenAI tool schemas with reordered dict keys canonicalize identically
    dummy_handler = object.__new__(HyperCacheProxyHandler)
    body_a = {
        "tools": [{"type": "function", "function": {"name": "exec_sql", "description": "Run query"}}],
        "messages": [{"role": "user", "content": "Select top 10 users"}],
    }
    body_b = {
        "tools": [{"function": {"description": "Run query", "name": "exec_sql"}, "type": "function"}],
        "messages": [{"role": "user", "content": "Select top 10 users"}],
    }
    extracted_a = dummy_handler._extract_prompt_text(body_a)
    extracted_b = dummy_handler._extract_prompt_text(body_b)
    assert extracted_a == extracted_b
    assert "<|im_start|>tools" in extracted_a


def test_radix_dag_zero_tier_capacity_and_l3_eviction_purge() -> None:
    # 1. Siloed APC mode: L1 holds 1 block (1024 B), L2 = 0, L3 = 0
    siloed_dag = GlobalRadixDAG(
        hasher=BlockHasher(block_size=4),
        l1_capacity_bytes_per_node=1024,
        l2_capacity_bytes_per_node=0,
        l3_capacity_bytes_per_node=0,
        raw_kv_bytes_per_token=256,  # 4 tokens * 256 = 1024 B per block
    )
    siloed_dag.lookup_sequence([1, 2, 3, 4], requesting_node="node-1")
    siloed_dag.lookup_sequence([5, 6, 7, 8], requesting_node="node-1")  # Evicts [1, 2, 3, 4]

    assert siloed_dag.l3_evictions == 1
    usage = siloed_dag.get_node_usage("node-1")
    assert usage[MemoryTier.L2_PINNED_RAM] == 0
    assert usage[MemoryTier.L3_NVME_SSD] == 0

    # Re-querying [1, 2, 3, 4] must be a complete MISS, not a phantom L3 hit
    plan_evicted = siloed_dag.lookup_sequence(
        [1, 2, 3, 4], requesting_node="node-1", auto_populate_on_miss=False
    )
    assert plan_evicted.block_hits[0].hit_type == "MISS"

    # 2. Bounded L3 capacity mode: L1 = 1024B (1 blk), L2 = 512B (1 FP8 blk), L3 = 256B (1 INT4 blk)
    bounded_dag = GlobalRadixDAG(
        hasher=BlockHasher(block_size=4),
        l1_capacity_bytes_per_node=1024,
        l2_capacity_bytes_per_node=512,
        l3_capacity_bytes_per_node=256,
        raw_kv_bytes_per_token=256,
    )
    bounded_dag.lookup_sequence([10, 11, 12, 13], requesting_node="node-1")
    bounded_dag.lookup_sequence([20, 21, 22, 23], requesting_node="node-1")
    bounded_dag.lookup_sequence([30, 31, 32, 33], requesting_node="node-1")
    bounded_dag.lookup_sequence([40, 41, 42, 43], requesting_node="node-1")

    assert bounded_dag.l3_evictions == 1
    b_usage = bounded_dag.get_node_usage("node-1")
    assert b_usage[MemoryTier.L3_NVME_SSD] <= 256
    plan_oldest = bounded_dag.lookup_sequence(
        [10, 11, 12, 13], requesting_node="node-1", auto_populate_on_miss=False
    )
    assert plan_oldest.block_hits[0].hit_type == "MISS"


def test_physical_storage_tier_promotion_dedup_and_l3_disk_lru() -> None:
    rng = np.random.default_rng(77)
    k = rng.standard_normal((1, 8, 2, 64), dtype=np.float32)
    v = rng.standard_normal((1, 8, 2, 64), dtype=np.float32)

    engine = PhysicalTieredStorageEngine(l1_max_blocks=1, l2_max_blocks=1, l3_max_blocks=1)
    try:
        # Use namespaced keys containing ':' and '/' to test filename sanitization
        engine.put_block("tenant:alpha/blk-0", k, v, cached_start_pos=0)
        engine.put_block("tenant:alpha/blk-1", k, v, cached_start_pos=64)
        engine.put_block("tenant:alpha/blk-2", k, v, cached_start_pos=128)
        # Now: blk-2 in L1, blk-1 in L2, blk-0 in L3

        # Promote blk-1 from L2 to L1: blk-1 should move to L1, blk-2 demotes to L2,
        # and blk-1 must NOT exist in both L1 and L2 simultaneously.
        res_prom = engine.get_block("tenant:alpha/blk-1", promote_to_l1=True)
        assert res_prom is not None and res_prom.source_tier == MemoryTier.L2_PINNED_RAM
        assert "tenant:alpha/blk-1" in engine._l1_store
        assert "tenant:alpha/blk-1" not in engine._l2_store
        assert "tenant:alpha/blk-2" in engine._l2_store

        # Insert blk-3 -> pushes blk-1 to L2, blk-2 to L3, evicting blk-0 from L3 disk!
        engine.put_block("tenant:alpha/blk-3", k, v, cached_start_pos=192)
        assert engine.l3_evictions == 1
        assert engine.get_block("tenant:alpha/blk-0") is None
        assert engine.get_block("tenant:alpha/blk-2") is not None
    finally:
        engine.close()


def test_int4_subchannel_group_quantization_outlier_resilience() -> None:
    rng = np.random.default_rng(1337)
    k = rng.standard_normal((1, 16, 4, 128), dtype=np.float32)
    v = rng.standard_normal((1, 16, 4, 128), dtype=np.float32)

    # Inject a realistic Transformer Key outlier channel in channel 0 (25x magnitude)
    k[..., 0] *= 25.0

    buf_per_head = KVQuantizer.quantize(k, v, mode=KVQuantMode.INT4_GROUP, group_size=128)
    buf_sub_grp = KVQuantizer.quantize(k, v, mode=KVQuantMode.INT4_GROUP, group_size=32)

    k_ph, _ = KVQuantizer.dequantize(buf_per_head)
    k_sg, _ = KVQuantizer.dequantize(buf_sub_grp)

    # Evaluate reconstruction fidelity on the non-outlier channels [32:]
    cos_per_head_non_outlier = KVQuantizer.cosine_similarity(k[..., 32:], k_ph[..., 32:])
    cos_sub_grp_non_outlier = KVQuantizer.cosine_similarity(k[..., 32:], k_sg[..., 32:])
    assert cos_sub_grp_non_outlier > cos_per_head_non_outlier
    assert cos_sub_grp_non_outlier > 0.992

    # Verify sub-block token slicing works seamlessly on multi-group INT4 buffers
    sliced_sg = buf_sub_grp.slice_tokens(start_token=4, num_tokens=8)
    k_sliced, _ = KVQuantizer.dequantize(sliced_sg)
    assert k_sliced.shape == (1, 8, 4, 128)
    assert np.allclose(k_sg[:, 4:12], k_sliced)
    assert KVQuantizer.cosine_similarity(k[:, 4:12], k_sliced) > 0.985


def test_vectorized_indexed_rope_rotation_exactness() -> None:
    rng = np.random.default_rng(2026)
    rotator = RoPEPhaseRotator(head_dim=64)
    k_sub = rng.standard_normal((12, 4, 64), dtype=np.float32)
    positions = np.array([0, 3, 7, 15, 31, 64, 65, 100, 128, 255, 512, 1024], dtype=np.int64)

    vec_out = rotator.apply_indexed_rope(k_sub, positions)
    loop_out = np.empty_like(k_sub)
    for i, pos in enumerate(positions):
        loop_out[i] = rotator.apply_absolute_rope(k_sub[i : i + 1][None, ...], start_pos=int(pos))[
            0, 0
        ]

    assert np.max(np.abs(vec_out - loop_out)) < 1e-7


if __name__ == "__main__":
    test_fns = [
        test_causal_chain_vs_modular_hash_and_tenant_isolation,
        test_hasher_vectorized_packing_and_chunk_stream,
        test_adversarial_middle_block_mutation_breaks_causal_chain,
        test_relative_rope_phase_rotation_exactness,
        test_fp8_and_int4_kv_quantization_compression,
        test_physical_l1_l2_l3_disk_storage_and_crc32_tamper_detection,
        test_physical_storage_async_watermark_eviction,
        test_live_tcp_socket_server_and_concurrent_clients_with_rope_shift,
        test_real_multilayer_causal_gqa_attention_reordered_rag_cacheblend,
        test_l1_to_l2_to_l3_demotion_and_cross_node_rdma_hit,
        test_vllm_connector_reordered_rag_chunks_and_break_even_speedup,
        test_proxy_transparent_interception_and_telemetry_headers,
        test_radix_trie_prefix_matching_and_branching,
        test_singleflight_request_coalescing_and_incast_prevention,
        test_cli_json_export_and_benchmark_machine_readable,
        test_proxy_deterministic_tokenization_and_tool_schema_prefix_caching,
        test_radix_dag_zero_tier_capacity_and_l3_eviction_purge,
        test_physical_storage_tier_promotion_dedup_and_l3_disk_lru,
        test_int4_subchannel_group_quantization_outlier_resilience,
        test_vectorized_indexed_rope_rotation_exactness,
    ]
    for fn in test_fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nAll {len(test_fns)} HyperCache test suites passed successfully.")
