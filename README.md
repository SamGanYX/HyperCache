# HyperCache

[![CI](https://github.com/SamGanYX/HyperCache/actions/workflows/ci.yml/badge.svg)](https://github.com/SamGanYX/HyperCache/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Distributed KV-cache management, wire quantization, and relative RoPE block reuse for multi-node LLM inference.

## Motivation

In multi-turn agent loops, codebase analysis, and RAG workloads, most prompt tokens are shared across requests (system prompts, tool definitions, repository files, and retrieved documents).

Current serving stacks handle this in one of three ways, each with a specific bottleneck:

1. **Local in-engine prefix caching (vLLM APC, SGLang RadixAttention):** Stores KV blocks in a prefix tree in local GPU VRAM. This works well when consecutive requests hit the same worker and fit in HBM. In a multi-node cluster behind a load balancer, a request routed to Node B cannot read KV blocks computed on Node A. Moreover, because prefix trees require exact token-for-token prefix alignment from index 0, reordered RAG chunks (`[Doc A, Doc B]` vs `[Doc B, Doc A]`) miss the cache entirely.
2. **Disaggregated prefill/decode architectures (Mooncake, DistServe, Splitwise):** Separate prefill nodes from decode nodes and stream KV caches over RDMA. Effective at large scale, but requires replacing the serving runtime and cluster scheduler rather than dropping into an existing vLLM deployment.
3. **Host-RAM and TCP offloading (LMCache, DeepSpeed-FastGen):** Evicts KV blocks over PCIe to host memory or streams them over standard kernel TCP sockets. The bottleneck here is the recompute break-even threshold: for a 70B model (320 KB/token in BF16), a 16k-token prefix is ~5 GB. Copying 5 GB across PCIe, host memory, and a standard TCP stack takes ~310-530 ms, which is roughly the same time an H100 takes to simply recompute the 16k tokens from scratch (~389 ms).

HyperCache is designed as an external cache layer ("memcached for KV tensors") that sits underneath existing inference workers.

```text
                        [ Incoming LLM Request ]
                                   |
                     +-------------+-------------+
                     |                           |
                     v                           v
          [ Prompt Tokenizer ]         [ Global Radix / Hash DAG ]
                     |                  (HMAC-BLAKE2b + Modular)
                     |                           |
                     |                      Cache Lookup
                     |                           |
                     v                           v
          +---------------------+      +---------------------+
          |  Compute New Suffix |      |  Fetch Shared KV    |
          |  + 15% CacheBlend   |      |  Prefix Tensors     |
          +----------+----------+      +----------+----------+
                     |                            |
                     |                 +----------+----------+
                     |                 |  RDMA / TCP-FP8     |
                     |                 | (Layer-Wise Overlap)|
                     |                 +----------+----------+
                     v                            v
                 +------------------------------------+
                 |       Decode Generation Loop       |
                 |   (vLLM / SGLang / TRT-LLM Hook)   |
                 +------------------------------------+
```

## Core Design

### 1. Dual-Mode Block Hashing (`hypercache/core/hasher.py`)

Standard Transformer self-attention is strictly causal: the K and V tensors at token $i$ depend on every preceding token $0 \dots i-1$. Fuzzy hashing (such as LSH) cannot be used for exact prefix matching without corrupting attention states. HyperCache computes two deterministic 128-bit `BLAKE2b` digests per token block (default 16 or 64 tokens):

* **Chained Causal Hash (`causal_hash`):**
  $$H_k = \text{BLAKE2b}_{K_{\text{tenant}}}(\text{"CAUSAL"} \parallel H_{k-1} \parallel \text{tokens}_k)$$
  Used for exact prefix hits with zero recomputation. If a middle block changes (`[A, B, C]` -> `[A, X, C]`), the chain breaks at `X` and `C` is never falsely matched as an exact causal hit.
* **Position-Independent Modular Hash (`modular_hash`):**
  $$M_k = \text{BLAKE2b}_{K_{\text{tenant}}}(\text{"MODULAR"} \parallel \text{tokens}_k)$$
  Used when a known document or file block appears at a new token offset or after a different predecessor.

All hashes are keyed by a per-tenant secret derived via `HMAC-SHA256` to prevent cross-tenant timing side-channel attacks (where a user infers whether another tenant has cached a proprietary file by measuring TTFT). An explicit `"public"` namespace is available for global system prompts and shared tool schemas.

### 2. Relative RoPE Phase Rotation Without Model Surgery (`hypercache/core/quantizer.py`)

When a cached block is reused at a new token offset $m_{\text{new}}$ instead of its original offset $m_{\text{old}}$, its keys have the wrong Rotary Position Embedding (RoPE) phase angles.

Because 2D RoPE rotations belong to $SO(2)$, they satisfy the group identity:
$$R(m_{\text{new}} \theta_i) = R((m_{\text{new}} - m_{\text{old}}) \theta_i) \, R(m_{\text{old}} \theta_i)$$

This means HyperCache does not need invasive forward-pass hooks to intercept pre-RoPE keys. It takes standard post-RoPE keys directly from the engine's PagedAttention buffer at $m_{\text{old}}$ and rotates each complex coordinate pair by the relative delta $\Delta m = m_{\text{new}} - m_{\text{old}}$ on fetch (`max_err ~ 4.7e-7` in float32).

To account for cross-chunk causal attention drift in layers $1 \dots L$ when chunks are reordered, HyperCache pairs relative RoPE rotation with selective boundary recomputation (`hypercache/core/attention_engine.py`): Layer 0 K/V depends only on token embeddings (making rotated Layer 0 KV exact), and the top ~15% of tokens with the largest Layer-0 hidden-state drift are selectively recomputed through deeper layers (following the CacheBlend approach).

### 3. Wire Quantization and Tiered Storage (`hypercache/core/storage_engine.py`)

Because network and PCIe links are an order of magnitude slower than HBM3e, blocks are compressed before leaving L1:

* **L1 (Active HBM / Working Memory):** Native `FP16` / `BF16`.
* **L2 (Peer Memory / Pinned Host RAM / Network Wire):** Per-token, per-head dynamic `FP8` (`int8` payload + `fp16` scales), cutting transfer volume by `1.97x` (`>0.9999` cosine similarity vs raw float32).
* **L3 (NVMe / Disk Spillover):** Symmetric `INT4` bit-packed (2 values per `uint8` byte + `fp16` scales) written to `.hckv` binary block files with a 32-bit `CRC32` frame checksum (`3.88x` compression, `>0.993` cosine similarity).

### 4. Layer-Wise Overlap and Break-Even Physics (`hypercache/transport/fabric.py`)

Waiting for all 80 layers of a 64k-token KV cache (~10 GB in FP8) to transfer before starting prefill leaves the GPU idle. By pipelining transfers layer-by-layer—fetching Layer $l+1$ into a staging buffer while the GPU computes suffix tokens for Layer $l$—most of the wire transfer time is hidden behind active suffix compute.

Modeled break-even comparison for Llama-3-70B across 80GB H100 nodes (400Gbps RoCEv2, ~45.5 GB/s effective):

| Context Length | Raw BF16 KV | Full GPU Recompute | Host-RAM + TCP (Uncompressed) | HyperCache (FP8 + RDMA + Layer Overlap) |
| :--- | :--- | :--- | :--- | :--- |
| 16k tokens | 5.00 GB | 389 ms | 530 ms | 22.4 ms |
| 32k tokens | 10.00 GB | 942 ms | 1,049 ms | 44.6 ms |
| 64k tokens | 20.00 GB | 2,546 ms | 2,096 ms | 95.7 ms |
| 128k tokens | 40.00 GB | 7,753 ms | 4,219 ms | 221.9 ms |

## Implementation Status (What is Real vs. Modeled in v0.1)

To keep the repository easy to inspect and test on any machine without requiring an InfiniBand cluster:

* **Implemented and executed locally (zero mocks):**
  * `hypercache/core/hasher.py`: Chained causal and modular `HMAC-BLAKE2b` block chunker and hasher.
  * `hypercache/core/quantizer.py`: Per-head `FP8` and bit-packed `INT4` KV quantizer/dequantizer, plus the $SO(2)$ relative RoPE phase rotator.
  * `hypercache/core/attention_engine.py`: A 4-layer Causal Grouped-Query Attention (GQA) + RoPE Transformer in NumPy that runs end-to-end forward passes on reordered chunks (`[Sys, DocB, DocA, Query]`), comparing ground-truth prefill against naive KV stitching, RoPE-only rotation, and RoPE + 15% selective recomputation.
  * `hypercache/core/storage_engine.py`: Thread-safe L1 -> L2 (`FP8`) -> L3 (`INT4` `.hckv` disk files with `CRC32` tamper detection) physical tensor storage engine.
  * `hypercache/transport/tcp_server.py`: Multi-threaded binary TCP socket server and client (`HyperCacheTCPServer` / `HyperCacheTCPClient`) with `TCP_NODELAY` framing that streams compressed KV blocks over the wire and applies relative RoPE shifts on fetch.
  * `hypercache/core/radix_dag.py`: Thread-safe distributed Radix/modular block directory with LRFU demotion and hot-spot tracking.
* **Analytical / simulated for cluster capacity planning:**
  * `hypercache/transport/fabric.py` and `hypercache/cli/auditor.py`: Analytical latency models for 400Gbps RoCEv2/InfiniBand GPUDirect RDMA, NVLink P2P, and H100/A100/B200 prefill compute scaling. Native `ibverbs` / UCX C++ bindings and CUDA stream hooks for vLLM's C++ block manager are planned for v0.2.

## Installation & Usage

Requires Python 3.10+ and `numpy`.

```bash
pip install -e .
```

### 1. Run the verification and break-even benchmark

Executes the local 4-layer GQA attention verification on reordered chunks, physical L1/L2/L3 `.hckv` disk round-trips, live loopback TCP socket transfers, and prints the cluster break-even table:

```bash
python -m hypercache benchmark --gpu h100
```

Example output:

```text
==========================================================================================
  PART 1: LIVE MEASURED NUMERICAL & SYSTEMS VERIFICATION (EXECUTED LOCALLY)
==========================================================================================
  A. Multi-Layer Causal GQA Attention + RoPE on Reordered RAG Chunks [Sys, DocB, DocA, Q]
     * Naive KV Stitching (No RoPE / No Recompute) : Cosine Sim = 0.7423 (Rel L2 Err: 0.715 -> BROKEN)
     * Relative RoPE Phase Rotation Only (0% Rec.) : Cosine Sim = 0.9701
     * HYPERCACHE (RoPE Shift + 15% CacheBlend)    : Cosine Sim = 0.9783 (Rel L2 Err: 0.208 | Saved 85.3% Prefix FLOPs)
------------------------------------------------------------------------------------------
  B. Physical Tensor Quantization, L3 Disk Spillover & Live TCP Socket Transport
     * Relative RoPE Shift Error (Pos 1024 -> 8192): 4.77e-07 (Exact SO(2) identity)
     * FP8 (E4M3) Wire Quantization                : 1.97x compression | Cosine Sim: 0.999979
     * INT4 (Grouped) L3 Disk (.hckv + CRC32)      : 3.88x compression | Cosine Sim: 0.993258 (67,653 B on disk)
     * Live TCP Socket Fetch + On-The-Fly RoPE     : 133,188 B frame in 1.14 ms | Cosine Sim vs GT: 0.999979
==========================================================================================
```

### 2. Run the test suite

Runs all 9 unit, adversarial, CRC32 tamper detection, multi-threaded TCP concurrency, and GQA attention tests:

```bash
python tests/test_hypercache.py
```

### 3. Audit prompt logs or simulated multi-node workloads

Evaluates cache hit rates and TTFT across Siloed APC, Host-RAM+TCP offload, and HyperCache on a JSONL log file (or a built-in multi-agent / RAG workload):

```bash
python -m hypercache audit --workload multi-agent-swe --nodes 4 --gpu h100
# Or replay your own prompt logs:
python -m hypercache audit --log-file requests.jsonl --nodes 4 --gpu h100
```

### 4. Terminal telemetry view

```bash
python -m hypercache top --duration 1.5
```

## License

MIT
