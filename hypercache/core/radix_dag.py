"""
Decoupled Control Plane: Distributed Radix / Hash DAG & Tiered Memory Directory.

Manages KV-cache block placement across a cluster of GPU nodes and memory tiers:
- Tier L1: Local GPU HBM (BF16/FP16 active PagedAttention blocks)
- Tier L2: Peer GPU HBM (via GPUDirect RDMA) & Local Pinned Host RAM (FP8 quantized)
- Tier L3: Local NVMe SSD via SPDK / io_uring (INT4 bit-packed spillover)

Includes:
- Dual-mode lookup (Exact Causal Prefix vs. Pre-RoPE Modular DAG)
- Frequency-weighted LRFU (Least Recently/Frequently Used) demotion (L1 -> L2 -> L3)
- Hot-spot P2P broadcast replication to prevent RDMA NIC incast storms
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from typing import Optional

from hypercache.core.hasher import BlockHasher, TokenBlock


class MemoryTier(str, Enum):
    L1_LOCAL_VRAM = "L1_LOCAL_VRAM"
    L2_PEER_RDMA = "L2_PEER_RDMA"
    L2_PINNED_RAM = "L2_PINNED_RAM"
    L3_NVME_SSD = "L3_NVME_SSD"
    MISS = "MISS"


@dataclass
class BlockReplica:
    """Tracks a physical copy of a KV-cache block on a specific node and tier."""

    node_id: str
    tier: MemoryTier
    size_bytes: int
    cached_token_offset: int
    tp_shards: int = 1
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    access_count: int = 1


@dataclass
class BlockLookupHit:
    """Result of querying a single TokenBlock in the global Radix/Hash DAG."""

    block: TokenBlock
    hit_type: str  # "CAUSAL_EXACT" | "MODULAR_ROPE_SHIFT" | "MISS"
    tier: MemoryTier
    source_node_id: Optional[str]
    position_delta: int = 0
    selective_recompute_ratio: float = 0.0
    bytes_transferred: int = 0


@dataclass
class SequenceLookupPlan:
    """Complete execution and prefetch plan for an incoming prompt sequence."""

    total_tokens: int
    exact_causal_tokens: int
    modular_reused_tokens: int
    recompute_tokens: int
    block_hits: list[BlockLookupHit]
    tier_breakdown: dict[MemoryTier, int]
    lookup_latency_us: float

    @property
    def total_cached_tokens(self) -> int:
        return self.exact_causal_tokens + int(
            round(self.modular_reused_tokens * (1.0 - 0.12))
        )

    @property
    def effective_cache_hit_ratio(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return min(1.0, self.total_cached_tokens / self.total_tokens)


class GlobalRadixDAG:
    """
    Distributed Control Plane Index tracking KV block locations across all cluster nodes.

    In production, this control plane runs as a lock-free Rust/C++ daemon communicating
    via shared memory locally and RDMA/gRPC gossip across nodes.
    """

    def __init__(
        self,
        hasher: Optional[BlockHasher] = None,
        l1_capacity_bytes_per_node: int = 24 * 1024**3,   # 24 GB KV pool per GPU node
        l2_capacity_bytes_per_node: int = 128 * 1024**3,  # 128 GB Pinned Host RAM
        l3_capacity_bytes_per_node: int = 1024 * 1024**3, # 1 TB NVMe pool
        raw_kv_bytes_per_token: int = 327_680,            # 320 KB/token (Llama-3-70B BF16)
        enable_modular_rope_reuse: bool = True,
        hotspot_replication_threshold: int = 8,
    ) -> None:
        self.hasher = hasher or BlockHasher(block_size=16)
        self.l1_capacity = l1_capacity_bytes_per_node
        self.l2_capacity = l2_capacity_bytes_per_node
        self.l3_capacity = l3_capacity_bytes_per_node
        self.raw_kv_bytes_per_token = raw_kv_bytes_per_token
        self.enable_modular_rope_reuse = enable_modular_rope_reuse
        self.hotspot_replication_threshold = hotspot_replication_threshold

        # Primary Index: causal_hash -> list[BlockReplica]
        self._causal_index: dict[str, list[BlockReplica]] = {}
        # Secondary Index (Position-Independent): (tenant_id, modular_hash) -> list[BlockReplica]
        self._modular_index: dict[tuple[str, str], list[BlockReplica]] = {}

        # Per-node memory accounting: node_id -> {tier: used_bytes}
        self._node_usage: dict[str, dict[MemoryTier, int]] = {}
        # Node -> hash entries for fast LRFU eviction
        self._node_blocks: dict[str, dict[str, BlockReplica]] = {}
        self._lock = threading.RLock()

        # Telemetry counters
        self.total_queries = 0
        self.hotspot_replications = 0
        self.l1_to_l2_demotions = 0
        self.l2_to_l3_demotions = 0

    def _ensure_node(self, node_id: str) -> None:
        if node_id not in self._node_usage:
            self._node_usage[node_id] = {
                MemoryTier.L1_LOCAL_VRAM: 0,
                MemoryTier.L2_PINNED_RAM: 0,
                MemoryTier.L3_NVME_SSD: 0,
            }
            self._node_blocks[node_id] = {}

    def get_node_usage(self, node_id: str) -> dict[MemoryTier, int]:
        with self._lock:
            self._ensure_node(node_id)
            return dict(self._node_usage[node_id])

    def _evict_or_demote_if_needed(self, node_id: str, required_l1_bytes: int) -> None:
        """
        Instead of dropping evicted KV blocks like standard vLLM APC, HyperCache
        demotes cold L1 VRAM blocks to L2 Pinned Host RAM (with 2x FP8 compression)
        and from L2 to L3 NVMe (with 4x INT4 compression).
        """
        self._ensure_node(node_id)
        usage = self._node_usage[node_id]
        node_map = self._node_blocks[node_id]

        while usage[MemoryTier.L1_LOCAL_VRAM] + required_l1_bytes > self.l1_capacity:
            # Find cold L1 replicas on this node using LRFU score
            l1_candidates = [
                (c_hash, rep)
                for c_hash, rep in node_map.items()
                if rep.tier == MemoryTier.L1_LOCAL_VRAM
            ]
            if not l1_candidates:
                break

            # Lowest access_count & oldest last_accessed gets demoted first
            victim_hash, victim = min(
                l1_candidates,
                key=lambda item: (item[1].access_count, item[1].last_accessed),
            )
            usage[MemoryTier.L1_LOCAL_VRAM] -= victim.size_bytes

            # Demote to L2 Pinned RAM at 50% size (FP8 quantization)
            fp8_bytes = victim.size_bytes // 2
            if usage[MemoryTier.L2_PINNED_RAM] + fp8_bytes > self.l2_capacity:
                self._demote_l2_to_l3(node_id, fp8_bytes)

            victim.tier = MemoryTier.L2_PINNED_RAM
            victim.size_bytes = fp8_bytes
            usage[MemoryTier.L2_PINNED_RAM] += fp8_bytes
            self.l1_to_l2_demotions += 1

    def _demote_l2_to_l3(self, node_id: str, required_l2_bytes: int) -> None:
        usage = self._node_usage[node_id]
        node_map = self._node_blocks[node_id]

        while usage[MemoryTier.L2_PINNED_RAM] + required_l2_bytes > self.l2_capacity:
            l2_candidates = [
                (c_hash, rep)
                for c_hash, rep in node_map.items()
                if rep.tier == MemoryTier.L2_PINNED_RAM
            ]
            if not l2_candidates:
                break

            _, victim = min(
                l2_candidates,
                key=lambda item: (item[1].access_count, item[1].last_accessed),
            )
            usage[MemoryTier.L2_PINNED_RAM] -= victim.size_bytes

            # Demote to L3 NVMe at 50% of FP8 size (= 25% of BF16 via INT4 packing)
            int4_bytes = max(1, victim.size_bytes // 2)
            victim.tier = MemoryTier.L3_NVME_SSD
            victim.size_bytes = int4_bytes
            usage[MemoryTier.L3_NVME_SSD] = min(
                self.l3_capacity, usage[MemoryTier.L3_NVME_SSD] + int4_bytes
            )
            self.l2_to_l3_demotions += 1

    def register_blocks(
        self,
        blocks: list[TokenBlock],
        node_id: str,
        tp_shards: int = 1,
    ) -> None:
        """Registers newly computed or prefetched KV blocks into a node's L1 VRAM."""
        with self._lock:
            self._ensure_node(node_id)
            now = time.monotonic()

            for blk in blocks:
                raw_bytes = blk.num_tokens * self.raw_kv_bytes_per_token
                existing_rep = self._node_blocks[node_id].get(blk.causal_hash)
                if existing_rep is not None:
                    existing_rep.last_accessed = now
                    existing_rep.access_count += 1
                    continue

                self._evict_or_demote_if_needed(node_id, raw_bytes)
                replica = BlockReplica(
                    node_id=node_id,
                    tier=MemoryTier.L1_LOCAL_VRAM,
                    size_bytes=raw_bytes,
                    cached_token_offset=blk.start_token_offset,
                    tp_shards=tp_shards,
                    created_at=now,
                    last_accessed=now,
                    access_count=1,
                )
                self._node_usage[node_id][MemoryTier.L1_LOCAL_VRAM] += raw_bytes
                self._node_blocks[node_id][blk.causal_hash] = replica
                self._causal_index.setdefault(blk.causal_hash, []).append(replica)

                effective_tenant = (
                    blk.tenant_id if self.hasher.enable_tenant_isolation else "public"
                )
                mod_key = (effective_tenant, blk.modular_hash)
                self._modular_index.setdefault(mod_key, []).append(replica)

    def _select_best_replica(
        self,
        replicas: list[BlockReplica],
        requesting_node: str,
    ) -> tuple[BlockReplica, MemoryTier]:
        """
        Selects the lowest-latency replica across the cluster:
        1. Local L1 VRAM (0 network hops)
        2. Peer L1 VRAM via GPUDirect RDMA (`L2_PEER_RDMA`)
        3. Local L2 Pinned Host RAM (`L2_PINNED_RAM`)
        4. Peer L2 Pinned Host RAM over RDMA (`L2_PEER_RDMA`)
        5. Local or Peer L3 NVMe (`L3_NVME_SSD`)
        """
        best_rep: Optional[BlockReplica] = None
        best_effective_tier = MemoryTier.MISS
        best_rank = 999

        for rep in replicas:
            if rep.node_id == requesting_node:
                if rep.tier == MemoryTier.L1_LOCAL_VRAM:
                    rank, eff_tier = 1, MemoryTier.L1_LOCAL_VRAM
                elif rep.tier == MemoryTier.L2_PINNED_RAM:
                    rank, eff_tier = 3, MemoryTier.L2_PINNED_RAM
                else:
                    rank, eff_tier = 5, MemoryTier.L3_NVME_SSD
            else:
                if rep.tier in (MemoryTier.L1_LOCAL_VRAM, MemoryTier.L2_PINNED_RAM):
                    rank, eff_tier = 2, MemoryTier.L2_PEER_RDMA
                else:
                    rank, eff_tier = 6, MemoryTier.L3_NVME_SSD

            if rank < best_rank:
                best_rank = rank
                best_rep = rep
                best_effective_tier = eff_tier

        assert best_rep is not None
        best_rep.access_count += 1
        best_rep.last_accessed = time.monotonic()

        # Hot-spot P2P Broadcast protection: if a remote block is heavily contended,
        # count a broadcast replication event.
        if (
            best_rep.node_id != requesting_node
            and best_rep.access_count >= self.hotspot_replication_threshold
        ):
            self.hotspot_replications += 1

        return best_rep, best_effective_tier

    def lookup_sequence(
        self,
        token_ids: list[int] | tuple[int, ...],
        requesting_node: str,
        tenant_id: str = "public",
        auto_populate_on_miss: bool = True,
    ) -> SequenceLookupPlan:
        """
        Performs a sub-millisecond lookup across the global Radix DAG for a prompt sequence.
        First matches exact causal prefix blocks, then falls back to position-independent
        Pre-RoPE modular blocks for shifted RAG/code chunks.
        """
        t0 = time.perf_counter()
        blocks = self.hasher.chunk_and_hash(token_ids, tenant_id=tenant_id)
        block_hits: list[BlockLookupHit] = []
        tier_counts: dict[MemoryTier, int] = {
            MemoryTier.L1_LOCAL_VRAM: 0,
            MemoryTier.L2_PEER_RDMA: 0,
            MemoryTier.L2_PINNED_RAM: 0,
            MemoryTier.L3_NVME_SSD: 0,
            MemoryTier.MISS: 0,
        }

        exact_tokens = 0
        modular_tokens = 0
        recompute_tokens = 0
        causal_chain_intact = True
        effective_tenant = (
            tenant_id if self.hasher.enable_tenant_isolation else "public"
        )

        with self._lock:
            self.total_queries += 1
            self._ensure_node(requesting_node)

            for blk in blocks:
                # 1. Try Exact Causal Prefix Match
                causal_reps = self._causal_index.get(blk.causal_hash)
                if causal_chain_intact and causal_reps:
                    rep, eff_tier = self._select_best_replica(causal_reps, requesting_node)
                    wire_bytes = (
                        0
                        if eff_tier == MemoryTier.L1_LOCAL_VRAM
                        else (blk.num_tokens * self.raw_kv_bytes_per_token) // 2
                    )
                    block_hits.append(
                        BlockLookupHit(
                            block=blk,
                            hit_type="CAUSAL_EXACT",
                            tier=eff_tier,
                            source_node_id=rep.node_id,
                            position_delta=0,
                            selective_recompute_ratio=0.0,
                            bytes_transferred=wire_bytes,
                        )
                    )
                    tier_counts[eff_tier] += blk.num_tokens
                    exact_tokens += blk.num_tokens
                    continue

                # Once a causal block misses, downstream blocks cannot be exact causal matches
                # UNLESS their full prefix existed on another node (checked via causal_reps above).
                causal_chain_intact = False

                # 2. Try Position-Independent Pre-RoPE Modular Match (for shifted RAG/code blocks)
                mod_reps = (
                    self._modular_index.get((effective_tenant, blk.modular_hash))
                    if self.enable_modular_rope_reuse
                    else None
                )
                if mod_reps:
                    rep, eff_tier = self._select_best_replica(mod_reps, requesting_node)
                    pos_delta = blk.start_token_offset - rep.cached_token_offset
                    wire_bytes = (
                        0
                        if eff_tier == MemoryTier.L1_LOCAL_VRAM
                        else (blk.num_tokens * self.raw_kv_bytes_per_token) // 2
                    )
                    block_hits.append(
                        BlockLookupHit(
                            block=blk,
                            hit_type="MODULAR_ROPE_SHIFT",
                            tier=eff_tier,
                            source_node_id=rep.node_id,
                            position_delta=pos_delta,
                            selective_recompute_ratio=0.12,  # 12% boundary attention blending
                            bytes_transferred=wire_bytes,
                        )
                    )
                    tier_counts[eff_tier] += blk.num_tokens
                    modular_tokens += blk.num_tokens
                    continue

                # 3. Complete Cache Miss -> Compute on GPU
                block_hits.append(
                    BlockLookupHit(
                        block=blk,
                        hit_type="MISS",
                        tier=MemoryTier.MISS,
                        source_node_id=None,
                        position_delta=0,
                        selective_recompute_ratio=1.0,
                        bytes_transferred=0,
                    )
                )
                tier_counts[MemoryTier.MISS] += blk.num_tokens
                recompute_tokens += blk.num_tokens

            if auto_populate_on_miss:
                self.register_blocks(blocks, node_id=requesting_node)

        elapsed_us = (time.perf_counter() - t0) * 1e6
        return SequenceLookupPlan(
            total_tokens=len(token_ids),
            exact_causal_tokens=exact_tokens,
            modular_reused_tokens=modular_tokens,
            recompute_tokens=recompute_tokens,
            block_hits=block_hits,
            tier_breakdown=tier_counts,
            lookup_latency_us=elapsed_us,
        )
