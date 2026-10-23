"""
Drop-In vLLM `KVConnector` & SGLang RadixCache Adapter for HyperCache.

Allows existing vLLM workers to query the global HyperCache Radix DAG, stream remote
KV blocks over GPUDirect RDMA with layer-wise CUDA stream overlap, and apply on-the-fly
relative RoPE phase shifts without forking the underlying serving engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from hypercache.core.hasher import BlockHasher
from hypercache.core.quantizer import KVQuantMode
from hypercache.core.radix_dag import GlobalRadixDAG, MemoryTier, SequenceLookupPlan
from hypercache.transport.fabric import (
    KernelBypassFabric,
    PrefillLatencyBreakdown,
    TransportBackend,
)


@dataclass
class HyperCacheConfig:
    """Configuration for a drop-in HyperCache node connector."""

    node_id: str = "gpu-node-01"
    gpu_profile: str = "h100"
    block_size: int = 16
    transport: str = "rdma_roce"
    quantization: str = "fp8_e4m3"
    enable_pre_rope_modular: bool = True
    layer_overlap_streams: bool = True
    tenant_hmac_isolation: bool = True
    l1_vram_gb: float = 24.0
    l2_pinned_ram_gb: float = 128.0
    l3_nvme_gb: float = 1024.0


@dataclass
class ConnectorLookupResult:
    """Result returned to the vLLM / SGLang scheduler before prefill execution."""

    request_id: str
    node_id: str
    tenant_id: str
    total_tokens: int
    cached_tokens: int
    exact_causal_tokens: int
    modular_reused_tokens: int
    recompute_tokens: int
    exposed_latency_ms: float
    full_recompute_ms: float
    legacy_host_tcp_ms: float
    saved_ms: float
    speedup_factor: float
    wire_bytes_transferred: int
    plan: SequenceLookupPlan
    latency_breakdown: PrefillLatencyBreakdown


class HyperCacheVLLMConnector:
    """
    Drop-in KV-Cache Connector implementing the vLLM `KVConnectorBase` lifecycle:
      1. `lookup_and_prefetch()` -> Queries global Radix DAG & schedules async RDMA layer pulls
      2. `register_computed_blocks()` -> Publishes newly computed suffix KV blocks to L1/global DAG
    """

    _TRANSPORT_MAP = {
        "nvlink": TransportBackend.NVLINK_P2P,
        "nvlink_p2p": TransportBackend.NVLINK_P2P,
        "rdma": TransportBackend.RDMA_ROCE_400G,
        "rdma_roce": TransportBackend.RDMA_ROCE_400G,
        "pinned_dma": TransportBackend.PINNED_PCIE_GEN5,
        "tcp_fp8": TransportBackend.CLOUD_TCP_COMPRESSED,
        "legacy_host_tcp": TransportBackend.LEGACY_HOST_TCP,
    }

    _QUANT_MAP = {
        "none": KVQuantMode.NONE,
        "bf16": KVQuantMode.NONE,
        "fp8": KVQuantMode.FP8_E4M3,
        "fp8_e4m3": KVQuantMode.FP8_E4M3,
        "int4": KVQuantMode.INT4_GROUP,
    }

    def __init__(
        self,
        config: Optional[HyperCacheConfig] = None,
        shared_dag: Optional[GlobalRadixDAG] = None,
    ) -> None:
        self.config = config or HyperCacheConfig()
        self.hasher = BlockHasher(
            block_size=self.config.block_size,
            enable_tenant_isolation=self.config.tenant_hmac_isolation,
        )
        self.dag = shared_dag or GlobalRadixDAG(
            hasher=self.hasher,
            l1_capacity_bytes_per_node=int(self.config.l1_vram_gb * (1024**3)),
            l2_capacity_bytes_per_node=int(self.config.l2_pinned_ram_gb * (1024**3)),
            l3_capacity_bytes_per_node=int(self.config.l3_nvme_gb * (1024**3)),
            enable_modular_rope_reuse=self.config.enable_pre_rope_modular,
        )
        transport_enum = self._TRANSPORT_MAP.get(
            self.config.transport.lower(), TransportBackend.RDMA_ROCE_400G
        )
        quant_enum = self._QUANT_MAP.get(
            self.config.quantization.lower(), KVQuantMode.FP8_E4M3
        )
        self.fabric = KernelBypassFabric(
            gpu_profile=self.config.gpu_profile,
            transport=transport_enum,
            quant_mode=quant_enum,
            layer_overlap=self.config.layer_overlap_streams,
        )

    def lookup_and_prefetch(
        self,
        request_id: str,
        token_ids: Sequence[int],
        tenant_id: str = "public",
        auto_populate: bool = True,
    ) -> ConnectorLookupResult:
        """
        Checks the global Radix DAG for cached causal or Pre-RoPE modular KV blocks,
        schedules layer-wise RDMA transfers into vLLM's PagedAttention buffers, and
        returns the execution plan and TTFT telemetry.
        """
        plan = self.dag.lookup_sequence(
            token_ids=list(token_ids),
            requesting_node=self.config.node_id,
            tenant_id=tenant_id,
            auto_populate_on_miss=auto_populate,
        )
        breakdown = self.fabric.evaluate_plan_latency(plan)
        return ConnectorLookupResult(
            request_id=request_id,
            node_id=self.config.node_id,
            tenant_id=tenant_id,
            total_tokens=plan.total_tokens,
            cached_tokens=plan.exact_causal_tokens + plan.modular_reused_tokens,
            exact_causal_tokens=plan.exact_causal_tokens,
            modular_reused_tokens=plan.modular_reused_tokens,
            recompute_tokens=plan.recompute_tokens,
            exposed_latency_ms=breakdown.hypercache_exposed_ttft_ms,
            full_recompute_ms=breakdown.full_recompute_ttft_ms,
            legacy_host_tcp_ms=breakdown.legacy_host_tcp_ttft_ms,
            saved_ms=breakdown.saved_ms,
            speedup_factor=breakdown.speedup_vs_recompute,
            wire_bytes_transferred=breakdown.wire_bytes_transferred,
            plan=plan,
            latency_breakdown=breakdown,
        )

    def node_memory_summary_gb(self) -> dict[str, float]:
        """Returns current L1/L2/L3 memory utilization on this worker node in GB."""
        usage = self.dag.get_node_usage(self.config.node_id)
        return {
            "L1_VRAM_GB": usage[MemoryTier.L1_LOCAL_VRAM] / (1024**3),
            "L2_PINNED_RAM_GB": usage[MemoryTier.L2_PINNED_RAM] / (1024**3),
            "L3_NVME_GB": usage[MemoryTier.L3_NVME_SSD] / (1024**3),
        }
