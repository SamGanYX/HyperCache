"""
`hypercache top`: Live Cluster Memory Fabric & RDMA Telemetry Dashboard.

Provides a real-time view of L1 (GPU VRAM), L2 (Peer RDMA / Pinned Host RAM),
and L3 (NVMe) utilization, active GPUDirect RDMA streams, P95 TTFT reduction,
and live GPU compute dollars saved.
"""

from __future__ import annotations

import sys
import time

from hypercache.cli.auditor import generate_workload
from hypercache.core.hasher import BlockHasher
from hypercache.core.quantizer import KVQuantMode
from hypercache.core.radix_dag import GlobalRadixDAG, MemoryTier
from hypercache.transport.fabric import (
    GPU_PROFILES,
    KernelBypassFabric,
    TransportBackend,
)


def _bar(ratio: float, width: int = 18) -> str:
    clamped = max(0.0, min(1.0, ratio))
    filled = int(round(clamped * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def run_top_dashboard(
    duration_sec: float = 4.0,
    num_nodes: int = 4,
    gpu_profile: str = "h100",
    refresh_interval: float = 0.5,
) -> None:
    """
    Runs a live simulation of the HyperCache distributed memory fabric across `num_nodes`,
    printing the telemetry dashboard as requests stream through the cluster.
    """
    # Ensure UTF-8 stdout on Windows PowerShell if supported
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    hw = GPU_PROFILES.get(gpu_profile.lower(), GPU_PROFILES["h100"])
    dag = GlobalRadixDAG(
        hasher=BlockHasher(block_size=64),
        l1_capacity_bytes_per_node=20 * 1024**3,
        l2_capacity_bytes_per_node=128 * 1024**3,
        l3_capacity_bytes_per_node=512 * 1024**3,
        enable_modular_rope_reuse=True,
    )
    fabric = KernelBypassFabric(
        gpu_profile=gpu_profile,
        transport=TransportBackend.RDMA_ROCE_400G,
        quant_mode=KVQuantMode.FP8_E4M3,
        layer_overlap=True,
    )

    requests = generate_workload(
        workload_type="multi-agent-swe",
        num_requests=240,
        num_nodes=num_nodes,
    )

    total_tokens = 0
    cached_tokens = 0
    total_saved_ms = 0.0
    ttft_samples: list[float] = []
    recompute_samples: list[float] = []
    recent_events: list[str] = []

    steps = max(1, int(duration_sec / max(0.1, refresh_interval)))
    reqs_per_step = max(1, len(requests) // steps)
    cursor = 0

    for step_idx in range(steps):
        batch = requests[cursor : cursor + reqs_per_step]
        cursor = (cursor + reqs_per_step) % len(requests)

        for req in batch:
            plan = dag.lookup_sequence(
                req.token_ids,
                requesting_node=req.assigned_node,
                tenant_id=req.tenant_id,
            )
            bd = fabric.evaluate_plan_latency(plan)
            total_tokens += plan.total_tokens
            cached_tokens += plan.total_cached_tokens
            total_saved_ms += bd.saved_ms
            ttft_samples.append(bd.hypercache_exposed_ttft_ms)
            recompute_samples.append(bd.full_recompute_ttft_ms)

            rdma_toks = plan.tier_breakdown.get(MemoryTier.L2_PEER_RDMA, 0)
            l1_toks = plan.tier_breakdown.get(MemoryTier.L1_LOCAL_VRAM, 0)
            if rdma_toks > 0:
                src_nodes = {
                    h.source_node_id
                    for h in plan.block_hits
                    if h.tier == MemoryTier.L2_PEER_RDMA and h.source_node_id
                }
                src_label = next(iter(src_nodes), "gpu-node-01")
                event = (
                    f"{src_label} ==(GPUDirect RDMA FP8 | {rdma_toks // 1024}k tok)==> "
                    f"{req.assigned_node} | TTFT: {bd.hypercache_exposed_ttft_ms:5.1f}ms "
                    f"(Saved {bd.saved_ms / 1000:.2f}s)"
                )
                recent_events.append(event)
            elif l1_toks > 0:
                event = (
                    f"{req.assigned_node} ==(Local L1 HBM Hit  | {l1_toks // 1024}k tok)==> "
                    f"{req.assigned_node} | TTFT: {bd.hypercache_exposed_ttft_ms:5.1f}ms "
                    f"(Saved {bd.saved_ms / 1000:.2f}s)"
                )
                recent_events.append(event)

        recent_events = recent_events[-5:]

        if step_idx < steps - 1 and duration_sec > 0:
            time.sleep(min(refresh_interval, 0.15))

    # Aggregate final cluster memory usage across all nodes
    l1_used = sum(
        dag.get_node_usage(f"gpu-node-{i + 1:02d}")[MemoryTier.L1_LOCAL_VRAM]
        for i in range(num_nodes)
    ) / (1024**3)
    l2_used = sum(
        dag.get_node_usage(f"gpu-node-{i + 1:02d}")[MemoryTier.L2_PINNED_RAM]
        for i in range(num_nodes)
    ) / (1024**3)
    l3_used = sum(
        dag.get_node_usage(f"gpu-node-{i + 1:02d}")[MemoryTier.L3_NVME_SSD]
        for i in range(num_nodes)
    ) / (1024**3)

    l1_cap = (dag.l1_capacity * num_nodes) / (1024**3)
    l2_cap = (dag.l2_capacity * num_nodes) / (1024**3)
    l3_cap = (dag.l3_capacity * num_nodes) / (1024**3)

    hit_rate = (cached_tokens / max(1, total_tokens)) * 100.0
    avg_ttft = sum(ttft_samples) / max(1, len(ttft_samples))
    avg_recomp = sum(recompute_samples) / max(1, len(recompute_samples))

    # Extrapolate to 24h production traffic at 250k req/day (4-GPU TP per node)
    scale_24h = 250_000 / max(1, len(ttft_samples))
    tokens_saved_24h_b = (cached_tokens * scale_24h) / 1e9
    gpu_hours_saved_24h = (((total_saved_ms * scale_24h) / 1000.0) / 3600.0) * 4.0
    usd_saved_24h = gpu_hours_saved_24h * hw.gpu_hourly_cost_usd

    dashboard = [
        "",
        "+------------------------------------------------------------------------------+",
        f"| HYPERCACHE v0.1.0 - Distributed GPU Memory Fabric [{num_nodes}x {hw.name}] |",
        "+--------------------------+--------------------------+------------------------+",
        f"| CLUSTER CACHE HIT RATE   | AVG TIME-TO-FIRST-TOKEN  | COMPUTE SAVED (24H)    |",
        f"| {hit_rate:5.1f}% (Causal+PreRoPE)  | {avg_ttft:5.1f} ms (was {avg_recomp:5.0f} ms)  | "
        f"{tokens_saved_24h_b:4.1f}B Tok (${usd_saved_24h:,.0f}/day) |",
        "+--------------------------+--------------------------+------------------------+",
        "| HETEROGENEOUS MEMORY TIERS (CLUSTER-WIDE POOL)                               |",
        f"|   L1 (GPU HBM BF16)   : {_bar(l1_used / max(1.0, l1_cap))} {l1_used:6.1f} / {l1_cap:5.1f} GB          |",
        f"|   L2 (Peer RDMA/RAM)  : {_bar(l2_used / max(1.0, l2_cap))} {l2_used:6.1f} / {l2_cap:5.1f} GB (FP8)    |",
        f"|   L3 (NVMe SPDK Pool) : {_bar(l3_used / max(1.0, l3_cap))} {l3_used:6.1f} / {l3_cap:5.1f} GB (INT4)   |",
        "+------------------------------------------------------------------------------+",
        "| LIVE GPUDIRECT RDMA & LAYER-OVERLAP STREAM TELEMETRY                         |",
    ]
    for ev in recent_events:
        dashboard.append(f"|   > {ev:<72} |")
    dashboard.append(
        "+------------------------------------------------------------------------------+"
    )
    dashboard.append("")
    print("\n".join(dashboard))
