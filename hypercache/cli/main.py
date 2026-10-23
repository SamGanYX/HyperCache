"""
Unified CLI Entrypoint for HyperCache:
  - `python -m hypercache audit`
  - `python -m hypercache top`
  - `python -m hypercache benchmark`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

from benchmarks.break_even_benchmark import run_break_even_benchmark
from hypercache.cli.auditor import format_audit_report, run_cluster_audit
from hypercache.cli.top import run_top_dashboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypercache",
        description="HyperCache: The Redis of GPU Memory — Distributed KV-Cache Fabric",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    # 1. `hypercache audit`
    p_audit = sub.add_parser(
        "audit",
        help="Run a zero-risk cluster prefix audit & CFO ROI report on simulated or JSONL logs",
    )
    p_audit.add_argument(
        "--workload",
        default="multi-agent-swe",
        choices=["multi-agent-swe", "enterprise-rag"],
        help="Built-in workload profile to simulate if --log-file is not provided",
    )
    p_audit.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to a .jsonl file of prompt requests",
    )
    p_audit.add_argument(
        "--nodes",
        type=int,
        default=4,
        help="Number of GPU nodes in the cluster (default: 4)",
    )
    p_audit.add_argument(
        "--gpu",
        default="h100",
        choices=["h100", "a100", "b200"],
        help="GPU hardware profile (default: h100)",
    )

    # 2. `hypercache top`
    p_top = sub.add_parser(
        "top",
        help="Launch the live cluster memory tier & GPUDirect RDMA telemetry dashboard",
    )
    p_top.add_argument(
        "--duration",
        type=float,
        default=1.5,
        help="Simulation duration in seconds (default: 1.5)",
    )
    p_top.add_argument(
        "--nodes",
        type=int,
        default=4,
        help="Number of GPU nodes in the cluster (default: 4)",
    )
    p_top.add_argument(
        "--gpu",
        default="h100",
        choices=["h100", "a100", "b200"],
        help="GPU hardware profile (default: h100)",
    )

    # 3. `hypercache benchmark`
    p_bench = sub.add_parser(
        "benchmark",
        help="Run the physics break-even & numerical kernel verification benchmark",
    )
    p_bench.add_argument(
        "--gpu",
        default="h100",
        choices=["h100", "a100", "b200"],
        help="GPU hardware profile (default: h100)",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cmd = args.command or "audit"
    if cmd == "audit":
        workload = getattr(args, "workload", "multi-agent-swe")
        nodes = getattr(args, "nodes", 4)
        gpu = getattr(args, "gpu", "h100")
        log_file = getattr(args, "log_file", None)
        report = run_cluster_audit(
            workload_type=workload,
            num_nodes=nodes,
            gpu_profile=gpu,
            jsonl_path=log_file,
        )
        print(format_audit_report(report))
        return 0

    if cmd == "top":
        run_top_dashboard(
            duration_sec=args.duration,
            num_nodes=args.nodes,
            gpu_profile=args.gpu,
        )
        return 0

    if cmd == "benchmark":
        run_break_even_benchmark(gpu_profile=args.gpu)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
