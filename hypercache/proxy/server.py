"""
Transparent OpenAI-compatible Reverse Proxy Middleware for HyperCache.

Intercepts `/v1/chat/completions` and `/v1/completions` to:
1. Parse prompt text or conversation messages into deterministic token blocks.
2. Query the Global Radix / Hash DAG for prefix hits across L1/L2/L3 tiers.
3. Forward the request to the upstream inference engine (vLLM, SGLang, Ollama, etc.).
4. Attach low-overhead telemetry headers:
   - `x-hypercache-hit-tokens`: Number of cached prefix tokens matched.
   - `x-hypercache-total-tokens`: Total prompt tokens analyzed.
   - `x-hypercache-hit-ratio`: Effective cache reuse ratio.
   - `x-hypercache-tier`: Dominant memory tier of cached state.
   - `x-hypercache-est-saved-ms`: Estimated TTFT compute latency saved.

Also exposes:
  `GET /v1/hypercache/stats`: Cluster-wide memory usage & hit-rate telemetry.
  `GET /v1/hypercache/health`: Readiness probe for Kubernetes / Docker.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import threading
import time
from typing import Any, Optional
import urllib.error
import urllib.request

from hypercache.core.hasher import BlockHasher
from hypercache.core.radix_dag import GlobalRadixDAG, MemoryTier
from hypercache.transport.fabric import KernelBypassFabric


class HyperCacheProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler implementing the transparent prefix caching proxy."""

    server: HyperCacheProxyServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default noisy access logs unless verbose
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)

    def _send_json_response(self, status_code: int, data: dict[str, Any]) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/v1/hypercache/health":
            self._send_json_response(200, {"status": "ok", "version": "0.1.0"})
            return

        if self.path == "/v1/hypercache/stats":
            stats = self.server.get_stats()
            self._send_json_response(200, stats)
            return

        # Forward any other GET request upstream
        self._proxy_generic_request("GET")

    def do_POST(self) -> None:
        if self.path in ("/v1/chat/completions", "/v1/completions"):
            self._handle_completion_request()
            return

        self._proxy_generic_request("POST")

    def _extract_prompt_text(self, body: dict[str, Any]) -> str:
        """Extracts linear prompt text from standard OpenAI chat or completion bodies."""
        if "messages" in body and isinstance(body["messages"], list):
            parts = []
            for msg in body["messages"]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle multimodal / structured message blocks
                    text_parts = [
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    content = " ".join(text_parts)
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            return "\n".join(parts)

        if "prompt" in body:
            prompt = body["prompt"]
            if isinstance(prompt, str):
                return prompt
            if isinstance(prompt, list):
                return " ".join(str(p) for p in prompt)

        return ""

    def _tokenize_fallback(self, text: str) -> list[int]:
        """
        Deterministic word-level tokenization fallback for proxy parsing when
        an engine-specific tokenizer is not loaded.
        """
        words = text.split()
        return [abs(hash(w)) % 100_000 for w in words]

    def _handle_completion_request(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            body = json.loads(raw_body.decode("utf-8"))
        except Exception:
            body = {}

        prompt_text = self._extract_prompt_text(body)
        token_ids = self._tokenize_fallback(prompt_text)

        tenant_id = self.headers.get("X-Tenant-ID", "public")
        node_id = self.server.node_id

        # Query HyperCache distributed Radix DAG
        plan = self.server.dag.lookup_sequence(
            token_ids=token_ids,
            requesting_node=node_id,
            tenant_id=tenant_id,
            auto_populate_on_miss=True,
        )
        breakdown = self.server.fabric.evaluate_plan_latency(plan)

        dominant_tier = MemoryTier.MISS
        for tier in (
            MemoryTier.L1_LOCAL_VRAM,
            MemoryTier.L2_PEER_RDMA,
            MemoryTier.L2_PINNED_RAM,
            MemoryTier.L3_NVME_SSD,
        ):
            if plan.tier_breakdown.get(tier, 0) > 0:
                dominant_tier = tier
                break

        # Forward request to upstream server
        upstream_url = f"{self.server.upstream_url.rstrip('/')}{self.path}"
        req = urllib.request.Request(
            upstream_url,
            data=raw_body,
            headers={
                k: v
                for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length")
            },
            method="POST",
        )
        req.add_header("Content-Length", str(len(raw_body)))

        try:
            with urllib.request.urlopen(req, timeout=self.server.upstream_timeout) as resp:
                resp_data = resp.read()
                self.send_response(resp.status)
                for header, value in resp.getheaders():
                    if header.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(header, value)

                # Inject HyperCache telemetry headers
                self.send_header("x-hypercache-hit-tokens", str(plan.total_cached_tokens))
                self.send_header("x-hypercache-total-tokens", str(plan.total_tokens))
                self.send_header(
                    "x-hypercache-hit-ratio", f"{plan.effective_cache_hit_ratio:.3f}"
                )
                self.send_header("x-hypercache-tier", dominant_tier.value)
                self.send_header("x-hypercache-est-saved-ms", f"{breakdown.saved_ms:.1f}")
                self.send_header("Content-Length", str(len(resp_data)))
                self.end_headers()
                self.wfile.write(resp_data)

        except urllib.error.HTTPError as e:
            err_data = e.read()
            self.send_response(e.code)
            for header, value in e.headers.items():
                if header.lower() not in ("transfer-encoding", "content-length"):
                    self.send_header(header, value)
            self.send_header("Content-Length", str(len(err_data)))
            self.end_headers()
            self.wfile.write(err_data)
        except Exception as e:
            self._send_json_response(
                502,
                {
                    "error": {
                        "message": f"HyperCache upstream connection error: {e}",
                        "type": "upstream_error",
                    }
                },
            )

    def _proxy_generic_request(self, method: str) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else None

        upstream_url = f"{self.server.upstream_url.rstrip('/')}{self.path}"
        req = urllib.request.Request(
            upstream_url,
            data=raw_body,
            headers={
                k: v
                for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length")
            },
            method=method,
        )
        if raw_body:
            req.add_header("Content-Length", str(len(raw_body)))

        try:
            with urllib.request.urlopen(req, timeout=self.server.upstream_timeout) as resp:
                resp_data = resp.read()
                self.send_response(resp.status)
                for header, value in resp.getheaders():
                    if header.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(header, value)
                self.send_header("Content-Length", str(len(resp_data)))
                self.end_headers()
                self.wfile.write(resp_data)
        except urllib.error.HTTPError as e:
            err_data = e.read()
            self.send_response(e.code)
            for header, value in e.headers.items():
                if header.lower() not in ("transfer-encoding", "content-length"):
                    self.send_header(header, value)
            self.send_header("Content-Length", str(len(err_data)))
            self.end_headers()
            self.wfile.write(err_data)
        except Exception as e:
            self._send_json_response(
                502, {"error": {"message": f"HyperCache upstream error: {e}"}}
            )


class HyperCacheProxyServer(socketserver.ThreadingMixIn, HTTPServer):
    """Multi-threaded transparent proxy server with embedded HyperCache DAG directory."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        upstream_url: str = "http://127.0.0.1:8080",
        node_id: str = "proxy-node-01",
        upstream_timeout: float = 60.0,
        verbose: bool = False,
        dag: Optional[GlobalRadixDAG] = None,
        fabric: Optional[KernelBypassFabric] = None,
    ) -> None:
        super().__init__((host, port), HyperCacheProxyHandler)
        self.upstream_url = upstream_url
        self.node_id = node_id
        self.upstream_timeout = upstream_timeout
        self.verbose = verbose
        self.dag = dag or GlobalRadixDAG(hasher=BlockHasher(block_size=16))
        self.fabric = fabric or KernelBypassFabric()
        self._thread: Optional[threading.Thread] = None

    def get_stats(self) -> dict[str, Any]:
        """Returns cluster cache metrics and memory tier breakdown."""
        usage = self.dag.get_node_usage(self.node_id)
        return {
            "node_id": self.node_id,
            "total_queries": self.dag.total_queries,
            "hotspot_replications": self.dag.hotspot_replications,
            "l1_to_l2_demotions": self.dag.l1_to_l2_demotions,
            "l2_to_l3_demotions": self.dag.l2_to_l3_demotions,
            "memory_usage_bytes": {
                "l1_vram": usage.get(MemoryTier.L1_LOCAL_VRAM, 0),
                "l2_pinned_ram": usage.get(MemoryTier.L2_PINNED_RAM, 0),
                "l3_nvme_ssd": usage.get(MemoryTier.L3_NVME_SSD, 0),
            },
        }

    def start_background(self) -> None:
        self._thread = threading.Thread(
            target=self.serve_forever, name="hypercache-proxy", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self.shutdown()
        self.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
