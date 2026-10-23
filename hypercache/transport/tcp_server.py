"""
Real Binary TCP / Loopback KV-Block Transport Server & Client (`hypercached` Data Plane).

Provides a working, zero-dependency network transport daemon so developers on any laptop,
workstation, or standard cloud VM (without InfiniBand hardware) can run a live multi-worker
KV-cache cluster over TCP sockets with wire-speed FP8/INT4 compression and CRC32 checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import socket
import socketserver
import struct
import threading
import time
from typing import Optional

import numpy as np

from hypercache.core.quantizer import KVQuantizer, KVQuantMode, RoPEPhaseRotator
from hypercache.core.storage_engine import PhysicalTieredStorageEngine


@dataclass
class WireTransferResult:
    """Real measured wall-clock network transfer telemetry for a KV tensor block."""

    block_hash: str
    found: bool
    k_tensor: Optional[np.ndarray]
    v_tensor: Optional[np.ndarray]
    wire_bytes: int
    raw_bf16_bytes: int
    roundtrip_ms: float
    effective_gbps: float


def _recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
    buf = bytearray(n_bytes)
    view = memoryview(buf)
    pos = 0
    while pos < n_bytes:
        n_read = sock.recv_into(view[pos:], n_bytes - pos)
        if n_read == 0:
            raise ConnectionError("Socket closed mid-frame.")
        pos += n_read
    return bytes(buf)


class _KVRequestHandler(socketserver.BaseRequestHandler):
    """
    Binary Wire Protocol:
      Request Header: `cmd (1B: 'P'=PUT, 'G'=GET) | key_len (2B uint16) | payload_len (4B uint32)`
      followed by `key_bytes` and (if PUT) `payload_bytes`.
      Response Header: `status (1B: 1=OK, 0=MISS/ERR) | payload_len (4B uint32)`
      followed by `payload_bytes`.
    """

    def handle(self) -> None:
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server: HyperCacheTCPServer = self.server  # type: ignore[assignment]
        try:
            while True:
                hdr = self.request.recv(7)
                if not hdr:
                    break
                if len(hdr) < 7:
                    hdr += _recv_exact(self.request, 7 - len(hdr))
                cmd_byte, key_len, payload_len = struct.unpack("<c H I", hdr)
                key = _recv_exact(self.request, key_len).decode("utf-8")

                if cmd_byte == b"P":
                    payload = _recv_exact(self.request, payload_len)
                    server.put_raw_frame(key, payload)
                    self.request.sendall(struct.pack("<B I", 1, 0))
                elif cmd_byte == b"G":
                    frame = server.get_raw_frame(key)
                    if frame is None:
                        self.request.sendall(struct.pack("<B I", 0, 0))
                    else:
                        self.request.sendall(struct.pack("<B I", 1, len(frame)) + frame)
                else:
                    self.request.sendall(struct.pack("<B I", 0, 0))
        except (ConnectionError, OSError):
            pass


class HyperCacheTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """
    Multi-threaded binary KV-block cache server storing compressed FP8/INT4 frames.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        super().__init__((host, port), _KVRequestHandler)
        self._lock = threading.RLock()
        self._frames: dict[str, bytes] = {}
        self._thread: Optional[threading.Thread] = None

    @property
    def bound_port(self) -> int:
        return int(self.server_address[1])

    def start_background(self) -> None:
        self._thread = threading.Thread(
            target=self.serve_forever, name="hypercached-tcp", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self.shutdown()
        self.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def put_raw_frame(self, key: str, frame: bytes) -> None:
        with self._lock:
            self._frames[key] = frame

    def get_raw_frame(self, key: str) -> Optional[bytes]:
        with self._lock:
            return self._frames.get(key)


class HyperCacheTCPClient:
    """
    Client that quantizes `(K, V)` blocks into FP8/INT4 frames, streams them to a remote
    `HyperCacheTCPServer`, and fetches + dequantizes + RoPE-rotates them on demand.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9470,
        quant_mode: KVQuantMode = KVQuantMode.FP8_E4M3,
        head_dim: int = 128,
    ) -> None:
        self.host = host
        self.port = port
        self.quant_mode = quant_mode
        self.rotator = RoPEPhaseRotator(head_dim=head_dim)

    def put_kv_block(
        self,
        block_hash: str,
        k_tensor: np.ndarray,
        v_tensor: np.ndarray,
        cached_start_pos: int = 0,
    ) -> int:
        """Quantizes a KV block, verifies CRC32 framing, and pushes it to the TCP daemon."""
        q_buf = KVQuantizer.quantize(
            k_tensor, v_tensor, mode=self.quant_mode, cached_start_pos=cached_start_pos
        )
        frame = PhysicalTieredStorageEngine.serialize_quantized_buffer(q_buf)
        key_bytes = block_hash.encode("utf-8")

        with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            hdr = struct.pack("<c H I", b"P", len(key_bytes), len(frame))
            sock.sendall(hdr + key_bytes + frame)
            resp = _recv_exact(sock, 5)
            status, _ = struct.unpack("<B I", resp)
            if status != 1:
                raise RuntimeError("Remote HyperCacheTCPServer rejected PUT frame.")
        return len(frame)

    def get_kv_block(
        self,
        block_hash: str,
        new_start_pos: Optional[int] = None,
    ) -> WireTransferResult:
        """
        Fetches a compressed KV block over TCP, verifies its CRC32 checksum, dequantizes
        to float32, and optionally applies an on-the-fly relative RoPE phase shift to
        `new_start_pos`.
        """
        key_bytes = block_hash.encode("utf-8")
        t0 = time.perf_counter()

        with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            hdr = struct.pack("<c H I", b"G", len(key_bytes), 0)
            sock.sendall(hdr + key_bytes)
            resp = _recv_exact(sock, 5)
            status, payload_len = struct.unpack("<B I", resp)
            if status != 1 or payload_len == 0:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return WireTransferResult(
                    block_hash=block_hash,
                    found=False,
                    k_tensor=None,
                    v_tensor=None,
                    wire_bytes=0,
                    raw_bf16_bytes=0,
                    roundtrip_ms=elapsed_ms,
                    effective_gbps=0.0,
                )
            frame = _recv_exact(sock, payload_len)

        q_buf = PhysicalTieredStorageEngine.deserialize_quantized_buffer(frame)
        k_f32, v_f32 = KVQuantizer.dequantize(q_buf)
        if new_start_pos is not None and new_start_pos != q_buf.cached_start_pos:
            k_f32 = self.rotator.shift_positions(
                k_f32, old_start_pos=q_buf.cached_start_pos, new_start_pos=new_start_pos
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        gbps = (len(frame) / 1e9) / max(1e-6, elapsed_ms / 1000.0)
        return WireTransferResult(
            block_hash=block_hash,
            found=True,
            k_tensor=k_f32,
            v_tensor=v_f32,
            wire_bytes=len(frame),
            raw_bf16_bytes=q_buf.raw_bf16_bytes,
            roundtrip_ms=elapsed_ms,
            effective_gbps=gbps,
        )
