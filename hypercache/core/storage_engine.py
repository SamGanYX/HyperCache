"""
Physical Multi-Tier KV Tensor Storage Engine (L1 Memory -> L2 FP8 Buffer -> L3 INT4 Disk/mmap).

Addresses the Reddit/HN Critique:
"Does the repo actually store, quantize, demote, and serialize real KV tensors to memory
and NVMe/disk, or does it only track block hashes?"

This module implements a real, thread-safe physical tensor store:
  - Tier L1: Uncompressed float16/float32 `(K, V)` numpy tensors in active working memory.
  - Tier L2: Automatically demotes evicted L1 tensors into `FP8_E4M3` quantized byte buffers
             in host memory (or pushes them over the wire via `HyperCacheTCPServer`).
  - Tier L3: Automatically spills evicted L2 tensors to binary `.hckv` block files on disk
             using `INT4_GROUP` bit-packing + CRC32 integrity verification, reading them
             back via `mmap` / binary I/O on demand.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import struct
import tempfile
import threading
from typing import Optional
import zlib

import numpy as np

from hypercache.core.quantizer import KVQuantizer, KVQuantMode, QuantizedKVBuffer
from hypercache.core.radix_dag import MemoryTier


@dataclass
class PhysicalFetchResult:
    """Returned when a physical KV tensor block is retrieved from L1, L2, or L3."""

    block_hash: str
    source_tier: MemoryTier
    k_tensor: np.ndarray
    v_tensor: np.ndarray
    cached_start_pos: int
    wire_or_disk_bytes: int


class PhysicalTieredStorageEngine:
    """
    Thread-safe physical KV tensor manager across L1 (FP16/FP32), L2 (FP8), and L3 (INT4 Disk).
    """

    MAGIC_HEADER = b"HCKV01"

    def __init__(
        self,
        l1_max_blocks: int = 32,
        l2_max_blocks: int = 128,
        l3_storage_dir: Optional[Path] = None,
    ) -> None:
        self.l1_max_blocks = max(1, l1_max_blocks)
        self.l2_max_blocks = max(1, l2_max_blocks)
        self._temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        if l3_storage_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="hypercache_l3_")
            self.l3_dir = Path(self._temp_dir.name)
        else:
            self.l3_dir = Path(l3_storage_dir)
            self.l3_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        # L1 LRU: block_hash -> (k_fp16, v_fp16, cached_start_pos)
        self._l1_store: OrderedDict[str, tuple[np.ndarray, np.ndarray, int]] = OrderedDict()
        # L2 LRU: block_hash -> QuantizedKVBuffer (FP8_E4M3)
        self._l2_store: OrderedDict[str, QuantizedKVBuffer] = OrderedDict()
        # L3 Disk Index: block_hash -> Path to `.hckv` binary INT4 file
        self._l3_index: dict[str, Path] = {}

        self.l1_hits = 0
        self.l2_hits = 0
        self.l3_hits = 0
        self.l1_to_l2_demotions = 0
        self.l2_to_l3_demotions = 0

    def put_block(
        self,
        block_hash: str,
        k_tensor: np.ndarray,
        v_tensor: np.ndarray,
        cached_start_pos: int = 0,
    ) -> None:
        """Stores a KV tensor block into L1, automatically demoting cold blocks to L2 -> L3."""
        with self._lock:
            if block_hash in self._l1_store:
                self._l1_store.move_to_end(block_hash)
                return

            while len(self._l1_store) >= self.l1_max_blocks:
                victim_hash, (vk, vv, vpos) = self._l1_store.popitem(last=False)
                self._demote_to_l2(victim_hash, vk, vv, vpos)

            self._l1_store[block_hash] = (
                k_tensor.astype(np.float16, copy=True),
                v_tensor.astype(np.float16, copy=True),
                cached_start_pos,
            )

    def _demote_to_l2(
        self,
        block_hash: str,
        k_tensor: np.ndarray,
        v_tensor: np.ndarray,
        cached_start_pos: int,
    ) -> None:
        while len(self._l2_store) >= self.l2_max_blocks:
            victim_hash, victim_buf = self._l2_store.popitem(last=False)
            self._demote_to_l3_disk(victim_hash, victim_buf)

        fp8_buf = KVQuantizer.quantize(
            k_tensor, v_tensor, mode=KVQuantMode.FP8_E4M3, cached_start_pos=cached_start_pos
        )
        self._l2_store[block_hash] = fp8_buf
        self.l1_to_l2_demotions += 1

    def _demote_to_l3_disk(self, block_hash: str, fp8_buf: QuantizedKVBuffer) -> None:
        """Dequantizes FP8 and bit-packs into INT4 on disk with a CRC32 checksum."""
        k_f32, v_f32 = KVQuantizer.dequantize(fp8_buf)
        int4_buf = KVQuantizer.quantize(
            k_f32,
            v_f32,
            mode=KVQuantMode.INT4_GROUP,
            cached_start_pos=fp8_buf.cached_start_pos,
        )
        payload = self.serialize_quantized_buffer(int4_buf)
        file_path = self.l3_dir / f"{block_hash}.hckv"
        file_path.write_bytes(payload)
        self._l3_index[block_hash] = file_path
        self.l2_to_l3_demotions += 1

    def get_block(
        self, block_hash: str, promote_to_l1: bool = False
    ) -> Optional[PhysicalFetchResult]:
        """Retrieves and dequantizes a physical KV tensor block from L1, L2, or L3 disk."""
        with self._lock:
            # 1. Check L1
            if block_hash in self._l1_store:
                self._l1_store.move_to_end(block_hash)
                k, v, pos = self._l1_store[block_hash]
                self.l1_hits += 1
                return PhysicalFetchResult(
                    block_hash=block_hash,
                    source_tier=MemoryTier.L1_LOCAL_VRAM,
                    k_tensor=k.astype(np.float32),
                    v_tensor=v.astype(np.float32),
                    cached_start_pos=pos,
                    wire_or_disk_bytes=0,
                )

            # 2. Check L2 (FP8 quantized in-memory pool)
            if block_hash in self._l2_store:
                self._l2_store.move_to_end(block_hash)
                fp8_buf = self._l2_store[block_hash]
                k_f32, v_f32 = KVQuantizer.dequantize(fp8_buf)
                self.l2_hits += 1
                if promote_to_l1:
                    self.put_block(block_hash, k_f32, v_f32, fp8_buf.cached_start_pos)
                return PhysicalFetchResult(
                    block_hash=block_hash,
                    source_tier=MemoryTier.L2_PINNED_RAM,
                    k_tensor=k_f32,
                    v_tensor=v_f32,
                    cached_start_pos=fp8_buf.cached_start_pos,
                    wire_or_disk_bytes=fp8_buf.compressed_bytes,
                )

            # 3. Check L3 (INT4 bit-packed binary file on disk)
            if block_hash in self._l3_index:
                file_path = self._l3_index[block_hash]
                if file_path.exists():
                    raw_bytes = file_path.read_bytes()
                    int4_buf = self.deserialize_quantized_buffer(raw_bytes)
                    k_f32, v_f32 = KVQuantizer.dequantize(int4_buf)
                    self.l3_hits += 1
                    if promote_to_l1:
                        self.put_block(block_hash, k_f32, v_f32, int4_buf.cached_start_pos)
                    return PhysicalFetchResult(
                        block_hash=block_hash,
                        source_tier=MemoryTier.L3_NVME_SSD,
                        k_tensor=k_f32,
                        v_tensor=v_f32,
                        cached_start_pos=int4_buf.cached_start_pos,
                        wire_or_disk_bytes=len(raw_bytes),
                    )

            return None

    @classmethod
    def serialize_quantized_buffer(cls, buf: QuantizedKVBuffer) -> bytes:
        """
        Packs a `QuantizedKVBuffer` into a zero-copy-friendly binary wire frame with a
        CRC32 checksum header (used by both L3 NVMe storage and TCP/RDMA transport).
        """
        mode_bytes = buf.mode.value.encode("ascii")
        ndim = len(buf.original_shape)
        shape_header = struct.pack(f"<{ndim}I", *buf.original_shape)

        scale_shape = buf.scale_k.shape
        scale_ndim = len(scale_shape)
        scale_shape_header = struct.pack(f"<{scale_ndim}I", *scale_shape)

        pk_bytes = buf.packed_k.tobytes()
        pv_bytes = buf.packed_v.tobytes()
        sk_bytes = buf.scale_k.tobytes()
        sv_bytes = buf.scale_v.tobytes()

        body = b"".join(
            [
                struct.pack(
                    "<B i B B I I I I",
                    len(mode_bytes),
                    buf.cached_start_pos,
                    ndim,
                    scale_ndim,
                    len(pk_bytes),
                    len(pv_bytes),
                    len(sk_bytes),
                    len(sv_bytes),
                ),
                mode_bytes,
                shape_header,
                scale_shape_header,
                pk_bytes,
                pv_bytes,
                sk_bytes,
                sv_bytes,
            ]
        )
        crc = zlib.crc32(body) & 0xFFFFFFFF
        return cls.MAGIC_HEADER + struct.pack("<I", crc) + body

    @classmethod
    def deserialize_quantized_buffer(cls, data: bytes) -> QuantizedKVBuffer:
        """Unpacks and verifies a CRC32-protected binary KV frame."""
        if len(data) < 10 or data[:6] != cls.MAGIC_HEADER:
            raise ValueError("Invalid HyperCache binary KV frame magic header.")

        (expected_crc,) = struct.unpack("<I", data[6:10])
        body = data[10:]
        actual_crc = zlib.crc32(body) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(
                f"CRC32 checksum mismatch in KV block frame: expected {expected_crc:#x}, got {actual_crc:#x}"
            )

        hdr_size = struct.calcsize("<B i B B I I I I")
        (
            mode_len,
            cached_start_pos,
            ndim,
            scale_ndim,
            pk_len,
            pv_len,
            sk_len,
            sv_len,
        ) = struct.unpack("<B i B B I I I I", body[:hdr_size])
        offset = hdr_size

        mode_str = body[offset : offset + mode_len].decode("ascii")
        offset += mode_len
        mode = KVQuantMode(mode_str)

        orig_shape = struct.unpack(f"<{ndim}I", body[offset : offset + 4 * ndim])
        offset += 4 * ndim

        scale_shape = struct.unpack(
            f"<{scale_ndim}I", body[offset : offset + 4 * scale_ndim]
        )
        offset += 4 * scale_ndim

        if mode == KVQuantMode.NONE:
            packed_dtype = np.float16
            packed_shape = orig_shape
        elif mode == KVQuantMode.FP8_E4M3:
            packed_dtype = np.int8
            packed_shape = orig_shape
        else:
            packed_dtype = np.uint8
            packed_shape = (*orig_shape[:-1], orig_shape[-1] // 2)

        pk = np.frombuffer(body[offset : offset + pk_len], dtype=packed_dtype).reshape(
            packed_shape
        )
        offset += pk_len
        pv = np.frombuffer(body[offset : offset + pv_len], dtype=packed_dtype).reshape(
            packed_shape
        )
        offset += pv_len

        sk = np.frombuffer(body[offset : offset + sk_len], dtype=np.float16).reshape(
            scale_shape
        )
        offset += sk_len
        sv = np.frombuffer(body[offset : offset + sv_len], dtype=np.float16).reshape(
            scale_shape
        )

        return QuantizedKVBuffer(
            mode=mode,
            packed_k=pk.copy(),
            packed_v=pv.copy(),
            scale_k=sk.copy(),
            scale_v=sv.copy(),
            original_shape=orig_shape,
            cached_start_pos=cached_start_pos,
        )

    def close(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
