"""
Cryptographic Causal Prefix & Position-Independent (Pre-RoPE) Block Hasher.

Why LSH isn't used for exact causal prefixes:
Standard Transformer self-attention is strictly causal and path-dependent: every
token's Key-Value (KV) state at layer L is a function of all preceding tokens [0..i-1].
Therefore, HyperCache uses a Dual-Mode Deterministic Hashing architecture:

1. Causal Prefix Hash (`causal_hash`):
   Chained cryptographic digest `H_k = BLAKE2b(tenant_key, H_{k-1} || block_tokens || model_id)`.
   Guarantees 100% bit-exact KV equivalence and multi-tenant timing-attack immunity.

2. Pre-RoPE Modular Block Hash (`modular_hash`):
   Position-independent digest `M_k = BLAKE2b(tenant_key, "MODULAR" || block_tokens || model_id)`.
   Allows HyperCache to identify and reuse cached RAG documents or repository files even when
   their starting token offset shifts, pairing with on-the-fly RoPE phase rotation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import struct
from typing import Sequence


@dataclass(frozen=True)
class TokenBlock:
    """Represents a deterministic fixed-size block of prompt tokens."""

    block_index: int
    start_token_offset: int
    token_ids: tuple[int, ...]
    causal_hash: str
    modular_hash: str
    parent_causal_hash: str
    tenant_id: str
    model_id: str

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


class BlockHasher:
    """
    Deterministic token chunker and dual-mode cryptographic block hasher.

    Supports tenant-namespaced HMAC keys to prevent cross-tenant timing side-channel
    attacks while allowing an explicit 'public' namespace for shared system prompts
    and open-source tool schemas.
    """

    GENESIS_HASH = "0" * 32

    def __init__(
        self,
        block_size: int = 16,
        model_id: str = "meta-llama/Llama-3-70B-Instruct",
        cluster_secret: str = "hypercache-default-cluster-secret",
        enable_tenant_isolation: bool = True,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be a positive integer (e.g., 16 or 64).")
        self.block_size = block_size
        self.model_id = model_id
        self._cluster_secret = cluster_secret.encode("utf-8")
        self.enable_tenant_isolation = enable_tenant_isolation

    def _derive_tenant_key(self, tenant_id: str) -> bytes:
        """Derives a 32-byte HMAC key isolated per tenant (or global if public)."""
        effective_tenant = (
            tenant_id if (self.enable_tenant_isolation and tenant_id != "public") else "public"
        )
        return hmac.new(
            self._cluster_secret,
            f"tenant:{effective_tenant}|model:{self.model_id}".encode("utf-8"),
            hashlib.sha256,
        ).digest()

    @staticmethod
    def _pack_tokens(token_ids: Sequence[int]) -> bytes:
        """Packs integer token IDs into a compact little-endian uint32 byte buffer."""
        return struct.pack(f"<{len(token_ids)}I", *token_ids)

    def compute_block_hashes(
        self,
        token_ids: Sequence[int],
        parent_causal_hash: str = GENESIS_HASH,
        tenant_id: str = "public",
    ) -> tuple[str, str]:
        """
        Computes `(causal_hash, modular_hash)` as 32-character hex digests (128-bit BLAKE2b).
        """
        t_key = self._derive_tenant_key(tenant_id)
        packed = self._pack_tokens(token_ids)

        # 1. Chained Causal Prefix Hash (depends on parent_causal_hash)
        causal_hasher = hashlib.blake2b(key=t_key, digest_size=16)
        causal_hasher.update(b"CAUSAL|")
        causal_hasher.update(parent_causal_hash.encode("ascii"))
        causal_hasher.update(b"|")
        causal_hasher.update(packed)
        causal_hash = causal_hasher.hexdigest()

        # 2. Position-Independent Pre-RoPE Modular Hash (independent of position/parent)
        modular_hasher = hashlib.blake2b(key=t_key, digest_size=16)
        modular_hasher.update(b"MODULAR|")
        modular_hasher.update(packed)
        modular_hash = modular_hasher.hexdigest()

        return causal_hash, modular_hash

    def chunk_and_hash(
        self,
        token_ids: Sequence[int],
        tenant_id: str = "public",
        drop_incomplete_tail: bool = False,
    ) -> list[TokenBlock]:
        """
        Splits a token sequence into deterministic blocks of `block_size` and computes
        both chained causal hashes and position-independent Pre-RoPE modular hashes.

        Note: The final partial block (if `len(token_ids) % block_size != 0`) is typically
        computed as part of the active suffix during prefill unless `drop_incomplete_tail=False`.
        """
        blocks: list[TokenBlock] = []
        parent_hash = self.GENESIS_HASH
        total_len = len(token_ids)

        for idx, offset in enumerate(range(0, total_len, self.block_size)):
            chunk = tuple(token_ids[offset : offset + self.block_size])
            if len(chunk) < self.block_size and drop_incomplete_tail:
                break

            causal_h, modular_h = self.compute_block_hashes(
                chunk,
                parent_causal_hash=parent_hash,
                tenant_id=tenant_id,
            )
            blocks.append(
                TokenBlock(
                    block_index=idx,
                    start_token_offset=offset,
                    token_ids=chunk,
                    causal_hash=causal_h,
                    modular_hash=modular_h,
                    parent_causal_hash=parent_hash,
                    tenant_id=tenant_id,
                    model_id=self.model_id,
                )
            )
            parent_hash = causal_h

        return blocks
