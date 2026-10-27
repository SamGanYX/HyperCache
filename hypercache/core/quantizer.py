"""
Wire-Speed KV-Cache Quantization (FP8 / INT4) & Relative RoPE Phase-Shift Engine.

Key Mathematical Insight for Position-Independent KV Reuse:
Because Rotary Position Embeddings (RoPE) apply an orthogonal 2D rotation matrix
`R(m * theta_i)` to each complex coordinate pair `(k_{2i}, k_{2i+1})` at token index `m`,
the group property of SO(2) rotations guarantees:
    R(m_new * theta_i) = R((m_new - m_old) * theta_i) @ R(m_old * theta_i)

Therefore, HyperCache does NOT require invasive model-surgery to intercept pre-RoPE Keys!
Any standard post-RoPE Key block cached in vLLM at token offset `m_old` can be directly
rotated into a new token offset `m_new` on fetch using only the relative shift
`delta_m = m_new - m_old`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np


class KVQuantMode(str, Enum):
    NONE = "bf16"       # 2 bytes / element (baseline)
    FP8_E4M3 = "fp8"    # 1 byte / element  (2.0x wire & memory compression)
    INT4_GROUP = "int4" # 0.5 bytes / elem  (3.8x wire & memory compression with scales)


class RoPEScalingType(str, Enum):
    DEFAULT = "default"
    LINEAR = "linear"
    YARN = "yarn"


@dataclass
class RoPEScalingConfig:
    """Configuration for extended-context RoPE scaling protocols (YaRN, Linear)."""

    scaling_type: RoPEScalingType = RoPEScalingType.DEFAULT
    factor: float = 1.0
    original_max_position_embeddings: int = 8192
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0


@dataclass
class QuantizedKVBuffer:
    """Compressed representation of a KV-cache block ready for RDMA or L2/L3 storage."""

    mode: KVQuantMode
    packed_k: np.ndarray
    packed_v: np.ndarray
    scale_k: np.ndarray
    scale_v: np.ndarray
    original_shape: tuple[int, ...]
    cached_start_pos: int

    @property
    def compressed_bytes(self) -> int:
        return (
            self.packed_k.nbytes
            + self.packed_v.nbytes
            + self.scale_k.nbytes
            + self.scale_v.nbytes
        )

    @property
    def raw_bf16_bytes(self) -> int:
        # 2 tensors (K and V) * 2 bytes per BF16/FP16 element
        num_elements = int(np.prod(self.original_shape))
        return 2 * num_elements * 2

    @property
    def compression_ratio(self) -> float:
        return self.raw_bf16_bytes / max(1, self.compressed_bytes)

    def slice_tokens(self, start_token: int, num_tokens: int) -> QuantizedKVBuffer:
        """
        Slices a subset of tokens [start_token .. start_token + num_tokens] from the buffer
        without full dequantization, enabling sub-block cache reuse.
        """
        if start_token < 0 or num_tokens <= 0:
            raise ValueError("Invalid slice parameters.")
        end_token = start_token + num_tokens
        new_shape = list(self.original_shape)
        new_shape[-3] = num_tokens

        sliced_k = np.ascontiguousarray(self.packed_k[..., start_token:end_token, :, :])
        sliced_v = np.ascontiguousarray(self.packed_v[..., start_token:end_token, :, :])

        if self.scale_k.ndim >= 3 and self.scale_k.shape[-3] > 1:
            sliced_sk = np.ascontiguousarray(self.scale_k[..., start_token:end_token, :, :])
            sliced_sv = np.ascontiguousarray(self.scale_v[..., start_token:end_token, :, :])
        else:
            sliced_sk = self.scale_k.copy()
            sliced_sv = self.scale_v.copy()

        return QuantizedKVBuffer(
            mode=self.mode,
            packed_k=sliced_k,
            packed_v=sliced_v,
            scale_k=sliced_sk,
            scale_v=sliced_sv,
            original_shape=tuple(new_shape),
            cached_start_pos=self.cached_start_pos + start_token,
        )


class KVQuantizer:
    """
    Quantizes and dequantizes KV-cache blocks between L1 (BF16/FP16),
    L2 (FP8 E4M3 simulated in int8 with per-head dynamic scales), and
    L3 (Grouped INT4 bit-packed into uint8).
    """

    FP8_MAX = 448.0  # Standard IEEE / OCP FP8 E4M3 max normal value

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Computes numerical cosine similarity between two tensors."""
        a_f = a.reshape(-1).astype(np.float64)
        b_f = b.reshape(-1).astype(np.float64)
        dot = np.dot(a_f, b_f)
        norm_a = np.linalg.norm(a_f)
        norm_b = np.linalg.norm(b_f)
        return float(dot / max(1e-12, norm_a * norm_b))

    @classmethod
    def quantize(
        cls,
        k_tensor: np.ndarray,
        v_tensor: np.ndarray,
        mode: KVQuantMode = KVQuantMode.FP8_E4M3,
        cached_start_pos: int = 0,
    ) -> QuantizedKVBuffer:
        """
        Quantizes K and V tensors of shape `(..., num_tokens, num_kv_heads, head_dim)`.
        """
        if k_tensor.shape != v_tensor.shape:
            raise ValueError(f"K and V shape mismatch: {k_tensor.shape} vs {v_tensor.shape}")

        orig_shape = k_tensor.shape
        k_f32 = k_tensor.astype(np.float32, copy=False)
        v_f32 = v_tensor.astype(np.float32, copy=False)

        if mode == KVQuantMode.NONE:
            return QuantizedKVBuffer(
                mode=mode,
                packed_k=k_f32.astype(np.float16),
                packed_v=v_f32.astype(np.float16),
                scale_k=np.ones((1,), dtype=np.float16),
                scale_v=np.ones((1,), dtype=np.float16),
                original_shape=orig_shape,
                cached_start_pos=cached_start_pos,
            )

        if mode == KVQuantMode.FP8_E4M3:
            # Per-token, per-head dynamic scaling along the last dimension (head_dim)
            max_k = np.maximum(np.max(np.abs(k_f32), axis=-1, keepdims=True), 1e-6)
            max_v = np.maximum(np.max(np.abs(v_f32), axis=-1, keepdims=True), 1e-6)
            scale_k = (max_k / 127.0).astype(np.float16)
            scale_v = (max_v / 127.0).astype(np.float16)

            q_k = np.clip(np.round(k_f32 / scale_k), -127, 127).astype(np.int8)
            q_v = np.clip(np.round(v_f32 / scale_v), -127, 127).astype(np.int8)

            return QuantizedKVBuffer(
                mode=mode,
                packed_k=q_k,
                packed_v=q_v,
                scale_k=scale_k,
                scale_v=scale_v,
                original_shape=orig_shape,
                cached_start_pos=cached_start_pos,
            )

        if mode == KVQuantMode.INT4_GROUP:
            # Symmetric INT4 [-7, 7] packed 2 values per uint8 byte along head_dim
            if orig_shape[-1] % 2 != 0:
                raise ValueError("head_dim must be even for INT4 byte packing.")

            max_k = np.maximum(np.max(np.abs(k_f32), axis=-1, keepdims=True), 1e-6)
            max_v = np.maximum(np.max(np.abs(v_f32), axis=-1, keepdims=True), 1e-6)
            scale_k = (max_k / 7.0).astype(np.float16)
            scale_v = (max_v / 7.0).astype(np.float16)

            q_k = np.clip(np.round(k_f32 / scale_k), -7, 7).astype(np.int8) + 8
            q_v = np.clip(np.round(v_f32 / scale_v), -7, 7).astype(np.int8) + 8

            packed_k = ((q_k[..., 0::2].astype(np.uint8) << 4) | (q_k[..., 1::2].astype(np.uint8) & 0x0F))
            packed_v = ((q_v[..., 0::2].astype(np.uint8) << 4) | (q_v[..., 1::2].astype(np.uint8) & 0x0F))

            return QuantizedKVBuffer(
                mode=mode,
                packed_k=packed_k,
                packed_v=packed_v,
                scale_k=scale_k,
                scale_v=scale_v,
                original_shape=orig_shape,
                cached_start_pos=cached_start_pos,
            )

        raise ValueError(f"Unsupported quantization mode: {mode}")

    @classmethod
    def dequantize(cls, buf: QuantizedKVBuffer) -> tuple[np.ndarray, np.ndarray]:
        """Dequantizes a `QuantizedKVBuffer` back into float32/float16 K and V tensors."""
        if buf.mode == KVQuantMode.NONE:
            return buf.packed_k.astype(np.float32), buf.packed_v.astype(np.float32)

        if buf.mode == KVQuantMode.FP8_E4M3:
            k_out = buf.packed_k.astype(np.float32) * buf.scale_k.astype(np.float32)
            v_out = buf.packed_v.astype(np.float32) * buf.scale_v.astype(np.float32)
            return k_out, v_out

        if buf.mode == KVQuantMode.INT4_GROUP:
            k_hi = ((buf.packed_k >> 4) & 0x0F).astype(np.int8) - 8
            k_lo = (buf.packed_k & 0x0F).astype(np.int8) - 8
            v_hi = ((buf.packed_v >> 4) & 0x0F).astype(np.int8) - 8
            v_lo = (buf.packed_v & 0x0F).astype(np.int8) - 8

            k_unpacked = np.empty(buf.original_shape, dtype=np.float32)
            v_unpacked = np.empty(buf.original_shape, dtype=np.float32)
            k_unpacked[..., 0::2] = k_hi
            k_unpacked[..., 1::2] = k_lo
            v_unpacked[..., 0::2] = v_hi
            v_unpacked[..., 1::2] = v_lo

            return (
                k_unpacked * buf.scale_k.astype(np.float32),
                v_unpacked * buf.scale_v.astype(np.float32),
            )

        raise ValueError(f"Unsupported quantization mode: {buf.mode}")


class RoPEPhaseRotator:
    """
    Applies relative Rotary Position Embedding (RoPE) phase shifts `R((m_new - m_old) * theta)`
    to cached Key blocks so that a document or code block cached at token offset `m_old`
    can be reused at an arbitrary new prompt offset `m_new`.
    """

    def __init__(
        self,
        head_dim: int = 128,
        rope_theta: float = 500000.0,
        scaling_config: Optional[RoPEScalingConfig] = None,
    ) -> None:
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be divisible by 2 for RoPE rotation.")
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.scaling_config = scaling_config or RoPEScalingConfig()

        dim_indices = np.arange(0, head_dim, 2, dtype=np.float64)
        base_inv_freq = 1.0 / (rope_theta ** (dim_indices / head_dim))

        cfg = self.scaling_config
        if cfg.scaling_type == RoPEScalingType.DEFAULT:
            self.inv_freq = base_inv_freq
        elif cfg.scaling_type == RoPEScalingType.LINEAR:
            factor = max(1e-4, cfg.factor)
            self.inv_freq = base_inv_freq / factor
        elif cfg.scaling_type == RoPEScalingType.YARN:
            factor = max(1e-4, cfg.factor)
            wavelengths = 2.0 * np.pi / base_inv_freq
            low_bound = cfg.beta_slow * cfg.original_max_position_embeddings
            high_bound = cfg.beta_fast * cfg.original_max_position_embeddings

            # Smooth interpolation ramp between low and high wavelengths
            # For wavelength < low_bound (high frequency), ramp is 0 (no interpolation)
            # For wavelength > high_bound (low frequency), ramp is 1 (full factor interpolation)
            ramp = np.clip(
                (wavelengths - low_bound) / max(1.0, high_bound - low_bound),
                0.0,
                1.0,
            )
            yarn_inv_freq = (1.0 - ramp) * base_inv_freq + ramp * (base_inv_freq / factor)
            self.inv_freq = yarn_inv_freq * cfg.mscale
        else:
            self.inv_freq = base_inv_freq

    def apply_absolute_rope(self, k_pre_rope: np.ndarray, start_pos: int = 0) -> np.ndarray:
        """
        Applies standard per-token Rotary Position Embedding (RoPE) across sequence
        positions `[start_pos, start_pos + 1, ..., start_pos + seq_len - 1]` for a
        tensor of shape `(..., seq_len, num_kv_heads, head_dim)`.
        """
        seq_len = k_pre_rope.shape[-3]
        positions = (start_pos + np.arange(seq_len, dtype=np.float64))[
            :, None, None
        ]  # (seq_len, 1, 1)
        angles = positions * self.inv_freq[None, None, :]  # (seq_len, 1, head_dim // 2)
        cos_a = np.cos(angles).astype(k_pre_rope.dtype)
        sin_a = np.sin(angles).astype(k_pre_rope.dtype)

        k_even = k_pre_rope[..., 0::2]
        k_odd = k_pre_rope[..., 1::2]

        out = np.empty_like(k_pre_rope)
        out[..., 0::2] = k_even * cos_a - k_odd * sin_a
        out[..., 1::2] = k_even * sin_a + k_odd * cos_a
        return out

    def shift_positions(
        self,
        k_cached: np.ndarray,
        old_start_pos: int,
        new_start_pos: int,
    ) -> np.ndarray:
        """
        Rotates a cached Key tensor of shape `(..., seq_len, num_kv_heads, head_dim)`
        from `old_start_pos` to `new_start_pos` using the SO(2) group identity:
            R(new_pos) = R(new_pos - old_pos) @ R(old_pos)
        """
        delta_pos = new_start_pos - old_start_pos
        if delta_pos == 0:
            return k_cached.copy()

        # Since delta_pos is constant across all tokens in the contiguous block:
        # (old_start_pos + i) -> (new_start_pos + i) shifts every token by exactly `delta_pos`!
        angles = delta_pos * self.inv_freq  # shape: (head_dim // 2,)
        cos_a = np.cos(angles).astype(k_cached.dtype)
        sin_a = np.sin(angles).astype(k_cached.dtype)

        k_even = k_cached[..., 0::2]
        k_odd = k_cached[..., 1::2]

        out = np.empty_like(k_cached)
        out[..., 0::2] = k_even * cos_a - k_odd * sin_a
        out[..., 1::2] = k_even * sin_a + k_odd * cos_a
        return out
