"""DCT-domain dithered-QIM utilities for diffusion initial latents.

The module is intentionally small and dependency-light:
* DCT-II basis / transform helpers
* key-permuted carrier selection
* binary dithered-QIM embedding
* hard-decision and soft-LLR extraction

The public functions are used by both the end-to-end steganography runner and
the DCT-domain inversion-channel sounder.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def dct_basis(
    n: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return the orthonormal DCT-II basis matrix ``B``."""
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cpu")

    k = torch.arange(n, device=device, dtype=dtype)[:, None]
    x = torch.arange(n, device=device, dtype=dtype)[None, :]
    basis = torch.cos(math.pi * (x + 0.5) * k / n) * math.sqrt(2.0 / n)
    basis[0] /= math.sqrt(2.0)
    return basis


def dct2(x: torch.Tensor, basis: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply a 2-D DCT over the last two dimensions: ``Y = B X B^T``."""
    if basis is None:
        basis = dct_basis(x.shape[-1], x.device, x.dtype)
    return torch.einsum("ik,bckl,jl->bcij", basis, x, basis)


def idct2(c: torch.Tensor, basis: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply a 2-D inverse DCT: ``X = B^T C B``."""
    if basis is None:
        basis = dct_basis(c.shape[-1], c.device, c.dtype)
    return torch.einsum("ik,bcij,jl->bckl", basis, c, basis)


def carrier_positions(
    channels: int,
    size: int,
    key: int,
    min_radius: float,
    max_radius: float,
) -> np.ndarray:
    """Return key-permuted ``(channel, y, x)`` DCT carrier positions."""
    y, x = np.mgrid[:size, :size]
    radius = np.hypot(y, x)
    mask = (radius >= min_radius) & (radius <= max_radius)
    coords = np.argwhere(mask)
    positions = np.asarray(
        [(ch, yy, xx) for ch in range(channels) for yy, xx in coords],
        dtype=np.int64,
    )
    rng = np.random.default_rng(key)
    return positions[rng.permutation(len(positions))]


def dithers(key: int, step: float, n: int) -> np.ndarray:
    """Deterministic dithers for QIM embedding / extraction."""
    return np.random.default_rng(key ^ 0x5EED).uniform(0.0, step, n)


def carrier_step_values(
    positions: np.ndarray,
    step: float,
    min_radius: float,
    max_radius: float,
    mode: str = "global",
    min_scale: float = 0.75,
    max_scale: float = 1.25,
) -> np.ndarray:
    """Return deterministic per-carrier QIM steps.

    Radial mode allocates a smaller step to lower spatial frequencies and a
    larger step to higher frequencies. The schedule depends only on public
    carrier geometry, so extraction stays synchronized without side data.
    """
    if mode not in ("global", "radial"):
        raise ValueError("step mode must be either 'global' or 'radial'")
    if step <= 0 or min_scale <= 0 or max_scale <= 0:
        raise ValueError("step and step scales must be positive")
    if min_scale > max_scale:
        raise ValueError("min_scale must not exceed max_scale")
    if mode == "global":
        return np.full(len(positions), float(step), dtype=np.float64)
    if max_radius <= min_radius:
        raise ValueError("max_radius must exceed min_radius for radial steps")
    radii = np.hypot(positions[:, 1], positions[:, 2])
    alpha = np.clip(
        (radii - float(min_radius)) / (float(max_radius) - float(min_radius)),
        0.0,
        1.0,
    )
    scales = float(min_scale) + alpha * (float(max_scale) - float(min_scale))
    return float(step) * scales


def _dithers_for_steps(key: int, steps: np.ndarray) -> np.ndarray:
    """Generate the same deterministic dither fraction for each carrier."""
    unit = np.random.default_rng(key ^ 0x5EED).uniform(0.0, 1.0, len(steps))
    return unit * np.asarray(steps, dtype=np.float64)


def _normal_cdf(value: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    z = (value - mean) / std
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def _normal_icdf(probability: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    z = math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)
    return mean + std * z


def _cdf_dithers(key: int, cdf_step: float, n: int) -> np.ndarray:
    return np.random.default_rng(key ^ 0x5EED).uniform(0.0, cdf_step, n)


def count_carriers(
    channels: int,
    size: int,
    min_radius: float,
    max_radius: float,
) -> int:
    """Number of available DCT carriers before key permutation."""
    y, x = np.mgrid[:size, :size]
    radius = np.hypot(y, x)
    spatial = int(np.count_nonzero((radius >= min_radius) & (radius <= max_radius)))
    return channels * spatial


def _validate(latent: torch.Tensor, n_bits: int, repeat: int, step: float) -> None:
    if latent.ndim != 4 or latent.shape[0] != 1:
        raise ValueError("expected square [1,C,H,W] latents")
    if latent.shape[-1] != latent.shape[-2]:
        raise ValueError("expected square latent spatial dimensions")
    if n_bits < 1 or repeat < 1 or step <= 0:
        raise ValueError("n_bits, repeat, and step must be positive")


def embed_message_in_latents(
    init_latents: torch.Tensor,
    message_bits: np.ndarray,
    key_seed: int,
    step: float = 0.8,
    repeat: int = 5,
    min_radius: float = 8.0,
    max_radius: float = 26.0,
    embedding_mode: str = "nearest",
    rounding_seed: int = None,
    step_mode: str = "global",
    step_min_scale: float = 0.75,
    step_max_scale: float = 1.25,
    cdf_step: float = 0.08,
    cdf_mean: float = 0.0,
    cdf_std: float = 1.0,
    cdf_clip: float = 1.0e-4,
) -> torch.Tensor:
    """Embed a binary message with dithered binary QIM in DCT domain.

    ``nearest`` reproduces the original nearest-lattice-point QIM.  In
    ``stochastic`` mode the two adjacent points of the target bit's QIM
    lattice are sampled with distance-proportional probabilities.  The
    selected point always belongs to the correct bit lattice, while the
    conditional quantization error has zero mean.  The receiver therefore
    needs no rounding seed and the existing extractor remains unchanged.
    """
    bits = np.asarray(message_bits, dtype=np.int8).reshape(-1)
    _validate(init_latents, len(bits), repeat, step)
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("message_bits must be binary")
    if embedding_mode not in ("nearest", "stochastic", "cdf"):
        raise ValueError(
            "embedding_mode must be 'nearest', 'stochastic', or 'cdf'"
        )
    if embedding_mode == "stochastic" and rounding_seed is None:
        raise ValueError("rounding_seed is required for stochastic QIM")
    if embedding_mode == "cdf" and not (0.0 < cdf_step < 1.0):
        raise ValueError("cdf_step must be between 0 and 1")
    if embedding_mode == "cdf" and (cdf_std <= 0 or not (0.0 < cdf_clip < 0.5)):
        raise ValueError("cdf_std must be positive and cdf_clip in (0, 0.5)")

    channels = init_latents.shape[1]
    size = init_latents.shape[-1]
    positions = carrier_positions(channels, size, key_seed, min_radius, max_radius)
    needed = len(bits) * repeat
    if needed > len(positions):
        raise ValueError(
            f"message too long: needs {needed} carriers, "
            f"capacity {len(positions)//repeat} bits"
        )

    work = init_latents.detach().double()
    basis = dct_basis(size, work.device, work.dtype)
    coeff = dct2(work, basis).clone()
    selected_positions = positions[:needed]
    local_steps = carrier_step_values(
        selected_positions, step, min_radius, max_radius, step_mode,
        step_min_scale, step_max_scale
    )
    ds = _dithers_for_steps(key_seed, local_steps)
    cdf_ds = _cdf_dithers(key_seed, cdf_step, needed)
    rounding_rng = (
        np.random.default_rng(rounding_seed)
        if embedding_mode == "stochastic"
        else None
    )

    for i, (ch, yy, xx) in enumerate(positions[:needed]):
        bit = int(bits[i // repeat])
        if embedding_mode == "cdf":
            probability = _normal_cdf(
                coeff[0, ch, yy, xx], cdf_mean, cdf_std
            ).clamp(cdf_clip, 1.0 - cdf_clip)
            offset = float(cdf_ds[i]) + bit * cdf_step / 2.0
            quantized_probability = (
                cdf_step
                * torch.round((probability - offset) / cdf_step)
                + offset
            ).clamp(cdf_clip, 1.0 - cdf_clip)
            coeff[0, ch, yy, xx] = _normal_icdf(
                quantized_probability, cdf_mean, cdf_std
            )
            continue
        local_step = float(local_steps[i])
        offset = float(ds[i]) + bit * local_step / 2.0
        normalized = (coeff[0, ch, yy, xx] - offset) / local_step
        if embedding_mode == "nearest":
            quantized_index = torch.round(normalized)
        else:
            lower_index = torch.floor(normalized)
            upper_probability = float((normalized - lower_index).item())
            choose_upper = rounding_rng.random() < upper_probability
            quantized_index = lower_index + int(choose_upper)
        coeff[0, ch, yy, xx] = local_step * quantized_index + offset

    return idct2(coeff, basis).to(init_latents.dtype)


def extract_message_details_from_latents(
    reversed_latents: torch.Tensor,
    n_bits: int,
    key_seed: int,
    step: float = 0.8,
    repeat: int = 5,
    min_radius: float = 8.0,
    max_radius: float = 26.0,
    step_mode: str = "global",
    step_min_scale: float = 0.75,
    step_max_scale: float = 1.25,
    cdf_step: float = 0.08,
    cdf_mean: float = 0.0,
    cdf_std: float = 1.0,
    cdf_clip: float = 1.0e-4,
    embedding_mode: str = "nearest",
) -> Dict[str, np.ndarray]:
    """Extract hard decisions and soft LLRs from DCT-domain QIM carriers.

    Returns
    -------
    dict
        * ``carrier_llrs``: per-carrier LLR, positive favours bit 0
        * ``carrier_hard``: per-carrier hard decision
        * ``message_llrs``: summed LLR for each message bit
        * ``soft_bits``: soft-decision bits from LLR sign
        * ``vote_bits``: majority-vote bits, matching the original decoder
    """
    _validate(reversed_latents, n_bits, repeat, step)

    channels = reversed_latents.shape[1]
    size = reversed_latents.shape[-1]
    positions = carrier_positions(channels, size, key_seed, min_radius, max_radius)
    needed = n_bits * repeat
    if needed > len(positions):
        raise ValueError(
            f"payload needs {needed} carriers, "
            f"capacity {len(positions)//repeat} bits"
        )

    work = reversed_latents.detach().double()
    basis = dct_basis(size, work.device, work.dtype)
    coeff = dct2(work, basis)
    selected_positions = positions[:needed]
    local_steps = carrier_step_values(
        selected_positions, step, min_radius, max_radius, step_mode,
        step_min_scale, step_max_scale
    )
    ds = _dithers_for_steps(key_seed, local_steps)
    cdf_ds = _cdf_dithers(key_seed, cdf_step, needed)

    carrier_llrs = np.empty(needed, dtype=np.float32)
    for i, (ch, yy, xx) in enumerate(positions[:needed]):
        value = coeff[0, ch, yy, xx]
        if embedding_mode == "cdf":
            probability = _normal_cdf(value, cdf_mean, cdf_std).clamp(
                cdf_clip, 1.0 - cdf_clip
            )
            o0 = float(cdf_ds[i])
            o1 = o0 + cdf_step / 2.0
            d0 = torch.abs(
                probability
                - (
                    cdf_step
                    * torch.round((probability - o0) / cdf_step)
                    + o0
                )
            )
            d1 = torch.abs(
                probability
                - (
                    cdf_step
                    * torch.round((probability - o1) / cdf_step)
                    + o1
                )
            )
            carrier_llrs[i] = float(d1 - d0)
            continue
        local_step = float(local_steps[i])
        o0 = float(ds[i])
        o1 = o0 + local_step / 2.0

        d0 = torch.abs(
            value - (local_step * torch.round((value - o0) / local_step) + o0)
        )
        d1 = torch.abs(
            value - (local_step * torch.round((value - o1) / local_step) + o1)
        )
        carrier_llrs[i] = float(d1 - d0)

    carrier_hard = (carrier_llrs < 0.0).astype(np.int8)
    message_llrs = carrier_llrs.reshape(n_bits, repeat).sum(axis=1)
    soft_bits = (message_llrs < 0.0).astype(np.int8)
    vote_bits = (
        carrier_hard.reshape(n_bits, repeat).sum(axis=1) >= (repeat / 2.0)
    ).astype(np.int8)

    return {
        "carrier_llrs": carrier_llrs,
        "carrier_hard": carrier_hard,
        "message_llrs": message_llrs.astype(np.float32),
        "soft_bits": soft_bits,
        "vote_bits": vote_bits,
    }


def extract_message_from_latents(
    reversed_latents: torch.Tensor,
    n_bits: int,
    key_seed: int,
    step: float = 0.8,
    repeat: int = 5,
    min_radius: float = 8.0,
    max_radius: float = 26.0,
) -> np.ndarray:
    """Original majority-vote decoder; kept for backward compatibility."""
    details = extract_message_details_from_latents(
        reversed_latents,
        n_bits,
        key_seed,
        step,
        repeat,
        min_radius,
        max_radius,
    )
    return details["vote_bits"]


def extract_message_soft_from_latents(
    reversed_latents: torch.Tensor,
    n_bits: int,
    key_seed: int,
    step: float = 0.8,
    repeat: int = 5,
    min_radius: float = 8.0,
    max_radius: float = 26.0,
) -> np.ndarray:
    """Soft-decision repetition decoder based on summed LLRs."""
    details = extract_message_details_from_latents(
        reversed_latents,
        n_bits,
        key_seed,
        step,
        repeat,
        min_radius,
        max_radius,
    )
    return details["soft_bits"]
