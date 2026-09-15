"""
ATLEF: Core Encoding Algorithm

Implements the two-layer error correction pipeline: Reed-Solomon outer
coding, adaptive repetition inner coding, deterministic random padding
(the padding correction described in Section 4.1.2), the per-channel
orthogonal transform, and the shuffle permutation that together map a
message bitstream to a latent tensor. Matches Section 3.2.3 and
Appendix A.2 of the thesis.
"""

import numpy as np
import torch

from .constants import (
    LATENT_C,
    LATENT_H,
    LATENT_W,
    NUM_LATENT_BITS,
    USE_ECC,
    RS_CODEC,
    device,
)

# ------------------------------------------------------------
# Bit / byte helpers
# ------------------------------------------------------------


def after_message_bits_to_bytes(bits: torch.Tensor):
    """Message bits (0/1) -> bytes, keeping pad bit count (to reach byte boundary)."""
    arr = bits.view(-1).detach().cpu().numpy().astype(np.uint8)
    pad = (-len(arr)) % 8
    if pad > 0:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    return bytes(np.packbits(arr)), int(pad)


def after_bytes_to_message_bits(b: bytes, pad_bits: int, device=device):
    """Bytes -> message bits (0/1), removing pad_bits from the end."""
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = np.unpackbits(arr)
    if pad_bits > 0:
        bits = bits[:-pad_bits]
    return torch.from_numpy(bits.astype(np.int64)).to(device)


def after_bytes_to_bits(b: bytes, device=device):
    """Bytes -> raw bit tensor (0/1)."""
    arr = np.frombuffer(b, dtype=np.uint8)
    return torch.from_numpy(np.unpackbits(arr).astype(np.int64)).to(device)


def after_bits_to_bytes(bits: torch.Tensor):
    """Raw bit tensor (0/1) -> bytes (length must be a multiple of 8)."""
    arr = bits.view(-1).detach().cpu().numpy().astype(np.uint8)
    return bytes(np.packbits(arr))


# ------------------------------------------------------------
# Repetition code (inner code)
# ------------------------------------------------------------


def after_repetition_encode(bits: torch.Tensor, r: int):
    if r <= 1:
        return bits.view(-1).long()
    return bits.view(-1).long().unsqueeze(1).repeat(1, r).view(-1)


def after_repetition_decode(bits: torch.Tensor, r: int):
    """Majority vote per group of r."""
    bits = bits.view(-1).long()
    if r <= 1:
        return bits
    g = bits.view(-1, r)
    votes = g.sum(dim=1)
    return (votes >= (r / 2)).long()


# ------------------------------------------------------------
# Transforms (deterministic by seed, shared with decoder)
# ------------------------------------------------------------


def after_generate_orthogonal_64(seed: int, device=device):
    """64x64 orthogonal matrix, deterministic by seed (QR decomposition)."""
    g = torch.Generator("cpu").manual_seed(int(seed) & 0xFFFFFFFF)
    A = torch.randn(64, 64, generator=g)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diag(R))
    return Q.to(device=device, dtype=torch.float32)


def after_make_permutation(seed: int, numel: int, device=device):
    """Permutation of [0..numel-1], deterministic by seed."""
    g = torch.Generator("cpu").manual_seed(int(seed) & 0xFFFFFFFF)
    return torch.randperm(int(numel), generator=g).to(device)


# ------------------------------------------------------------
# Adaptive repetition factor
# ------------------------------------------------------------


def after_adaptive_rep(ecc_len: int, max_caps: int = NUM_LATENT_BITS, max_r: int = 5):
    """Return r in [1..max_r] such that r * ecc_len <= max_caps, else 0."""
    ecc_len = int(ecc_len)
    max_caps = int(max_caps)
    max_r_allowed = max_caps // ecc_len
    return int(min(max_r, max_r_allowed)) if max_r_allowed >= 1 else 0


# ------------------------------------------------------------
# Core encoder
# ------------------------------------------------------------


def after_encode_core(
    msg_bits: torch.Tensor,
    key_C: int,
    key_shuffle: int,
    rep: int,
    use_ecc: bool,
    scale: float,
    verbose: bool = False,
):
    msg_bits = msg_bits.view(-1).long().to(device)
    orig_len = int(len(msg_bits))

    meta = {
        "orig_num_bits": orig_len,
        "scale": float(scale),
        "rep_factor": int(rep),
        "use_ecc": bool(use_ecc),
    }

    # Step 1: Reed-Solomon outer encoding
    if use_ecc and USE_ECC and (RS_CODEC is not None):
        msg_bytes, pad = after_message_bits_to_bytes(msg_bits)
        enc_bytes = RS_CODEC.encode(msg_bytes)
        ecc_bits = after_bytes_to_bits(enc_bytes, device=device)
        meta["pad_bits"] = int(pad)
        meta["enc_bits_len"] = int(len(ecc_bits))
    else:
        ecc_bits = msg_bits
        meta["pad_bits"] = 0
        meta["enc_bits_len"] = int(len(ecc_bits))

    # Step 2: adaptive repetition inner encoding
    rep_bits = after_repetition_encode(ecc_bits, int(rep))

    if len(rep_bits) > NUM_LATENT_BITS:
        raise ValueError("payload exceeds latent capacity")

    # Step 3: deterministic random padding.
    # Zero-padding biases the bipolar latent mean to -0.61 at 1,024 bits,
    # breaking the N(0, I) assumption and degrading inversion under diffuse
    # attacks (Gaussian noise, Gaussian blur). Random padding restores the
    # balanced {-1, +1} distribution. The seed is deterministic so the
    # decoder can reproduce and strip it. Note: this seed combination is
    # for reproducible padding only, not a cryptographic KDF (Section 5.7.5).
    pad_tail = int(NUM_LATENT_BITS - len(rep_bits))
    pad_seed = int(key_C) ^ int(key_shuffle) ^ 0xDEADBEEF
    if pad_tail > 0:
        g_pad = torch.Generator(device=device).manual_seed(pad_seed & 0xFFFFFFFF)
        random_pad = torch.randint(0, 2, (pad_tail,), generator=g_pad, device=device).long()
        rep_bits = torch.cat([rep_bits, random_pad])

    meta["pad_tail_bits"] = pad_tail
    meta["pad_seed"] = pad_seed

    # Step 4: bipolar conversion {0,1} -> {-1,+1}
    M = rep_bits.view(LATENT_C, LATENT_H, LATENT_W).float()
    M = M * 2.0 - 1.0

    # Step 5: per-channel orthogonal transform z[c] = C . M[c] . C^T
    C = after_generate_orthogonal_64(key_C, device=device)
    Z = torch.empty_like(M)
    for c in range(LATENT_C):
        Z[c] = C @ M[c] @ C.t()

    # Step 6: shuffle permutation
    perm = after_make_permutation(key_shuffle, NUM_LATENT_BITS, device=device)
    Z = Z.view(-1)[perm].view(LATENT_C, LATENT_H, LATENT_W)

    if verbose:
        print(f"[ENC] bits={orig_len} ecc={meta['enc_bits_len']} rep={rep}")

    return float(scale) * Z, meta


# ------------------------------------------------------------
# Public modes
# ------------------------------------------------------------


def encode_after_capacity(
    msg_bits: torch.Tensor,
    key_C: int,
    key_shuffle: int,
    scale: float = 1.0,
    verbose: bool = True,
):
    """Capacity mode: no ECC, no repetition, maximum raw payload."""
    msg_bits = msg_bits.view(-1)
    if len(msg_bits) > NUM_LATENT_BITS:
        raise ValueError("payload exceeds 16,384-bit capacity")

    return after_encode_core(
        msg_bits=msg_bits,
        key_C=key_C,
        key_shuffle=key_shuffle,
        rep=1,
        use_ecc=False,
        scale=scale,
        verbose=verbose,
    )


def encode_after_robust(
    msg_bits: torch.Tensor,
    key_C: int,
    key_shuffle: int,
    scale: float = 1.0,
    verbose: bool = True,
):
    """Robust mode: Reed-Solomon + adaptive repetition (r >= 2)."""
    msg_bits = msg_bits.view(-1)

    if not USE_ECC or (RS_CODEC is None):
        raise RuntimeError("ECC not available")

    # preview ECC length (bits) to choose repetition factor
    msg_bytes, _ = after_message_bits_to_bytes(msg_bits.long().to(device))
    enc_preview = RS_CODEC.encode(msg_bytes)
    ecc_len = int(len(np.unpackbits(np.frombuffer(enc_preview, dtype=np.uint8))))

    rep = after_adaptive_rep(ecc_len)

    # policy: require repetition >= 2 for "robust" mode
    if rep < 2:
        raise ValueError(
            f"[ROBUST DISABLED] payload={len(msg_bits)} ecc_bits={ecc_len} rep={rep}"
        )

    return after_encode_core(
        msg_bits=msg_bits,
        key_C=key_C,
        key_shuffle=key_shuffle,
        rep=rep,
        use_ecc=True,
        scale=scale,
        verbose=verbose,
    )
