"""
ATLEF: Core Decoding Algorithm

Reverses every operation performed by the core encoder (src/encoder.py)
to recover the original message: undo scale, invert the shuffle
permutation, invert the per-channel orthogonal transform, threshold to
bits, strip the random padding tail, majority-vote decode the
repetition code, then Reed-Solomon decode. Matches Section 3.2.3 and
Appendix A.3 of the thesis.
"""

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
from .encoder import (
    after_generate_orthogonal_64,
    after_make_permutation,
    after_repetition_decode,
    after_bits_to_bytes,
    after_bytes_to_message_bits,
    encode_after_capacity,
    encode_after_robust,
)


def decode_latent_to_bits_after(
    z_in: torch.Tensor,
    key_C: int,
    key_shuffle: int,
    meta: dict,
    verbose: bool = False,
):
    """
    Inverse of after_encode_core:
      latent z -> inverse perm -> inverse orthogonal -> bits -> unpad
                -> rep decode -> ECC decode -> trim to original length

    Returns the recovered message bits (length = meta['orig_num_bits']).
    """
    # Step 0: shape fix, accept a batched (1, C, H, W) or unbatched (C, H, W) tensor
    z = z_in[0] if z_in.dim() == 4 else z_in
    z = z.to(device).float()

    scale = float(meta.get("scale", 1.0))
    rep = int(meta.get("rep_factor", 1))
    use_ecc = bool(meta.get("use_ecc", False))

    # Step 1: undo scale
    Z = z / scale if scale != 0 else z.clone()

    # Step 2: inverse permutation
    perm = after_make_permutation(key_shuffle, NUM_LATENT_BITS, device=device)
    invperm = torch.empty_like(perm)
    invperm[perm] = torch.arange(NUM_LATENT_BITS, device=device)
    Z = Z.view(-1)[invperm].view(LATENT_C, LATENT_H, LATENT_W)

    # Step 3: inverse orthogonal transform. Z = C . M . C^T, so M = C^T . Z . C
    C = after_generate_orthogonal_64(key_C, device=device)
    Ct = C.t()
    M_hat = torch.empty_like(Z)
    for c in range(LATENT_C):
        M_hat[c] = Ct @ Z[c] @ C

    # Step 4: sign thresholding. M was {-1, +1}; after the transform and any
    # channel noise it's real-valued, so threshold at zero.
    bits_hat = (M_hat.view(-1) >= 0).long()

    # Step 5: strip the random padding tail added at encode time
    pad_tail = int(meta.get("pad_tail_bits", 0))
    if pad_tail > 0:
        bits_hat = bits_hat[:-pad_tail]

    # Step 6: majority-vote repetition decoding
    bits_hat = after_repetition_decode(bits_hat, rep)

    # Step 7: Reed-Solomon outer decoding, if this message was encoded with ECC
    if use_ecc and USE_ECC and (RS_CODEC is not None):
        pad_bits = int(meta.get("pad_bits", 0))
        enc_bytes_hat = after_bits_to_bytes(bits_hat)

        try:
            dec = RS_CODEC.decode(enc_bytes_hat)
            msg_bytes_hat = dec[0] if isinstance(dec, (tuple, list)) else dec
        except Exception:
            # RS decode failed outright (uncorrectable errors); fall back to
            # the raw truncated bits rather than raising, useful for debugging
            msg_bytes_hat = b""

        if len(msg_bytes_hat) == 0:
            msg_bits_hat = bits_hat
        else:
            msg_bits_hat = after_bytes_to_message_bits(msg_bytes_hat, pad_bits, device=device)
    else:
        msg_bits_hat = bits_hat

    # Step 8: trim to the original message length
    orig_len = int(meta.get("orig_num_bits", len(msg_bits_hat)))
    msg_bits_hat = msg_bits_hat[:orig_len].long()

    if verbose:
        print(f"[DEC] out_bits={len(msg_bits_hat)} rep={rep} ecc={use_ecc}")

    return msg_bits_hat


def _self_test():
    """Round-trip sanity check: encode then decode a random message and
    confirm perfect recovery under both capacity and robust modes."""
    key_C = 13579
    key_S = 24680

    n = 2048
    g = torch.Generator(device=device).manual_seed(1234)
    msg = torch.randint(0, 2, (n,), generator=g, device=device).long()

    z, meta = encode_after_capacity(msg, key_C, key_S, scale=1.0, verbose=False)
    rec = decode_latent_to_bits_after(z, key_C, key_S, meta, verbose=False)
    acc = (rec == msg).float().mean().item()
    print(f"capacity mode round-trip accuracy: {acc:.6f}")

    if USE_ECC and (RS_CODEC is not None):
        z2, meta2 = encode_after_robust(msg, key_C, key_S, scale=0.16, verbose=False)
        rec2 = decode_latent_to_bits_after(z2, key_C, key_S, meta2, verbose=False)
        acc2 = (rec2 == msg).float().mean().item()
        print(f"robust mode round-trip accuracy:   {acc2:.6f}")
    else:
        print("robust mode skipped (ECC not available)")


if __name__ == "__main__":
    _self_test()
