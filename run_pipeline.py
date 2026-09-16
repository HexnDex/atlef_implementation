"""
ATLEF: End-to-End Pipeline Orchestration Script

Ties together message encoding, Stable Diffusion generation, channel
attacks, DPM-Solver++ inversion, and message decoding into a single
runnable round trip. This is the top-level script referenced in
Appendix A / Section 3.5.4.

Requires Hu et al.'s repository (github.com/HXX5656/mas_GRDH) for the
Stable Diffusion 1.5 pipeline and DPM-Solver++ sampler, since ATLEF's
own contribution is the encoding/decoding/ECC layer, not the diffusion
model itself (see README, "Built on"). This script is the seam between
the two: it imports ATLEF's own components from src/, and expects the
diffusion model and sampler objects to be set up per Hu et al.'s
instructions before running.

Usage:
    python run_pipeline.py --prompt "a cat sitting on a chair" \
        --payload-bits 1024 --attack jpeg50 --mode robust

Setup (one-time, before running):
    1. Clone https://github.com/HXX5656/mas_GRDH and follow its README
       to download the SD 1.5 checkpoint and CLIP encoder.
    2. Load the model and build `sampler` (generation) and
       `sampler_is_after` (inversion) DPM-Solver++ samplers exactly as
       shown in their scripts/txt2img.py. Pass both into
       PipelineContext below.
"""

import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from src.constants import LATENT_C, LATENT_H, LATENT_W, device
from src.encoder import encode_after_capacity, encode_after_robust
from src.decoder import decode_latent_to_bits_after
from src.attacks import ATTACKS


# ------------------------------------------------------------
# Reproducibility: global seed
# ------------------------------------------------------------
# All reported results use a fixed global seed for run-level determinism
# (library-level RNG state), on top of the per-sample key_C / key_shuffle
# seeds that control the actual encoding (see src/encoder.py). Changing
# GLOBAL_SEED does not change what a given (key_C, key_shuffle) pair
# encodes, it only affects incidental randomness elsewhere (e.g. any
# non-deterministic library behavior not already pinned by the keys).
GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# External pipeline context (Hu et al.'s model + samplers)
# ------------------------------------------------------------
@dataclass
class PipelineContext:
    """Holds the Stable Diffusion model and the two DPM-Solver++ samplers
    (generation and inversion), set up per Hu et al.'s repository. This
    script does not construct these itself, see the module docstring.
    """
    model: object            # loaded Stable Diffusion 1.5 model
    sampler: object           # DPM-Solver++ sampler for generation
    sampler_is_after: object  # DPM-Solver++ sampler for inversion


# ------------------------------------------------------------
# Step 1: Encode message -> initial latent z_T
# ------------------------------------------------------------
def encode(payload_bits: torch.Tensor, key_C: int, key_shuffle: int, mode: str = "robust"):
    if mode == "robust":
        z_T, meta = encode_after_robust(payload_bits, key_C, key_shuffle, scale=1.0, verbose=False)
    elif mode == "capacity":
        z_T, meta = encode_after_capacity(payload_bits, key_C, key_shuffle, scale=1.0, verbose=False)
    else:
        raise ValueError("mode must be 'robust' or 'capacity'")
    meta.update({"mode": mode, "key_C": key_C, "key_shuffle": key_shuffle})
    return z_T, meta


# ------------------------------------------------------------
# Step 2: Generate stego image from z_T (Hu et al. pipeline)
# ------------------------------------------------------------
def generate_image(ctx: PipelineContext, z_T: torch.Tensor, prompt: str,
                    steps: int = 20, guidance: float = 5.0) -> Image.Image:
    dev = next(ctx.model.parameters()).device
    z_T_batch = z_T.to(dev).unsqueeze(0)  # [1, C, H, W]

    with torch.no_grad():
        uc = ctx.model.get_learned_conditioning([""])
        c = ctx.model.get_learned_conditioning([prompt])
        samples, _ = ctx.sampler.sample(
            steps=steps, conditioning=c, batch_size=1,
            shape=[LATENT_C, LATENT_H, LATENT_W], verbose=False,
            unconditional_guidance_scale=guidance, unconditional_conditioning=uc,
            eta=0.0, order=2, x_T=z_T_batch, width=512, height=512, DPMencode=False,
        )
        x_samples = ctx.model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)

    img_np = (x_samples[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(img_np)


# ------------------------------------------------------------
# Step 3: Apply a channel attack
# ------------------------------------------------------------
def apply_attack(img: Image.Image, attack_name: str) -> Image.Image:
    attack_fn, kwargs = ATTACKS[attack_name]
    return attack_fn(img, **kwargs)


# ------------------------------------------------------------
# Step 4: Invert the (possibly attacked) image back to a latent
# ------------------------------------------------------------
def invert_image(ctx: PipelineContext, image: Image.Image, prompt: str, meta: dict,
                  steps_inv: int = 20, guidance: float = 5.0, order: int = 2) -> torch.Tensor:
    from src.encoder import after_generate_orthogonal_64  # noqa: F401 (kept for parity with notebook)

    dev = next(ctx.model.parameters()).device

    # image -> latent z0 (VAE encode)
    img_t = torch.from_numpy(np.array(image).astype(np.float32) / 127.5 - 1.0)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(dev)
    with torch.no_grad():
        z0 = ctx.model.get_first_stage_encoding(ctx.model.encode_first_stage(img_t))

    with torch.no_grad():
        uc = ctx.model.get_learned_conditioning([""])
        c = ctx.model.get_learned_conditioning([prompt])
        z_enc, _ = ctx.sampler_is_after.sample(
            steps=steps_inv, unconditional_conditioning=uc, conditioning=c,
            batch_size=1, shape=(LATENT_C, LATENT_H, LATENT_W), verbose=False,
            unconditional_guidance_scale=guidance, eta=0.0, order=order,
            x_T=z0, width=512, height=512, DPMencode=True,
        )
    return z_enc[0]  # [C, H, W]


# ------------------------------------------------------------
# Step 5: Decode recovered latent -> message bits
# ------------------------------------------------------------
def decode(z_T_hat: torch.Tensor, meta: dict) -> torch.Tensor:
    return decode_latent_to_bits_after(
        z_T_hat, key_C=meta["key_C"], key_shuffle=meta["key_shuffle"], meta=meta, verbose=False
    )


# ------------------------------------------------------------
# Full round trip
# ------------------------------------------------------------
def run_round_trip(ctx: PipelineContext, prompt: str, payload_bits: torch.Tensor,
                    key_C: int, key_shuffle: int, attack_name: str = "clean",
                    mode: str = "robust"):
    z_T, meta = encode(payload_bits, key_C, key_shuffle, mode=mode)
    img = generate_image(ctx, z_T, prompt)
    attacked_img = apply_attack(img, attack_name)
    z_T_hat = invert_image(ctx, attacked_img, prompt, meta)
    recovered_bits = decode(z_T_hat, meta)

    n = min(len(recovered_bits), len(payload_bits))
    accuracy = (recovered_bits[:n] == payload_bits.to(recovered_bits.device)[:n]).float().mean().item()

    return {
        "prompt": prompt,
        "payload_bits": int(len(payload_bits)),
        "attack": attack_name,
        "mode": mode,
        "key_C": key_C,
        "key_shuffle": key_shuffle,
        "accuracy": accuracy,
        "stego_image": img,
        "attacked_image": attacked_img,
        "recovered_bits": recovered_bits,
    }


def main():
    parser = argparse.ArgumentParser(description="Run one ATLEF encode/attack/decode round trip.")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--payload-bits", type=int, default=1024, choices=[1024, 2048, 4096])
    parser.add_argument("--attack", default="clean", choices=list(ATTACKS.keys()))
    parser.add_argument("--mode", default="robust", choices=["robust", "capacity"])
    parser.add_argument("--key-c", type=int, default=13579)
    parser.add_argument("--key-shuffle", type=int, default=24680)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = parser.parse_args()

    set_global_seed(args.seed)

    print("This script defines the pipeline; it does not load a model itself.")
    print("See the module docstring for one-time setup using Hu et al.'s repository,")
    print("then construct a PipelineContext and call run_round_trip().")
    print()
    print(f"Example: run_round_trip(ctx, {args.prompt!r}, "
          f"torch.randint(0, 2, ({args.payload_bits},)), "
          f"{args.key_c}, {args.key_shuffle}, {args.attack!r}, {args.mode!r})")


if __name__ == "__main__":
    main()
