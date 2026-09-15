"""
ATLEF: Global Constants and Configuration

Defines the latent space geometry and error correction parameters used
throughout the ATLEF encoding and decoding pipeline. These values match
Section 3.2 and Appendix A.1 of the thesis.
"""

import torch
from reedsolo import RSCodec

# ------------------------------------------------------------
# Latent geometry (Stable Diffusion 1.5, 512x512 generation)
# ------------------------------------------------------------
LATENT_C = 4          # channels
LATENT_H = 64          # height in latent space
LATENT_W = 64          # width in latent space
NUM_LATENT_BITS = LATENT_C * LATENT_H * LATENT_W  # 16,384 for SD 1.5

# ------------------------------------------------------------
# Error correction: Reed-Solomon outer code
# ------------------------------------------------------------
USE_ECC = True
RS_NSYM = 32  # parity bytes (ECC strength)

try:
    RS_CODEC = RSCodec(RS_NSYM)
except Exception as e:
    print(f"WARNING: reedsolo could not be loaded, ECC disabled. ({e})")
    USE_ECC = False
    RS_CODEC = None

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    print(f"Latent geometry: {LATENT_C}x{LATENT_H}x{LATENT_W} -> {NUM_LATENT_BITS} bits")
    print(f"Reed-Solomon ECC: {'enabled' if USE_ECC else 'disabled'} (nsym={RS_NSYM})")
    print(f"Device: {device}")
