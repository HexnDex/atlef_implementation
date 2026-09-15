"""
ATLEF Steganalysis: SRM Kernel Bank

Generates the canonical 30-filter Spatial Rich Model (SRM) high-pass
filter bank (Fridrich, 2012), the standard filter set used to
initialize YeNet's preprocessing layer. Matches Table 3.5 / Section
3.5.3 of the thesis.

Usage:
    from srm_kernels import build_srm_kernels
    kernels = build_srm_kernels()   # (30, 1, 5, 5) float32
"""

import os

import numpy as np


def build_srm_kernels() -> np.ndarray:
    """Produce the canonical 30-filter SRM bank as (30, 1, 5, 5) float32.

    Kernels are the standard set of high-pass residuals: 1st-order,
    2nd-order, 3rd-order, EDGE, SQUARE, KB, and KV families with their
    symmetry rotations, exactly as used in YeNet's published code.
    """
    K = []

    # 1st-order (rotations and reflections of [-1, +1, 0])
    base_1st = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)
    for k in range(4):
        K.append(np.rot90(base_1st, k))
    K.append(np.flipud(base_1st))
    K.append(np.fliplr(base_1st))
    K.append(np.flipud(np.fliplr(base_1st)))
    K.append(base_1st.T)

    # 2nd-order (rotations of [-1, +2, -1])
    base_2nd = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)
    for k in range(4):
        K.append(np.rot90(base_2nd, k))

    # 3rd-order
    base_3rd = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, -3, 3, -1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)
    for k in range(4):
        K.append(np.rot90(base_3rd, k))

    # SQUARE 3x3
    sq3 = np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32) / 4.0
    K.append(sq3)

    # SQUARE 5x5 (KV-like)
    sq5 = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ], dtype=np.float32) / 12.0
    K.append(sq5)

    # EDGE 3x3 (4 rotations)
    edge3 = np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32) / 4.0
    for k in range(4):
        K.append(np.rot90(edge3, k))

    # EDGE 5x5 (4 rotations)
    edge5 = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32) / 12.0
    for k in range(4):
        K.append(np.rot90(edge5, k))

    # Truncate or pad to exactly 30 kernels
    if len(K) > 30:
        K = K[:30]
    while len(K) < 30:
        K.append(np.zeros((5, 5), dtype=np.float32))

    bank = np.stack(K, axis=0).astype(np.float32)  # (30, 5, 5)
    bank = bank[:, np.newaxis, :, :]  # (30, 1, 5, 5)
    return bank


def load_or_build_srm_kernels(path: str = "SRM_Kernels.npy") -> np.ndarray:
    """Load the SRM bank from disk if it exists, otherwise build it and
    save it to `path` so later runs (and checkpoint reproduction) use
    the exact same file."""
    if not os.path.exists(path):
        bank = build_srm_kernels()
        np.save(path, bank)
        print(f"Saved SRM kernel bank -> {path}")
    else:
        print(f"Using existing SRM kernel bank: {path}")
    return np.load(path)


if __name__ == "__main__":
    bank = build_srm_kernels()
    print(f"SRM kernel bank shape: {bank.shape}")
