"""
ATLEF: Exact Seeds and Experimental Configuration

Documents the deterministic seed-generation formula and experimental
parameters used to produce the 16,200 robustness evaluation instances
reported in Chapter 4 (3 datasets x 3 payload levels x 100 samples x
18 attack conditions). Every quantity below is reproducible from these
formulas alone, no external seed file was used or is needed.
"""

# ------------------------------------------------------------
# Experimental grid
# ------------------------------------------------------------
DATASETS = ["MS-COCO", "Flickr8K", "DeskGPT"]
PAYLOAD_LEVELS = [1024, 2048, 4096]     # bits
SAMPLES_PER_PAYLOAD = 100                # per dataset, per payload level
# 3 datasets x 3 payloads x 100 samples x 18 attacks = 16,200 instances

# ------------------------------------------------------------
# Generation / inversion parameters (matched, per Section 4.1.1)
# ------------------------------------------------------------
ENCODE_STEPS = 20        # DPM-Solver++ steps, generation
INVERSION_STEPS = 20     # DPM-Solver++ steps, inversion (matched to encode)
SOLVER_ORDER = 2         # DPM-Solver++ second-order
GUIDANCE_SCALE = 5.0

# ------------------------------------------------------------
# Deterministic per-sample seed formulas
# ------------------------------------------------------------
# For sample index s (0 to SAMPLES_PER_PAYLOAD - 1) and payload_len in
# PAYLOAD_LEVELS, every quantity below is a pure function of (s, payload_len):
#
#   message_bits_seed = 999 + s + payload_len
#   key_C              = 70000 + s
#   key_shuffle        = 80000 + s
#   prompt             = dataset_prompts[s % len(dataset_prompts)]
#
# message_bits_seed drives a torch.Generator used to sample the random
# payload bits themselves (torch.randint(0, 2, (payload_len,), generator=...)).
# key_C and key_shuffle are passed directly into encode_after_robust /
# encode_after_capacity (src/encoder.py), which derive the orthogonal
# transform matrix and shuffle permutation from them (see
# after_generate_orthogonal_64 and after_make_permutation).
#
# key_C and key_shuffle depend only on the sample index, not on the
# payload level, so the same sample index uses the same keys across all
# three payload levels; message_bits_seed does depend on payload_len so
# that different payload levels don't sample identical bit prefixes.


def message_bits_seed(sample_index: int, payload_bits: int) -> int:
    return 999 + sample_index + payload_bits


def key_c_for_sample(sample_index: int) -> int:
    return 70000 + sample_index


def key_shuffle_for_sample(sample_index: int) -> int:
    return 80000 + sample_index


def prompt_for_sample(sample_index: int, dataset_prompts: list) -> str:
    """Cyclic prompt assignment: sample index wraps around the dataset's
    prompt list if there are fewer prompts than samples."""
    return dataset_prompts[sample_index % len(dataset_prompts)]


if __name__ == "__main__":
    print(f"Experimental grid: {len(DATASETS)} datasets x {len(PAYLOAD_LEVELS)} "
          f"payloads x {SAMPLES_PER_PAYLOAD} samples = "
          f"{len(DATASETS) * len(PAYLOAD_LEVELS) * SAMPLES_PER_PAYLOAD} stego images")
    print(f"Sample 0, 1024 bits -> message seed {message_bits_seed(0, 1024)}, "
          f"key_C {key_c_for_sample(0)}, key_shuffle {key_shuffle_for_sample(0)}")
