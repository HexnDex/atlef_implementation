# Dataset and Prompt Manifests

## Prompt sources (main robustness evaluation, 16,200 instances)

- `coco_prompts.json` — MS-COCO 2017 validation captions (standard COCO annotation format: `info`, `licenses`, `images`, `annotations`).
- `flickr8k_prompts.txt` — Flickr8k captions, tab-separated `image_id#caption_number<TAB>caption`.
- `deskgpt_prompts.txt` — DeskGPT prompt set, one prompt per line.

These are the raw prompt corpora each dataset's prompts are drawn from. For a given sample index `s` and dataset, the prompt actually used is `dataset_prompts[s % len(dataset_prompts)]` (cyclic assignment), combined with the deterministic seed formulas in `experiment_config.py` at the repository root. Together, a prompt source file plus `experiment_config.py` fully determines every input to the main robustness evaluation, no separate per-sample seed lookup table is needed for this part.

## Steganalysis training/test dataset (separate from the robustness evaluation above)

- `train_images_index.csv` — 1,202 rows. Every image used to train the SRNet/YeNet/XuNet detectors: dataset, payload level, cover/stego label, prompt, seed, key_seed, image path, and BRISQUE score.
- `test_images_index.csv` — 906 rows. Same structure, held-out test split, used to produce the steganalysis results in Table 4.11 (`results/detectability_results.csv`).

Unlike the main robustness evaluation, this dataset's seeds are stored explicitly per row (column `seed`) rather than following a formula, since it was generated as a separate, fixed dataset independent of the main evaluation's sample-index scheme.
