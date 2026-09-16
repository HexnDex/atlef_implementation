# Principal Result Files

The CSVs underlying Chapter 4's headline tables and figures, with SHA-256 checksums in `checksums.txt` to verify integrity.

| File | Rows | Source for |
|---|---|---|
| `fast_after_COCO.csv` | 5,400 | Tables 4.5, 4.7–4.10 (extraction accuracy by attack, MS-COCO) |
| `fast_after_Flickr8K.csv` | 5,400 | Same, Flickr8K |
| `fast_after_DeskGPT.csv` | 5,400 | Same, DeskGPT |
| `detectability_results.csv` | 25 | Table 4.11 (steganalysis detection accuracy, pooled across datasets) |
| `fid_results.csv` | 8 | Section 4.3 (FID discussion) |
| `kid_results.csv` | 9 | Table 4.1 / Section 4.3 (KID discussion) |

Each `fast_after_*.csv` row is one (sample, payload, attack) combination: 100 samples x 3 payload levels x 18 attacks = 5,400 rows per dataset, 16,200 total across all three, matching the instance count cited throughout the thesis.

## Verifying integrity

```bash
sha256sum -c checksums.txt
```

## Cross-checked against the thesis

Before publishing, these files were spot-checked against the thesis's own reported numbers:
- `fast_after_COCO.csv`: clean/1,024-bit mean accuracy = 100.00%, matches Table 4.5.
- `fast_after_COCO.csv`: JPEG50/1,024-bit mean accuracy = 97.64%, matches Table 4.7.
- `detectability_results.csv`: SRNet at 1,024 bits = 87.2%, at 0 bits (cover-only) = 70.9%, overall = 67.8%, all match Table 4.11 exactly.
