"""
ATLEF: Result File Checksums

Computes SHA-256 checksums for the principal result files (the CSVs
underlying Chapter 4's tables and figures) and writes them to
checksums.txt. Run this once you have the result CSVs in a local
folder, then commit checksums.txt alongside them.

Usage:
    python compute_checksums.py /path/to/results/folder
"""

import hashlib
import sys
from pathlib import Path

# The principal result files underlying Chapter 4's tables and figures.
PRINCIPAL_RESULT_FILES = [
    "fast_after_COCO.csv",       # source for Tables 4.5, 4.7-4.10 (accuracy by attack)
    "fast_after_Flickr8K.csv",
    "fast_after_DeskGPT.csv",
    "detectability_results.csv",  # source for Table 4.11 (steganalysis detection accuracy)
    "fid_results.csv",            # source for FID discussion, Section 4.3
    "kid_results.csv",            # source for Table 4.1 / KID discussion, Section 4.3
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) != 2:
        print("Usage: python compute_checksums.py /path/to/results/folder")
        sys.exit(1)

    folder = Path(sys.argv[1])
    lines = []
    missing = []

    for name in PRINCIPAL_RESULT_FILES:
        path = folder / name
        if path.exists():
            digest = sha256_of(path)
            lines.append(f"{digest}  {name}")
            print(f"OK    {name}")
        else:
            missing.append(name)
            print(f"MISS  {name}")

    out_path = folder / "checksums.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWrote {len(lines)} checksums to {out_path}")
    if missing:
        print(f"{len(missing)} file(s) not found, update PRINCIPAL_RESULT_FILES "
              f"if your file names differ: {missing}")


if __name__ == "__main__":
    main()
