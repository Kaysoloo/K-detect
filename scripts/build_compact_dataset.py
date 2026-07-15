"""
build_compact_dataset.py
========================
FGrade 10-class (0-9) -> K-Detect 3-class mapping:
  Classes 0-2  -> Good    (fresh, firm)
  Classes 3-6  -> Medium  (at-risk, minor defects)
  Classes 7-9  -> Poor    (rotten, spoiled)

Output:
  tomato_dataset/
  ├── train/{Good,Medium,Poor}/
  └── test/{Good,Medium,Poor}/
"""

import argparse, os, random, shutil
from pathlib import Path

CLASS_MAP = {
    "0": "Good", "1": "Good", "2": "Good",
    "3": "Medium", "4": "Medium", "5": "Medium", "6": "Medium",
    "7": "Poor", "8": "Poor", "9": "Poor",
}

VALID_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def collect_images(fgrade_data: Path) -> dict:
    collected = {str(i): [] for i in range(0, 10)}
    for split in ["Training_set", "Testing_set"]:
        split_dir = fgrade_data / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name not in collected:
                continue
            for img in class_dir.iterdir():
                if img.suffix in VALID_EXT:
                    collected[class_dir.name].append(img)
    return collected


def build_compact(collected, output_path, samples_per_class, train_split, seed):
    random.seed(seed)
    grouped = {"Good": [], "Medium": [], "Poor": []}
    for label, paths in collected.items():
        target = CLASS_MAP.get(label)
        if target:
            grouped[target].extend(paths)

    compact = {}
    for label, paths in grouped.items():
        n = len(paths)
        if n <= samples_per_class:
            compact[label] = paths
            print(f"  {label}: {n} images (all available)")
        else:
            compact[label] = random.sample(paths, samples_per_class)
            print(f"  {label}: {samples_per_class} sampled from {n}")

    for label, paths in compact.items():
        random.shuffle(paths)
        split_idx = int(len(paths) * train_split)
        for subset, subset_paths in [("train", paths[:split_idx]), ("test", paths[split_idx:])]:
            dest_dir = output_path / subset / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in subset_paths:
                dest = dest_dir / src.name
                if dest.exists():
                    stem, ext = os.path.splitext(src.name)
                    dest = dest_dir / f"{stem}_{src.parent.name}{ext}"
                shutil.copy2(src, dest)
        print(f"    -> {len(paths[:split_idx])} train / {len(paths[split_idx:])} test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fgrade-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fgrade_data = args.fgrade_path / "data"
    if not fgrade_data.exists():
        print(f"ERROR: FGrade data not found at {fgrade_data}")
        return

    print(f"Source:     {args.fgrade_path}")
    print(f"Output:     {args.output_path}")
    print(f"Target:     {args.samples_per_class} images/class, {args.train_split} train split")
    print()

    collected = collect_images(fgrade_data)
    total = sum(len(v) for v in collected.values())
    print(f"Found {total} total images across 10 classes:")
    for k in sorted(collected, key=int):
        print(f"  Class {k}: {len(collected[k])} images")
    print()

    build_compact(collected, args.output_path, args.samples_per_class, args.train_split, args.seed)
    print(f"\nDataset built -> {args.output_path.resolve()}")


if __name__ == "__main__":
    main()
