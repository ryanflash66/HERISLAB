"""
One-shot ingestion of Matias's transformer dataset + motor GT annotations
into CA_Training_Data.

Source: OneDrive_1_4-14-2026/
- Transformer healthy (22 BMPs from p1_Noload): 18 train / 4 test (every 5th to test)
- Transformer fault (233 BMPs from p2..p9): all to test/fault/transformer
- Motor GT annotations (40 images + 40 masks): stashed in data/ground_truth/

The induction motor and electric motor datasets are already ingested, so
this script only touches the net-new transformer + ground truth data.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_BASE = ROOT / "OneDrive_1_4-14-2026"
DEST = ROOT / "data" / "CA_Training_Data"
GT_DEST = ROOT / "data" / "ground_truth" / "induction_motor_40_gt"

TRANS_SRC = SRC_BASE / "Thermal images dataset, Transformer, 1 phase dry type" / "IR_trans_bmp"
GT_SRC = SRC_BASE / "Thermal image of equipment (Induction Motor) + 40 Ground Truths added" / "40_GT_with_Refrences"

FAULT_FOLDERS = ["p2_80", "p3_160", "p4_240", "p5_320", "p6_400", "p7_480", "p8_560", "p9_600"]


def copy_transformer_healthy():
    """22 files from p1_Noload. Every 5th (p1005, p1010, p1015, p1020) -> test, rest -> train."""
    src = TRANS_SRC / "p1_Noload"
    train_dest = DEST / "train" / "normal" / "transformer"
    test_dest = DEST / "test" / "normal" / "transformer"

    files = sorted(src.glob("*.bmp"))
    test_idx = {4, 9, 14, 19}  # 0-indexed -> p1005, p1010, p1015, p1020

    train_count = 0
    test_count = 0
    for i, f in enumerate(files):
        if i in test_idx:
            shutil.copy2(f, test_dest / f.name)
            test_count += 1
        else:
            shutil.copy2(f, train_dest / f.name)
            train_count += 1

    print(f"Transformer healthy: {train_count} train / {test_count} test")
    return train_count, test_count


def copy_transformer_fault():
    """All 233 fault images (p2_80..p9_600) -> test/fault/transformer/ with prefix."""
    dest = DEST / "test" / "fault" / "transformer"
    total = 0
    for folder in FAULT_FOLDERS:
        src = TRANS_SRC / folder
        files = sorted(src.glob("*.bmp"))
        for f in files:
            # Prefix with folder name to avoid collisions across fault severities
            shutil.copy2(f, dest / f"{folder}_{f.name}")
            total += 1
        print(f"  {folder}: {len(files)} copied")
    print(f"Transformer fault: {total} total")
    return total


def copy_ground_truth():
    """80 files total: 40 thermal + 40 GT masks. Preserve relative structure."""
    GT_DEST.mkdir(parents=True, exist_ok=True)
    # Copy the txt reference
    for item in GT_SRC.iterdir():
        if item.is_file():
            shutil.copy2(item, GT_DEST / item.name)
        elif item.is_dir():
            # Copy subdirectory contents preserving structure
            sub_dest = GT_DEST / item.name
            sub_dest.mkdir(exist_ok=True)
            for f in item.iterdir():
                if f.is_file():
                    shutil.copy2(f, sub_dest / f.name)
    # Count
    total = sum(1 for _ in GT_DEST.rglob("*") if _.is_file())
    print(f"Ground truth: {total} files copied to {GT_DEST}")
    return total


def main():
    print("=" * 60)
    print("Ingesting Matias's new data into CA_Training_Data")
    print("=" * 60)

    th_train, th_test = copy_transformer_healthy()
    print()
    tf_total = copy_transformer_fault()
    print()
    gt_total = copy_ground_truth()

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  train/normal/transformer:     {th_train}")
    print(f"  test/normal/transformer:      {th_test}")
    print(f"  test/fault/transformer:       {tf_total}")
    print(f"  data/ground_truth/*:          {gt_total}")
    print()
    print("Next: update preprocess.py to include the new folders.")


if __name__ == "__main__":
    main()
