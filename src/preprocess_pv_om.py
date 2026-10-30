"""
PV-specialist preprocessing, O&M-only variant (HER-114 retrain).

Same pipeline, resize, normalisation and seed-42 90/10 split logic as
`preprocess_pv.py`, but the healthy pool is `pv_om_inspection` only: the
`pv_thermal_inspection` (THM) aerial survey frames are excluded from both the
train and the holdout side. Nothing under data/CA_Preprocessed/pv/ is touched.

  data/CA_Preprocessed/pv_om/
    train_normal.npy     O&M train frames (z-scored with O&M-only stats)
    test_normal.npy      O&M holdout frames
    test_fault.npy       PVMD faults (data/CA_Training_Data/test/fault/pv)
    norm_stats.npy       {"mean", "std"} from the O&M train frames
    split_manifest.txt   [TRAIN] / [HOLDOUT] file lists

Usage:
    venv/Scripts/python.exe src/preprocess_pv_om.py
"""

from pathlib import Path

import preprocess_pv as P

OUTPUT_DIR = P.ROOT / "data" / "CA_Preprocessed" / "pv_om"
ORIGINAL_DIR = P.ROOT / "data" / "CA_Preprocessed" / "pv"


def main():
    assert OUTPUT_DIR != ORIGINAL_DIR
    P.OUTPUT_DIR = OUTPUT_DIR
    P.PV_TRAIN_FOLDERS = ["pv_om_inspection"]
    # RANDOM_SEED (42), HOLDOUT_FRACTION (0.10), TARGET_SIZE and the loaders
    # are inherited unchanged from preprocess_pv.
    print(f"O&M-only variant: folders={P.PV_TRAIN_FOLDERS}, seed={P.RANDOM_SEED}, "
          f"holdout={P.HOLDOUT_FRACTION:.0%}, out={OUTPUT_DIR}")
    P.main()


if __name__ == "__main__":
    main()
