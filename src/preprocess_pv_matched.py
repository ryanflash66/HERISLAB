"""
PV preprocessing with one transform for both classes (HER-107).

Same healthy folders, seed-42 90/10 split and PVMD fault folder as
`preprocess_pv.py`, so the frames and their split are pv/'s. Only the
transform changes, and it is the same for healthy and fault files:

  fit="crop"                centre-crop to 4:3 before resizing, so neither
                            640x512 (5:4) nor PVMD's 512x512 (1:1) is squashed
  representation="equalize" per-image rank of each pixel, scaled to 0-255,
                            instead of a min-max stretch for 16-bit files only

  data/CA_Preprocessed/pv_matched/
    train_normal.npy, test_normal.npy, test_fault.npy, norm_stats.npy, split_manifest.txt

What this removes and what it does not: equalisation leaves every image with
the same flat value distribution, so global brightness statistics (the
1st-percentile shortcut, AUROC 0.972) cannot separate the formats any more. It
does not give PVMD temperatures: those files are colour-palette renders with
no scale bar, and their channel mean is not monotonic in temperature. Scene
content (aerial site views vs close-up modules) also still differs. The valid
test is the same-source India set (HER-95); this variant is the check that the
CA-PV vs PVMD numbers were driven by format.

Usage:
    venv/Scripts/python.exe src/preprocess_pv_matched.py
"""

import preprocess_pv as P
from hlab.datasets import PV_FOLDERS
from hlab.paths import PREPROCESSED

OUTPUT_DIR = PREPROCESSED / "pv_matched"
REPRESENTATION = "equalize"
FIT = "crop"


def main():
    assert OUTPUT_DIR != P.OUTPUT_DIR
    P.main(output_dir=OUTPUT_DIR, folders=PV_FOLDERS["pv_matched"], representation=REPRESENTATION, fit=FIT)


if __name__ == "__main__":
    main()
