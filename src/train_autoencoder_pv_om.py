"""
Train the PV-specialist autoencoder on O&M-only healthy frames (HER-114 retrain).

Identical to `train_autoencoder_pv.py` (same ThermalAutoencoder, batch 32,
Adam 1e-3, ReduceLROnPlateau, patience 10, max 50 epochs, internal val split
with seed 43); only the data source and output names differ:

  Input:  data/CA_Preprocessed/pv_om/train_normal.npy
  Output: models/autoencoder_pv_om_best.pth
          models/autoencoder_pv_om_final.pth
          results/training_log_pv_om.txt

models/autoencoder_pv_best.pth (the THM+O&M checkpoint) is never written.

Usage:
    venv/Scripts/python.exe src/train_autoencoder_pv_om.py
"""

import train_autoencoder_pv as T
from hlab.paths import PREPROCESSED

DATA_DIR = PREPROCESSED / "pv_om"
BEST_PATH = T.MODEL_DIR / "autoencoder_pv_om_best.pth"
FINAL_PATH = T.MODEL_DIR / "autoencoder_pv_om_final.pth"
LOG_PATH = T.RESULTS_DIR / "training_log_pv_om.txt"


def main():
    for new, old in [(BEST_PATH, T.BEST_PATH), (FINAL_PATH, T.FINAL_PATH), (LOG_PATH, T.LOG_PATH)]:
        assert new != old, f"would overwrite {old}"
    T.DATA_DIR = DATA_DIR
    T.BEST_PATH = BEST_PATH
    T.FINAL_PATH = FINAL_PATH
    T.LOG_PATH = LOG_PATH
    print(f"O&M-only retrain: data={DATA_DIR}  best={BEST_PATH.name}  "
          f"batch={T.BATCH_SIZE} epochs={T.EPOCHS} lr={T.LEARNING_RATE} patience={T.PATIENCE}")
    T.main()


if __name__ == "__main__":
    main()
