"""
Repeated training runs of the PV specialist (HER-120, reviewer comments 6 and 13).

Trains the "pv" configuration of config/training.json once per torch seed, with
everything else fixed: the same data, the same validation holdout (RandomFrames
seed 43), the same recipe (lr 1e-3, no clipping, patience 10). Each run's best
checkpoint is then scored on the same test set the paper uses (the pv holdout
and the 1,000 PVMD fault files), so docs/paper/audit_eval_validity.py can report
the spread across seeds without loading any weights.

Outputs per seed s (models/ and results/ entries of the paper's model are not touched):
  models/her-120/autoencoder_pv_seed<s>_{best,final}.pth
  results/her-120/training_log_pv_seed<s>.txt
  results/her-120/pv_seed<s>_normal_errors.npy     one error per held-out healthy frame
  results/her-120/pv_seed<s>_fault_errors.npy      one error per PVMD file, manifest order

A seed whose error files already exist is skipped, so the script can be rerun
after an interruption.

Usage:
    venv/Scripts/python.exe src/experiments/seed_runs_pv.py --seeds 0 1 2 3 4
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.bundle import Bundle, resolve  # noqa: E402
from hlab.datasets import Dataset  # noqa: E402
from hlab.paths import MODELS, RESULTS  # noqa: E402
from hlab.train import fit_config  # noqa: E402

CONFIG = "pv"
OUT = RESULTS / "her-120"
MODEL_DIR = MODELS / "her-120"


def error_paths(seed, out=OUT):
    return out / f"pv_seed{seed}_normal_errors.npy", out / f"pv_seed{seed}_fault_errors.npy"


def run_seed(seed):
    neg_path, pos_path = error_paths(seed)
    if neg_path.exists() and pos_path.exists():
        print(f"seed {seed}: errors already saved, skipping")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    _, history = fit_config(CONFIG, model_dir=MODEL_DIR, seed=seed, out_name=f"pv_seed{seed}",
                            log=f"her-120/training_log_pv_seed{seed}.txt")
    bundle = Bundle.from_paths(history["best_path"], resolve(CONFIG)["stats"], name=f"pv_seed{seed}")
    ds = Dataset(CONFIG)
    neg = bundle.score_normalized(ds.split("holdout").x)
    pos = bundle.score_normalized(ds.split("fault").x)
    np.save(neg_path, neg)
    np.save(pos_path, pos)
    print(f"seed {seed}: best val_loss {history['best_val_loss']:.6f} at epoch {history['best_epoch']}, "
          f"{history['epochs_run']} epochs, {history['minutes']:.1f} min; "
          f"saved {len(neg)} healthy and {len(pos)} fault errors")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args(argv)
    for seed in args.seeds:
        run_seed(seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
