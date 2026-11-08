"""
THM-only PV specialist: the "train on THM, test on unseen O&M" half of the leave-one-source-out
check (HER-118, reviewer comment 12). The other half (train on O&M, score unseen THM) is the
HER-114 O&M-only model (bundle "pv_om") and needs no new training.

Data (healthy frames only; nothing in the paper's split or models is touched):
  THM   the 1,075 frames of data/CA_Training_Data/train/normal/pv_thermal_inspection, loaded and
        transformed exactly as preprocess_pv.py does (hlab.transform defaults: scale_by_max + stretch)
  in-source holdout   the last 10% of THM frames by frame number (one contiguous block at the end of
        the flight); the 25 frame numbers before it are dropped (a gap: THM frame correlation falls to
        the random-pair level by a lag of about 25 frames); TRAIN is every THM frame before the gap
  norm stats          mean / std over the THM TRAIN pixels only
  unseen source       every O&M healthy frame of the pv dataset (TRAIN + HOLDOUT, 7,836 frames),
        taken from the stored pv arrays: denormalised with the pv stats, re-normalised with the THM stats
  faults              the 1,000 PVMD files of the pv dataset, same re-normalisation

Recipe: the "pv" entry of config/training.json (lr 1e-3, no clipping, 50 epochs, patience 10,
best checkpoint, RandomFrames seed 43 early-stopping holdout inside TRAIN), with torch seed 0 and
batch 16 instead of 32 (the GPU is shared with other training jobs).

Outputs (new folders only):
  models/her-118/autoencoder_pv_thm_only_{best,final}.pth, models/her-118/thm_only_norm_stats.npy
  results/her-118/training_log_pv_thm_only.txt
  results/her-118/thm_only_errors.npz   thm_frames, thm_role (train/gap/holdout), thm_err,
                                        om_frames, om_err, fault_err (pv fault manifest order)

Usage:
    venv/Scripts/python.exe src/experiments/thm_only_pv.py
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset, InMemoryDataset, Split  # noqa: E402
from hlab.paths import MODELS, RAW, RESULTS  # noqa: E402
from hlab.train import RandomFrames, fit  # noqa: E402
from hlab.transform import denormalize, load_gray, normalize, to_model_input  # noqa: E402

THM_DIR = RAW / "train" / "normal" / "pv_thermal_inspection"
OUT = RESULTS / "her-118"
MODEL_DIR = MODELS / "her-118"
HOLDOUT_FRACTION = 0.10
GAP = 25
SEED = 0
BATCH = 16


def thm_number(name):
    return int(re.match(r"THM_\d+_(\d+)", name).group(1))


def thm_roles(names):
    """train / gap / holdout per THM frame: the last 10% by frame number is the holdout block."""
    nums = np.array([thm_number(n) for n in names])
    n_ho = int(round(HOLDOUT_FRACTION * len(nums)))
    first_ho = np.sort(nums)[-n_ho]
    return np.where(nums >= first_ho, "holdout", np.where(nums >= first_ho - GAP, "gap", "train"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in THM_DIR.iterdir() if p.is_file())
    names = [p.name for p in files]
    role = thm_roles(names)
    print(f"THM frames {len(files)}: " + "  ".join(f"{r} {(role == r).sum()}" for r in ("train", "gap", "holdout")))
    raw = np.stack([to_model_input(load_gray(p)) for p in files]).astype(np.float32)
    stats = {"mean": float(raw[role == "train"].mean()), "std": float(raw[role == "train"].std())}
    np.save(MODEL_DIR / "thm_only_norm_stats.npy", stats)
    print(f"THM TRAIN norm stats: mean {stats['mean']:.4f}  std {stats['std']:.4f}")
    x = normalize(raw, stats)
    tr = role == "train"
    ids = [f"data/CA_Training_Data/train/normal/pv_thermal_inspection/{n}" for n in names]
    train = Split("train", x[tr], np.zeros(int(tr.sum()), int), [f for f, k in zip(ids, tr) if k],
                  np.array(["thm"] * int(tr.sum())), [{} for _ in range(int(tr.sum()))])
    ds = InMemoryDataset("pv_thm_only", {"train": train}, stats=stats)
    _, history = fit(ds, "pv_thm_only", holdout=RandomFrames(seed=43, fraction=0.1), loss="mse", lr=1e-3,
                     clip=None, seed=SEED, epochs=50, batch=BATCH, patience=10, final="best",
                     model_dir=MODEL_DIR, log=OUT / "training_log_pv_thm_only.txt")
    bundle = Bundle("pv_thm_only", history["best_path"], stats, None)
    thm_err = bundle.score_normalized(x)

    pv = Dataset("pv")
    pv_stats = pv.stats
    om_frames, om_err = [], []
    for split in ("train", "holdout"):
        s = pv.split(split)
        keep = s.source == "om"
        om_frames += [f for f, k in zip(s.frame_ids, keep) if k]
        om_err.append(bundle.score_normalized(normalize(denormalize(s.x[keep], pv_stats), stats)))
        del s
    fault = pv.split("fault")
    fault_err = bundle.score_normalized(normalize(denormalize(fault.x, pv_stats), stats))
    np.savez(OUT / "thm_only_errors.npz", thm_frames=np.array(names), thm_role=role, thm_err=thm_err,
             om_frames=np.array(om_frames), om_err=np.concatenate(om_err), fault_err=fault_err,
             fault_frames=np.array(fault.frame_ids))
    print(f"best val_loss {history['best_val_loss']:.6f} at epoch {history['best_epoch']}, "
          f"{history['epochs_run']} epochs, {history['minutes']:.1f} min; saved errors: THM {len(thm_err)}, "
          f"O&M {sum(len(e) for e in om_err)}, faults {len(fault_err)}")


if __name__ == "__main__":
    main()
