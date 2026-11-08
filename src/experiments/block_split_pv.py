"""
PV specialist on a temporally separated split (HER-117, reviewer comment 2).

The paper's PV split (data/CA_Preprocessed/pv/split_manifest.txt, preprocess_pv.py) is random per
frame, so 884 of 892 held-out healthy frames have an immediately adjacent frame in TRAIN and all 892
have one within two frames. This script rebuilds the healthy split from the frame order, retrains the
"pv" recipe on it and scores the new holdout and the PVMD faults.

Frame order. Every healthy PV frame belongs to one of three continuous recordings, read from the file
name: O&M double-row video (dr_frame_NNNNNN, 2,541 frames), O&M single-row video (sr_frame_NNNNNN,
5,295 frames) and the THM aerial survey (THM_00078_NNNNN, 1,075 frames, 8 frame numbers missing).
There are only two O&M videos, so holding out whole sequences would leave one O&M video for training
and test on a different row layout; contiguous time blocks with a gap are used instead.

Split rule (no scores are looked at). Each recording is cut into round(n / 900) equal segments. In the
middle of each segment a contiguous block of 10% of the segment's frames is held out; GAP = 100 frame
numbers on each side of the block go to neither side; the rest is TRAIN. The gap is chosen from the
frame-correlation curve (median Pearson r between frames k apart): r falls to the level of a random
pair of the same recording by k = 50 (O&M) and k = 25 (THM), so 100 is past it with margin.
Each held-out block is split in two for the threshold protocol of HER-97: its first half is VAL
(where the threshold is chosen), its second half TEST. The PVMD faults keep their VAL / TEST halves
from docs/data_manifests/pv_calibration_split.csv.

Model units. The stored pv arrays are z-scored with the old TRAIN statistics; they are mapped back to
0-255 model input (hlab.transform.denormalize) and z-scored again with the new TRAIN statistics, which
is what preprocess_pv.py would have written for this split (the z-score is the only split-dependent
step). Training: config/training.json "pv" (lr 1e-3, no clipping, patience 10, early-stopping holdout
RandomFrames seed 43 inside TRAIN) with torch seed 0, so it pairs with HER-120 seed 0 trained on the
random split.

Outputs (the shared results/ and models/ entries of the paper's model are not touched):
  docs/data_manifests/pv_block_split.csv            frame_id, recording, frame_no, split, role
  models/her-117/autoencoder_pv_block_seed0_{best,final}.pth
  results/her-117/norm_stats_pv_block.npy
  results/her-117/training_log_pv_block_seed0.txt
  results/her-117/pv_block_seed0_normal_errors.npy  one error per held-out healthy frame, manifest order
  results/her-117/pv_block_seed0_fault_errors.npy   one error per PVMD file, pv fault order

Usage:
    venv/Scripts/python.exe src/experiments/block_split_pv.py            # split + train + score
    venv/Scripts/python.exe src/experiments/block_split_pv.py --split-only
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset, InMemoryDataset, Split  # noqa: E402
from hlab.paths import MANIFESTS, MODELS, RESULTS  # noqa: E402
from hlab.train import RandomFrames, load_training_config, fit  # noqa: E402
from hlab.transform import denormalize, normalize  # noqa: E402

OUT = RESULTS / "her-117"
MODEL_DIR = MODELS / "her-117"
SPLIT_CSV = MANIFESTS / "pv_block_split.csv"
SEGMENT = 900        # frames per segment; one held-out block per segment
HOLDOUT_FRACTION = 0.10
GAP = 100            # frame numbers dropped on each side of a held-out block
SEED = 0
NAME = f"pv_block_seed{SEED}"


def frame_key(frame_id):
    """(recording, frame number) from a PV healthy file name."""
    name = frame_id.split("/")[-1]
    m = re.match(r"(dr|sr)_frame_(\d+)", name)
    if m:
        return m.group(1), int(m.group(2))
    m = re.match(r"THM_\d+_(\d+)", name)
    if m:
        return "thm", int(m.group(1))
    raise ValueError(f"no frame order in {frame_id}")


def block_split(frame_ids):
    """{frame_id: (recording, frame_no, split, role)}; split in train | holdout | gap, role val | test | ''."""
    by_rec = {}
    for f in frame_ids:
        rec, n = frame_key(f)
        by_rec.setdefault(rec, []).append((n, f))
    out = {}
    for rec, rows in by_rec.items():
        rows.sort()
        nums = np.array([n for n, _f in rows])
        lo, hi = nums.min(), nums.max() + 1
        n_seg = max(1, round((hi - lo) / SEGMENT))
        edges = np.linspace(lo, hi, n_seg + 1)
        blocks = []
        for a, b in zip(edges[:-1], edges[1:]):
            length = max(2, int(round((b - a) * HOLDOUT_FRACTION)))
            start = int(round((a + b) / 2 - length / 2))
            blocks.append((start, start + length))
        for n, f in rows:
            split, role = "train", ""
            for s, e in blocks:
                if s <= n < e:
                    split, role = "holdout", ("val" if n < s + (e - s) // 2 else "test")
                    break
                if s - GAP <= n < e + GAP:
                    split = "gap"
            out[f] = (rec, int(n), split, role)
    return out


def write_split(split, path=SPLIT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_id", "recording", "frame_no", "split", "role"])
        for f, (rec, n, sp, role) in sorted(split.items(), key=lambda kv: (kv[1][0], kv[1][1])):
            w.writerow([f, rec, n, sp, role])


def read_split(path=SPLIT_CSV):
    with path.open(encoding="utf-8") as fh:
        return {r["frame_id"]: (r["recording"], int(r["frame_no"]), r["split"], r["role"]) for r in csv.DictReader(fh)}


def load_healthy():
    """(frame_ids, 0-255 model-input arrays) of every healthy PV frame, old TRAIN then old HOLDOUT."""
    pv = Dataset("pv")
    tr, ho = pv.split("train"), pv.split("holdout")
    ids = list(tr.frame_ids) + list(ho.frame_ids)
    x = denormalize(np.concatenate([tr.x, ho.x]), pv.stats)
    return ids, x, pv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--score-only", action="store_true",
                        help="score with the best checkpoint saved so far (no training)")
    args = parser.parse_args(argv)

    ids, x, pv = load_healthy()
    split = block_split(ids)
    write_split(split)
    counts = {}
    for rec, _n, sp, _r in split.values():
        counts[(rec, sp)] = counts.get((rec, sp), 0) + 1
    print("frames by recording and split:", dict(sorted(counts.items())))
    if args.split_only:
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    tr_mask = np.array([split[f][2] == "train" for f in ids])
    ho_mask = np.array([split[f][2] == "holdout" for f in ids])
    stats = {"mean": float(x[tr_mask].mean(dtype=np.float64)), "std": float(x[tr_mask].std(dtype=np.float64))}
    np.save(OUT / "norm_stats_pv_block.npy", stats)
    print(f"new TRAIN stats: mean {stats['mean']:.3f}  std {stats['std']:.3f}  (old {pv.stats})")

    def mk(name, mask):
        sel = [f for f, k in zip(ids, mask) if k]
        return Split(name, normalize(x[mask], stats), np.zeros(int(mask.sum()), int), sel,
                     np.array(["thm" if split[f][0] == "thm" else "om" for f in sel]),
                     [{"sequence": split[f][0], "role": split[f][3]} for f in sel])

    holdout = mk("holdout", ho_mask)
    if args.score_only:
        del x
        history = {"best_path": MODEL_DIR / f"autoencoder_{NAME}_best.pth", "best_val_loss": float("nan"),
                   "best_epoch": "?", "epochs_run": "?", "minutes": float("nan")}
    else:
        train = mk("train", tr_mask)
        del x
        cfg = load_training_config("pv")
        ds = InMemoryDataset("pv_block", {"train": train}, stats=stats)
        _bundle, history = fit(ds, NAME, holdout=RandomFrames(**{k: cfg["holdout"][k] for k in ("seed", "fraction")}),
                               loss=cfg["loss"], lr=cfg["lr"], clip=cfg["clip"], seed=SEED, epochs=cfg["epochs"],
                               batch=cfg["batch"], patience=cfg["patience"], final=cfg["final"],
                               model_dir=MODEL_DIR, log=OUT / f"training_log_{NAME}.txt")
    bundle = Bundle(NAME, history["best_path"], stats, None)
    neg = bundle.score_normalized(holdout.x)
    fault_x = normalize(denormalize(pv.split("fault").x, pv.stats), stats)
    pos = bundle.score_normalized(fault_x)
    np.save(OUT / f"{NAME}_normal_errors.npy", neg)
    np.save(OUT / f"{NAME}_fault_errors.npy", pos)
    with (OUT / f"{NAME}_holdout_ids.txt").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(holdout.frame_ids) + "\n")
    print(f"best val_loss {history['best_val_loss']:.6f} at epoch {history['best_epoch']}, "
          f"{history['epochs_run']} epochs, {history['minutes']:.1f} min; "
          f"saved {len(neg)} healthy and {len(pos)} fault errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
