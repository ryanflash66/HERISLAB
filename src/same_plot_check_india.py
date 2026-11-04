"""
Is the plot split to blame for the low scores? Check on plot 17 ITSELF.

The 3 flights held out during training (17.9, 17.14, 17.11) were never trained on
(only used to decide when to stop). Score their 966 healthy frames and their 139
faulty frames (which were set aside, unused). Same plot, same days as training.
If the split between plots were the problem, scores here should be much higher
than on plots 16 and 18.

Output: results/india/same_plot_check_log.txt (stdout)
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_india import WEIGHTS, auroc, boot_ci, errors  # noqa: E402
from flir_rjpg import read_rjpg  # noqa: E402
from preprocess_india import to_model_input  # noqa: E402
from score_peak_india import scores  # noqa: E402
from train_autoencoder import ThermalAutoencoder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "CA_Preprocessed" / "india"
HOLD = {"17.9", "17.14", "17.11"}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalAutoencoder().to(dev); model.load_state_dict(torch.load(WEIGHTS, map_location=dev)); model.eval()
    ns = np.load(D / "norm_stats.npy", allow_pickle=True).item()
    xt = np.load(D / "train.npy", mmap_mode="r")
    tm = list(csv.DictReader(open(D / "train_meta.csv", encoding="utf-8")))
    hi = [i for i, m in enumerate(tm) if m["flight"] in HOLD]
    xh = np.stack([xt[i] for i in hi])
    rows = [r for r in csv.DictReader(open(ROOT / "docs" / "data_manifests" / "india_plot_split.csv", encoding="utf-8"))
            if r["split"] == "train_faulty_unused" and r["flight"] in HOLD]
    xf = np.stack([(to_model_input(read_rjpg(ROOT / r["rel_path"])[0]) - ns["mean"]) / ns["std"] for r in rows]).astype(np.float32)
    x = np.concatenate([xh, xf]); y = np.r_[np.zeros(len(xh), int), np.ones(len(xf), int)]
    e = errors(model, x, dev); p = scores(model, x, dev)
    print(f"SAME-PLOT CHECK, plot 17 held-out flights {sorted(HOLD)}: {len(xf)} faulty / {len(xh)} healthy")
    for name, v, v16, v18 in [("ae_mean (original)", e["mean"], 0.376, 0.373),
                              ("ae_peak3", p["peak3"], 0.640, 0.597),
                              ("input_peak3 (no model)", p["input_peak3 (no model)"], 0.600, 0.585)]:
        a = auroc(v, y); lo, hi_ = boot_ci(v, y)
        print(f"  {name:24s} AUROC {a:.3f} (95% CI {lo:.3f} to {hi_:.3f})   [plot 16: {v16:.3f}, plot 18: {v18:.3f}]")


if __name__ == "__main__":
    main()
