"""
Label audit sheet: the 24 plot-16 frames labelled HEALTHY with the largest
relative hot spot (rel_local), shown in real temperature, spot circled.
For Ryan to judge by eye: real fault the annotators missed, or not?
Output: results/india/label_audit_healthy_top24.png and .csv (fill in the 'ryan_says' column)
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_india import load  # noqa: E402
from reltemp_india import RES  # noqa: E402
from reltemp_where import spot  # noqa: E402


def main():
    dev = torch.device("cuda")
    x, meta, y = load("val")
    masks = np.load(RES / "val_panel_mask.npy")
    sc = {r["rel_path"]: float(r["rel_local_K"]) for r in csv.DictReader(open(RES / "val_reltemp_scores.csv", encoding="utf-8"))}
    he = [i for i in range(len(meta)) if y[i] == 0]
    top = sorted(he, key=lambda i: -sc[meta[i]["rel_path"]])[:24]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 6, figsize=(20, 11.5))
    rows = []
    for n, i in enumerate(top):
        t, m, r, c, dk, dist = spot(i, meta, masks, dev)
        A = ax[n // 6, n % 6]; A.imshow(t, cmap="inferno", vmin=np.percentile(t, 1), vmax=np.percentile(t, 99.9))
        A.plot(c, r, "o", mfc="none", mec="cyan", ms=18, mew=2)
        p = Path(meta[i]["rel_path"])
        A.set_title(f"#{n+1} {p.parts[-3]} {p.name}\n+{dk:.0f} K above nearby panel", fontsize=8); A.axis("off")
        rows.append([n + 1, meta[i]["rel_path"], meta[i]["flight"], f"{dk:.1f}", f"{dist:.0f}", ""])
    plt.suptitle("Labelled HEALTHY on plot 16, biggest relative hot spots. Is the circled spot a real fault?", fontsize=13)
    plt.tight_layout(); plt.savefig(RES / "label_audit_healthy_top24.png", dpi=75)
    with open(RES / "label_audit_healthy_top24.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["n", "rel_path", "flight", "spot_K", "px_from_ground", "ryan_says"]); w.writerows(rows)
    print("saved results/india/label_audit_healthy_top24.png and .csv")


if __name__ == "__main__":
    main()
