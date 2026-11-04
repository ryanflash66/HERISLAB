"""
Side-by-side for the label audit: zoomed crops in real temperature.
Top row: what the annotators DID box (labelled faulty).
Rows 2-3: the hottest frames labelled HEALTHY.
Same colour scale everywhere: degrees above the frame's typical panel temperature.
Output: results/india/label_audit_compare.png
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_india import ROOT, load  # noqa: E402
from reltemp_india import RES  # noqa: E402
from reltemp_where import spot  # noqa: E402

H = 70  # crop half-size in pixels of the 640x512 frame


def crop(t, r, c):
    r0, c0 = max(0, r - H), max(0, c - H)
    return t[r0:r0 + 2 * H, c0:c0 + 2 * H], r - r0, c - c0


def main():
    dev = torch.device("cuda")
    x, meta, y = load("val")
    masks = np.load(RES / "val_panel_mask.npy")
    sc = {r["rel_path"]: float(r["rel_local_K"]) for r in csv.DictReader(open(RES / "val_reltemp_scores.csv", encoding="utf-8"))}
    fa = [i for i in range(len(meta)) if y[i] == 1]
    he = [i for i in range(len(meta)) if y[i] == 0 and masks[i].mean() > 0.2]
    fa_top = sorted(fa, key=lambda i: -sc[meta[i]["rel_path"]])[:6]
    he_top = sorted(he, key=lambda i: -sc[meta[i]["rel_path"]])[:12]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 6, figsize=(18, 10))
    for n, i in enumerate(fa_top + he_top):
        t, m, r, c, dk, _ = spot(i, meta, masks, dev)
        base = np.median(t[m > 0]) if (m > 0).any() else np.median(t)
        if y[i] == 1:  # centre on the biggest annotator box instead of our spot
            sh = max(json.load(open((ROOT / meta[i]["rel_path"]).with_suffix(".json"), encoding="utf-8"))["shapes"],
                     key=lambda s: abs(s["points"][1][0] - s["points"][0][0]) * abs(s["points"][1][1] - s["points"][0][1]))
            (x1, y1), (x2, y2) = sh["points"]; r, c = int((y1 + y2) / 2), int((x1 + x2) / 2)
        cr, rr, cc = crop(t - base, r, c)
        A = ax[n // 6, n % 6]; im = A.imshow(cr, cmap="inferno", vmin=-5, vmax=40)
        if y[i] == 1:
            A.add_patch(plt.Rectangle((min(x1, x2) - (c - cc) - 2, min(y1, y2) - (r - rr) - 2), abs(x2 - x1) + 4, abs(y2 - y1) + 4, fill=False, ec="cyan", lw=2))
            A.set_title(f"LABELLED FAULTY (annotator box)\n{Path(meta[i]['rel_path']).name}", fontsize=9, color="tab:green")
        else:
            A.plot(cc, rr, "o", mfc="none", mec="cyan", ms=22, mew=2)
            A.set_title(f"LABELLED HEALTHY  #{n - 5}\n{Path(meta[i]['rel_path']).parts[-3]} {Path(meta[i]['rel_path']).name}", fontsize=9, color="tab:red")
        A.axis("off")
    fig.colorbar(im, ax=ax, shrink=.6, label="degrees above the frame's typical panel temperature")
    plt.suptitle("Top row: faults the annotators boxed.  Rows 2-3: frames labelled healthy.  Do rows 2-3 look like faults too?", fontsize=13)
    plt.savefig(RES / "label_audit_compare.png", dpi=75, bbox_inches="tight")
    with open(RES / "label_audit_compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["n", "rel_path", "ryan_says (fault / not fault / unsure)"])
        for n, i in enumerate(he_top, 1): w.writerow([n, meta[i]["rel_path"], ""])
    print("saved results/india/label_audit_compare.png and .csv")


if __name__ == "__main__":
    main()
