"""
Where does the hottest relative spot sit, and does it look like a fault?
Three views of the same thing (plot 16, the reltemp_india.py panel masks and
rel_local scores), all in real temperature:

  where    diagnostic for reltemp_india.py (no new score): healthy frames show ~11 K
           spots, so check whether they are at the panel/ground border (mask leak)
           or inside panels. Output: stdout, results/india/reltemp_where.png
  sheet    the 24 plot-16 frames labelled HEALTHY with the largest relative hot spot,
           spot circled, for Ryan to judge by eye: real fault the annotators missed, or
           not? Output: results/india/label_audit_healthy_top24.png and .csv
  compare  side by side: zoomed crops of what the annotators DID box (top row) and the
           hottest frames labelled HEALTHY (rows 2-3), one colour scale (degrees above
           the frame's typical panel temperature). Output: results/india/label_audit_compare.png and .csv

  python src/experiments/label_audit_india.py where|sheet|compare
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.datasets import Dataset  # noqa: E402
from hlab.panels import MIN_PANEL_SHARE, MODULE_WIN, box_mask, hottest_spot, largest_box, resolve_device, scale_point  # noqa: E402
from hlab.paths import RESULTS, ROOT  # noqa: E402
from hlab.rjpg import raw_to_celsius, read_rjpg  # noqa: E402

RES = RESULTS / "india"
CROP = 70  # compare: crop half-size in pixels of the 640x512 frame


class Spots:
    """VAL frames of plot 16 with their panel masks and rel_local scores (from reltemp_india.py)."""

    def __init__(self, dev):
        self.dev = dev
        s = Dataset("india").split("val")
        self.meta, self.y = s.meta, s.labels
        self.masks = np.load(RES / "val_panel_mask.npy")
        sc = {r["rel_path"]: float(r["rel_local_K"]) for r in csv.DictReader(open(RES / "val_reltemp_scores.csv", encoding="utf-8"))}
        self.score = np.array([sc[m["rel_path"]] for m in self.meta])

    def spot(self, i):
        """(temperature C (512, 640), panel mask (512, 640), row, col, excess K, px from ground) of frame i."""
        raw, calib = read_rjpg(ROOT / self.meta[i]["rel_path"])
        t = torch.from_numpy(raw_to_celsius(raw, calib)).to(self.dev)[None, None]
        m = Fn.interpolate(torch.from_numpy(self.masks[i]).float()[None, None].to(self.dev), size=t.shape[2:], mode="nearest")
        r, c, dk, dist = hottest_spot(t, m, MODULE_WIN, MIN_PANEL_SHARE)
        return t[0, 0].cpu().numpy(), m[0, 0].cpu().numpy(), r, c, dk, dist


def where(S):
    he = np.where(S.y == 0)[0]; fa = np.where(S.y == 1)[0]
    rng = np.random.default_rng(2)
    samp = {"healthy": rng.choice(he, 200, replace=False), "faulty": fa}
    for name, idx in samp.items():
        dd = np.array([S.spot(i)[5] for i in idx])
        print(f"{name}: hottest relative spot within 14 px (one patch) of ground: {100*np.mean(dd <= 14):.0f}% of {len(dd)} frames; median distance {np.median(dd):.0f} px")
    hits = 0
    for i in fa:
        _, _, r, c, _, _ = S.spot(i)
        bm = box_mask((ROOT / S.meta[i]["rel_path"]).with_suffix(".json"), 0)
        hits += bool(bm[scale_point(r, c)])
    print(f"faulty: hottest relative spot lands in a fault box {hits}/{len(fa)}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    v = S.score
    top_h = he[np.argsort(-v[he])[:4]]; top_f = fa[np.argsort(-v[fa])[:4]]
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.5))
    for n, i in enumerate(list(top_h) + list(top_f)):
        t, m, r, c, dk, dist = S.spot(i)
        A = ax[n // 4, n % 4]; A.imshow(t, cmap="inferno")
        A.contour(m, levels=[.5], colors="lime", linewidths=.7)
        A.plot(c, r, "o", mfc="none", mec="cyan", ms=16, mew=2)
        A.set_title(f"{S.meta[i]['label']} {Path(S.meta[i]['rel_path']).name}  +{dk:.1f} K, {dist:.0f}px from ground", fontsize=8); A.axis("off")
    plt.tight_layout(); plt.savefig(RES / "reltemp_where.png", dpi=80)
    print("figure: results/india/reltemp_where.png  (top row: 4 healthy frames with the biggest spots; bottom: 4 faulty)")


def sheet(S):
    he = [i for i in range(len(S.meta)) if S.y[i] == 0]
    top = sorted(he, key=lambda i: -S.score[i])[:24]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 6, figsize=(20, 11.5))
    rows = []
    for n, i in enumerate(top):
        t, m, r, c, dk, dist = S.spot(i)
        A = ax[n // 6, n % 6]; A.imshow(t, cmap="inferno", vmin=np.percentile(t, 1), vmax=np.percentile(t, 99.9))
        A.plot(c, r, "o", mfc="none", mec="cyan", ms=18, mew=2)
        p = Path(S.meta[i]["rel_path"])
        A.set_title(f"#{n+1} {p.parts[-3]} {p.name}\n+{dk:.0f} K above nearby panel", fontsize=8); A.axis("off")
        rows.append([n + 1, S.meta[i]["rel_path"], S.meta[i]["flight"], f"{dk:.1f}", f"{dist:.0f}", ""])
    plt.suptitle("Labelled HEALTHY on plot 16, biggest relative hot spots. Is the circled spot a real fault?", fontsize=13)
    plt.tight_layout(); plt.savefig(RES / "label_audit_healthy_top24.png", dpi=75)
    with open(RES / "label_audit_healthy_top24.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["n", "rel_path", "flight", "spot_K", "px_from_ground", "ryan_says"]); w.writerows(rows)
    print("saved results/india/label_audit_healthy_top24.png and .csv")


def _crop(t, r, c):
    r0, c0 = max(0, r - CROP), max(0, c - CROP)
    return t[r0:r0 + 2 * CROP, c0:c0 + 2 * CROP], r - r0, c - c0


def compare(S):
    fa = [i for i in range(len(S.meta)) if S.y[i] == 1]
    he = [i for i in range(len(S.meta)) if S.y[i] == 0 and S.masks[i].mean() > 0.2]
    fa_top = sorted(fa, key=lambda i: -S.score[i])[:6]
    he_top = sorted(he, key=lambda i: -S.score[i])[:12]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 6, figsize=(18, 10))
    for n, i in enumerate(fa_top + he_top):
        t, m, r, c, dk, _ = S.spot(i)
        base = np.median(t[m > 0]) if (m > 0).any() else np.median(t)
        if S.y[i] == 1:  # centre on the biggest annotator box instead of our spot
            x1, y1, x2, y2 = largest_box((ROOT / S.meta[i]["rel_path"]).with_suffix(".json"))
            r, c = int((y1 + y2) / 2), int((x1 + x2) / 2)
        cr, rr, cc = _crop(t - base, r, c)
        A = ax[n // 6, n % 6]; im = A.imshow(cr, cmap="inferno", vmin=-5, vmax=40)
        if S.y[i] == 1:
            A.add_patch(plt.Rectangle((x1 - (c - cc) - 2, y1 - (r - rr) - 2), x2 - x1 + 4, y2 - y1 + 4, fill=False, ec="cyan", lw=2))
            A.set_title(f"LABELLED FAULTY (annotator box)\n{Path(S.meta[i]['rel_path']).name}", fontsize=9, color="tab:green")
        else:
            A.plot(cc, rr, "o", mfc="none", mec="cyan", ms=22, mew=2)
            A.set_title(f"LABELLED HEALTHY  #{n - 5}\n{Path(S.meta[i]['rel_path']).parts[-3]} {Path(S.meta[i]['rel_path']).name}", fontsize=9, color="tab:red")
        A.axis("off")
    fig.colorbar(im, ax=ax, shrink=.6, label="degrees above the frame's typical panel temperature")
    plt.suptitle("Top row: faults the annotators boxed.  Rows 2-3: frames labelled healthy.  Do rows 2-3 look like faults too?", fontsize=13)
    plt.savefig(RES / "label_audit_compare.png", dpi=75, bbox_inches="tight")
    with open(RES / "label_audit_compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["n", "rel_path", "ryan_says (fault / not fault / unsure)"])
        for n, i in enumerate(he_top, 1): w.writerow([n, S.meta[i]["rel_path"], ""])
    print("saved results/india/label_audit_compare.png and .csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["where", "sheet", "compare"]); ap.add_argument("--device", default=None)
    a = ap.parse_args()
    S = Spots(resolve_device(a.device))
    {"where": where, "sheet": sheet, "compare": compare}[a.mode](S)
