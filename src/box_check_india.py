"""
Box check (HER-80 diagnostic): does the autoencoder's error map light up
inside the annotators' fault boxes?

Run on VAL only (plot 16, 121 faulty frames). Test (plot 18) stays untouched.

For each faulty frame:
  - error map  = (reconstruction - input)^2 at 240x320
  - boxes      = labelme rectangles from the frame's .json (640x512 coords),
                 scaled to 320x240, padded by PAD pixels
  - err_ratio  = mean error inside boxes / mean error outside
  - hit        = is the single highest-error pixel inside a box?
  - chance     = share of the frame the boxes cover (hit rate if error were random)
  - img_ratio  = mean input brightness inside / outside (alignment sanity check:
                 hot spots should be brighter, so this should be > 1)

Outputs: results/india/box_check_val.csv, results/india/box_check_val.png, stdout summary
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_autoencoder import ThermalAutoencoder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "CA_Preprocessed" / "india"
RES = ROOT / "results" / "india"
WEIGHTS = ROOT / "models" / "autoencoder_india_best.pth"
SRC_W, SRC_H, W, H = 640, 512, 320, 240


def box_mask(json_path, pad):
    m = np.zeros((H, W), bool)
    for s in json.load(open(json_path, encoding="utf-8"))["shapes"]:
        (x1, y1), (x2, y2) = s["points"]
        x0, x1_ = sorted((x1, x2)); y0, y1_ = sorted((y1, y2))
        c0 = max(0, math.floor(x0 * W / SRC_W) - pad); c1 = min(W, math.ceil((x1_ + 1) * W / SRC_W) + pad)
        r0 = max(0, math.floor(y0 * H / SRC_H) - pad); r1 = min(H, math.ceil((y1_ + 1) * H / SRC_H) + pad)
        m[r0:r1, c0:c1] = True
    return m


def sign_test_p(k, n):
    """Two-sided binomial test vs p=0.5."""
    pk = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pk if p <= pk[k] + 1e-15))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalAutoencoder().to(dev)
    model.load_state_dict(torch.load(WEIGHTS, map_location=dev)); model.eval()

    x = np.load(DATA / "val.npy")
    meta = list(csv.DictReader(open(DATA / "val_meta.csv", encoding="utf-8")))
    idx = [i for i, m in enumerate(meta) if m["label"] == "faulty"]
    xb = torch.from_numpy(x[idx]).unsqueeze(1).to(dev)
    with torch.no_grad():
        rec = torch.cat([model(xb[i:i + 32]) for i in range(0, len(xb), 32)])
    err = ((rec - xb) ** 2)[:, 0].cpu().numpy()
    rec = rec[:, 0].cpu().numpy()

    rows, keep = [], []
    for pad in (0, 2):
        for j, i in enumerate(idx):
            jp = (ROOT / meta[i]["rel_path"]).with_suffix(".json")
            m = box_mask(jp, pad)
            e, im = err[j], x[i]
            # rank of the box's mean error among all pixels (0.5 = typical pixel)
            rows.append({
                "pad": pad, "rel_path": meta[i]["rel_path"], "n_box_px": int(m.sum()),
                "chance": m.mean(), "err_in": e[m].mean(), "err_out": e[~m].mean(),
                "err_ratio": e[m].mean() / e[~m].mean(),
                "hit": bool(m.flat[e.argmax()]),
                "box_err_pct": (e < e[m].mean()).mean(),
                "img_ratio": (im[m].mean() - im.min()) / (im[~m].mean() - im.min() + 1e-9),
            })
            if pad == 2 and len(keep) < 6:
                keep.append((im, rec[j], e, m, Path(meta[i]["rel_path"]).name))

    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "box_check_val.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f"BOX CHECK on VAL (plot 16): {len(idx)} faulty frames")
    for pad in (0, 2):
        r = [q for q in rows if q["pad"] == pad]
        ratio = np.array([q["err_ratio"] for q in r]); n = len(r); k = int((ratio > 1).sum())
        hits = sum(q["hit"] for q in r); chance = np.mean([q["chance"] for q in r])
        print(f"\n-- box padding {pad} px  (boxes cover {100*chance:.2f}% of a frame on average)")
        print(f"  error inside/outside, median ratio : {np.median(ratio):.2f}x")
        print(f"  frames with MORE error inside box  : {k}/{n}  (sign test p = {sign_test_p(k, n):.2g})")
        print(f"  box mean error beats this % of px  : median {100*np.median([q['box_err_pct'] for q in r]):.0f}%")
        print(f"  top-error pixel lands in a box     : {hits}/{n} = {100*hits/n:.1f}%  (chance {100*chance:.2f}%)")
        print(f"  sanity: input brightness in/out    : median {np.median([q['img_ratio'] for q in r]):.2f}x  (should be > 1)")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(keep), 3, figsize=(12, 3.2 * len(keep)))
        for r_, (im, rc, e, m, name) in enumerate(keep):
            for c_, (a, t) in enumerate([(im, "input " + name), (rc, "reconstruction"), (e, "error map")]):
                ax[r_, c_].imshow(a, cmap="inferno" if c_ == 2 else "gray"); ax[r_, c_].set_title(t, fontsize=9)
                ax[r_, c_].contour(m, levels=[0.5], colors="cyan", linewidths=0.8); ax[r_, c_].axis("off")
        plt.tight_layout(); plt.savefig(RES / "box_check_val.png", dpi=90)
        print("\nfigure: results/india/box_check_val.png")
    except Exception as ex:  # figure is optional
        print("figure skipped:", ex)


if __name__ == "__main__":
    main()
