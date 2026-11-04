"""
Visual check of the no-model roughness panel mask (hlab.panels.panel_mask) on a
few plot-16 frames: healthy and faulty, with the annotators' boxes drawn.

  python src/experiments/panel_mask_check_india.py   -> results/india/panel_mask_check_v<version>.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.datasets import Dataset  # noqa: E402
from hlab.panels import MASK_VERSION, box_mask, panel_mask  # noqa: E402
from hlab.paths import RESULTS, ROOT  # noqa: E402


def main(version=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ds = Dataset("india")
    x = np.load(ds.array_path("val"), mmap_mode="r")
    meta = ds.meta("val")
    fa = [i for i, m in enumerate(meta) if m["label"] == "faulty"]
    he = [i for i, m in enumerate(meta) if m["label"] == "healthy"]
    rng = np.random.default_rng(1)
    pick = list(rng.choice(he, 4, replace=False)) + list(rng.choice(fa, 4, replace=False))
    b = torch.from_numpy(np.stack([x[i] for i in pick])).unsqueeze(1)
    m = panel_mask(b, version)[:, 0].numpy()
    fig, ax = plt.subplots(4, 4, figsize=(14, 11))
    for k, i in enumerate(pick):
        a = ax[k // 4 * 2, k % 4]
        a.imshow(x[i], cmap="gray"); a.set_title(f"{meta[i]['label']} {Path(meta[i]['rel_path']).name}", fontsize=8); a.axis("off")
        a2 = ax[k // 4 * 2 + 1, k % 4]
        a2.imshow(x[i], cmap="gray"); a2.imshow(np.ma.masked_where(m[k], np.ones_like(m[k])), cmap="autumn", alpha=.45, vmin=0, vmax=1)
        if meta[i]["label"] == "faulty":
            a2.contour(box_mask((ROOT / meta[i]["rel_path"]).with_suffix(".json"), 2), levels=[.5], colors="cyan", linewidths=1)
        a2.set_title(f"orange = masked out ({100*(1-m[k].mean()):.0f}%)", fontsize=8); a2.axis("off")
    v = version or MASK_VERSION
    plt.tight_layout(); plt.savefig(RESULTS / "india" / f"panel_mask_check_v{v}.png", dpi=80)
    print(f"saved results/india/panel_mask_check_v{v}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--version", type=int, default=None, help="mask version (default: the current one)")
    main(ap.parse_args().version)
