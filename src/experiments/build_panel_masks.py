"""
Stage 1 of the pipeline: panel masks for every India frame (Oct 26).

Uses the DINOv2 panel/ground split of panel_dino_india.py (hlab.panels.dino_panel_fit:
k-means on healthy plot 17 patches, no fault labels), computed from the stretched
8-bit input in data/CA_Preprocessed/india (same frame order as india_temp; cross_site
is matched by rel_path because india_temp drops the 17 FLIR Vue Pro frames).

Output: data/CA_Preprocessed/india_temp/{split}_panelmask.npy  bool (N, 240, 320)
        + a one-line summary per split
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hlab.datasets import Dataset  # noqa: E402
from hlab.panels import dino_panel_fit, resolve_device  # noqa: E402
from hlab.transform import TARGET_SIZE  # noqa: E402

SEED, N_CLUSTER_FRAMES, N_BANK_FRAMES = 0, 300, 1500      # the panel_dino_india.py fit


def main(device=None):
    dev = resolve_device(device)
    ds, temp = Dataset("india"), Dataset("india_temp")
    xt = np.load(ds.array_path("train"), mmap_mode="r")
    pm = dino_panel_fit(xt, ds.stats, seed=SEED, n_bank_frames=N_BANK_FRAMES, n_cluster_frames=N_CLUSTER_FRAMES, device=dev)

    size = (TARGET_SIZE[1], TARGET_SIZE[0])
    for split in ("train", "val", "test", "cross_site"):
        x = np.load(ds.array_path(split), mmap_mode="r")
        idx = {m["rel_path"]: i for i, m in enumerate(ds.meta(split))}
        order = [idx[m["rel_path"]] for m in temp.meta(split)]        # align with india_temp
        masks = np.concatenate([pm.masks(np.stack([x[j] for j in order[i:i + 128]]), size=size)
                                for i in range(0, len(order), 128)])
        np.save(temp.dir / f"{split}_panelmask.npy", masks)
        print(f"{split}: {masks.shape}, panel share median {100*np.median(masks.mean((1, 2))):.0f}%, "
              f"frames with <10% panel: {int((masks.mean((1, 2)) < 0.10).sum())}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default=None)
    main(ap.parse_args().device)
