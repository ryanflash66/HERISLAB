"""
Plot-level split for the Thermal-Library radiometric data (HER-18, HER-95).

Decision (Oct 23): hold out whole 50 MW blocks of the Kurnool park so no test
panel has a near-copy in training.

  plot 17 -> train       (healthy only; its faulty frames are kept but unused)
  plot 16 -> val         (healthy + faulty, used to choose the alarm threshold)
  plot 18 -> test        (healthy + faulty, reported once)
  certified (2 other sites) -> cross_site check

Inputs (gitignored, see data/README.md):
  data/sources/thermal_library_meta/manifest.csv        labels (Carter)
  data/sources/thermal_library_india/Plot_NN_50MW/...   R-JPG frames
  data/sources/thermal_library_certified/dataset_N/...  R-JPG frames

Output (tracked):
  docs/data_manifests/india_plot_split.csv

Frames are keyed on (plot, day, flight, full filename): the manifest writes
flight IDs two ways, and 232 frames are named like "DJI_0325_R (2).JPG" but are
different shots from their namesakes.
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "sources"
MANIFEST = SRC / "thermal_library_meta" / "manifest.csv"
INDIA = SRC / "thermal_library_india"
CERT = SRC / "thermal_library_certified"
OUT = ROOT / "docs" / "data_manifests" / "india_plot_split.csv"

PLOT_SPLIT = {"17": "train", "16": "val", "18": "test"}
INDIA_SETS = {
    "healthy_reviewed_plot16_1743": ("healthy", "reviewed_no_box"),
    "healthy_reviewed_plot17_7790": ("healthy", "reviewed_no_box"),
    "healthy_reviewed_plot18_2687": ("healthy", "reviewed_no_box"),
    "faulty_india_boxes_967": ("faulty", "annotator_boxes"),
}
CERT_SETS = {
    "healthy_certified_97": ("healthy", "certified"),
    "faulty_certified_40": ("faulty", "certified"),
}


def frame_key(text, filename):
    """(plot, day, flight, FILENAME) from any path or manifest fields."""
    m = re.search(r"Plot_(\d+)\.(\d+)", text)
    d = re.search(r"Day(\d+)", text)
    if not (m and d):
        return None
    return (m.group(1), d.group(1), m.group(2), filename.upper())


def index_india():
    files = {}
    for p in INDIA.rglob("*"):
        if p.is_file() and re.search(r"_R( \(\d+\))?\.JPG$", p.name, re.I):
            k = frame_key(p.as_posix(), p.name)
            if k:
                if k in files:
                    sys.exit(f"duplicate key on disk: {k}")
                files[k] = p
    return files


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8", errors="replace")))
    india_files = index_india()
    out, missing = [], []

    for r in rows:
        ds = r["dataset"]
        if ds in INDIA_SETS:
            label, strength = INDIA_SETS[ds]
            name = Path(r["path"]).name
            k = frame_key(" ".join([r["path"], r["day"], r["flight"]]), name)
            p = india_files.get(k)
            if p is None:
                missing.append(r["path"]); continue
            plot, day, flight, _ = k
            split = PLOT_SPLIT[plot]
            if split == "train" and label == "faulty":
                split = "train_faulty_unused"
            out.append({"rel_path": p.relative_to(ROOT).as_posix(), "source": "india",
                        "plot": plot, "day": day, "flight": f"{plot}.{flight}",
                        "label": label, "label_strength": strength, "split": split,
                        "camera": "DJI XT2"})
        elif ds in CERT_SETS:
            label, strength = CERT_SETS[ds]
            sub, name = r["path"].split("/")[-2], Path(r["path"]).name
            p = CERT / sub / "images" / name
            if not p.exists():
                missing.append(r["path"]); continue
            out.append({"rel_path": p.relative_to(ROOT).as_posix(), "source": "certified",
                        "plot": "", "day": "", "flight": sub, "label": label,
                        "label_strength": strength, "split": "cross_site",
                        "camera": r["camera"].split(" 640")[0].strip()})

    if missing:
        sys.exit(f"{len(missing)} manifest rows have no file, e.g. {missing[:3]}")

    # Leakage guard: no flight may appear in more than one split
    fl = {}
    for o in out:
        if o["source"] == "india":
            s = "train" if o["split"].startswith("train") else o["split"]
            if fl.setdefault(o["flight"], s) != s:
                sys.exit(f"flight {o['flight']} appears in two splits")

    out.sort(key=lambda o: (o["split"], o["rel_path"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    c = Counter((o["split"], o["label"]) for o in out)
    print(f"wrote {len(out)} rows -> {OUT.relative_to(ROOT)}")
    for (s, l), n in sorted(c.items()):
        print(f"  {s:20s} {l:8s} {n:6d}")
    print("flights per split:", dict(Counter(fl.values())))


if __name__ == "__main__":
    main()
