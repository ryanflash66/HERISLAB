"""
HER-114: screen the THM (Kaggle "Photovoltaic system thermal inspection",
senseFly eBee + ThermoMAP, 1,075 frames) set that the PV specialist treats as
healthy.

Step 1 (this script, automatic):
  * per-frame statistics from the raw 16-bit TIFFs: min / median / MAD / max,
    and a hot-spot score = max of a top-hat response
    (3x3-median-filtered frame minus its 31x31 local mean), in frame-MAD units
  * contact sheets of every frame (48 per sheet, frame index printed) for
    visual review of PV content
  * a top-k sheet of 96x96 crops centred on each frame's hottest blob

Step 2 (manual): review the sheets, fill results/her-114/thm_visual_labels.csv
  (frame, pv_content, hotspot, note), then run with --merge to write the
  committed manifest docs/data_manifests/thm_screened_frames.csv.

Outputs (new files only; nothing in the existing split is touched):
  results/her-114/thm_frame_metrics.csv
  results/her-114/sheets/thm_sheet_NN.png
  results/her-114/thm_tophat_top96.png
  docs/data_manifests/thm_screened_frames.csv   (with --merge)
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
THM = ROOT / "data" / "CA_Training_Data" / "train" / "normal" / "pv_thermal_inspection"
OUT = ROOT / "results" / "her-114"
MANIFEST = ROOT / "docs" / "data_manifests" / "thm_screened_frames.csv"
SPLITS = ROOT / "docs" / "data_manifests" / "pv_normal_splits.csv"


def median3(a):
    t = torch.from_numpy(a)[None, None]
    u = F.unfold(F.pad(t, (1, 1, 1, 1), mode="reflect"), 3)
    return u.median(dim=1).values.reshape(a.shape).numpy()


def box_mean(a, k):
    t = torch.from_numpy(a)[None, None]
    return F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="reflect"), k, stride=1)[0, 0].numpy()


def frame_metrics(a):
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)) * 1.4826) + 1e-6
    m3 = median3(a)
    tophat = m3 - box_mean(m3, 31)
    iy, ix = np.unravel_index(int(tophat.argmax()), tophat.shape)
    return dict(min=int(a.min()), max=int(a.max()), median=med, mad=round(mad, 1),
                med3_max=int(m3.max()), max_minus_median=int(m3.max() - med),
                tophat_max=round(float(tophat.max()), 1),
                tophat_z=round(float(tophat.max() / mad), 2), hot_y=int(iy), hot_x=int(ix))


def to_u8(a, lo_p=1, hi_p=99.8):
    lo, hi = np.percentile(a, [lo_p, hi_p])
    return (np.clip((a - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true", help="merge visual labels into the manifest")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in THM.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    split_of = {}
    with SPLITS.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            split_of[row["filename"]] = row["split"]

    if args.merge:
        metrics = {r["frame"]: r for r in csv.DictReader((OUT / "thm_frame_metrics.csv").open(encoding="utf-8"))}
        arr = {r["frame"]: r for r in csv.DictReader((OUT / "thm_array_tophat.csv").open(encoding="utf-8"))}
        labels = {r["frame"]: r for r in csv.DictReader((OUT / "thm_visual_labels.csv").open(encoding="utf-8"))}
        missing = [f for f in files if f not in labels]
        if missing:
            raise SystemExit(f"{len(missing)} frames without a visual label, e.g. {missing[:5]}")
        with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "split", "flag", "reason", "reviewer", "pv_content", "hotspot", "tophat_z", "arr_tophat_z", "note"])
            counts = {}
            for f in files:
                lab, met = labels[f], metrics[f]
                pv = lab["pv_content"]
                reasons = []
                if pv == "none":
                    reasons.append("no PV modules in frame (parking, field, road, roof, building)")
                if lab["hotspot"] == "yes":
                    reasons.append(f"hot-spot candidate on a module, visually confirmed (array top-hat z={arr[f]['arr_tophat_z']})")
                if lab["hotspot"] == "candidate":
                    reasons.append(f"hot-spot candidate by metric only, not visually confirmed (array top-hat z={arr[f]['arr_tophat_z']})")
                if pv == "partial":
                    reasons.append("PV modules occupy less than half of the frame")
                flag = ("drop" if (pv == "none" or lab["hotspot"] == "yes")
                        else "review" if (pv == "partial" or lab["hotspot"] == "candidate") else "keep")
                counts[flag] = counts.get(flag, 0) + 1
                w.writerow([f, split_of.get(f, "?"), flag, "; ".join(reasons) or "PV array fills frame, no hot-spot candidate",
                            lab.get("reviewer", "Claude (visual review of contact sheets)"), pv, lab["hotspot"],
                            met["tophat_z"], arr[f]["arr_tophat_z"], lab.get("note", "")])
        print("wrote", MANIFEST, counts)
        return

    rows = []
    thumbs = []
    for i, f in enumerate(files):
        a = np.array(Image.open(THM / f)).astype(np.float32)
        m = frame_metrics(a)
        m.update(frame=f, idx=i, split=split_of.get(f, "?"), height=a.shape[0], width=a.shape[1])
        rows.append(m)
        thumbs.append((f, i, to_u8(a), m))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)}")

    with (OUT / "thm_frame_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ["frame", "idx", "split", "height", "width", "min", "median", "mad", "max", "med3_max",
                "max_minus_median", "tophat_max", "tophat_z", "hot_y", "hot_x"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    # contact sheets, 8 x 6 per sheet, thumbnail 200 x 160
    TW, TH, NC, NR = 200, 160, 8, 6
    per = NC * NR
    for s in range(0, len(thumbs), per):
        sheet = Image.new("L", (TW * NC, TH * NR), 0)
        d = ImageDraw.Draw(sheet)
        for j, (f, i, u8, m) in enumerate(thumbs[s:s + per]):
            t = Image.fromarray(u8).resize((TW - 2, TH - 2))
            x, y = (j % NC) * TW, (j // NC) * TH
            sheet.paste(t, (x, y))
            tag = f"{i} {f[-9:-4]}" + (" H" if m["split"] == "holdout" else "")
            d.rectangle([x, y, x + 8 * len(tag), y + 11], fill=0)
            d.text((x + 1, y), tag, fill=255)
        sheet.save(OUT / "sheets" / f"thm_sheet_{s // per:02d}.png")

    # top-k top-hat crops
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["tophat_z"])[:96]
    CS, NC2 = 96, 12
    sheet = Image.new("L", (CS * NC2, CS * (len(order) // NC2)), 0)
    d = ImageDraw.Draw(sheet)
    for j, i in enumerate(order):
        f, _, u8, m = thumbs[i]
        y0 = min(max(m["hot_y"] - CS // 2, 0), u8.shape[0] - CS)
        x0 = min(max(m["hot_x"] - CS // 2, 0), u8.shape[1] - CS)
        crop = u8[y0:y0 + CS, x0:x0 + CS]
        x, y = (j % NC2) * CS, (j // NC2) * CS
        sheet.paste(Image.fromarray(crop), (x, y))
        tag = f"{i} z{m['tophat_z']:.0f}"
        d.rectangle([x, y, x + 7 * len(tag), y + 11], fill=0)
        d.text((x + 1, y), tag, fill=255)
    sheet.save(OUT / "thm_tophat_top96.png")
    print("done:", len(files), "frames")


if __name__ == "__main__":
    main()
