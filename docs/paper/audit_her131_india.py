"""
HER-131: India audit with no model. Every number comes from audit.py checks; nothing is trained
except the 16-statistic logistic regression of the no-model baseline, and nothing is tuned.

  1. baselines   audit.global_features on all 13,187 India R-JPGs, read from the raw files in two
                 readings: "minmax" (the raw counts min-max stretched to 0-255 at 320x240, what the
                 india/ arrays and the 0.597 model hold) and "celsius" (degrees C on the fixed
                 0-120 C scale of india_temp/, mapped to 0-255). audit.check_india_baselines: 5
                 folds of whole flights over all frames; fit on plot 17, scored on plots 16 and 18
                 (the india_plot_split.csv split); fit on all India, scored on the certified frames
                 of the two other sites (cross-site; the 17 FLIR Vue Pro frames are left out of the
                 celsius reading, their calibration is unverified, as in preprocess_india_temp.py).
  2. spread      For every labelme box of the 967 India faulty frames (3,258 boxes), max - median
                 and max - min of the degrees C inside it at 640x512 (audit.box_spread), against
                 the same box placed in one healthy frame of the same flight (audit.control_boxes,
                 seed 0). audit.check_box_spread.
  3. duplicates  Boxed frames of one flight in consecutive shots (every R-JPG on disk in the flight
                 folder, ordered by EXIF capture time, audit.shot_key) counted once:
                 audit.check_fault_duplicates at gaps 1, 2 and 5 shots.
  4. preview     Check 1 on the 8-bit preview arrays (data/CA_Preprocessed/india_preview/, HER-104).

Usage:
    venv\\Scripts\\python.exe docs\\paper\\audit_her131_india.py [--workers N]
Writes results/her-131/report.md and boxes.csv (spreads per box); the per-frame pass (features of
every frame, spreads of every box) is cached in results/her-131/frames.npz and reused when present.
"""
import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit  # noqa: E402  (puts src/ on sys.path)
from audit import FEAT, INDIA_READINGS, SEED  # noqa: E402

from hlab.panels import boxes  # noqa: E402
from hlab.paths import MANIFESTS, PREPROCESSED, RESULTS, ROOT  # noqa: E402
from hlab.rjpg import raw_to_celsius, read_rjpg  # noqa: E402

OUT = RESULTS / "her-131"
CACHE = OUT / "frames.npz"
PREVIEW = PREPROCESSED / "india_preview"
UNVERIFIED_CELSIUS = "FLIR Vue Pro"     # camera whose temperatures preprocess_india_temp.py leaves out
# The paper's distinct-fault count (decided on HER-131, Nov 7): the very next shot, the adjacent-frame
# definition the PV leakage check uses (check_temporal_leakage, within=1). Gaps 2 and 5 are the
# sensitivity row; the paper states the range. Grouping uses capture order only, not the hot spot itself.
HEADLINE_GAP = 1


def manifest():
    return list(csv.DictReader(open(MANIFESTS / "india_plot_split.csv", encoding="utf-8")))


# ---------------------------------------------------------------- per-frame pass (parallel)
def _frame(job):
    rel, fault_boxes, controls = job
    raw, calib = read_rjpg(ROOT / rel)
    t = raw_to_celsius(raw, calib)
    feats = audit.india_frame_features(raw, t)
    return (rel, [feats[k] for k in INDIA_READINGS],
            [audit.box_spread(t, b) for b in fault_boxes], [(i, audit.box_spread(t, b)) for i, b in controls])


def frame_pass(rows, workers):
    """Features of every frame, spreads of every fault box and of its healthy control box."""
    india = [r for r in rows if r["source"] == "india"]
    fault_boxes = [(r["rel_path"], r["flight"], r["plot"], b) for r in india if r["label"] == "faulty"
                   for b in boxes((ROOT / r["rel_path"]).with_suffix(".json"))]
    ctrl, fallback = audit.control_boxes(fault_boxes, [r for r in india if r["label"] == "healthy"], seed=SEED)
    by_frame = {}
    for p, _f, _pl, b in fault_boxes:
        by_frame.setdefault(p, []).append(b)
    jobs = [(r["rel_path"], by_frame.get(r["rel_path"], []), ctrl.get(r["rel_path"], [])) for r in rows]
    F = np.zeros((len(INDIA_READINGS), len(rows), len(FEAT)))
    fault_sp, ctrl_sp = [], np.zeros((len(fault_boxes), 3))
    row_of = {r["rel_path"]: i for i, r in enumerate(rows)}
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for n, (rel, feats, fsp, csp) in enumerate(ex.map(_frame, jobs, chunksize=16), 1):
            F[:, row_of[rel]] = feats
            fault_sp += [(rel, k, *v) for k, v in enumerate(fsp)]
            for i, v in csp:
                ctrl_sp[i] = v
            if n % 1000 == 0:
                print(f"  {n}/{len(jobs)} frames, {time.time() - t0:.0f}s", flush=True)
    fault_sp.sort(key=lambda r: (row_of[r[0]], r[1]))
    # box_fault[j] and box_ctrl[j] are the j-th box of fault_boxes: manifest frame order, then labelme order
    return dict(rel=np.array([r["rel_path"] for r in rows]), F=F,
                box_rel=np.array([p for p, *_ in fault_boxes]), box_xyxy=np.array([b for *_, b in fault_boxes]),
                box_fault=np.array([v[2:] for v in fault_sp]), box_ctrl=ctrl_sp, fallback=fallback)


# ---------------------------------------------------------------- shot order on disk
def _taken(path):
    try:
        with Image.open(path) as im:
            return str(im.getexif().get(306, "")) or None
    except OSError:
        return None


def shot_order(folders):
    """({folder: [every readable R-JPG name in it, in capture order]}, [files without a capture time]).

    Capture time, not the file number: a folder can hold two counters (odd and even numbers) shot
    minutes apart, so DJI_0002 can come 17 minutes after DJI_0003.
    """
    out, unreadable = {}, []
    for d in folders:
        names = [p.name for p in (ROOT / d).iterdir() if audit._SHOT.search(p.name)]
        taken = {n: _taken(ROOT / d / n) for n in names}
        unreadable += [f"{d}/{n}" for n, v in taken.items() if v is None]
        out[d] = sorted((n for n in names if taken[n]), key=lambda n: audit.shot_key(n, taken[n]))
    return out, unreadable


# ---------------------------------------------------------------- report
def fmt_row(name, r):
    ci = "[{:.3f}, {:.3f}]".format(*r["ci"])
    cf = "[{:.3f}, {:.3f}]".format(*r["ci_flight"]) if "ci_flight" in r else "n/a"
    return f"| {name} | {r['n_healthy']} | {r['n_faulty']} | {r['auroc']:.3f} | {ci} | {cf} |"


BASELINE_ROWS = [("group_cv", "in-site, all India, 5 folds of whole flights"),
                 ("plot16", "in-site, fit plot 17, score plot 16 (val)"),
                 ("plot18", "in-site, fit plot 17, score plot 18 (test)"),
                 ("plot16_18", "in-site, fit plot 17, score plots 16 + 18"),
                 ("cross_site", "cross-site, fit all India, score certified"),
                 ("random_cv", "for contrast: frames shuffled into 5 folds (HER-104 protocol)")]


def baseline_block(title, b):
    top = sorted(b["single"].items(), key=lambda kv: -kv[1])[:3]
    md = [f"### {title}", "", "| protocol | healthy | faulty | AUROC | 95% CI (frames) | 95% CI (flights) |",
          "|---|---|---|---|---|---|"]
    md += [fmt_row(lab, b[k]) for k, lab in BASELINE_ROWS if k in b]
    md += ["", "best single statistics, all India frames: " + ", ".join(f"{k} {v:.3f}" for k, v in top), ""]
    return md


def check1(cache, rows):
    india = np.array([r["source"] == "india" for r in rows])
    cert = ~india
    y = np.array([r["label"] == "faulty" for r in rows], int)
    flight = np.array([r["flight"] for r in rows]); split = np.array([r["split"] for r in rows])
    cert_ok = {"minmax": cert, "celsius": cert & np.array([r["camera"] != UNVERIFIED_CELSIUS for r in rows])}
    out = {}
    for k, reading in enumerate(INDIA_READINGS):
        F = cache["F"][k]
        out[reading] = audit.check_india_baselines(F[india], y[india], flight[india], split[india],
                                                   F[cert_ok[reading]], y[cert_ok[reading]], seed=SEED)
    return out


def check4(rows):
    """Check 1 on the preview arrays, rows aligned to the manifest by rel_path. None if the arrays are missing."""
    if not (PREVIEW / "norm_stats.npy").exists():
        return None
    stats = np.load(PREVIEW / "norm_stats.npy", allow_pickle=True).item()
    feats = {}
    for s in ("train", "val", "test", "cross_site", "train_faulty_unused"):
        meta = list(csv.DictReader(open(PREVIEW / f"{s}_meta.csv", encoding="utf-8")))
        F = audit.global_features(np.load(PREVIEW / f"{s}.npy", mmap_mode="r"), stats)
        feats.update({m["rel_path"]: f for m, f in zip(meta, F)})
    missing = [r["rel_path"] for r in rows if r["rel_path"] not in feats]
    if missing:
        raise RuntimeError(f"{len(missing)} manifest frames are not in {PREVIEW}, e.g. {missing[:2]}")
    F = np.array([feats[r["rel_path"]] for r in rows])
    india = np.array([r["source"] == "india" for r in rows])
    y = np.array([r["label"] == "faulty" for r in rows], int)
    flight = np.array([r["flight"] for r in rows]); split = np.array([r["split"] for r in rows])
    return audit.check_india_baselines(F[india], y[india], flight[india], split[india], F[~india], y[~india], seed=SEED)


def spread_block(sp, n_frames, fallback):
    md = [f"{sp['max_minus_median']['fault']['n']} boxes on {n_frames} faulty frames; control = the same box in one "
          f"healthy frame of the same flight ({fallback} boxes fell back to a healthy frame of the same plot).", "",
          "| measure | boxes | mean | p5 | p25 | median | p75 | p95 | under 2 C |", "|---|---|---|---|---|---|---|---|---|"]
    for m, r in sp.items():
        for who in ("fault", "control"):
            s = r[who]
            md.append(f"| {m}, {who} | {s['n']} | {s['mean']:.2f} | {s['p5']:.2f} | {s['p25']:.2f} | {s['p50']:.2f} | "
                      f"{s['p75']:.2f} | {s['p95']:.2f} | {s['frac_under']:.1%} |")
    md += [""] + [f"AUROC fault box vs control box, {m}: {r['auroc']:.3f}" for m, r in sp.items()] + [""]
    return md


def check3(rows):
    faulty = [r for r in rows if r["source"] == "india" and r["label"] == "faulty"]
    folder = lambda r: str(Path(r["rel_path"]).parent.as_posix())  # noqa: E731
    boxed = {}
    for r in faulty:
        boxed.setdefault(folder(r), []).append(Path(r["rel_path"]).name)
    order, unreadable = shot_order(sorted(boxed))
    split_of = {(folder(r), Path(r["rel_path"]).name): r["split"] for r in faulty}
    return audit.check_fault_duplicates(boxed, order, split_of), sum(len(v) for v in order.values()), len(order), unreadable


def dup_block(d, n_disk, n_folders, unreadable):
    md = [f"Shot order: every R-JPG on disk in the {n_folders} flight folders that hold a boxed frame "
          f"({n_disk} frames), sorted by EXIF capture time; {len(unreadable)} file(s) on disk could not be "
          f"read and are left out ({', '.join(unreadable) or 'none'}).", "",
          "| joined when at most | boxed frames | distinct faults | frames per fault | largest group | "
          "val (plot 16) | test (plot 18) | plot 17 (unused) |", "|---|---|---|---|---|---|---|---|"]
    for g, r in d.items():
        bs = r["by_split"]
        cell = lambda s: f"{bs[s][0]} -> {bs[s][1]}" if s in bs else "0"  # noqa: E731
        lab = f"{g} shot{'s' if g > 1 else ''}" + (" (headline)" if g == HEADLINE_GAP else "")
        md.append(f"| {lab} | {r['boxed']} | {r['distinct']} | {r['boxed'] / r['distinct']:.2f} | "
                  f"{r['largest']} | {cell('val')} | {cell('test')} | {cell('train_faulty_unused')} |")
    lo, hi = min(r["distinct"] for r in d.values()), max(r["distinct"] for r in d.values())
    md += ["", f"Headline: {d[HEADLINE_GAP]['distinct']} distinct faults ({HEADLINE_GAP}-shot rule); range over the "
               f"settings {lo} to {hi}. Limitation: grouping uses capture order only and does not check that the "
               "grouped frames show the same module."]
    md += ["", "group sizes (frames: groups): " + "; ".join(
        f"gap {g}: " + ", ".join(f"{k}: {v}" for k, v in r["sizes"].items()) for g, r in d.items()), ""]
    return md


def main(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = manifest()
    if CACHE.exists():
        cache = dict(np.load(CACHE, allow_pickle=False))
        if list(cache["rel"]) != [r["rel_path"] for r in rows]:
            raise SystemExit(f"{CACHE} does not match the manifest; move it away and rerun")
        print(f"using cached frame pass {CACHE}")
    else:
        print(f"frame pass over {len(rows)} frames with {workers} workers")
        cache = frame_pass(rows, workers)
        np.savez(CACHE, **cache)
    india_n = sum(r["source"] == "india" for r in rows)
    md = ["# HER-131 India audit (no model)", "",
          f"Manifest: docs/data_manifests/india_plot_split.csv, {india_n} India frames, "
          f"{len(rows) - india_n} certified frames from two other sites. Seed {SEED}. "
          "CIs: 2,000 bootstrap draws, frames resampled within class (hlab.metrics.auroc_ci) and whole flights "
          "resampled (audit.auroc_cluster_ci).", "", "## 1. 16-statistic logistic regression, radiometric", ""]
    b1 = check1(cache, rows)
    md += baseline_block("Reading minmax (raw counts stretched per frame, the india/ model input)", b1["minmax"])
    md += baseline_block("Reading celsius (fixed 0-120 C scale, the india_temp/ input; certified without the 17 "
                         "FLIR Vue Pro frames)", b1["celsius"])
    sp = audit.check_box_spread({"max_minus_median": cache["box_fault"][:, 0], "max_minus_min": cache["box_fault"][:, 1]},
                                {"max_minus_median": cache["box_ctrl"][:, 0], "max_minus_min": cache["box_ctrl"][:, 1]})
    n_faulty = len(set(cache["box_rel"]))
    md += ["## 2. Temperature spread inside the fault boxes (degrees C, 640x512)", ""]
    md += spread_block(sp, n_faulty, int(cache["fallback"]))
    d, n_disk, n_folders, unreadable = check3(rows)
    md += ["## 3. Near-duplicate fault frames", ""] + dup_block(d, n_disk, n_folders, unreadable)
    b4 = check4(rows)
    md += ["## 4. 16-statistic logistic regression, 8-bit previews (india_preview/)", ""]
    md += baseline_block("Preview (palette RGB averaged, min-max stretched, 320x240)", b4) if b4 else ["arrays not built", ""]
    text = "\n".join(md)
    (OUT / "report.md").write_text(text + "\n", encoding="utf-8")
    with open(OUT / "boxes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rel_path", "x0", "y0", "x1", "y1", "fault_max_minus_median", "fault_max_minus_min", "fault_px",
                    "control_max_minus_median", "control_max_minus_min", "control_px"])
        for p, b, f, c in zip(cache["box_rel"], cache["box_xyxy"], cache["box_fault"], cache["box_ctrl"]):
            w.writerow([p, *[round(float(v), 1) for v in b], *[round(float(v), 3) for v in f[:2]], int(f[2]),
                        *[round(float(v), 3) for v in c[:2]], int(c[2])])
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args().workers)
