"""
Validation / test halves of the PV and general-model test pools (HER-97).

The alarm threshold used to be the best-F1 point on the whole test set, so the
reported F1 was tuned on the data it was reported on. This script cuts each
test pool once into a validation half (the threshold is chosen there, then
frozen) and a test half (scored once at that threshold). It reads file names
and pixels only, never a model score, and its output is tracked, so the halves
are fixed before any testing.

  PV healthy (holdout)  per video sequence (O&M dr, O&M sr, THM), the held-out
                        frames in frame order are cut into 10 contiguous blocks;
                        even blocks -> val, odd blocks -> test
  PV faults (PVMD)      by pixel-unique image (all copies of an image share a
                        half), stratified by the image's label mix (C/H/S),
                        half of the images -> val (seed 0)
  v2 test_normal +      by pixel-identical image across both classes (a copy
  test_fault            filed under both labels goes wholly to one half),
                        stratified by (class, equipment folder), half -> val (seed 0)

Outputs (tracked):
  docs/data_manifests/pv_calibration_split.csv   frame_id, label, stratum, group, role
  docs/data_manifests/v2_calibration_split.csv

Run: venv/Scripts/python.exe src/build_calibration_split.py
"""
import re
from collections import Counter

from hlab.datasets import (CALIBRATION_FILE, Dataset, blocked_roles, group_roles, pixel_hash,
                           pvmd_unique_groups, write_calibration)
from hlab.paths import MANIFESTS, ROOT


def frame_number(frame_id):
    return int(re.search(r"(\d+)\.[A-Za-z]+$", frame_id).group(1))


def pv_rows():
    ds = Dataset("pv")
    rows = []
    holdout = [(f, src) for f, split, src in ds.manifest() if split == "holdout"]
    stratum = [src if src == "thm" else "om_" + f.split("/")[-1].split("_")[0] for f, src in holdout]
    for s in sorted(set(stratum)):
        idx = [i for i, x in enumerate(stratum) if x == s]
        roles = blocked_roles([frame_number(holdout[i][0]) for i in idx])
        rows += [dict(frame_id=holdout[i][0], label="healthy", stratum=s,
                      group=holdout[i][0].split("/")[-1], role=r) for i, r in zip(idx, roles)]
    faults = [f for f, split, _src in ds.manifest() if split == "fault"]
    names, gid, _groups = pvmd_unique_groups()
    if [f.split("/")[-1] for f in faults] != names:
        raise RuntimeError("PV manifest faults and the PVMD folder differ in order or content")
    mix = {g: "".join(sorted({n[0].upper() for n, gg in zip(names, gid) if gg == g})) for g in set(gid)}
    roles = group_roles(gid, [mix[g] for g in gid])
    rows += [dict(frame_id=f, label="fault", stratum=mix[g], group=f"pvmd_{g:03d}", role=r)
             for f, g, r in zip(faults, gid, roles)]
    return rows


def v2_rows():
    ds = Dataset("v2")
    items = [(f, split, src) for f, split, src in ds.manifest() if split in ("test_normal", "test_fault")]
    hashes = [pixel_hash(ROOT / f) for f, _s, _src in items]
    label = ["healthy" if s == "test_normal" else "fault" for _f, s, _src in items]
    strata = [f"{lab}_{src}" for lab, (_f, _s, src) in zip(label, items)]
    roles = group_roles(hashes, strata)
    gnum = {h: i for i, h in enumerate(dict.fromkeys(hashes))}
    return [dict(frame_id=f, label=lab, stratum=st, group=f"img_{gnum[h]:03d}", role=r)
            for (f, _s, _src), lab, st, h, r in zip(items, label, strata, hashes, roles)]


def report(name, rows):
    print(f"{name}: {len(rows)} frames")
    files = Counter((r["label"], r["stratum"], r["role"]) for r in rows)
    images = Counter(k for k in {(r["label"], r["stratum"], r["role"], r["group"]) for r in rows})
    images = Counter((lab, st, role) for lab, st, role, _g in images)
    for lab, st in sorted({(r["label"], r["stratum"]) for r in rows}):
        print(f"  {lab:7s} {st:22s} files val {files[(lab, st, 'val')]:4d} test {files[(lab, st, 'test')]:4d}"
              f"   groups val {images[(lab, st, 'val')]:3d} test {images[(lab, st, 'test')]:3d}")


def main():
    for name, build in (("pv", pv_rows), ("v2", v2_rows)):
        rows = build()
        path = MANIFESTS / CALIBRATION_FILE[name]
        write_calibration(path, rows)
        report(name, rows)
        print(f"  wrote {path.relative_to(ROOT).as_posix()}\n")


if __name__ == "__main__":
    main()
