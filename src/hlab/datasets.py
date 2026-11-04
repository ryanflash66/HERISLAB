"""
Datasets (HER-127): directory layout, stored representation and manifests behind one seam.

    ds = Dataset("pv")
    holdout = ds.split("holdout")        # Split: x in model units, labels, frame_ids, source, meta
    thm = ds.frames("holdout", source="thm")   # boolean selector, no "THM_" in-name tests elsewhere
    ds.manifest()                        # [(frame_id, split, source), ...]
    ds.stats                             # {"mean", "std"} the arrays were z-scored with

Datasets and what the preprocess scripts write for them (all under data/CA_Preprocessed/):

  v2          train_normal.npy, test_normal.npy, test_fault.npy, norm_stats.npy
              splits: train, test_normal, test_fault, test (normal + fault, labelled)
              source = equipment folder (electric_motor, induction_motor, transformer, pv_om_inspection, pv_thermal_inspection)
              manifest: split_manifest.txt when preprocess.py has written one; otherwise derived from the raw
              folders in the order preprocess.py reads them, checked against the array lengths
  pv, pv_om   pv/ and pv_om/: train_normal.npy, test_normal.npy, test_fault.npy, norm_stats.npy, split_manifest.txt
              splits: train, holdout, fault, test (holdout + fault, labelled)
              source = "om" | "thm" for healthy frames, "pvmd" for faults; meta carries the O&M sequence (dr / sr)
  india       india/: {train,val,test,cross_site}.npy + {split}_meta.csv (rel_path, label, plot, flight, t_*)
  india_temp  india_temp/: same, stored in degrees C (z-scored here with its own norm_stats), plus {split}_panelmask.npy
              splits: train, val, test, cross_site; labels from meta "label" == "faulty"; source = plot

Split.x is always what a model consumes: z-scored float32 (N, 240, 320). Nothing here
regenerates arrays; the preprocess scripts stay the writers (they use write_manifest).

InMemoryDataset is the in-process adapter for tests: same interface, arrays passed in.
"""
import csv
import hashlib
import os
from dataclasses import dataclass, field, replace

import numpy as np
from PIL import Image, ImageOps

from hlab.paths import PREPROCESSED, RAW, ROOT
from hlab.transform import normalize

NAMES = ("v2", "pv", "pv_om", "india", "india_temp")

# The one declaration of which raw folders feed which split (preprocess.py / preprocess_pv.py read these).
V2_FOLDERS = {
    "train": ("electric_motor", "induction_motor", "transformer", "pv_om_inspection", "pv_thermal_inspection"),
    "test_normal": ("electric_motor", "induction_motor", "transformer"),
    "test_fault": ("electric_motor", "induction_motor", "transformer"),
}
PV_FOLDERS = {"pv": ("pv_om_inspection", "pv_thermal_inspection"), "pv_om": ("pv_om_inspection",)}
PV_FAULT_FOLDER = "pv"                     # data/CA_Training_Data/test/fault/pv (PVMD)
PV_SOURCE = {"pv_om_inspection": "om", "pv_thermal_inspection": "thm"}

_V2_RAW_SUBDIR = {"train": "train/normal", "test_normal": "test/normal", "test_fault": "test/fault"}
_V2_ARRAY = {"train": "train_normal.npy", "test_normal": "test_normal.npy", "test_fault": "test_fault.npy"}
_PV_ARRAY = {"train": "train_normal.npy", "holdout": "test_normal.npy", "fault": "test_fault.npy"}
_PV_SECTION = {"train": "TRAIN", "holdout": "HOLDOUT", "fault": "FAULT"}
_INDIA_SPLITS = ("train", "val", "test", "cross_site")


# ---------------------------------------------------------------- manifests
def write_manifest(path, sections, header=()):
    """Write a split manifest: comment header, then one [SECTION] per split with one frame id per line."""
    with open(path, "w", encoding="utf-8") as fh:
        for line in header:
            fh.write(f"# {line}\n")
        if header:
            fh.write("\n")
        for i, (section, ids) in enumerate(sections.items()):
            if i:
                fh.write("\n")
            fh.write(f"[{section}]\n")
            for frame_id in ids:
                fh.write(f"{frame_id}\n")


def read_manifest(path):
    """Inverse of write_manifest: {SECTION: [frame_id, ...]} in file order."""
    sections, current = {}, None
    for line in open(path, encoding="utf-8").read().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def raw_files(subdir, folder):
    """Repo-relative posix paths of the files preprocess reads from RAW/subdir/folder, in its order."""
    d = RAW / subdir / folder
    if not d.exists():
        return []
    return [(d / f).relative_to(ROOT).as_posix() for f in sorted(os.listdir(d)) if (d / f).is_file()]


def flight_holdout(meta, fraction=0.1, seed=42):
    """Hold out whole flights (meta "flight") until `fraction` of the frames are held out.

    The early-stopping split of the India training runs (HER-18): the flight
    list is shuffled once with `seed`, flights are taken in that order until the
    target is reached. Returns (train_idx, held_idx sorted, chosen flights).
    """
    import random
    by_flight = {}
    for i, m in enumerate(meta):
        by_flight.setdefault(m["flight"], []).append(i)
    flights = sorted(by_flight)
    random.Random(seed).shuffle(flights)
    target, held, chosen = fraction * len(meta), [], []
    for f in flights:
        if len(held) >= target:
            break
        held += by_flight[f]
        chosen.append(f)
    held_set = set(held)
    train_idx = [i for i in range(len(meta)) if i not in held_set]
    return np.array(train_idx), np.array(sorted(held)), chosen


def adjacent_frame(frame_id, step):
    """Frame id of the video neighbour `step` frames away (same zero-padded width), or None."""
    import re
    m = re.search(r"(\d+)(\.[A-Za-z]+)$", frame_id)
    if not m:
        return None
    n = int(m.group(1)) + step
    if n < 0:
        return None
    return frame_id[:m.start(1)] + str(n).zfill(len(m.group(1))) + m.group(2)


# ---------------------------------------------------------------- PVMD duplicates
def pvmd_unique_groups(exif_transpose=False, fault_dir=None):
    """Pixel-identical groups of the PVMD fault files (sorted file order, as preprocess_pv reads them).

    Returns (files, gid, groups): file names, the group index of every file, and
    {hash: [file, ...]}. With exif_transpose the EXIF orientation tag is applied
    first (PVMD stores some rotations only as that tag).
    """
    fault_dir = fault_dir or RAW / "test" / "fault" / PV_FAULT_FOLDER
    files = sorted(p.name for p in fault_dir.iterdir() if p.is_file())
    groups, pix = {}, {}
    for f in files:
        img = Image.open(fault_dir / f)
        if exif_transpose:
            img = ImageOps.exif_transpose(img)
        h = hashlib.md5(np.array(img).tobytes()).hexdigest()
        pix[f] = h
        groups.setdefault(h, []).append(f)
    gid_of = {h: i for i, h in enumerate(groups)}
    gid = np.array([gid_of[pix[f]] for f in files])
    return files, gid, groups


# ---------------------------------------------------------------- splits
@dataclass
class Split:
    """One split in model units. `source` and `meta` are per frame and aligned with `x`."""
    name: str
    x: np.ndarray                      # float32 (N, 240, 320), z-scored
    labels: np.ndarray                 # int, 1 = fault
    frame_ids: list                    # repo-relative posix path per frame ("" when unknown)
    source: np.ndarray                 # str per frame
    meta: list = field(default_factory=list)   # dict per frame (empty dicts when there is no metadata)

    def __len__(self):
        return len(self.x)

    def select(self, mask):
        """Sub-split by boolean mask (keep-masks for the audits)."""
        mask = np.asarray(mask, bool)
        return replace(self, x=self.x[mask], labels=self.labels[mask],
                       frame_ids=[f for f, k in zip(self.frame_ids, mask) if k],
                       source=self.source[mask], meta=[m for m, k in zip(self.meta, mask) if k])


def _concat(name, a, b):
    return Split(name, np.concatenate([a.x, b.x]), np.concatenate([a.labels, b.labels]),
                 a.frame_ids + b.frame_ids, np.concatenate([a.source, b.source]), a.meta + b.meta)


class Dataset:
    """The stored arrays of one dataset, by name. `root` overrides data/CA_Preprocessed (tests)."""

    def __init__(self, name, root=None):
        if name not in NAMES:
            raise ValueError(f"unknown dataset {name!r}; known: {', '.join(NAMES)}")
        self.name = name
        base = root if root is not None else PREPROCESSED
        self.dir = base if name == "v2" else base / name
        self._cache = {}

    def __repr__(self):
        return f"Dataset({self.name!r}, {self.dir})"

    @property
    def stats(self):
        """{"mean", "std"} the arrays were (or, for india_temp, are) z-scored with."""
        return np.load(self.dir / "norm_stats.npy", allow_pickle=True).item()

    def splits(self):
        if self.name == "v2":
            return ("train", "test_normal", "test_fault", "test")
        if self.name in ("pv", "pv_om"):
            return ("train", "holdout", "fault", "test")
        return _INDIA_SPLITS

    def has_split(self, split):
        try:
            return self.array_path(split).exists()
        except KeyError:
            return split in ("test",) and all(self.has_split(s) for s in self._test_parts())

    def _test_parts(self):
        return ("test_normal", "test_fault") if self.name == "v2" else ("holdout", "fault")

    def array_path(self, split):
        """Path of the stored array of `split` (degrees C for india_temp, model units elsewhere)."""
        if self.name == "v2":
            return self.dir / _V2_ARRAY[split]
        if self.name in ("pv", "pv_om"):
            return self.dir / _PV_ARRAY[split]
        if split not in _INDIA_SPLITS:
            raise KeyError(split)
        return self.dir / f"{split}.npy"

    # ---- manifests ----
    def manifest(self):
        """[(frame_id, split, source), ...] for every frame the arrays hold, in array order."""
        if self.name == "v2":
            return self._v2_manifest()
        if self.name in ("pv", "pv_om"):
            return self._pv_manifest()
        out = []
        for split in _INDIA_SPLITS:
            for m in self._india_meta(split):
                out.append((m["rel_path"], split, str(m.get("plot", ""))))
        return out

    def _v2_manifest(self):
        path = self.dir / "split_manifest.txt"
        if path.exists():
            sec = read_manifest(path)
            return [(f, split, f.split("/")[-2]) for split, key in
                    (("train", "TRAIN"), ("test_normal", "TEST_NORMAL"), ("test_fault", "TEST_FAULT"))
                    for f in sec.get(key, [])]
        # Pre-manifest arrays: preprocess.py read the folders in V2_FOLDERS order, sorted file names.
        out = []
        for split, folders in V2_FOLDERS.items():
            for folder in folders:
                out.extend((f, split, folder) for f in raw_files(_V2_RAW_SUBDIR[split], folder))
        return out

    def _pv_manifest(self):
        sec = read_manifest(self.dir / "split_manifest.txt")
        out = [(f, split, PV_SOURCE.get(f.split("/")[-2], "om"))
               for split, key in (("train", "TRAIN"), ("holdout", "HOLDOUT")) for f in sec.get(key, [])]
        if "FAULT" in sec:
            faults = sec["FAULT"]
        else:   # manifests written before HER-127 list healthy frames only; faults are the sorted PVMD folder
            faults = raw_files("test/fault", PV_FAULT_FOLDER)
        out.extend((f, "fault", "pvmd") for f in faults)
        return out

    def _india_meta(self, split):
        key = ("meta", split)
        if key not in self._cache:
            with open(self.dir / f"{split}_meta.csv", encoding="utf-8") as fh:
                self._cache[key] = list(csv.DictReader(fh))
        return self._cache[key]

    def meta(self, split):
        """One dict per frame of `split`, in array order (the India *_meta.csv rows; sequence for PV)."""
        if self.name in ("india", "india_temp"):
            return list(self._india_meta(split))
        return self.split(split).meta

    # ---- splits ----
    def split(self, split):
        if split == "test" and self.name in ("v2", "pv", "pv_om"):
            a, b = (self.split(s) for s in self._test_parts())
            return _concat("test", a, b)
        if split not in self.splits():
            raise ValueError(f"{self.name} has no split {split!r}; splits: {', '.join(self.splits())}")
        x = np.load(self.array_path(split))
        if self.name == "india_temp":
            x = normalize(x, self.stats)              # stored in degrees C
        if self.name in ("india", "india_temp"):
            meta = self._india_meta(split)
            labels = np.array([1 if m["label"] == "faulty" else 0 for m in meta], int)
            source = np.array([str(m.get("plot", "")) for m in meta])
            ids = [m["rel_path"] for m in meta]
        else:
            rows = [r for r in self.manifest() if r[1] == split]
            ids = [r[0] for r in rows]
            source = np.array([r[2] for r in rows])
            labels = np.full(len(x), 1 if split in ("test_fault", "fault") else 0, int)
            meta = [{"sequence": os.path.basename(f).split("_")[0] if src == "om" else ""} for f, src in zip(ids, source)]
        if len(ids) != len(x):
            raise RuntimeError(f"{self}: {split} has {len(x)} frames but the manifest lists {len(ids)}")
        return Split(split, x, labels, ids, source, meta)

    def frames(self, split, source=None, **where):
        """Boolean selector over `split`: by source and/or meta key=value (e.g. sequence="dr", plot="18")."""
        s = self.split(split) if isinstance(split, str) else split
        keep = np.ones(len(s), bool)
        if source is not None:
            keep &= s.source == source
        for k, v in where.items():
            keep &= np.array([str(m.get(k, "")) == str(v) for m in s.meta])
        return keep

    def masks(self, split):
        """Panel masks aligned with `split` (bool (N, 240, 320)). india_temp stores them; india frames are
        aligned by rel_path, and frames india_temp dropped (no FLIR data) get an all-False mask."""
        if self.name == "india_temp":
            return np.load(self.dir / f"{split}_panelmask.npy")
        if self.name != "india":
            raise ValueError(f"{self.name} has no panel masks")
        temp = Dataset("india_temp", root=self.dir.parent)
        stored = temp.masks(split)
        row_of = {m["rel_path"]: i for i, m in enumerate(temp._india_meta(split))}
        out = np.zeros((len(self._india_meta(split)),) + stored.shape[1:], bool)
        for i, m in enumerate(self._india_meta(split)):
            j = row_of.get(m["rel_path"])
            if j is not None:
                out[i] = stored[j]
        return out


class InMemoryDataset:
    """Test adapter with the Dataset interface: splits (and optional masks) are given, nothing is read from disk."""

    def __init__(self, name, splits, stats=None, masks=None):
        self.name = name
        self._splits = dict(splits)
        self._stats = stats or {"mean": 0.0, "std": 1.0}
        self._masks = dict(masks or {})

    @property
    def stats(self):
        return dict(self._stats)

    def splits(self):
        return tuple(self._splits)

    def has_split(self, split):
        return split in self._splits

    def split(self, split):
        if split not in self._splits:
            raise ValueError(f"{self.name} has no split {split!r}; splits: {', '.join(self._splits)}")
        return self._splits[split]

    def manifest(self):
        return [(f, s.name, src) for s in self._splits.values() for f, src in zip(s.frame_ids, s.source)]

    def meta(self, split):
        return list(self.split(split).meta)

    def frames(self, split, source=None, **where):
        return Dataset.frames(self, split, source, **where)

    def masks(self, split):
        if split not in self._masks:
            raise ValueError(f"{self.name} has no panel masks for {split!r}")
        return self._masks[split]
