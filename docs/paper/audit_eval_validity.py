"""
Evaluation-validity audit for the HERIS Lab thermal anomaly detectors.

Reproduces every number in the paper draft (Problem Statement, Method, Results) that is
not already printed by evaluate_autoencoder.py / evaluate_autoencoder_pv.py.
It never retrains or re-runs a model: it reads the saved reconstruction errors
in results/, the preprocessed arrays and split manifests through
hlab.datasets, the saved thresholds through hlab.bundle (weights are never
loaded), and the raw PVMD files.

Checks (one function each in audit.py; this file picks the inputs and prints):
  1. PVMD duplicates: pixel-identical files, and files whose pixels appear
     under more than one fault label
  2. Temporal leakage: held-out PV normals with an adjacent video frame in TRAIN
  3. Source shortcut: AE error split by acquisition source (O&M vs THM)
  4. Trivial baselines: single global intensity statistics and a 5-fold
     logistic regression on 16 global statistics (no model)
  5. Headline metrics with bootstrap 95% CIs, on the full and deduplicated set
  6. Rule layer: tier distribution and ensemble verdicts on the test sets,
     through hlab.rules.evaluate per image (the same code as the demo)

Usage:
    venv\\Scripts\\python.exe docs\\paper\\audit_eval_validity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import (PVInputs, SEED, check_headline, check_pvmd_duplicates, check_rule_layer,  # noqa: E402
                   check_source_shortcut, check_temporal_leakage, check_trivial_baselines, global_features,
                   rounded, saved_errors, unique_first)
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import Dataset  # noqa: E402
from hlab.paths import ROOT  # noqa: E402
from hlab.rules import RuleConfig  # noqa: E402
from hlab.transform import denormalize  # noqa: E402


def main():
    # The evaluators' frozen best-F1 thresholds (results/*_eval_metrics.npy), read through the bundles.
    pv, v2 = Bundle.load("pv"), Bundle.load("v2")
    pv_thr, v2_thr = pv.require_threshold(), v2.require_threshold()

    print("=" * 70, "\n1. PVMD duplicates\n" + "=" * 70)
    dup = check_pvmd_duplicates()
    print(f"files: {dup['n_files']}   pixel-unique images: {dup['n_unique']}")
    print(f"files whose pixels also appear under another fault label: {dup['multi_label']}")
    print("label mix of unique images:", dup["label_mix"])
    print(f"files with a non-default EXIF orientation tag: {dup['exif_nondefault']}")
    print(f"unique images with EXIF orientation applied: {dup['n_unique_exif']}   "
          f"files sharing an image with another label: {dup['multi_label_exif']}")
    print("label mix with EXIF applied:", dup["label_mix_exif"])

    print("\n" + "=" * 70, "\n2. Temporal leakage in the PV normal split\n" + "=" * 70)
    pvds = Dataset("pv")
    D = PVInputs(pvds, pv, gid=dup["gid"])
    adj, n_ho = check_temporal_leakage(pvds, D.holdout)
    print(f"held-out normals with an immediately adjacent frame in TRAIN: {adj} / {n_ho}")

    print("\n" + "=" * 70, "\n3. Source shortcut (PV specialist)\n" + "=" * 70)
    src = check_source_shortcut(D.neg, D.src, D.fu, pv_thr, order=("om", "thm"))
    for s, r in src["rows"].items():
        print(f"healthy {s:3s}: n={r['n']:4d}  median err {r['median']:.4f}  flagged at thr {r['flagged']}")
    print(f"AUROC healthy O&M vs healthy THM : {src['auroc_om_thm']:.3f}")
    print(f"AUROC healthy O&M vs unique fault: {src['auroc_om_fault']:.3f}")
    print(f"AUROC healthy THM vs unique fault: {src['auroc_thm_fault']:.3f}")

    print("\n" + "=" * 70, "\n4/5. Headline metrics, CIs, trivial baselines\n" + "=" * 70)
    gn, gf = saved_errors(v2)
    for name, neg, pos, thr in [("v2 general", gn, gf, v2_thr), ("PV, 1000 files", D.neg, D.pos, pv_thr),
                                ("PV, 140 unique", D.neg, D.fu, pv_thr)]:
        h = check_headline(neg, pos, thr, seed=SEED)
        print(f"\n{name}: AUROC {h['auroc']:.4f} [{h['ci'][0]:.3f}, {h['ci'][1]:.3f}]   "
              f"flag-everything F1 = {h['trivial_f1']:.4f}")
        print("  at saved threshold :", rounded(h["saved"]))
        print("  re-tuned best-F1   :", rounded(h["retuned"]))

    print("\nComputing global image statistics (no model)...")
    pvnF, pvfF_u = D.features()
    v2ds = Dataset("v2")
    v2n, v2f = v2ds.split("test_normal"), v2ds.split("test_fault")
    gnF = global_features(v2n.x, v2ds.stats)
    gfF = global_features(v2f.x, v2ds.stats)
    for name, a, b in [("PV (unique faults)", pvnF, pvfF_u), ("v2 general", gnF, gfF)]:
        base = check_trivial_baselines(a, b, seed=SEED)
        print(f"{name}: best single statistics {[(k, round(v, 3)) for k, v in base['top']]}")
        print(f"{name}: 5-fold logistic regression on 16 statistics AUROC {base['logreg']:.3f}")

    config = RuleConfig.load()
    print("\n" + "=" * 70, f"\n6. Rule layer ({Path(config.source).relative_to(ROOT).as_posix()} defaults)\n" + "=" * 70)
    first = unique_first(D.gid)
    for name, imgs, e in [("PV healthy", denormalize(D.holdout.x, pvds.stats), D.neg),
                          ("PV unique faults", denormalize(D.fault.x[first], pvds.stats), D.fu)]:
        tiers, final, _ = check_rule_layer(config, imgs, "pv", e > pv_thr)
        print(f"{name}: tiers {tiers}  ensemble {final}")
    ceiling = float(config.ambient_c) + float(config.equipment("transformer")["ceiling_over_ambient_c"])
    for s in ("transformer",):
        for lab, split, e in [("healthy", v2n, gn), ("fault", v2f, gf)]:
            keep = v2ds.frames(split, source=s)
            _, _, flagged = check_rule_layer(config, denormalize(split.x[keep], v2ds.stats), s, e[keep] > v2_thr)
            print(f"{s} {lab}: n={int(keep.sum())} AE flags {(e[keep] > v2_thr).sum()}  "
                  f"top-oil rule (>{ceiling:.0f} C) flags {flagged}")


if __name__ == "__main__":
    main()
