"""
Evaluation-validity audit for the HERIS Lab thermal anomaly detectors: the script the paper cites.

Prints every number in the paper draft (Abstract, Method, Results, Tables III to VII) and
writes both data figures (Fig. 2 and Fig. 3, single PV panels) to docs/paper/figs/. Its output
is the source of truth: where the paper and this printout differ, the paper changes (HER-110).
It never retrains or re-runs a model: it reads the saved reconstruction errors in results/,
the training log, the preprocessed arrays and split manifests through hlab.datasets, the saved
thresholds through hlab.bundle (weights are never loaded), and the raw image files.

Sections (one or more check functions each in audit.py; this file picks the inputs and prints):
  0. Data, split and training: split sizes, normalisation statistics, THM share of TRAIN, training log
  1. PVMD duplicates: byte-identical and pixel-identical files, files under more than one fault label
  2. Test-set hashes: duplicates, cross-label copies and training-pool copies on every test set
     (the PV holdout and PVMD against the PV training pool; every motor and transformer test folder
     against the general model's training pool, HER-105)
  3. Temporal leakage: held-out PV frames with an adjacent (and a two-away) frame in TRAIN, and the
     frame-correlation sample
  4. Source shortcut: AE error by acquisition source (O&M vs THM), source AUROCs with bootstrap CIs
  5. Headline metrics (Tables III and IV): AUROC and AUPRC with bootstrap CIs, TPR at FPR <= 5%,
     confusion at the saved and re-tuned thresholds, the cross-validated threshold
  6. Trivial baselines (Table VI): single global statistics and the 16-statistic logistic regression,
     and the HER-107 same-transform table (the trivial baselines under each hlab.transform variant)
  7. Rule layer (Table VII): dT-proxy AUROC, tier distribution and ensemble verdicts via hlab.rules
  8. Figures: docs/paper/figs/fig_pv_error_by_source.png (Fig. 2), fig_roc_vs_trivial_baselines.png (Fig. 3),
     fig_pr_original_vs_dedup.png (precision-recall, original and deduplicated test sets, HER-116)

Usage:
    venv\\Scripts\\python.exe docs\\paper\\audit_eval_validity.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit  # noqa: E402
from audit import (AUPRC_DEF, PVInputs, SEED, check_cv_threshold, check_frame_correlation, check_headline,  # noqa: E402
                   check_pvmd_duplicates, check_rule_layer, check_source_shortcut, check_temporal_leakage,
                   check_test_hashes, check_transform_baselines, image_formats, check_trivial_baselines, fig_error_by_source,
                   fig_pr_curves, fig_roc_vs_baselines, global_features, logreg_cv_scores, pv_frame_paths, read_training_log,
                   rounded, rule_signals, saved_errors, source_auroc_cis, unique_first)
from hlab.bundle import Bundle  # noqa: E402
from hlab.datasets import V2_FOLDERS, Dataset, raw_files  # noqa: E402
from hlab.metrics import auroc, from_groups  # noqa: E402
from hlab.paths import RAW, RESULTS, ROOT  # noqa: E402
from hlab.rules import RuleConfig  # noqa: E402
from hlab.transform import denormalize  # noqa: E402

FIGS = Path(__file__).resolve().parent / "figs"


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def ci(v):
    return f"[{v[0]:.3f}, {v[1]:.3f}]"


def main():
    # The evaluators' frozen best-F1 thresholds (results/*_eval_metrics.npy), read through the bundles.
    pv, v2 = Bundle.load("pv"), Bundle.load("v2")
    pv_thr, v2_thr = pv.require_threshold(), v2.require_threshold()
    pvds = Dataset("pv")
    manifest = pvds.manifest()

    header("0. Data, split and training (PV specialist)")
    n_split = {s: sum(1 for _f, sp, _src in manifest if sp == s) for s in ("train", "holdout", "fault")}
    n_train_thm = sum(1 for _f, sp, src in manifest if sp == "train" and src == "thm")
    n_healthy_src = {s: sum(1 for _f, sp, src in manifest if sp != "fault" and src == s) for s in ("om", "thm")}
    print(f"healthy PV images: {n_split['train'] + n_split['holdout']} (O&M {n_healthy_src['om']}, "
          f"THM {n_healthy_src['thm']})   TRAIN {n_split['train']}   HOLDOUT {n_split['holdout']}   "
          f"PVMD fault files {n_split['fault']}")
    print(f"excluded healthy module crops (train/normal/solar_modules): "
          f"{len(raw_files('train/normal', 'solar_modules'))}")
    print(f"THM frames in TRAIN: {n_train_thm} / {n_split['train']} = {n_train_thm / n_split['train']:.1%}")
    for lab, sp in (("healthy", ("train", "holdout")), ("PVMD fault", ("fault",))):
        fmt = image_formats([ROOT / f for f, s, _src in manifest if s in sp])
        print(f"{lab} image formats (PIL mode, size): {fmt}")
    st = pvds.stats
    print(f"normalisation statistics of TRAIN: mean {st['mean']:.2f}  std {st['std']:.2f}")
    print(f"saved F1-optimal threshold tau: {pv_thr:.6f} ({pv_thr:.2e})")
    log = read_training_log(RESULTS / "training_log_pv.txt")
    print(f"training log: train {log['n_train']}  val {log['n_val']}  parameters {log['parameters']:,}")
    print(f"  lowest val loss {log['best_val_loss']:.4f} at epoch {log['best_epoch']}; largest val loss "
          f"{log['spike_val_loss']:.2f} at epoch {log['spike_epoch']}; early stop at epoch {log['stop_epoch']}; "
          f"{log['minutes']} min")

    header("1. PVMD duplicates")
    dup = check_pvmd_duplicates()
    print(f"files: {dup['n_files']}   byte-unique files: {dup['n_byte_unique']}   "
          f"pixel-unique images: {dup['n_unique']}")
    print("byte-unique files of files, by label:",
          {k: f"{u} of {n}" for k, (u, n) in dup["byte_unique_by_label"].items()})
    print(f"files whose pixels also appear under another fault label: {dup['multi_label']}")
    print("label mix of unique images:", dup["label_mix"])
    print(f"files with a non-default EXIF orientation tag: {dup['exif_nondefault']}")
    print(f"unique images with EXIF orientation applied: {dup['n_unique_exif']}   "
          f"files sharing an image with another label: {dup['multi_label_exif']}")
    print("label mix with EXIF applied:", dup["label_mix_exif"])

    header("2. Test-set hashes (decoded pixels; every test set against its model's training pool)")
    pv_sets = {"PV holdout (healthy)": ([ROOT / f for f, sp, _s in manifest if sp == "holdout"], 0),
               "PVMD (fault)": ([ROOT / f for f, sp, _s in manifest if sp == "fault"], 1)}
    pv_pool = {"PV TRAIN": [ROOT / f for f, sp, _s in manifest if sp == "train"]}
    v2_man = Dataset("v2").manifest()
    v2_sets = {f"{kind}/{folder}": ([ROOT / f for f in raw_files(f"test/{kind}", folder)], int(kind == "fault"))
               for kind, split in (("normal", "test_normal"), ("fault", "test_fault"))
               for folder in V2_FOLDERS[split]}
    v2_sets["fault/induction_motor_ambiguous"] = ([ROOT / f for f in raw_files("test/fault", "induction_motor_ambiguous")], 1)
    v2_pool = {}
    for f, sp, src in v2_man:
        if sp == "train":
            v2_pool.setdefault(f"train/{src}", []).append(ROOT / f)
    for title, sets, pool in (("PV specialist test set vs PV TRAIN", pv_sets, pv_pool),
                              ("general model test folders vs its TRAIN pool", v2_sets, v2_pool)):
        hs = check_test_hashes(sets, pool)
        print(f"\n{title}:")
        for name, r in hs["sets"].items():
            print(f"  {name:34s} files {r['n_files']:5d}  distinct {r['n_unique']:5d}  "
                  f"opposite-label copies {r['opposite_label']:4d}  in TRAIN pool {r['in_train']:4d} "
                  f"{r['in_train_by_pool'] or ''}  shared {r['shared'] or ''}")
        t = hs["total"]
        print(f"  all of the above: files {t['n_files']}  distinct images {t['n_unique']}  in TRAIN pool {t['in_train']}")
    v2_test = {k: v for k, v in v2_sets.items() if "ambiguous" not in k}
    t = check_test_hashes(v2_test, v2_pool)["total"]
    print(f"general model test set as evaluated (test_normal + test_fault): files {t['n_files']}  "
          f"distinct images {t['n_unique']}  in TRAIN pool {t['in_train']}")

    header("3. Temporal leakage in the PV normal split")
    D = PVInputs(pvds, pv, gid=dup["gid"])
    adj, n_ho = check_temporal_leakage(pvds, D.holdout)
    adj2, _ = check_temporal_leakage(pvds, D.holdout, within=2)
    print(f"held-out normals with an immediately adjacent frame in TRAIN: {adj} / {n_ho}")
    print(f"held-out normals with a TRAIN frame within two frames: {adj2} / {n_ho}")
    train_ids = [f for f, sp, _s in manifest if sp == "train"]
    train_x = np.load(pvds.array_path("train"), mmap_mode="r")
    corr = check_frame_correlation(D.holdout, train_ids, train_x, n_per_sequence=15, seed=SEED)
    for seq, r in corr.items():
        name = {"dr": "double-row", "sr": "single-row"}.get(seq, seq)
        print(f"O&M {seq} ({name}): n={r['n']}  median r with TRAIN neighbour {r['median_adjacent']:.3f}   "
              f"with a random TRAIN frame of the sequence {r['median_random']:.3f}")

    header("4. Source shortcut (PV specialist)")
    src = check_source_shortcut(D.neg, D.src, D.fu, pv_thr, order=("om", "thm"))
    cis = source_auroc_cis(D.neg, D.src, D.fu, seed=SEED)
    for s, r in src["rows"].items():
        print(f"healthy {s:3s}: n={r['n']:4d}  median err {r['median']:.4f}  flagged at thr {r['flagged']}")
    print(f"false positives at tau: THM {src['fp_thm']}  O&M {src['fp_om']}  "
          f"THM share {src['fp_thm'] / (src['fp_thm'] + src['fp_om']):.1%}")
    print(f"unique faults: median err {np.median(D.fu):.4f}  flagged at thr {int((D.fu > pv_thr).sum())} / {len(D.fu)}")
    print(f"AUROC healthy O&M vs healthy THM : {src['auroc_om_thm']:.3f} {ci(cis['om_thm'])}")
    print(f"AUROC healthy O&M vs unique fault: {src['auroc_om_fault']:.3f} {ci(cis['om_fault'])}")
    print(f"AUROC healthy THM vs unique fault: {src['auroc_thm_fault']:.3f} {ci(cis['thm_fault'])}")

    header("5. Headline metrics (Tables III and IV)")
    print(f"AUPRC = {AUPRC_DEF}.\n95% CIs (AUROC and AUPRC): 2,000 bootstrap draws, healthy and fault resampled "
          f"separately, seed {SEED}")
    gn, gf = saved_errors(v2)
    for name, neg, pos, thr, groups in [(f"PV, {len(D.pos)} files", D.neg, D.pos, pv_thr, D.gid),
                                        (f"PV, {len(D.fu)} unique", D.neg, D.fu, pv_thr, None),
                                        ("v2 general (not in the paper, HER-105)", gn, gf, v2_thr, None)]:
        h = check_headline(neg, pos, thr, seed=SEED)
        print(f"\n{name}: n healthy {len(neg)}  n fault {len(pos)}")
        print(f"  AUROC {h['auroc']:.4f} {ci(h['ci'])}   AUPRC {h['auprc']:.3f} {ci(h['auprc_ci'])} "
              f"(fault prevalence {h['prevalence']:.3f})"
              f"   TPR at FPR <= 5% {h['tpr_at_fpr5']:.3f}")
        print(f"  mean error healthy / fault {h['mean_neg']:.4f} / {h['mean_pos']:.4f}   "
              f"flag-everything F1 {h['trivial_f1']:.4f}")
        print("  at saved threshold :", rounded(h["saved"]))
        print(f"  re-tuned best-F1 (tau {h['retuned_threshold']:.6f}):", rounded(h["retuned"]))
        if name.startswith("PV"):
            cv = check_cv_threshold(neg, pos, k=5, repeats=200, seed=SEED, pos_groups=groups)
            by = "folds keep each unique fault's copies together" if groups is not None else "folds by image"
            print(f"  cross-validated tau ({cv['k']}-fold x {cv['repeats']}, {by}): pooled F1 {cv['f1_mean']:.4f} "
                  f"[{cv['f1_lo']:.4f}, {cv['f1_hi']:.4f}]   mean per-fold F1 {cv['f1_mean_fold']:.4f}   "
                  f"change vs F1 at tau {cv['f1_mean'] - h['saved']['f1']:+.4f}")

    header("6. Trivial baselines (Table VI) and the HER-107 same-transform table")
    pvnF, pvfF_u = D.features()
    j = audit.FEAT.index("p1")
    print(f"PV p1 median healthy / unique fault (0-255): {np.median(pvnF[:, j]):.1f} / {np.median(pvfF_u[:, j]):.1f}")
    base = check_trivial_baselines(pvnF, pvfF_u, seed=SEED)
    print(f"PV ({len(D.neg)} healthy vs {len(D.fu)} unique): autoencoder AUROC {auroc(*from_groups(D.neg, D.fu)):.3f}")
    print("PV single-statistic AUROC:", {k: round(v, 3) for k, v in base["single"].items()})
    print(f"PV best single statistics {[(k, round(v, 3)) for k, v in base['top']]}")
    print(f"PV 5-fold logistic regression on 16 statistics AUROC {base['logreg']:.3f}")
    v2ds = Dataset("v2")
    v2n, v2f = v2ds.split("test_normal"), v2ds.split("test_fault")
    b2 = check_trivial_baselines(global_features(v2n.x, v2ds.stats), global_features(v2f.x, v2ds.stats), seed=SEED)
    print(f"v2 general (not in the paper): best single {[(k, round(v, 3)) for k, v in b2['top']]}  "
          f"logreg {b2['logreg']:.3f}")

    healthy_paths, fault_paths = pv_frame_paths(pvds, dup["gid"])
    print("\nHER-107: no-model baselines under each transform, applied the same way to both classes "
          f"(raw files of the {len(healthy_paths)} held-out frames and the {len(fault_paths)} unique faults)")
    rows = check_transform_baselines(healthy_paths, fault_paths, seed=SEED)
    cols = audit.TRANSFORM_COLS
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")

    config = RuleConfig.load()
    header(f"7. Rule layer (Table VII; {Path(config.source).relative_to(ROOT).as_posix()} defaults)")
    first = unique_first(D.gid)
    healthy_imgs = denormalize(D.holdout.x, pvds.stats)
    fault_imgs = denormalize(D.fault.x[first], pvds.stats)
    sig_h, sig_f = rule_signals(config, healthy_imgs, "pv"), rule_signals(config, fault_imgs, "pv")
    print(f"dT proxy alone, healthy vs unique fault: AUROC {auroc(*from_groups(sig_h, sig_f)):.3f}")
    for name, imgs, e in [("PV healthy", healthy_imgs, D.neg), ("PV unique faults", fault_imgs, D.fu)]:
        ae = e > pv_thr
        tiers, final, _ = check_rule_layer(config, imgs, "pv", ae)
        n = len(e)
        moved = final.get("MONITOR", 0)
        print(f"{name} (n={n}): autoencoder alone flagged {int(ae.sum())}, passed {int((~ae).sum())}")
        print(f"  rule tiers {tiers}  ({', '.join(f'{k} {v / n:.1%}' for k, v in tiers.items())})")
        print(f"  ensemble {final}  (Normal to Monitor: {moved} = {moved / n:.1%})")
    ceiling = float(config.ambient_c) + float(config.equipment("transformer")["ceiling_over_ambient_c"])
    for lab, split, e in [("healthy", v2n, gn), ("fault", v2f, gf)]:
        keep = v2ds.frames(split, source="transformer")
        _, _, flagged = check_rule_layer(config, denormalize(split.x[keep], v2ds.stats), "transformer", e[keep] > v2_thr)
        print(f"transformer {lab} (not in the paper): n={int(keep.sum())} AE flags {(e[keep] > v2_thr).sum()}  "
              f"top-oil rule (>{ceiling:.0f} C) flags {flagged}")

    header("8. Figures")
    FIGS.mkdir(parents=True, exist_ok=True)
    is_thm = D.src == "thm"
    print(fig_error_by_source(D.neg[~is_thm], D.neg[is_thm], D.fu, pv_thr, FIGS / "fig_pv_error_by_source.png")
          .relative_to(ROOT).as_posix())
    lr_s, lr_y = logreg_cv_scores(pvnF, pvfF_u, seed=SEED)
    p1_s, p1_y = from_groups(pvnF[:, j], pvfF_u[:, j])
    if auroc(p1_s, p1_y) < 0.5:                      # oriented as the baselines are (max(A, 1 - A))
        p1_s = -p1_s
    curves = [("Autoencoder error", *from_groups(D.neg, D.fu)), ("1st-pct. intensity (no model)", p1_s, p1_y),
              ("16 global stats, LR (5-fold CV)", lr_s, lr_y)]
    print(fig_roc_vs_baselines(curves, FIGS / "fig_roc_vs_trivial_baselines.png",
                               title=f"PV specialist: {len(D.neg)} healthy vs {len(D.fu)} unique faults")
          .relative_to(ROOT).as_posix())
    pr = [(f"Original, {len(D.pos):,} fault files", *from_groups(D.neg, D.pos)),
          (f"Deduplicated, {len(D.fu)} distinct faults", *from_groups(D.neg, D.fu))]
    print(fig_pr_curves(pr, FIGS / "fig_pr_original_vs_dedup.png",
                        title=f"PV specialist, autoencoder error ({len(D.neg)} healthy)")
          .relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
