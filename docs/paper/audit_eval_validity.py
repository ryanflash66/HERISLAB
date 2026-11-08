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
  5. Headline metrics (Tables III and IV): threshold-free AUROC with bootstrap CI, AUPRC and TPR at
     FPR <= 5% on every image; precision, recall and F1 on the test half at the threshold chosen on
     the validation half (HER-97), with the test-tuned threshold and the cross-validated one for contrast
  5b. Repeated runs and intervals (HER-120): bootstrap intervals over the 892 healthy frames and the
     140 unique faults (threshold statistics over the test half), and the per-seed table of
     src/experiments/seed_runs_pv.py, each seed with its own validation threshold

Threshold protocol (HER-97): every alarm threshold is the best-F1 point of the validation half of
docs/data_manifests/{pv,v2}_calibration_split.csv, frozen by the evaluator, and every count that
depends on it (false positives by source, precision, recall, F1, rule-layer ensemble) is taken
over the test half only.
  6. Trivial baselines (Table VI): single global statistics and the 16-statistic logistic regression,
     and the HER-107 same-transform table (the trivial baselines under each hlab.transform variant)
  7. Rule layer (Table VII): dT-proxy AUROC, tier distribution and ensemble verdicts via hlab.rules
  8. Figures: docs/paper/figs/fig_pv_error_by_source.png (Fig. 2), fig_roc_vs_trivial_baselines.png (Fig. 3)

Usage:
    venv\\Scripts\\python.exe docs\\paper\\audit_eval_validity.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit  # noqa: E402
from audit import (PVInputs, SEED, check_cv_threshold, check_frozen_threshold, check_seed_runs,  # noqa: E402
                   check_unique_intervals, check_frame_correlation, check_headline,
                   check_pvmd_duplicates, check_rule_layer, check_source_shortcut, check_temporal_leakage,
                   check_test_hashes, check_transform_baselines, image_formats, check_trivial_baselines, fig_error_by_source,
                   fig_roc_vs_baselines, global_features, logreg_cv_scores, pv_frame_paths, read_training_log,
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
    # The evaluators' thresholds (results/*_eval_metrics.npy), read through the bundles: each is the
    # best-F1 point of its validation half, frozen (HER-97); check_frozen_threshold re-derives it.
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
    print(f"alarm threshold tau, best F1 on the validation half, frozen: {pv_thr:.6f} ({pv_thr:.2e})")
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

    D = PVInputs(pvds, pv, gid=dup["gid"])
    hv, ht = D.neg_role == "val", D.neg_role == "test"
    fv, ft = D.fu_role == "val", D.fu_role == "test"
    print("validation / test halves (docs/data_manifests/pv_calibration_split.csv): "
          f"healthy {hv.sum()} / {ht.sum()} (THM {(hv & (D.src == 'thm')).sum()} / {(ht & (D.src == 'thm')).sum()})   "
          f"unique faults {fv.sum()} / {ft.sum()}   fault files {(D.pos_role == 'val').sum()} / {(D.pos_role == 'test').sum()}")

    header("3. Temporal leakage in the PV normal split")
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
    src_t = check_source_shortcut(D.neg[ht], D.src[ht], D.fu[ft], pv_thr, order=("om", "thm"))
    cis = source_auroc_cis(D.neg, D.src, D.fu, seed=SEED)
    for s, r in src["rows"].items():
        print(f"healthy {s:3s}: n={r['n']:4d}  median err {r['median']:.4f}   test half: "
              f"n={src_t['rows'][s]['n']:4d}  flagged at tau {src_t['rows'][s]['flagged']}")
    n_fp = src_t["fp_thm"] + src_t["fp_om"]
    print(f"false positives at tau (test half): THM {src_t['fp_thm']}  O&M {src_t['fp_om']}  "
          f"THM share {src_t['fp_thm'] / n_fp if n_fp else 0:.1%}")
    print(f"unique faults: median err {np.median(D.fu):.4f}   test half: flagged at tau "
          f"{int((D.fu[ft] > pv_thr).sum())} / {int(ft.sum())}")
    print(f"AUROC healthy O&M vs healthy THM : {src['auroc_om_thm']:.3f} {ci(cis['om_thm'])}")
    print(f"AUROC healthy O&M vs unique fault: {src['auroc_om_fault']:.3f} {ci(cis['om_fault'])}")
    print(f"AUROC healthy THM vs unique fault: {src['auroc_thm_fault']:.3f} {ci(cis['thm_fault'])}")

    header("5. Headline metrics (Tables III and IV)")
    gn, gf = saved_errors(v2)
    v2ds = Dataset("v2")
    v2_neg_role, v2_pos_role = v2ds.calibration("test_normal"), v2ds.calibration("test_fault")
    print("tau is the best-F1 point of the validation half, frozen; the confusion is the test half's, once.")
    print(f"The PV tau was chosen with one score per unique fault (the {len(D.fu)} unique row is the headline).")
    for name, neg, pos, nr, pr, thr, groups in [
            (f"PV, {len(D.fu)} unique", D.neg, D.fu, D.neg_role, D.fu_role, pv_thr, None),
            (f"PV, {len(D.pos)} files", D.neg, D.pos, D.neg_role, D.pos_role, pv_thr, D.gid),
            ("v2 general (not in the paper, HER-105)", gn, gf, v2_neg_role, v2_pos_role, v2_thr, None)]:
        h = check_headline(neg, pos, thr, seed=SEED)
        print(f"\n{name}: n healthy {len(neg)}  n fault {len(pos)}")
        print(f"  every image, threshold-free: AUROC {h['auroc']:.4f} {ci(h['ci'])}   AUPRC {h['auprc']:.3f} "
              f"(fault prevalence {h['prevalence']:.3f})   TPR at FPR <= 5% {h['tpr_at_fpr5']:.3f}")
        print(f"  mean error healthy / fault {h['mean_neg']:.4f} / {h['mean_pos']:.4f}")
        fz = check_frozen_threshold(neg, pos, nr, pr, thr) if pos is not D.pos else None
        if fz is None:    # the PV tau came from the unique-fault validation half; here it is only applied
            t_neg, t_pos = neg[nr == "test"], pos[pr == "test"]
            s_t, y_t = from_groups(t_neg, t_pos)
            fz = dict(n_test=(len(t_neg), len(t_pos)), test=audit.confusion(s_t, y_t, thr), test_auroc=auroc(s_t, y_t),
                      test_trivial_f1=audit.trivial_f1(y_t))
        else:
            print(f"  validation half (healthy {fz['n_val'][0]}, fault {fz['n_val'][1]}) at tau {thr:.6f}:",
                  rounded(fz["val"]))
        print(f"  TEST HALF (healthy {fz['n_test'][0]}, fault {fz['n_test'][1]}) at tau {thr:.6f}:", rounded(fz["test"]))
        print(f"    test-half AUROC {fz['test_auroc']:.4f}   test-half flag-everything F1 {fz['test_trivial_f1']:.4f}")
        if "test_retuned" in fz:
            print(f"  for contrast, tau re-tuned on the test half ({fz['test_retuned_threshold']:.6f}, optimistic):",
                  rounded(fz["test_retuned"]))
            print(f"  for contrast, tau re-tuned on every image ({h['retuned_threshold']:.6f}, the pre-HER-97 protocol):",
                  rounded(h["retuned"]))
        if name.startswith("PV"):
            cv = check_cv_threshold(neg, pos, k=5, repeats=200, seed=SEED, pos_groups=groups)
            by = "folds keep each unique fault's copies together" if groups is not None else "folds by image"
            print(f"  cross-validated tau over every image ({cv['k']}-fold x {cv['repeats']}, {by}): pooled F1 "
                  f"{cv['f1_mean']:.4f} [{cv['f1_lo']:.4f}, {cv['f1_hi']:.4f}]   mean per-fold F1 {cv['f1_mean_fold']:.4f}")

    header("5b. Repeated runs and intervals on the deduplicated set (HER-120)")
    print(f"95% stratified bootstrap, 2,000 resamples, seed {SEED}: {len(D.neg)} healthy frames and "
          f"{len(D.fu)} unique fault images (one mean error each) resampled, not the {len(D.pos)} files")
    iv = check_unique_intervals(D.neg, D.fu, seed=SEED)
    for k, (p, lo, hi) in iv.items():
        print(f"  {k:12s} {p:.4f} [{lo:.4f}, {hi:.4f}]   (every image)")
    iv_t = check_unique_intervals(D.neg[ht], D.fu[ft], thr=pv_thr, seed=SEED)
    print(f"test half only ({int(ht.sum())} healthy, {int(ft.sum())} unique faults), tau {pv_thr:.6f} from the "
          f"validation half, held fixed across resamples:")
    for k, (p, lo, hi) in iv_t.items():
        print(f"  {k:12s} {p:.4f} [{lo:.4f}, {hi:.4f}]")
    seed_runs = sorted((RESULTS / "her-120").glob("pv_seed*_normal_errors.npy"),
                       key=lambda q: int(q.name[len("pv_seed"):].split("_")[0]))
    runs = []
    for q in seed_runs:
        s = q.name[len("pv_seed"):].split("_")[0]
        neg_s = np.load(q)
        pos_s = np.load(q.with_name(f"pv_seed{s}_fault_errors.npy"))
        runs.append((f"seed {s}", neg_s, pos_s, read_training_log(RESULTS / "her-120" / f"training_log_pv_seed{s}.txt")))
    roles = (D.neg_role, D.fu_role)
    recorded = check_seed_runs([("recorded (paper)", D.neg, D.pos, log)], D.gid, roles=roles)["rows"][0]
    print(f"\nrepeated training runs: config/training.json 'pv' recipe (lr 1e-3, no clipping, val holdout seed 43), "
          f"torch seed varied; every run takes its own tau on the validation half and is scored on the same test half "
          f"({int(ht.sum())} healthy frames, {int(ft.sum())} unique faults)")
    print(f"spike = an epoch whose val loss is at least {audit.SPIKE_RATIO:g}x the lowest val loss before it")
    if runs:
        sr = check_seed_runs(runs, D.gid, roles=roles)
        print("| run | best epoch | best val loss | epochs run | largest val-loss jump | spike | AUROC | AUPRC | "
              "TPR at FPR <= 5% | tau (val) | precision | recall | F1 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in [recorded] + sr["rows"]:
            jump = f"{r['jump']:.1f}x at epoch {r['jump_epoch']}" if r["jump_epoch"] else "-"
            print(f"| {r['label']} | {r['best_epoch']} | {r['best_val_loss']:.4f} | {r['epochs_run']} | {jump} | "
                  f"{'yes' if r['spike'] else 'no'} | {r['auroc']:.4f} | {r['auprc']:.4f} | {r['tpr_at_fpr5']:.4f} | "
                  f"{r['threshold']:.6f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")
        print(f"\n{sr['n']} seeded runs, {sr['n_spike']} with a spike. mean, SD, range over the seeded runs (test half):")
        for k, (m, sd, lo, hi) in sr["summary"].items():
            fmt = "{:.1f}" if k == "best_epoch" else "{:.6f}" if k == "threshold" else "{:.4f}"
            print(f"  {k:13s} mean {fmt.format(m)}  SD {fmt.format(sd)}  range {fmt.format(lo)} to {fmt.format(hi)}")
    else:
        print("no seed runs under results/her-120 (src/experiments/seed_runs_pv.py): single-run numbers only")
    print(f"\nsingle-run vs multi-run: every number in sections 0 to 8 except the seed table above comes from the "
          f"single recorded training run; the seed table is the only multi-run result ({len(runs)} seeds).")

    header("6. Trivial baselines (Table VI) and the HER-107 same-transform table")
    pvnF, pvfF_u = D.features()
    j = audit.FEAT.index("p1")
    print(f"PV p1 median healthy / unique fault (0-255): {np.median(pvnF[:, j]):.1f} / {np.median(pvfF_u[:, j]):.1f}")
    base = check_trivial_baselines(pvnF, pvfF_u, seed=SEED)
    print(f"PV ({len(D.neg)} healthy vs {len(D.fu)} unique): autoencoder AUROC {auroc(*from_groups(D.neg, D.fu)):.3f}")
    print("PV single-statistic AUROC:", {k: round(v, 3) for k, v in base["single"].items()})
    print(f"PV best single statistics {[(k, round(v, 3)) for k, v in base['top']]}")
    print(f"PV 5-fold logistic regression on 16 statistics AUROC {base['logreg']:.3f}")
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
    print("test half only (the autoencoder flag uses tau from the validation half)")
    for name, imgs, e in [("PV healthy", healthy_imgs[ht], D.neg[ht]), ("PV unique faults", fault_imgs[ft], D.fu[ft])]:
        ae = e > pv_thr
        tiers, final, _ = check_rule_layer(config, imgs, "pv", ae)
        n = len(e)
        moved = final.get("MONITOR", 0)
        print(f"{name} (n={n}): autoencoder alone flagged {int(ae.sum())}, passed {int((~ae).sum())}")
        print(f"  rule tiers {tiers}  ({', '.join(f'{k} {v / n:.1%}' for k, v in tiers.items())})")
        print(f"  ensemble {final}  (Normal to Monitor: {moved} = {moved / n:.1%})")
    ceiling = float(config.ambient_c) + float(config.equipment("transformer")["ceiling_over_ambient_c"])
    for lab, split, e, role in [("healthy", v2n, gn, v2_neg_role), ("fault", v2f, gf, v2_pos_role)]:
        keep = v2ds.frames(split, source="transformer") & (role == "test")
        _, _, flagged = check_rule_layer(config, denormalize(split.x[keep], v2ds.stats), "transformer", e[keep] > v2_thr)
        print(f"transformer {lab}, test half (not in the paper): n={int(keep.sum())} AE flags {(e[keep] > v2_thr).sum()}  "
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


if __name__ == "__main__":
    main()
