"""
Evaluate the India autoencoder (milestone 06: HER-97, HER-98).

  - anomaly score (primary, as planned) = mean squared reconstruction error per image
  - secondary scores (added Oct 25 before seeing model results): p99 and p99.9
    of the per-pixel error map, because faults are small (median 1 box per frame)
    and a whole-image mean can drown them; thresholds for every score come from val
  - threshold chosen on VAL (plot 16) at best F1, then frozen (HER-97)
  - reported once on TEST (plot 18) and on CROSS_SITE (certified, other sites)
  - next to every result (HER-98): flag-everything F1, specificity, and
    no-model baselines whose direction is also fixed on val:
      image stats on the model input (mean, std, p1, p99)
      temperature stats (p99, max, hotspot delta = p99 - median)  [no model at all]
  - test AUROC with a 95% bootstrap interval

Outputs: results/india/eval_log.txt (stdout), results/india/{split}_errors.npy
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_autoencoder import ThermalAutoencoder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "CA_Preprocessed" / "india"
RES = ROOT / "results" / "india"
WEIGHTS = ROOT / "models" / "autoencoder_india_best.pth"


def auroc(score, y):
    """Mann-Whitney AUROC, ties counted half. y: 1 = faulty."""
    score, y = np.asarray(score, float), np.asarray(y, int)
    order = score.argsort(); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):                       # average ranks for ties
        m = score == v
        if m.sum() > 1: ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), len(y) - y.sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(score, y, n=1000, seed=0):
    rng, out = np.random.default_rng(seed), []
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        out.append(auroc(score[idx], y[idx]))
    return np.percentile(out, [2.5, 97.5])


def at_threshold(score, y, thr):
    p = score > thr
    tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum())
    fn = int((~p & (y == 1)).sum()); tn = int((~p & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, specificity=spec)


def best_f1_threshold(score, y):
    cands = np.unique(score)
    if len(cands) > 2000:
        cands = np.quantile(score, np.linspace(0, 1, 2001))
    return max(cands, key=lambda t: at_threshold(score, y, t)["f1"])


def load(split):
    x = np.load(DATA / f"{split}.npy")
    meta = list(csv.DictReader(open(DATA / f"{split}_meta.csv", encoding="utf-8")))
    y = np.array([1 if m["label"] == "faulty" else 0 for m in meta])
    return x, meta, y


def errors(model, x, dev):
    """Return dict of per-image scores from the per-pixel squared error map."""
    out = {"mean": [], "p99": [], "p99.9": []}
    with torch.no_grad():
        for i in range(0, len(x), 64):
            b = torch.from_numpy(x[i:i + 64]).unsqueeze(1).to(dev)
            e = ((model(b) - b) ** 2).flatten(1)
            out["mean"].append(e.mean(1).cpu().numpy())
            q = torch.quantile(e.float(), torch.tensor([0.99, 0.999], device=dev), dim=1)
            out["p99"].append(q[0].cpu().numpy()); out["p99.9"].append(q[1].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def baselines(x, meta):
    flat = x.reshape(len(x), -1)
    f = lambda k: np.array([float(m[k]) for m in meta])
    return {
        "img mean": flat.mean(1), "img std": flat.std(1),
        "img p1": np.percentile(flat, 1, axis=1), "img p99": np.percentile(flat, 99, axis=1),
        "temp p99 (C)": f("t_p99"), "temp max (C)": f("t_max"),
        "hotspot delta p99-median (C)": f("t_p99") - f("t_median"),
    }


def main():
    RES.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalAutoencoder().to(dev)
    model.load_state_dict(torch.load(WEIGHTS, map_location=dev)); model.eval()

    data = {s: load(s) for s in ("val", "test", "cross_site")}
    err = {}
    for s, (x, meta, y) in data.items():
        err[s] = errors(model, x, dev)
        np.save(RES / f"{s}_errors.npy", err[s]["mean"])
        np.save(RES / f"{s}_errors_p999.npy", err[s]["p99.9"])

    yv = data["val"][2]
    for score in ("mean", "p99", "p99.9"):
        tag = "PRIMARY" if score == "mean" else "secondary"
        thr = best_f1_threshold(err["val"][score], yv)
        print(f"######## score = {score} pixel error ({tag}); threshold from VAL: {thr:.6f}\n")
        for s in ("val", "test", "cross_site"):
            y = data[s][2]; e = err[s][score]
            m = at_threshold(e, y, thr)
            flag_all = at_threshold(np.ones(len(y)), y, 0)["f1"]
            ci = boot_ci(e, y) if s == "test" else None
            print(f"== {s.upper()}  ({len(y)-y.sum()} healthy / {y.sum()} faulty)")
            print(f"  AUROC {auroc(e, y):.3f}" + (f"  (95% CI {ci[0]:.3f} to {ci[1]:.3f})" if ci is not None else ""))
            print(f"  at frozen threshold: precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
                  f"F1 {m['f1']:.3f}  specificity {m['specificity']:.3f}")
            print(f"  TP {m['tp']}  FP {m['fp']}  FN {m['fn']}  TN {m['tn']}")
            print(f"  flag-everything F1: {flag_all:.3f}\n")

    yt = data["test"][2]
    print("== NO-MODEL BASELINES (direction fixed on val, AUROC on test)")
    bv = baselines(data["val"][0], data["val"][1]); bt = baselines(data["test"][0], data["test"][1])
    for k in bv:
        if np.isnan(bv[k]).any():
            continue
        sign = 1 if auroc(bv[k], yv) >= 0.5 else -1
        print(f"  {k:32s} val {auroc(sign*bv[k], yv):.3f}   test {auroc(sign*bt[k], yt):.3f}")
    for score in ("mean", "p99", "p99.9"):
        print(f"  {'AUTOENCODER ' + score + ' error':32s} val {auroc(err['val'][score], yv):.3f}   test {auroc(err['test'][score], yt):.3f}")


if __name__ == "__main__":
    main()
