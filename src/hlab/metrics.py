"""
Detection metrics on per-image anomaly scores (HER-122).

One convention for every function: ``scores`` is a 1-D float array (higher =
more anomalous) and ``labels`` is a 1-D int array of the same length with
1 = fault / positive and 0 = healthy / negative. A prediction is positive when
``score > threshold``. ``from_groups`` builds that pair from the two-array
form (healthy errors, fault errors) the evaluators keep in separate files.

Numpy only. Nothing here rounds, prints, or keeps a module-level random
state: every bootstrap takes an explicit ``seed``.
"""
import numpy as np


def from_groups(neg, pos):
    """(healthy scores, fault scores) -> (scores, labels), negatives first."""
    neg = np.asarray(neg, dtype=float).ravel()
    pos = np.asarray(pos, dtype=float).ravel()
    scores = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(len(neg), dtype=int), np.ones(len(pos), dtype=int)])
    return scores, labels


def _check(scores, labels):
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    if len(scores) != len(labels):
        raise ValueError(f"scores ({len(scores)}) and labels ({len(labels)}) differ in length")
    if labels.size and not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 (healthy) or 1 (fault)")
    return scores, labels


def auroc(scores, labels):
    """Mann-Whitney AUROC: P(score of a random fault > score of a random healthy).

    Ties get the average rank, so a tied pair counts one half.
    """
    s, y = _check(scores, labels)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auroc needs at least one fault and one healthy score")
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _stratified_indices(labels, rng):
    i1 = np.flatnonzero(labels == 1)
    i0 = np.flatnonzero(labels == 0)
    return np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))])


def auroc_ci(scores, labels, n_boot=2000, seed=0):
    """95% bootstrap interval for ``auroc``: (lo, hi) at the 2.5 / 97.5 percentiles.

    Resampling is stratified (healthy and fault resampled separately, with
    replacement) and driven by ``np.random.default_rng(seed)``.
    """
    s, y = _check(scores, labels)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stratified_indices(y, rng)
        vals[b] = auroc(s[idx], y[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_cis(scores, labels, stats, n_boot=2000, seed=0):
    """95% stratified bootstrap intervals of several statistics on the same resamples.

    ``stats`` maps a name to ``f(scores, labels) -> float``. Returns
    {name: (point, lo, hi)}. The resampling is the one ``auroc_ci`` uses (same
    generator, same draws), so ``stats={"auroc": auroc}`` reproduces ``auroc_ci``
    with the same seed. The resampling unit is one entry of ``scores``: pass one
    score per distinct image to make the interval reflect the distinct images.
    """
    s, y = _check(scores, labels)
    rng = np.random.default_rng(seed)
    vals = {k: np.empty(n_boot) for k in stats}
    for b in range(n_boot):
        idx = _stratified_indices(y, rng)
        for k, f in stats.items():
            vals[k][b] = f(s[idx], y[idx])
    out = {}
    for k, f in stats.items():
        lo, hi = np.percentile(vals[k], [2.5, 97.5])
        out[k] = (float(f(s, y)), float(lo), float(hi))
    return out


def paired_delta_ci(scores_a, scores_b, labels, n_boot=2000, seed=0):
    """AUROC(b) - AUROC(a) with a paired bootstrap interval: (delta, lo, hi).

    Both score vectors are scored on the same images, so each bootstrap draw
    uses one index set for both. ``delta > 0`` means ``scores_b`` separates
    better. A vector against itself gives exactly (0.0, 0.0, 0.0).
    """
    a, y = _check(scores_a, labels)
    b, _ = _check(scores_b, labels)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = _stratified_indices(y, rng)
        vals[i] = auroc(b[idx], y[idx]) - auroc(a[idx], y[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(auroc(b, y) - auroc(a, y)), float(lo), float(hi)


def confusion(scores, labels, threshold):
    """Counts and rates at ``score > threshold``. Rates are 0.0 when undefined."""
    s, y = _check(scores, labels)
    pred = s > threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                f1=f1, specificity=specificity)


def best_f1_threshold(scores, labels):
    """Threshold that maximises F1, searched over every distinct score.

    Above 2000 distinct values the candidates are 2001 evenly spaced quantiles
    of the scores. Among equal F1 the lowest threshold wins.
    """
    s, y = _check(scores, labels)
    cands = np.unique(s)
    if len(cands) > 2000:
        cands = np.quantile(s, np.linspace(0, 1, 2001))
    pos = np.sort(s[y == 1])
    neg = np.sort(s[y == 0])
    tp = len(pos) - np.searchsorted(pos, cands, side="right")
    fp = len(neg) - np.searchsorted(neg, cands, side="right")
    fn = len(pos) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(cands)), where=denom > 0)
    return float(cands[int(np.argmax(f1))])


def _counts_above(scores, labels, cands):
    """(tp, fp) at ``score > t`` for every threshold in ``cands``."""
    pos = np.sort(scores[labels == 1])
    neg = np.sort(scores[labels == 0])
    tp = len(pos) - np.searchsorted(pos, cands, side="right")
    fp = len(neg) - np.searchsorted(neg, cands, side="right")
    return tp, fp


def roc_points(scores, labels):
    """(fpr, tpr) of ``score > t`` over every distinct score plus one threshold below the minimum,
    ordered from (0, 0) to (1, 1)."""
    s, y = _check(scores, labels)
    cands = np.r_[np.unique(s)[::-1], -np.inf]
    tp, fp = _counts_above(s, y, cands)
    return fp / max(int((y == 0).sum()), 1), tp / max(int(y.sum()), 1)


def tpr_at_fpr(scores, labels, max_fpr):
    """Highest true-positive rate over the thresholds whose false-positive rate is at most ``max_fpr``."""
    fpr, tpr = roc_points(scores, labels)
    return float(tpr[fpr <= max_fpr + 1e-12].max())


def average_precision(scores, labels):
    """AUPRC as average precision: sum over distinct thresholds of (recall step) x precision.

    Each distinct score is one threshold (ties enter together), so the value does not depend on
    the order of tied images. Chance level is the fault prevalence.
    """
    s, y = _check(scores, labels)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("average_precision needs at least one fault")
    cands = np.unique(s)[::-1]
    pos = np.sort(s[y == 1])
    neg = np.sort(s[y == 0])
    tp = len(pos) - np.searchsorted(pos, cands, side="left")      # score >= t
    fp = len(neg) - np.searchsorted(neg, cands, side="left")
    recall = tp / n_pos
    precision = tp / (tp + fp)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def trivial_f1(labels):
    """F1 of flagging every image: 2p / (1 + p) with p the fault prevalence."""
    y = np.asarray(labels).ravel().astype(int)
    p = y.mean() if y.size else 0.0
    return float(2 * p / (1 + p)) if p else 0.0
