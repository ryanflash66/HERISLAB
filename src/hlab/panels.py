"""
Panel primitives for the India PV frames (HER-129): one definition each of what
the India experiment scripts used to copy.

  resolve_device(name)                     torch.device; None = CUDA when available
  boxes / box_mask / largest_box / scale_point
                                           labelme fault boxes (640x512 R-JPG coordinates) on the 240x320 frame
  panel_mask(b, version)                   the no-model roughness mask (v2: median 5, window 9, closing 21)
  blank_ground(x, mask)                    ground pixels set to the frame's panel median
  masked_peak / peak_ratio / peak_standout the k x k peak statistics over the whole frame or the panel pixels
  peak_scores(model, x, device)            the peak score family of score_peak_india (peak3, peak5, peakmax, input_peak3)
  smoothed / local_mean / local_excess / hottest_spot
                                           "degrees above the local panel mean" and where the hottest spot sits
  feats / kmeans2 / panel_of / patch_anomaly
                                           DINOv2 ViT-S/14 patch features and the two-cluster panel/ground split
  dino_panel_fit(train_frames, stats)      the fit as panel_dino_india declared it; returns a PanelModel
  PanelModel.masks(frames, size)           panel masks for any frames, at patch grid or pixel size
  block_mask / hidden_block_predict        the hidden-patch (inpainting) prediction stitched from 4 patterns
  error_maps(model, x, device)             3x3-averaged squared reconstruction error maps

Conventions: images are (N, 1, H, W) float tensors in model units unless the
docstring says numpy; masks are bool with True = panel. Every function that
touches the GPU takes `device`; nothing here reads a file or a module global of
another script. The numbers are the ones the experiments produced: the peak
statistics fall back to the whole frame when a mask keeps 50 pixels or fewer,
the box filter of local_mean is replicate-padded, the k x k smoothing of
masked_peak and error_maps is zero-padded (avg_pool2d), as the scripts had it.
"""
import json
import math

import numpy as np
import torch
import torch.nn.functional as Fn

from hlab.transform import TARGET_SIZE

H, W = TARGET_SIZE[1], TARGET_SIZE[0]
SRC_SHAPE = (512, 640)          # (rows, cols) of the R-JPG the labelme boxes were drawn on


def resolve_device(name=None):
    """torch.device from a name or a device; None picks CUDA when available."""
    if name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return name if isinstance(name, torch.device) else torch.device(name)


# ---------------------------------------------------------------- labelme boxes
def boxes(json_path):
    """Rectangles of a labelme file as (x0, y0, x1, y1) in source pixels, corners sorted."""
    out = []
    for s in json.load(open(json_path, encoding="utf-8"))["shapes"]:
        (xa, ya), (xb, yb) = s["points"]
        x0, x1 = sorted((xa, xb))
        y0, y1 = sorted((ya, yb))
        out.append((x0, y0, x1, y1))
    return out


def box_mask(json_path, pad=0, shape=(H, W), src_shape=SRC_SHAPE):
    """Bool (rows, cols) mask of the boxes scaled from src_shape to shape, padded by `pad` pixels."""
    rows, cols = shape
    sr, sc = src_shape
    m = np.zeros((rows, cols), bool)
    for x0, y0, x1, y1 in boxes(json_path):
        c0 = max(0, math.floor(x0 * cols / sc) - pad); c1 = min(cols, math.ceil((x1 + 1) * cols / sc) + pad)
        r0 = max(0, math.floor(y0 * rows / sr) - pad); r1 = min(rows, math.ceil((y1 + 1) * rows / sr) + pad)
        m[r0:r1, c0:c1] = True
    return m


def largest_box(json_path):
    """The box with the largest area, (x0, y0, x1, y1) in source pixels."""
    return max(boxes(json_path), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def scale_point(r, c, src_shape=SRC_SHAPE, shape=(H, W)):
    """(row, col) of a source-frame pixel on the model frame."""
    sr, sc = src_shape
    rows, cols = shape
    return min(rows - 1, r * rows // sr), min(cols - 1, c * cols // sc)


# ---------------------------------------------------------------- roughness panel mask (no model)
MASK_SETTINGS = {1: dict(med=0, win=7, close=11, shave=3), 2: dict(med=5, win=9, close=21, shave=0)}
MASK_VERSION = 2


def _otsu(v, bins=128):
    h, e = np.histogram(v, bins=bins); c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = ((h * c).sum() - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def _median_filter(b, k):
    p = k // 2; out = []
    for i in range(0, len(b), 8):
        u = Fn.unfold(Fn.pad(b[i:i + 8], (p, p, p, p), mode="reflect"), k)
        out.append(u.median(1).values.view(-1, 1, *b.shape[2:]))
    return torch.cat(out)


def panel_mask(b, version=None):
    """Panels are smooth, gravel and plants are rough: a per-frame Otsu split of the log local roughness.

    b: (N, 1, H, W) float tensor (model input). Returns bool (N, 1, H, W), True = panel.
    Version 2 (the one in use): 5x5 median first (erases hot spots a few pixels wide),
    roughness window 9, 21x21 closing, no edge shave. Version 1: window 7, closing 11, 3 px shave.
    """
    cfg = MASK_SETTINGS[version or MASK_VERSION]
    win, close, shave = cfg["win"], cfg["close"], cfg["shave"]
    src = _median_filter(b, cfg["med"]) if cfg["med"] else b
    p = win // 2
    xp = Fn.pad(src, (p, p, p, p), mode="reflect")
    mu = Fn.avg_pool2d(xp, win, 1); var = Fn.avg_pool2d(xp ** 2, win, 1) - mu ** 2
    lr = torch.log(var.clamp_min(1e-8).sqrt())
    thr = torch.tensor([_otsu(f.flatten().cpu().numpy()) for f in lr], device=b.device).view(-1, 1, 1, 1)
    m = (lr < thr).float()
    c = close // 2
    m = Fn.max_pool2d(m, close, 1, c)                     # dilate
    m = -Fn.max_pool2d(-m, close, 1, c)                   # erode  (= closing)
    if shave:
        m = -Fn.max_pool2d(-m, 2 * shave + 1, 1, shave)   # shave edges
    return m > 0.5


# ---------------------------------------------------------------- ground blanking
def blank_ground(x, mask):
    """Copy of x (numpy (N, H, W)) with every non-panel pixel set to that frame's panel median.

    Frames whose mask holds no panel pixel are left as they are.
    """
    out = np.array(x, copy=True)
    for i in range(len(out)):
        if mask[i].any():
            out[i][~mask[i]] = np.median(out[i][mask[i]])
    return out


# ---------------------------------------------------------------- peak statistics
def masked_peak(v, m=None, k=3):
    """Per frame: (max of the k x k average inside m, median inside m, std inside m).

    v: (N, 1, H, W) map; m: bool (N, 1, H, W) or None for the whole frame. The median and
    std fall back to the whole frame when the mask keeps 50 pixels or fewer, and the max to
    the whole frame when the mask is empty.
    """
    v3 = Fn.avg_pool2d(v, k, 1, k // 2)
    if m is None:
        flat = v.flatten(1)
        return v3.flatten(1).max(1).values, flat.median(1).values, flat.std(1)
    mx, md, sd = [], [], []
    for vi, v3i, mi in zip(v, v3, m):
        s = vi[mi] if mi.sum() > 50 else vi.flatten()
        mx.append(v3i[mi].max() if mi.any() else v3i.max()); md.append(s.median()); sd.append(s.std())
    return torch.stack(mx), torch.stack(md), torch.stack(sd)


def peak_ratio(v, m=None, k=3, eps=1e-12):
    """max k x k average / median, inside m: the autoencoder peak score (ae_peak3)."""
    mx, md, _ = masked_peak(v, m, k)
    return mx / md.clamp_min(eps)


def peak_standout(v, m=None, k=3, eps=1e-6):
    """(max k x k average - median) / std, inside m: the no-model input peak (input_peak3)."""
    mx, md, sd = masked_peak(v, m, k)
    return (mx - md) / sd.clamp_min(eps)


@torch.no_grad()
def peak_scores(model, x, device, batch=64):
    """The peak score family of score_peak_india on z-scored frames x (numpy (N, H, W)).

    peak3 (primary): max 3x3-averaged error / median error; peak5: 5x5; peakmax: max / mean;
    "input_peak3 (no model)": the same 3x3 peak on the input, in units of its spread.
    """
    out = {"peak3": [], "peak5": [], "peakmax": [], "input_peak3 (no model)": []}
    for i in range(0, len(x), batch):
        b = torch.from_numpy(np.ascontiguousarray(x[i:i + batch])).unsqueeze(1).to(device)
        e = (model(b) - b) ** 2
        flat = e.flatten(1)
        out["peak3"].append(peak_ratio(e, None, 3).cpu().numpy())
        out["peak5"].append(peak_ratio(e, None, 5).cpu().numpy())
        out["peakmax"].append((flat.max(1).values / flat.mean(1)).cpu().numpy())
        out["input_peak3 (no model)"].append(peak_standout(b, None, 3).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------- degrees above the local panel mean
MODULE_WIN, MIN_PANEL_SHARE = 31, 0.30   # reltemp_india, declared Oct 25: one module ~ 31 px, >= 30% of the window panel


def smoothed(t, k=3):
    """Replicate-padded k x k average of t (N, 1, H, W); k = 0 returns t."""
    if not k:
        return t
    p = k // 2
    return Fn.avg_pool2d(Fn.pad(t, (p, p, p, p), mode="replicate"), k, 1)


def local_mean(t, m, win, eps=1e-6, count=False):
    """(mean of t over the panel pixels of the win x win window, panel share of that window).

    t: (N, 1, H, W); m: (N, 1, H, W) float or bool. Replicate-padded box filter; a window
    without panel pixels gets mean 0. With count=True the second value is the panel pixel
    COUNT of the window and `eps` clamps that count (the image_score_v2 form: eps=1.0).
    """
    m = m.float()
    k = torch.ones(1, 1, win, win, device=t.device)
    if not count:
        k = k / win ** 2
    p = win // 2
    num = Fn.conv2d(Fn.pad(t * m, (p, p, p, p), mode="replicate"), k)
    den = Fn.conv2d(Fn.pad(m, (p, p, p, p), mode="replicate"), k)
    return num / den.clamp_min(eps), den


def local_excess(t, m, win, smooth=3):
    """(smoothed t - local panel mean, panel share of the window): how far a spot stands above
    the panel around it. smooth = k for a replicate-padded k x k average of t first, 0 for none.
    """
    ref, cov = local_mean(t, m, win)
    return smoothed(t, smooth) - ref, cov


def hottest_spot(t, m, win, min_panel):
    """(row, col, excess, distance to the nearest non-panel pixel) of the largest local excess.

    t, m: (1, 1, H, W). Pixels count only where they are panel and at least `min_panel` of
    their window is panel; distance is 999 when the frame has no non-panel pixel.
    """
    excess, cov = local_excess(t, m, win)
    ok = (m > 0) & (cov >= min_panel)
    d = torch.where(ok, excess, torch.full_like(t, -1e9))
    j = int(d.flatten().argmax()); r, c = divmod(j, t.shape[3])
    nm = (m[0, 0] == 0).nonzero().float()
    dist = float(((nm - torch.tensor([r, c], device=t.device).float()) ** 2).sum(1).min().sqrt()) if len(nm) else 999.0
    return r, c, float(d.max()), dist


# ---------------------------------------------------------------- DINOv2 panel / ground split
SIZE, GRID, PS = (476, 630), (34, 45), 14      # DINOv2 ViT-S/14 input, patch grid, patch size
FEAT_DIM = 384
DINO_HUB = ("facebookresearch/dinov2", "dinov2_vits14")


def load_dino(device):
    return torch.hub.load(*DINO_HUB).to(device).eval()


def to01(x, stats, device):
    """z-scored frames -> (N, 1, *SIZE) in 0..1, bilinear."""
    b = torch.from_numpy(np.ascontiguousarray(x)).to(device).float() * float(stats["std"]) + float(stats["mean"])
    return Fn.interpolate((b / 255).clamp(0, 1).unsqueeze(1), size=SIZE, mode="bilinear", align_corners=False)


@torch.no_grad()
def feats(net, x, stats, device, bs=16):
    """(normalised patch features (N, P, 384) fp16, per-patch pixel spread (N, P))."""
    F_, S_ = [], []
    m = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    s = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    for i in range(0, len(x), bs):
        g = to01(x[i:i + bs], stats, device)
        S_.append(Fn.unfold(g, PS, stride=PS).std(1))
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = net.forward_features((g.repeat(1, 3, 1, 1) - m) / s)["x_norm_patchtokens"]
        F_.append(Fn.normalize(f.float(), dim=-1).half())
    return torch.cat(F_), torch.cat(S_)


def kmeans2(z, iters=50, restarts=5, seed=0):
    """Two centroids of z (M, D) by k-means with `restarts` seeded starts; the lowest inertia wins."""
    g = torch.Generator(device=z.device).manual_seed(seed); best = None
    for _ in range(restarts):
        c = z[torch.randint(len(z), (2,), generator=g, device=z.device)].clone()
        for _ in range(iters):
            a = torch.cdist(z, c).argmin(1)
            c = torch.stack([z[a == k].mean(0) if (a == k).any() else c[k] for k in range(2)])
        inert = torch.cdist(z, c).min(1).values.pow(2).sum()
        if best is None or inert < best[0]: best = (inert, c)
    return best[1]


def panel_of(f, cent, panel_k):
    """f: (N, P, D) -> bool (N, *GRID) panel mask; a patch also counts as panel when >= 5 of its 8 neighbours do."""
    a = torch.cdist(f.float().reshape(-1, f.shape[-1]), cent).argmin(1).view(len(f), *GRID)
    m = (a == panel_k).float().unsqueeze(1)
    nb = Fn.conv2d(Fn.pad(m, (1, 1, 1, 1), mode="replicate"), torch.ones(1, 1, 3, 3, device=m.device)) - m
    return ((m > 0) | (nb >= 5))[:, 0]


@torch.no_grad()
def patch_anomaly(q, bank, chunk=2048):
    """q: (N, P, D) normalised features -> (N, P) = 1 - max cosine similarity to the bank."""
    N, P, D = q.shape; flat = q.reshape(-1, D); out = torch.empty(len(flat), device=q.device)
    for i in range(0, len(flat), chunk):
        out[i:i + chunk] = 1 - (flat[i:i + chunk] @ bank.T).max(1).values.float()
    return out.view(N, P)


class PanelModel:
    """A fitted panel / ground split: the DINOv2 net, the two centroids and which one is panel."""

    def __init__(self, net, stats, cent, panel_k, device, bank=None, info=None):
        self.net, self.stats, self.cent, self.panel_k, self.device = net, dict(stats), cent, panel_k, device
        self.bank = bank            # (B, 384) panel patch features, when the fit kept a bank
        self.info = info or {}

    def feats(self, x, bs=16):
        return feats(self.net, x, self.stats, self.device, bs)

    def panel_of(self, f):
        return panel_of(f, self.cent, self.panel_k)

    def masks(self, x, size=None, batch=128):
        """Bool panel masks of z-scored frames x: (N, *GRID), or (N, *size) nearest-upsampled."""
        out = []
        for i in range(0, len(x), batch):
            f, _ = self.feats(x[i:i + batch])
            m = self.panel_of(f)
            if size is not None:
                m = Fn.interpolate(m.float()[:, None], size=size, mode="nearest")[:, 0] > 0.5
            out.append(m.cpu().numpy())
        return np.concatenate(out)


def dino_panel_fit(train_frames, stats, seed=0, n_bank_frames=1500, n_cluster_frames=300, device=None,
                   net=None, bank_patches=0):
    """The panel / ground fit of panel_dino_india, declared Oct 25: k-means (k = 2) on the DINOv2
    patches of `n_cluster_frames` random healthy frames out of `n_bank_frames` random ones; the
    panel cluster is the one with the lower pixel spread inside its patches. The random draws
    are the script's (frames first, then the cluster subset, then the bank), so a given seed
    gives the masks the experiments were run with.

    bank_patches > 0 also keeps that many random PANEL patches of the bank frames (PatchCore-style
    memory for dino_panel_max). train_frames may be a memory-mapped array.
    """
    device = resolve_device(device)
    rng = np.random.default_rng(seed)
    net = net or load_dino(device)
    pick = np.sort(rng.choice(len(train_frames), n_bank_frames, replace=False))
    cl = rng.choice(n_bank_frames, n_cluster_frames, replace=False)
    if bank_patches:
        ft, st = feats(net, train_frames[pick], stats, device)
        z, sp = ft[cl].reshape(-1, FEAT_DIM).float(), st[cl].reshape(-1)
    else:
        f, s = feats(net, train_frames[pick[cl]], stats, device)
        z, sp = f.reshape(-1, FEAT_DIM).float(), s.reshape(-1)
    cent = kmeans2(z, seed=seed)
    a = torch.cdist(z, cent).argmin(1)
    spread = [float(sp[a == k].mean()) for k in range(2)]
    panel_k = int(np.argmin(spread))
    info = dict(n_patches=len(z), sizes=[(a == k).float().mean().item() for k in range(2)], spread=spread)
    del z, sp, a
    bank = None
    if bank_patches:
        tm = panel_of(ft, cent, panel_k).reshape(-1)
        pool = ft.reshape(-1, FEAT_DIM)[tm.to(device)]
        bank = pool[torch.from_numpy(rng.choice(len(pool), min(bank_patches, len(pool)), replace=False)).to(device)]
        del ft, pool
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return PanelModel(net, stats, cent, panel_k, device, bank=bank, info=info)


# ---------------------------------------------------------------- hidden-block (inpainting) prediction
def block_mask(grid, block=8):
    """grid: (N, GH, GW) bool -> (N, 1, GH * block, GW * block) float, 1 = hidden."""
    n, gh, gw = grid.shape
    return grid.float().view(n, 1, gh, 1, gw, 1).expand(n, 1, gh, block, gw, block).reshape(n, 1, gh * block, gw * block)


@torch.no_grad()
def hidden_block_predict(model, b, block=8):
    """Every block predicted from its surroundings: 4 disjoint (row + col) % 4 patterns, stitched."""
    gh, gw = b.shape[2] // block, b.shape[3] // block
    bi, bj = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    pat = ((bi + bj) % 4).to(b.device)
    out = torch.zeros_like(b)
    for k in range(4):
        hid = block_mask((pat == k)[None].expand(len(b), -1, -1), block)
        out += model(b * (1 - hid)) * hid
    return out


# ---------------------------------------------------------------- error maps
@torch.no_grad()
def error_maps(model, x, device, batch=32, inpaint=False, smooth=3):
    """(N, H, W) squared reconstruction error of z-scored frames x, `smooth` x `smooth` averaged (0 = raw)."""
    out = []
    for i in range(0, len(x), batch):
        b = torch.from_numpy(np.ascontiguousarray(x[i:i + batch])).unsqueeze(1).to(device)
        r = hidden_block_predict(model, b) if inpaint else model(b)
        e = (r - b) ** 2
        if smooth:
            e = Fn.avg_pool2d(e, smooth, 1, smooth // 2)
        out.append(e[:, 0].cpu().numpy())
    return np.concatenate(out)
