"""
Model bundles (HER-126): weights, norm stats and the calibrated threshold resolved together.

    bundle = Bundle.load("pv")              # or "v2", "pv_om", "india", "india_temp"
    errors = bundle.score(images_0_255)     # normalize, batch, per-image mean MSE
    errors = bundle.score_normalized(x)     # same, for arrays already in model units (Dataset splits)
    recon_0_255, error_maps = bundle.reconstruct(images_0_255)
    bundle.threshold                        # float from results/<name>_eval_metrics.npy, or None

Resolution (until the layout moves again, this table is the only place that knows it):

  name        weights (models/)                 stats (data/CA_Preprocessed/)   threshold (results/)
  v2          autoencoder_best.pth              norm_stats.npy                  eval_metrics.npy
  pv          autoencoder_pv_best.pth           pv/norm_stats.npy               pv_eval_metrics.npy
  pv_om       autoencoder_pv_om_best.pth        pv_om/norm_stats.npy            pv_om_eval_metrics.npy
  india       autoencoder_india_best.pth        india/norm_stats.npy            india_eval_metrics.npy
  india_temp  autoencoder_india_temp_best.pth   india_temp/norm_stats.npy       india_temp_eval_metrics.npy
  india_panel autoencoder_india_panel_best.pth  india_temp/norm_stats.npy       india_panel_eval_metrics.npy
  india_inpaint inpaint_india_best.pth          india_temp/norm_stats.npy       india_inpaint_eval_metrics.npy

A missing eval-metrics file gives threshold None; callers that need one call
require_threshold(), which says which evaluator to run. No threshold literal
lives anywhere else. Stats and threshold are read at load; the weights load
on first use (torch is imported then), so reading a threshold costs nothing.
torch.load happens in exactly one place: Bundle.model.

FakeBundle is the in-memory adapter for tests: a threshold and a scoring function.
"""
import numpy as np

from hlab.paths import MODELS, PREPROCESSED, RESULTS
from hlab.transform import denormalize, normalize

BUNDLES = {
    "v2": ("autoencoder_best.pth", "", "eval_metrics.npy"),
    "pv": ("autoencoder_pv_best.pth", "pv", "pv_eval_metrics.npy"),
    "pv_om": ("autoencoder_pv_om_best.pth", "pv_om", "pv_om_eval_metrics.npy"),
    "india": ("autoencoder_india_best.pth", "india", "india_eval_metrics.npy"),
    "india_temp": ("autoencoder_india_temp_best.pth", "india_temp", "india_temp_eval_metrics.npy"),
    "india_panel": ("autoencoder_india_panel_best.pth", "india_temp", "india_panel_eval_metrics.npy"),
    "india_inpaint": ("inpaint_india_best.pth", "india_temp", "india_inpaint_eval_metrics.npy"),
}


def resolve(name):
    """{"weights", "stats", "metrics"} paths for a bundle name."""
    if name not in BUNDLES:
        raise ValueError(f"unknown bundle {name!r}; known: {', '.join(BUNDLES)}")
    weights, stats_dir, metrics = BUNDLES[name]
    return {"weights": MODELS / weights,
            "stats": (PREPROCESSED / stats_dir if stats_dir else PREPROCESSED) / "norm_stats.npy",
            "metrics": RESULTS / metrics}


def read_threshold(metrics_path):
    """The saved best-F1 threshold of an evaluator run, or None when the file is absent."""
    if not metrics_path.exists():
        return None
    return float(np.load(metrics_path, allow_pickle=True).item()["threshold"])


class Bundle:
    def __init__(self, name, weights, stats, threshold, device=None, model=None, metrics_path=None):
        self.name = name
        self.weights = weights
        self.stats = dict(stats)
        self.threshold = threshold
        self.metrics_path = metrics_path
        self._device = device
        self._model = model

    @classmethod
    def load(cls, name, device=None):
        """Bundle by name from the resolution table. `device`: "cpu", "cuda" or None (CUDA when available)."""
        p = resolve(name)
        if not p["weights"].exists():
            raise FileNotFoundError(f"bundle {name!r}: weights not found at {p['weights']}")
        if not p["stats"].exists():
            raise FileNotFoundError(f"bundle {name!r}: norm stats not found at {p['stats']}")
        stats = np.load(p["stats"], allow_pickle=True).item()
        return cls(name, p["weights"], stats, read_threshold(p["metrics"]), device, metrics_path=p["metrics"])

    @classmethod
    def from_paths(cls, weights, stats_path, metrics_path=None, name=None, device=None):
        """Explicit files instead of a table entry."""
        stats = np.load(stats_path, allow_pickle=True).item()
        threshold = read_threshold(metrics_path) if metrics_path is not None else None
        return cls(name or str(weights), weights, stats, threshold, device, metrics_path=metrics_path)

    # ---- lazy torch ----
    @property
    def device(self):
        if self._device is None:
            import torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(self._device, str):
            import torch
            self._device = torch.device(self._device)
        return self._device

    @property
    def model(self):
        """The autoencoder in eval mode on `device`; weights load on first access (the one torch.load)."""
        if self._model is None:
            import torch
            from hlab.model import ThermalAutoencoder
            model = ThermalAutoencoder().to(self.device)
            model.load_state_dict(torch.load(self.weights, map_location=self.device, weights_only=True))
            model.eval()
            self._model = model
        return self._model

    def require_threshold(self):
        if self.threshold is None:
            hint = f" (run the evaluator that writes {self.metrics_path})" if self.metrics_path else ""
            raise RuntimeError(f"bundle {self.name!r} has no calibrated threshold{hint}")
        return self.threshold

    # ---- scoring ----
    def _batches(self, x, batch_size):
        import torch
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[np.newaxis]
        if x.ndim == 3:
            x = x[:, np.newaxis]
        model = self.model
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = torch.from_numpy(np.ascontiguousarray(x[i:i + batch_size])).float().to(self.device)
                yield batch, model(batch)

    def score_normalized(self, x, batch_size=32):
        """Per-image mean squared reconstruction error of arrays already in model units."""
        errors = []
        for batch, output in self._batches(x, batch_size):
            errors.extend(((batch - output) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
        return np.array(errors)

    def score(self, images_0_255, batch_size=32):
        """Per-image error of 0-255 model-input images (hlab.transform.to_model_input output)."""
        return self.score_normalized(normalize(images_0_255, self.stats), batch_size)

    def reconstruct_normalized(self, x, batch_size=32):
        """(reconstruction, squared-error map) per image, both in model units."""
        recon, err = [], []
        for batch, output in self._batches(x, batch_size):
            r = output.cpu().numpy()[:, 0]
            recon.append(r)
            err.append((batch.cpu().numpy()[:, 0] - r) ** 2)
        return np.concatenate(recon), np.concatenate(err)

    def reconstruct(self, images_0_255, batch_size=32):
        """(reconstruction in 0-255, squared-error map in model units) per image."""
        recon, err = self.reconstruct_normalized(normalize(images_0_255, self.stats), batch_size)
        return denormalize(recon, self.stats), err


class FakeBundle:
    """In-memory bundle for tests: `score_fn(x) -> errors` stands in for the model."""

    def __init__(self, threshold, score_fn, stats=None, name="fake"):
        self.name = name
        self.threshold = threshold
        self.stats = dict(stats or {"mean": 0.0, "std": 1.0})
        self.device = "cpu"
        self.weights = None
        self.metrics_path = None
        self._score_fn = score_fn

    def require_threshold(self):
        if self.threshold is None:
            raise RuntimeError(f"bundle {self.name!r} has no calibrated threshold")
        return self.threshold

    def score_normalized(self, x, batch_size=32):
        x = np.asarray(x, dtype=np.float32)
        return np.asarray(self._score_fn(x[np.newaxis] if x.ndim == 2 else x), dtype=float)

    def score(self, images_0_255, batch_size=32):
        return self.score_normalized(normalize(images_0_255, self.stats), batch_size)

    def reconstruct_normalized(self, x, batch_size=32):
        """Identity reconstruction; the error map is flat at score_fn's value so mean(err) == score."""
        x = np.asarray(x, dtype=np.float32)
        x = x[np.newaxis] if x.ndim == 2 else x
        err = np.repeat(self.score_normalized(x)[:, None, None], 1, axis=1) * np.ones_like(x)
        return x.copy(), err.astype(np.float32)

    def reconstruct(self, images_0_255, batch_size=32):
        recon, err = self.reconstruct_normalized(normalize(images_0_255, self.stats), batch_size)
        return denormalize(recon, self.stats), err
