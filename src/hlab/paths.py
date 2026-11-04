"""
Repository layout (HER-127): the one place that knows where things live.

Every path is relative to ROOT, the checkout that contains this package.
Scripts import these instead of declaring their own ``ROOT = Path(__file__)...``.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "CA_Training_Data"          # delivered images, gitignored
PREPROCESSED = DATA / "CA_Preprocessed"  # .npy arrays and stats the preprocess scripts write, gitignored
SOURCES = DATA / "sources"               # raw source datasets (India thermal library, PVMD)
MODELS = ROOT / "models"                 # checkpoints, gitignored
RESULTS = ROOT / "results"               # evaluation outputs, gitignored
CONFIG = ROOT / "config"                 # rule-layer threshold configs, tracked
MANIFESTS = ROOT / "docs" / "data_manifests"   # committed split and screening manifests
