"""
Train the v2 multi-equipment autoencoder (Track A).

The training loop lives in hlab.train and this model's recipe is the "v2"
entry of config/training.json (HER-128); running this file is the same as

    venv/Scripts/python.exe src/train.py --config v2

The file stays because the India experiment scripts still import
ThermalAutoencoder from here; the class itself lives in hlab.model.
"""
from hlab.model import ThermalAutoencoder  # noqa: F401
from hlab.train import fit_config


def main():
    fit_config("v2")


if __name__ == "__main__":
    main()
