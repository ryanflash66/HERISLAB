"""
Train one configuration from config/training.json (HER-128).

    venv/Scripts/python.exe src/train.py --config v2
    venv/Scripts/python.exe src/train.py --config pv_om --epochs 2 --model-dir /tmp/check

Every trained model is an entry in config/training.json; the loop is
hlab.train.fit. --model-dir and --log-dir redirect the outputs (a retrain check
that must not touch models/ and results/), --epochs and --device override the
entry.
"""
import argparse
from pathlib import Path

from hlab.train import fit_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="entry name in config/training.json (v2, pv, pv_om, ...)")
    parser.add_argument("--epochs", type=int, default=None, help="override the entry's max epochs")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: cuda when available)")
    parser.add_argument("--model-dir", type=Path, default=None, help="where to write the checkpoints (default models/)")
    parser.add_argument("--log-dir", type=Path, default=None, help="where to write the training log (default results/)")
    args = parser.parse_args(argv)
    overrides = {"epochs": args.epochs} if args.epochs is not None else {}
    bundle, history = fit_config(args.config, device=args.device, model_dir=args.model_dir,
                                 log_dir=args.log_dir, **overrides)
    print(f"\n{bundle.name}: best val_loss {history['best_val_loss']:.6f} at epoch {history['best_epoch']} "
          f"({history['epochs_run']} epochs, {history['minutes']:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
