import argparse
import shutil
from pathlib import Path
from churn.config import MODEL_DIR

ARTIFACTS = ["model.pkl", "preprocessor.pkl", "metrics.json", "config.json", "proxy_input.csv"]


def promote(source: Path, target: Path = MODEL_DIR / "current") -> None:
    """Copy a trained model's artifacts into the folder the API and batch predictions use."""
    missing = [name for name in ARTIFACTS if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"{source} is missing artifacts: {missing}")

    target.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        shutil.copy2(source / name, target / name)


def main():
    parser = argparse.ArgumentParser(description="Promote a trained model to models/current")
    parser.add_argument("model_dir", type=str, help="Folder of the trained model, e.g. models/v1/<timestamp>")
    args = parser.parse_args()

    promote(Path(args.model_dir))
    print(f"Promoted {args.model_dir} to {MODEL_DIR / 'current'}")


if __name__ == "__main__":
    main()
