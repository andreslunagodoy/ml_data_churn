import argparse
from pathlib import Path
import pandas as pd
from src.config import INPUT_DIR, OUTPUT_DIR
from src.model_loader import load_model
from src.predict import predict_df

def main():
    parser = argparse.ArgumentParser(description="Run batch churn predictions")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV (default: input/new_data.csv)")
    parser.add_argument("--output", type=str, default=None, help="Path to output CSV (default: output/<timestamp>/predictions.csv)")
    args = parser.parse_args()

    input_data_path = Path(args.input) if args.input else INPUT_DIR / "new_data.csv"
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "predictions.csv"

    model, preprocessor = load_model()

    df_new = pd.read_csv(input_data_path)

    predictions = predict_df(df_new, model, preprocessor)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()