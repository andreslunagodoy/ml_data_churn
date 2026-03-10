import pandas as pd
from src.config import INPUT_DIR, OUTPUT_DIR
from src.model_loader import load_model
from src.predict import predict_df

def main():

    input_data_path = INPUT_DIR / "new_data.csv"
    output_path = OUTPUT_DIR / "predictions.csv"

    model, preprocessor = load_model()

    df_new = pd.read_csv(input_data_path)

    predictions = predict_df(df_new, model, preprocessor)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()