# scripts/predict.py
import pandas as pd
from src.config import INPUT_DIR
from src.predict import predict
import random

def main():
    
    input_data_path = INPUT_DIR / "new_data.csv"

    #From the loaded data in df_new create one random dictionary
    # Load new data
    df_new = pd.read_csv(input_data_path)
    # Pick a random row index
    random_idx = random.choice(df_new.index)
    # Convert that row to a dictionary
    input_data = df_new.loc[random_idx].to_dict()
    
    #Make a prediction and print it
    print(input_data)
    print(predict(input_data))

if __name__ == "__main__":
    main()