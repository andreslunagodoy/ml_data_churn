import pandas as pd
from src.config import RAW_DATA_PATH

def load_raw_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path)
    return df