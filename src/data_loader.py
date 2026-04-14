import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_PATH

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Data file is empty: {path}")

    return df
