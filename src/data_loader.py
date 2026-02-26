import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.logger import logger

def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    logger.info(f"Loading raw data from {path}")
    return pd.read_csv(path)

def load_processed_data(path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    logger.info(f"Loading processed data from {path}")
    return pd.read_csv(path)