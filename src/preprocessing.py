import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from src.logger import logger

def preprocess_data(df: pd.DataFrame, fit: bool = True, scaler_path=None, encoder_path=None):
    # Example: separate numerical & categorical
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        df_cat = pd.DataFrame(encoder.fit_transform(df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))
        joblib.dump(encoder, encoder_path)
        logger.info(f"Saved encoder to {encoder_path}")
    else:
        scaler = joblib.load(scaler_path)
        df[num_cols] = scaler.transform(df[num_cols])

        encoder = joblib.load(encoder_path)
        df_cat = pd.DataFrame(encoder.transform(df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))

    df_final = pd.concat([df[num_cols], df_cat], axis=1)
    return df_final
