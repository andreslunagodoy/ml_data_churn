from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.config import CONFIG, MODEL_PATH, MODELS_FAMILY
from src.logger import setup_logger
from src.data_loader import load_raw_data
from src.preprocessing import get_preprocessor
from src.train_model import train_model
from src.evaluate import evaluate_model

timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
logger = setup_logger(f"train_pipeline_{timestamp}")

def main():
    # Start logger
    logger.info("Starting training pipeline")

    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Load data
    df_raw = load_raw_data()

    logger.info(f"Loaded data with shape {df_raw.shape}")

    # Split features and target
    X_raw = df_raw.drop(columns=CONFIG['target_column'])
    y = df_raw[CONFIG['target_column']].map(CONFIG['target_map'])

    logger.info(f"Split features and target {CONFIG['target_column']}")

    # Split train and test sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=CONFIG['test_size'], stratify=y, random_state=CONFIG['random_state']
    )

    logger.info(f"Split train and test set of size {CONFIG['test_size']}")

    # Preprocessing
    ## FEATURES
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_revenue']
    categorical_features = ['Contract', 'PaymentMethod', 'InternetService', 'tenure_group']
    engineered_features = [
        'is_new_customer', 'is_long_term', 'auto_pay_flag', 'num_services', 'high_monthly_flag', 'family_flag', 
        'fiber_flag', 'electronic_check_flag','new_echeck_interaction', 'fiber_highcharge_interaction', 
        'loyal_engaged_interaction']

    full_pipeline_linear, full_pipeline_tree = get_preprocessor(numeric_features, categorical_features, engineered_features)

    if MODELS_FAMILY[CONFIG['model_type']] == "linear":
        preprocessor = full_pipeline_linear
    elif MODELS_FAMILY[CONFIG['model_type']] == "tree":
        preprocessor = full_pipeline_tree

    preprocessor.fit(X_train_raw, y_train)
    X_train = preprocessor.transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)

    logger.info(f"Finished preprocessing for a {MODELS_FAMILY[CONFIG['model_type']]}-type model")

    # Train model
    model = train_model(X_train, y_train, CONFIG['model_type'])
    logger.info(f"Completed training a {CONFIG['model_type']} model")

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"Metrics: {metrics}")

    # Save artifacts
    joblib.dump(model, MODEL_PATH / "model.pkl")
    #joblib.dump(preprocessor, MODEL_PATH / "preprocessor.pkl")

    logger.info(f"Saved model and preprocessor to {MODEL_PATH}")
    logger.info("Training pipeline finished successfully")

if __name__ == "__main__":
    main()
