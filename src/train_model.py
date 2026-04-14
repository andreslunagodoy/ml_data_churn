import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_model(X: np.ndarray, y: np.ndarray, model_type: str) -> BaseEstimator:

    if model_type == "logistic_regression":
        # Instantiate baseline logistic regression
        model = LogisticRegression(random_state=42, max_iter=1000)

    elif model_type == "random_forest":
        # Instantiate baseline random forest
        model = RandomForestClassifier(random_state=42, n_estimators=100)

    elif model_type == "gradient_boosting":
        # Instantiate baseline gradient boosting
        model = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1)

    model.fit(X,y)
    return model