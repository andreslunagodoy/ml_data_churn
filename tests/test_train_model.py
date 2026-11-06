import numpy as np
from sklearn.datasets import make_classification
from churn.train_model import train_model


def _proba(random_state):
    X, y = make_classification(n_samples=200, n_features=10, random_state=0)
    return train_model(X, y, "random_forest", random_state=random_state).predict_proba(X)


def test_same_seed_gives_same_model():
    assert np.array_equal(_proba(1), _proba(1))


def test_different_seed_gives_different_model():
    assert not np.array_equal(_proba(1), _proba(2))
