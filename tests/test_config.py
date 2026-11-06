import pytest
from src.config import validate_model_type
from src.train_model import train_model


def test_validate_model_type_accepts_valid_types():
    for model_type in ["logistic_regression", "random_forest", "gradient_boosting"]:
        validate_model_type(model_type)  # should not raise


def test_validate_model_type_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown model_type 'xyz'") as exc:
        validate_model_type("xyz")
    assert "logistic_regression" in str(exc.value)  # lists the valid options


def test_train_model_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown model_type"):
        train_model([[0], [1]], [0, 1], "xyz")
