import os
import subprocess
import sys

import pytest
from src.config import BASE_DIR, MODEL_DIR, get_model_path, load_config, validate_model_type
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


def _override(tmp_path, text):
    path = tmp_path / "override.yaml"
    path.write_text(text)
    return path


def test_load_config_has_timestamp():
    assert "timestamp" in load_config()


def test_override_changes_only_given_keys(tmp_path):
    config = load_config(_override(tmp_path, "random_state: 7\n"))
    assert config["random_state"] == 7
    assert config["test_size"] == load_config()["test_size"]


def test_override_rejects_unknown_key(tmp_path):
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(_override(tmp_path, "random_stat: 1\n"))


def test_override_rejects_invalid_values(tmp_path):
    with pytest.raises(ValueError, match="test_size"):
        load_config(_override(tmp_path, "test_size: 5\n"))
    with pytest.raises(ValueError, match="Unknown model_type"):
        load_config(_override(tmp_path, 'model_type: "xyz"\n'))


def test_model_path_follows_override(tmp_path):
    config = load_config(_override(tmp_path, 'model_version: "v2"\n'))
    assert get_model_path(config) == MODEL_DIR / "v2" / config["timestamp"]


def test_current_model_dir_can_be_set_by_env_var(tmp_path):
    code = "from src.config import CURRENT_MODEL_DIR; print(CURRENT_MODEL_DIR)"
    env = {**os.environ, "CHURN_MODEL_DIR": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=BASE_DIR, env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == str(tmp_path)
