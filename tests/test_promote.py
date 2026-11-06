import pytest
from scripts.promote import ARTIFACTS, promote


def _make_model_dir(path):
    path.mkdir()
    for name in ARTIFACTS:
        (path / name).write_text(name)
    return path


def test_promote_copies_all_artifacts(tmp_path):
    source = _make_model_dir(tmp_path / "source")
    target = tmp_path / "current"
    promote(source, target)
    for name in ARTIFACTS:
        assert (target / name).read_text() == name


def test_promote_overwrites_existing_files(tmp_path):
    source = _make_model_dir(tmp_path / "source")
    target = tmp_path / "current"
    target.mkdir()
    (target / "model.pkl").write_text("old")
    promote(source, target)
    assert (target / "model.pkl").read_text() == "model.pkl"


def test_promote_refuses_incomplete_model_dir(tmp_path):
    source = _make_model_dir(tmp_path / "source")
    (source / "model.pkl").unlink()
    target = tmp_path / "current"
    with pytest.raises(FileNotFoundError, match="model.pkl"):
        promote(source, target)
    assert not target.exists()
