import pytest
from pathlib import Path
from src.data_loader import load_raw_data

def test_load_raw_data_success():
    df = load_raw_data()
    assert not df.empty
    assert "customerID" in df.columns

def test_load_raw_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_raw_data(Path("nonexistent/path.csv"))
