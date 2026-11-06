import pandas as pd
from src.preprocessing import FeatureEngineer


def _groups(tenures):
    X = pd.DataFrame({"tenure": tenures, "TotalCharges": [0.0] * len(tenures)})
    return FeatureEngineer().fit_transform(X)["tenure_group"].tolist()


def test_tenure_group_covers_every_tenure():
    groups = _groups([0, 1, 12, 13, 24, 25, 48, 49, 72, 80])
    assert groups == ["0-1yr", "0-1yr", "0-1yr", "1-2yr", "1-2yr",
                      "2-4yr", "2-4yr", "4-6yr", "4-6yr", "4-6yr"]


def test_no_missing_tenure_group():
    assert not pd.Series(_groups([0, 5, 72, 100])).isna().any()
