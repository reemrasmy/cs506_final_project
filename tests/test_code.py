import pandas as pd

data_path = "tests/test_data.csv"


def test_processed_data_loads():
    df = pd.read_csv(data_path)
    assert df.shape[0] > 0
    assert df.shape[1] > 2


def test_required_columns_exist():
    df = pd.read_csv(data_path)
    assert "sampleID" in df.columns
    assert "subtype" in df.columns


def test_expected_subtypes_present():
    df = pd.read_csv(data_path)
    expected = {"LumA", "LumB", "Basal", "Her2"}
    assert expected.issubset(set(df["subtype"].unique()))


def test_no_missing_subtypes():
    df = pd.read_csv(data_path)
    assert df["subtype"].isna().sum() == 0


