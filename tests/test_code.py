import pandas as pd

data_path = "data/processed_brca_data.csv"

# processed data exists
def test_processed_data_loads():
    df = pd.read_csv(data_path)
    assert df.shape[0] > 0
    assert df.shape[1] > 2

# make sure sampleID and subtype columns from metadata are there
def test_required_columns_exist():
    df = pd.read_csv(data_path)
    assert "sampleID" in df.columns
    assert "subtype" in df.columns

# all the expected subtypes are present in data
def test_expected_subtypes_present():
    df = pd.read_csv(data_path)
    expected_subtypes = {"LumA", "LumB", "Basal", "Her2"}
    observed_subtypes = set(df["subtype"].unique())
    assert expected_subtypes.issubset(observed_subtypes)

# no sample has a missing subtype
def test_no_missing_subtypes():
    df = pd.read_csv(data_path)
    assert df["subtype"].isna().sum() == 0