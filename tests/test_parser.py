from churn.parsing import DataManager
import pandas as pd
from pathlib import Path

RAW_DATA_SAMPLE_PATH = "./data/case_churn.csv"


def test_processed_data():
    manager = DataManager(raw_data_path=RAW_DATA_SAMPLE_PATH)
    assert isinstance(manager.data, pd.DataFrame)
