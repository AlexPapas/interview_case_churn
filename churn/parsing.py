from pathlib import Path
from typing import Union
import pandas as pd
from churn.utils import adapt_colnames
from churn.logger import logger
import logging
from typing import Dict

logger.setLevel(logging.INFO)

# Define how to cast datatypes in raw data.
# Keys are expected column names after converting them to snake case.
# TODO: Should use configuration to perform this.
parsing_fns_dict = {
    "customerid": lambda df: df["customerid"].astype("category"),
    "seniorcitizen": lambda df: df["seniorcitizen"].astype("category"),
    "partner": lambda df: df["partner"].astype("category"),
    "dependents": lambda df: df["dependents"].astype("category"),
    "age": lambda df: pd.to_numeric(df["age"], errors="coerce"),
    "tenure": lambda df: pd.to_numeric(df["tenure"], errors="coerce"),
    "busines_loan": lambda df: df["busines_loan"].astype("category"),
    "multiplebusinessloans": lambda df: df["multiplebusinessloans"].astype("category"),
    "creditline": lambda df: df["creditline"].astype("category"),
    "online_banking": lambda df: df["online_banking"].astype("category"),
    "mortgage": lambda df: df["mortgage"].astype("category"),
    "stocks": lambda df: df["stocks"].astype("category"),
    "forex": lambda df: df["forex"].astype("category"),
    "contract": lambda df: df["contract"].astype("category"),
    "paperlessbilling": lambda df: df["paperlessbilling"].astype("category"),
    "paymentmethod": lambda df: df["paymentmethod"].astype("category"),
    "monthlycharges": lambda df: pd.to_numeric(df["monthlycharges"], errors="coerce"),
    "totalcharges": lambda df: pd.to_numeric(df["totalcharges"], errors="coerce"),
    "churn_within_a_month": lambda df: df["churn_within_a_month"].astype("category"),
    "satisfactory_onboarding_form": lambda df: pd.to_numeric(
        df["satisfactory_onboarding_form"]
    ),
}

# TODO: Check data quality, missing features, etc.
class DataManager:
    """
    Reads raw data and prepares it for model.
    """

    def __init__(self, raw_data_path: Union[str, Path]):
        logger.info("DataManager initialized.")
        self._raw_data_path = raw_data_path

    @property
    def raw_data(self) -> pd.DataFrame:
        logger.info("Reading raw data.")
        return pd.read_csv(self._raw_data_path, sep=";")

    @property
    def data(self) -> pd.DataFrame:
        data = self.raw_data.pipe(adapt_colnames)
        cast_dict = dict(
            filter(lambda x: x[0] in data.columns, parsing_fns_dict.items())
        )
        logger.info("Processing raw data.")
        return data.assign(**cast_dict)
