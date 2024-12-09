import pandas as pd


def adapt_colnames(dataf: pd.DataFrame) -> pd.DataFrame:
    """Converts colnames to snake-case.

    Args:
        dataf (pd.DataFrame): Dataframe with messy column names.

    Returns:
        pd.DataFrame: Dataframe with less messy column names.
    """
    dataf = dataf.copy()
    dataf.columns = [col.replace(" ", "_").lower() for col in dataf.columns]
    return dataf
