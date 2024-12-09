from churn.parsing import DataManager
from churn.model import ChurnModel
from pathlib import Path
import yaml
from churn.logger import logger


RAW_DATA_PATH = "./data/case_churn.csv"
CONFIG_PATH = "./config/config.yml"


with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    data_manager = DataManager(RAW_DATA_PATH)

    # Initialize model:
    model = ChurnModel(
        numeric_features_list=config["model_dtype_treatment"]["numeric"],
        categorical_features_list=config["model_dtype_treatment"]["categorical"],
        response_col=config["model_dtype_treatment"]["response"],
    )

    # Train model
    model.train(data_manager.data)
