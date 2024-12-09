import typer
from churn.parsing import DataManager
from churn.model import ChurnModel
import yaml

app = typer.Typer(name="churn", add_completion=False)

# Some sensible defaults:
DEFAULT_DATA_TRAIN_PATH = "./data/input/case_churn.csv"
DEFAULT_MODEL_PATH = "./artefacts/models/model.joblib"
DEFAULT_REPORT_PATH = "./artefacts/reports"
DEFAULT_DATA_INFERENCE_PATH = "./data/input/case_churn.csv"
DEFAULT_PREDICTION_PATH = "./data/output/inference.csv"

# DECISION THRESHOLD
DEFAULT_DECISION_THRESHOLD = 0.55

# Configuration we should keep version controlled:
CONFIG_PATH = "./config/config.yml"

# Read configuration
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# CLI
@app.command()
def train_model(
    path_to_data: str = typer.Option(None, help="Path to trainig data (str).")
):
    data_manager = DataManager(path_to_data or DEFAULT_DATA_TRAIN_PATH)

    model = ChurnModel(
        numeric_features_list=config["model_dtype_treatment"]["numeric"],
        categorical_features_list=config["model_dtype_treatment"]["categorical"],
        response_col=config["model_dtype_treatment"]["response"],
    )

    # Using default hypereparameters (None==default):
    model.train(train_df=data_manager.data, hyperparam_dict=None)

    # Save model artefact:
    model.save_model(DEFAULT_MODEL_PATH)

    # Save classification reports:
    model.save_report(DEFAULT_REPORT_PATH)


@app.command()
def predict_model(
    path_to_model: str = typer.Option(None, help="Path to model (str)"),
    path_to_data: str = typer.Option(None, help="Path to data (str)"),
    save_inference_path: str = typer.Option(
        None, help="Path to save inference result."
    ),
    decision_threshold: float = typer.Option(None, help="Decision threshold (float)."),
):
    model = ChurnModel(
        numeric_features_list=config["model_dtype_treatment"]["numeric"],
        categorical_features_list=config["model_dtype_treatment"]["categorical"],
        response_col=config["model_dtype_treatment"]["response"],
        decision_threshold=decision_threshold or DEFAULT_DECISION_THRESHOLD,
    ).load_model(path_to_model or DEFAULT_MODEL_PATH)

    data_manager = DataManager(path_to_data or DEFAULT_DATA_INFERENCE_PATH)

    inference_data = model.predict(data_manager.data)

    inference_data.to_csv(save_inference_path or DEFAULT_PREDICTION_PATH, index=False)


if __name__ == "__main__":
    app()
