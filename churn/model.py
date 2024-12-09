import logging
from pathlib import Path
from typing import List, Union
from scipy import stats
from sklearn import model_selection
from typing import Dict, Union
from joblib import dump, load

import numpy as np
import pandas as pd
from churn.logger import logger
from sklearn import compose, feature_selection, impute, pipeline, preprocessing, metrics
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

# This is used throughout this module for reproducibility!
rng = np.random.seed(101)

logger.setLevel(logging.INFO)

PIPELINE_PARAMETERS_TUNING = {
    "regressor__learning_rate": stats.uniform(0.01, 0.5),
    "regressor__max_depth": stats.randint(3, 8),
    "regressor__max_iter": stats.randint(100, 800),
    "regressor__l2_regularization": stats.uniform(0.0, 15),
    "regressor__max_bins": stats.randint(50, 255),
    "regressor__class_weight": ["balanced", None],
    "regressor__early_stopping": [True],
    "regressor__n_iter_no_change": [5],
    # "rfe__n_features_to_select": stats.randint(5, 15),
}


class ModelNotFittedError(Exception):
    pass


class ChurnModel:
    def __init__(
        self,
        numeric_features_list: List[str],
        categorical_features_list: List[str],
        response_col: str,
        decision_threshold: float = 0.55,
    ):
        """Churn model.

        Args:
            numeric_features_list (List[str]): Numeric features.
            categorical_features_list (List[str]): Categorical features.
            response_col (str): Response column.
            decision_threshold (float, optional):Threshold to use to mark potential churn. Defaults to 0.55.
        """
        self._numeric = numeric_features_list
        self._categorical = categorical_features_list
        self._response_col = response_col
        self._decision_threshold = decision_threshold
        logger.info("ChurnModel initialized.")

    # TODO: This method is too big, need to refactor!!!
    # (direct copy from notebook)
    def train(
        self,
        train_df: pd.DataFrame,
        hyperparam_dict: Union[Dict[str, object], None] = None,
        max_iter: int = 300,
    ):
        """Trains and validates model.

        Args:
            train_df (pd.DataFrame): Data for training.
            hyperparam_dict (Union[Dict[str, object], None], optional): Hyperparameters. Defaults to None.
            max_iter (int, optional): Number of iterations for tuning. Defaults to 300.
        """
        hyperparams = hyperparam_dict or PIPELINE_PARAMETERS_TUNING
        logger.info("Creating pipeline.")
        self.pipeline = self._make_cls_pipeline()
        logger.info("Finished pipeline creation.")

        X = train_df[[col for col in train_df.columns if col != self._response_col]]
        y = train_df[self._response_col]

        # Using sklearn.compose.TransformedTargetRegressor makes
        # validation more challenging, so not using it here.
        label_enc = preprocessing.LabelEncoder().fit(y.to_numpy().ravel())

        y = label_enc.transform(y.to_numpy().ravel())

        # Save part of data as hold-out set:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, train_size=0.80, random_state=rng
        )

        # TODO: Perhaps use bayesian search (optuna, scikit-optimize, ..)
        logger.info("Parameter search started. This may take a while.")
        cv_res = model_selection.RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=hyperparams,
            n_jobs=-1,
            scoring="f1",
            random_state=rng,
            n_iter=max_iter,
        ).fit(X_train, y_train)
        logger.info("Hyperparameter search completed.")

        # We take the best estimator for later use!
        model = cv_res.best_estimator_

        logger.info("Generating validation report.")
        preds_train = model.predict_proba(X_train)[:, 1] >= self._decision_threshold
        preds_test = model.predict_proba(X_test)[:, 1] >= self._decision_threshold

        f1_score = metrics.f1_score(y_test, preds_test)
        logger.info(f"f1 score on hold-out: {f1_score:.2f}")

        self.report_train = metrics.classification_report(y_train, preds_train)
        self.report_test = metrics.classification_report(y_test, preds_test)

        logger.info("Finalizing model for inference.")
        self.model = model.fit(X, y)

        logger.info("Model trained and ready for inference.")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Make predictions.

        Args:
            X (pd.DataFrame): Data on which to make predictions.

        Raises:
            ModelNotFittedError: If model is not fitted.

        Returns:
            pd.DataFrame: Input frame with predictions.
        """
        if not getattr(self, "model", None):
            raise ModelNotFittedError("Train or load model first!")
        preds = self.model.predict_proba(X)[:, 1] >= self._decision_threshold
        return X.assign(preds=preds)

    def save_model(self, model_save_path: Union[str, Path]):
        """Saves model to specified location.

        Args:
            model_save_path (Union[str, Path]): Where to save model.

        Raises:
            ModelNotFittedError: If model was not fitted.
        """
        if not getattr(self, "model", None):
            raise ModelNotFittedError("Train or load model first!")
        logger.info(f"Saving model to {model_save_path}")
        dump(self.model, Path(model_save_path).absolute())
        logger.info("Model saved.")

    def load_model(self, model_path: Union[str, Path]) -> object:
        """Loads model from path for inference.

        Args:
            model_path (Union[str, Path]): Path to model object.
        """
        logger.info(f"Loading model from {model_path}")
        self.model = load(model_path)
        logger.info("Model loaded.")
        return self

    def save_report(self, report_path_folder: Union[str, Path]):
        """Saves report.

        Args:
            report_path_folder (Union[str, Path]): Where to save report.

        Raises:
            AttributeError: Model is not trained.
        """
        if not (
            getattr(self, "report_train", None) or getattr(self, "report_test", None)
        ):
            raise AttributeError("Run `train` first!")

        with open(Path(report_path_folder).absolute() / "report_train.txt", "w") as f:
            f.write(str(self.report_train))

        with open(Path(report_path_folder).absolute() / "report_test.txt", "w") as f:
            f.write(str(self.report_test))

    def _make_scaler(self):
        logger.info("Creating scaler.")
        self.scaler = preprocessing.StandardScaler()
        logger.info("Scaler created.")
        return self.scaler

    def _make_numeric_transformer(self):
        logger.info("Creating numeric transformed created.")
        self.numeric_transformer = pipeline.Pipeline(
            steps=[
                ("imputer_num", impute.KNNImputer(n_neighbors=5, add_indicator=True)),
                ("scaler", self._make_scaler()),
            ]
        )
        logger.info("Numeric transformed created.")
        return self.numeric_transformer

    def _make_cat_encoder(self):
        logger.info("Creating one hot encoder.")
        self.ohe = preprocessing.OneHotEncoder(
            sparse_output=False,
            handle_unknown="infrequent_if_exist",
            min_frequency=10,
            drop="if_binary",
        )
        logger.info("One hot encoder created.")
        return self.ohe

    def _make_cat_transformer(self):
        logger.info("Creating categorical transformer.")
        self.cat_transformer = pipeline.Pipeline(
            steps=[
                ("cat_impute", impute.SimpleImputer(strategy="most_frequent")),
                ("one_hot", self._make_cat_encoder()),
            ]
        )
        logger.info("Categorical transformer created.")
        return self.cat_transformer

    def _make_transformer(self):
        logger.info("Creating num + cat transformer.")
        self.transformer = compose.ColumnTransformer(
            transformers=[
                ("numeric_preprocess", self._make_numeric_transformer(), self._numeric),
                (
                    "categorical_preprocess",
                    self._make_cat_transformer(),
                    self._categorical,
                ),
            ],
            remainder="drop",
        )
        logger.info("Num + cat transformer created.")
        return self.transformer

    def _make_rfe_selector(self):
        logger.info("Creating feature selector.")
        self.rfe_selector = RandomForestClassifier(
            random_state=rng, class_weight="balanced", n_jobs=1, max_depth=5
        )
        logger.info("Feature selector created.")
        return self.rfe_selector

    def _make_cls_pipeline(self):
        logger.info("Creating classification pipeline.")
        self.cls_pipeline = pipeline.Pipeline(
            steps=[
                ("transformer", self._make_transformer()),
                # ("rfe", feature_selection.RFE(estimator=self._make_rfe_selector())),
                ("regressor", HistGradientBoostingClassifier()),
            ]
        )
        logger.info("Classification pipeline created.")

        return self.cls_pipeline
