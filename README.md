# Churn
Package for training and batch prediction for customer churn in a banking setting.

Since predictions are likely to be run in batch mode, command line app might be most convenient way to deploy this application.

Package provides CLI for easier deployment in setup like AzureML.

## Virtual environment setup
Please install conda or miniconda. You can find instructions [here](https://docs.anaconda.com/anaconda/install/index.html).

Once anaconda/miniconda is installed, open up the terminal and run:
```shell
make environment
```
This will install `churn` package dependencies into a conda environment `churn_env`.
Please make sure to use this environment to run any commands related to this package.

Once the environment is installed, run the following command from your terminal:
```shell
conda activate churn_env
```

## Run jupyter notebook
Activate `churn_env` (see above).
Jupyter notebooks are nice for explaration. 
To run notebook, from the terminal simply run:
```shell
make jupyter
```

## Train the model
Activate `churn_env` (see above).
To (re)train the model, simply run:
```shell
python main.py train-model path-to-data <relative path to train data file>
```
If `<relative path to train data file>` is not provided, default one will be used (`./data/input/case_churn.csv`)

You can also check `python main.py predict-model --help` for help.

## Make predictions
Activate `churn_env` (see above).
To make predictions, simply run:
```shell
python main.py predict-model --path-to-model <relative path to trained model object> --path-to-data <relative path to data> --save-inference-path <relative path to save inference data> --decision-threshold <decision threshold to flag churners>
```
You can also check `python main.py predict-model --help` for help.

## Configuration
Configuration is haldled using yaml stored in `./config/config.yml`. Package expects configuration to be stored at this location.

At the moment configuration contains features and type of those features to be considered by the model. Feature naming convention is snake case.


## TODOs (incomplete):
Needless to say, this package was created in relatively quick and diry way due to time constraints. This could serve as an MVP.
Number of things could be improved to make the codebase production-ready.
- Complete docstrings.
- Linting (flake8 or pylint).
- Dockerize application.
- Model and artefact versioning (MLFlow or AzureML).
- Would be good to request more granular dataset from the business, so that the problem can be treated as a time-series. 
- Hyperparameter optimization should be done better. Random search is not great.
- Survival analysis and Recency-Frequency-Monetary Value analysis could be powerful here. Still, it requires information that we do not have (time series; or differently aggregated data). Understanding Customer Lifetime Value would also be useful to prioritize high-impact customers.
- Consider performing cross-validation using custom metric that takes into account monthly charges.
- Model can be considerably improved by better treatment of categoricals (having model handle categoricals, using Target Encoder, etc.). By default this is difficult to implement as a part of sklearn pipeline, but there is a neat trick to [tackle this issue](https://medium.com/analytics-vidhya/scikit-learn-pipeline-transformers-the-hassle-of-transforming-target-variables-part-1-6dfb714e2aad#:~:text=You%20either%20transform%20one%20or%20another.&text=That%20happens%20because%20the%20y,passed%20on%20through%20the%20pipeline.). I didn't implement this here to keep things simple.
- Classes are not balanced. I've used class weighting and stratification (CV), but more can be done. For example, see [here](https://imbalanced-learn.org/stable/) for inspiration.
- More models could be tested.  For now I just used baseline and gradient boosting. Perhaps something like regularized logistic regression could already do the trick?
- Try autoML as a benchmark -> PyCaret is a good option.
- I have not done much feature engineering, there is likely room for improvement here.
- MOAR unit tests!! MOAR integration tests (there aren't any) :)
- Refactor `ChurnModel` using staticmethods to make testing easier.
- Requirements should be split for development and deployment purposes.
- `ChurnModel` train method is way too overcrowded. There is a lot to improve separation of concerns-wise.
- We could consider providing model decision threshold instead of binary target.
- More artefacts should be generated (diagnostic plots, reports, etc) when the model is trained. This is much easier to achieve using a framework such as MLFlow (not implemented here).
- Include input data validation (Great Expectations or similar). Exports can be nasty.
- Have option not to do hyperparameter tuning every time that the model is retrained.
- More efficient hyperparameter tuning using packages like `hyperopt`, `scikit-optimize` or `optuna`. Latter is ultra popular on Kaggle lately.
- Use config file for data casting and column checks.
- Extend `ChurnModel` with model explanability (now part of notebook). Ideally deploy interactive app to explore more easily and play with what-if type of scenarios (package [`explainerdashboard`](https://explainerdashboard.readthedocs.io/en/latest/) makes this a low-hanging fruit).
- Consider creating api, [`fastapi`](`https://fastapi.tiangolo.com/`) is a very popular framework.