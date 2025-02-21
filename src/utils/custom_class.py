import os

import mlflow

from sklearn.base import BaseEstimator, TransformerMixin

from dotenv import load_dotenv
load_dotenv()

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        ...

    def fit(self, X, y=None):


        return self

    def transform(self, X):
        new_X = X.copy()

        # Feature engineering
        new_X['CREDIT_INCOME_PERCENT'] = new_X['AMT_CREDIT'] / new_X['AMT_INCOME_TOTAL']
        new_X['ANNUITY_INCOME_PERCENT'] = new_X['AMT_ANNUITY'] / new_X['AMT_INCOME_TOTAL']
        new_X['CREDIT_TERM'] = new_X['AMT_ANNUITY'] / new_X['AMT_CREDIT']
        new_X['DAYS_EMPLOYED_PERCENT'] = new_X['DAYS_EMPLOYED'] / new_X['DAYS_BIRTH']

        return new_X
    
class MLFExperimentTracker:
    def __init__(self, use_mlflow=True, experiment_name="oc_pret_a_depenser"):
        self.use_mlflow = use_mlflow
        self.mlflow_url = os.getenv("MLFLOW_URL")
        if self.use_mlflow:
            mlflow.set_tracking_uri(self.mlflow_url)
            mlflow.set_experiment(experiment_name)

    def start_run(self, run_name=None):
        if self.use_mlflow:
            self.run = mlflow.start_run(run_name=run_name)
        else:
            self.run = None

    def log_param(self, key, value):
        if self.use_mlflow:
            mlflow.log_param(key, value)

    def log_params(self, dict_params):
        if self.use_mlflow:
            mlflow.log_params(dict_params)

    def log_metric(self, key, value):
        if self.use_mlflow:
            mlflow.log_metric(key, value)

    def log_model(self, model, model_name):
        if self.use_mlflow:
            mlflow.sklearn.log_model(model, model_name)

    def end_run(self):
        if self.use_mlflow:
            mlflow.end_run()