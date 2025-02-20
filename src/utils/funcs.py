import os
import json

import regex as re

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, fbeta_score, roc_auc_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV

from src.utils.custom_class import MLFExperimentTracker

import warnings
warnings.filterwarnings('ignore')

def get_data_from_files(train_filepath = "data/application_train.csv", test_filepath = "data/application_test.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(train_filepath):
        raise FileNotFoundError(f"Training file not found ('{train_filepath}')")
    if not os.path.exists(test_filepath):
        raise FileNotFoundError(f"Testing file not found ({test_filepath})")

    app_train = pd.read_csv(train_filepath)
    app_test = pd.read_csv(test_filepath)

    return app_train, app_test

def get_data_from_files_test(test_filepath = "data/application_test.csv"):
    if not os.path.exists(test_filepath):
        raise FileNotFoundError(f"Testing file not found ({test_filepath})")

    app_test = pd.read_csv(test_filepath)

    return app_test

def prepare_data_test(df_test):
    # Clean anomalies
    clean_anomalies(
        df_test,
        col='DAYS_EMPLOYED',
        anom=365243,
        value=np.nan
    )

    df_test = df_test[df_test['CODE_GENDER'] != 'XNA']

    # Group reccuring object
    df_test = group_organization(df_test)

    # Add Feature Engineering
    df_test = feature_engineering(df_test)

    return df_test

def clean_anomalies(df, col, anom, value=np.nan):
    if col not in df:
        raise KeyError(f"{col} not in DataFrame columns")
    
    df[col].replace({anom:value}, inplace = True)

def group_organization(df):
    col_organization_type = "ORGANIZATION_TYPE"
    sub_df = df.copy()
    
    # ORGANIZATION_TYPE::contains_colon
    sub_df.loc[
        sub_df[col_organization_type].str.contains(':')
    , col_organization_type] = sub_df.loc[:, col_organization_type].apply(lambda x: x.split(':')[0] if len(x.split(':'))>1 else x)

    # ORGANIZATION_TYPE::contains_Business
    sub_df.loc[
        sub_df[col_organization_type].str.contains('Business')
    , col_organization_type] = "Business"


    return sub_df

def prepare_data(df_train, df_test):
    # Clean anomalies
    clean_anomalies(
        df_train,
        col='DAYS_EMPLOYED',
        anom=365243,
        value=np.nan
    )
    clean_anomalies(
        df_test,
        col='DAYS_EMPLOYED',
        anom=365243,
        value=np.nan
    )

    df_train = df_train[df_train['CODE_GENDER'] != 'XNA']
    df_test = df_test[df_test['CODE_GENDER'] != 'XNA']

    # Group reccuring object
    df_train = group_organization(df_train)
    df_test = group_organization(df_test)

    # Add Feature Engineering
    df_train = feature_engineering(df_train)
    df_test = feature_engineering(df_test)

    return df_train, df_test

def feature_engineering(df):
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    return df

def make_preprocess_pipeline(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler(feature_range = (0, 1))),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ('onehot_encoder', OneHotEncoder())
    ])

    preprocess_pipeline = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
    ])

    return preprocess_pipeline

def prepare_X_y(app_train = None, app_test = None, reduced=False):
    if (app_train is None) and (app_test is None):
        app_train, app_test = get_data_from_files()

    if reduced:
        app_train = select_top_features(app_train, with_target=True)
        app_test = select_top_features(app_test, with_target=False)

    target = app_train["TARGET"]

    if 'TARGET' in app_train.columns:
        train = app_train.drop(columns = ['TARGET'])
    else:
        train = app_train.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        train,
        target,
        train_size=0.8,
        shuffle=True,
        stratify=target,
        random_state=14
    )

    return X_train, X_test, y_train, y_test

def opti_rf_hyperparams(pipeline, X_train, y_train, stratified=False, reduced=False):
    param_grid = {
        'clf__n_estimators': [10**i for i in range(1, 3)],
        'clf__max_depth': [2**i for i in range(5, 8)],
        'clf__class_weight': ("balanced", None),
    }

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if stratified else KFold(n_splits=5, shuffle=True, random_state=42)
    print("Folds ready!")

    print("Fitting...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=folds,
        scoring=make_scorer(fbeta_score, beta=5),
        verbose=10
    )
    grid_search.fit(X_train, y_train)
    print("Meilleures hyperparamètres : ", grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)

    return grid_search

def train_model(model, model_params, train = None, test = None, opti_hyper=False, reduced=False):
        X_train, X_test, y_train, y_test = prepare_X_y(train, test, reduced=reduced)

        print(f"{X_train.shape=}")

        preprocess_pipeline = make_preprocess_pipeline(X_train)

        model_pipeline = ImbPipeline(steps=[
            ('preprocess', preprocess_pipeline),
            ('smote', SMOTE()),
            ("clf", model(
                **model_params
            ))
        ])

        if opti_hyper:
            print("Optimizing ON")
            print("Start Optimizer")
            search = opti_rf_hyperparams(
                model_pipeline,
                X_train,
                y_train,
                stratified=False,
                reduced=reduced
            )

            model_pipeline = ImbPipeline(steps=[
                ('preprocess', preprocess_pipeline),
                ('smote', SMOTE()),
                ("clf", model(
                    **{x.split('__')[1]:y for x,y in search.best_params_.items() if 'clf' in x}
                ))
            ])

        model_pipeline.fit(
            X_train,
            y_train
        )

        model_y_pred_train = model_pipeline.predict(X_train)
        model_y_proba_train = model_pipeline.predict_proba(X_train)[:, 1]

        model_y_pred_test = model_pipeline.predict(X_test)
        model_y_proba_test = model_pipeline.predict_proba(X_test)[:, 1]

        model_df_scores_train, dict_score_train = get_scores(
            y_train,
            model_y_pred_train,
            model_y_proba_train,
            'model_train' if not reduced else 'model_train_reduced'
        )
        model_df_scores_test, dict_score_test = get_scores(
            y_test,
            model_y_pred_test,
            model_y_proba_test,
            'model_test' if not reduced else 'model_test_reduced'
        )

        scores = {
            "train" : dict_score_train,
            "test" : dict_score_test,
        }

        return model_pipeline, scores

def predict_output(df_test, model):
    model_submit_proba = model.predict_proba(df_test)[:, 1]

    submit = df_test[['SK_ID_CURR']]
    submit['TARGET'] = model_submit_proba
    submit['DECISION'] = submit['TARGET'].apply(lambda proba: "OK" if proba < 0.23 else "NOT OK")

    return submit

def get_scores(y_test, y_pred, y_proba, name_df_col="dataset"):
    dict_scores = {
        "acuracy": accuracy_score(y_true=y_test, y_pred=y_pred),
        "precision": precision_score(y_true=y_test, y_pred=y_pred),
        "recall": recall_score(y_true=y_test, y_pred=y_pred),
        "f1": fbeta_score(y_true=y_test, y_pred=y_pred, beta=1),
        "f5": fbeta_score(y_true=y_test, y_pred=y_pred, beta=5),
        "ROC_AUC": roc_auc_score(y_true=y_test, y_score=y_proba),
    }

    df = pd.DataFrame(dict_scores, index=dict_scores.keys(), columns=[name_df_col])

    return df, dict_scores

def test_run_with_randomforest(test_top_features=False, tracking=True, optimize=False):
    print()
    # Read data
    train, test = prepare_data(*get_data_from_files())
    print("Data Ready")

    tracker = MLFExperimentTracker(
        use_mlflow=tracking,
    )

    BASE_NAME = "RandomForestClassifier"
    run_name = f"{BASE_NAME}-15features" if test_top_features else BASE_NAME
    run_name = f"{run_name}-optimized" if optimize else BASE_NAME

    tracker.start_run(run_name=run_name)

    try:
        # Setup model
        rf_params = {
            "n_estimators" : 10,
            "max_depth" : 32,
            "class_weight": None
        }
        rf_model = RandomForestClassifier

        tracker.log_params(rf_params)

        print("Model setup !")

        print("Training...")
        # Train model
        model_pipeline, scores = train_model(rf_model, rf_params, train, test, reduced=test_top_features, opti_hyper=optimize)
        print("Training DONE!")

        dict_score_train, dict_score_test = scores.values()

        for name, score in dict_score_train.items():
            mlflow.log_metric(f"{name}_train", score)
        for name, score in dict_score_test.items():
            mlflow.log_metric(f"{name}_test", score)

        tracker.log_model(model_pipeline, "RandomForestClassifier" if not test_top_features else "RandomForestClassifier-15features")

        # Predict
        print("Predicting...")
        output = predict_output(test, model_pipeline)

        print(pd.concat([
            output["DECISION"].value_counts(),
            output["DECISION"].value_counts(normalize=True),
        ], axis=1))

        return model_pipeline, scores, output
    except Exception as err:
        print(err)
    finally:
        tracker.end_run()

def test_run_with_lgbm(test_top_features=False, tracking=True, optimize=False):
    print()
    # Read data
    train, test = prepare_data(*get_data_from_files())
    print("Data Ready")

    tracker = MLFExperimentTracker(
        use_mlflow=tracking,
    )

    BASE_NAME = "LGBM"
    run_name = f"{BASE_NAME}-15features" if test_top_features else BASE_NAME
    run_name = f"{run_name}-optimized" if optimize else BASE_NAME

    tracker.start_run(run_name=run_name)

    try:
        # Setup model
        lgbm_params = {
            "boosting_type" : "gbdt",
            "class_weight" : "balanced",
            "learning_rate" : 0.02,
            "n_estimators" : 10000,
            "objective" : "binary",
        }
        lgbm_model = LGBMClassifier

        tracker.log_params(lgbm_params)

        print("Model setup !")

        print("Training...")
        # Train model
        model_pipeline, scores = train_model(lgbm_model, lgbm_params, train, test, reduced=test_top_features, opti_hyper=optimize)
        print("Training DONE!")

        dict_score_train, dict_score_test = scores.values()

        for name, score in dict_score_train.items():
            mlflow.log_metric(f"{name}_train", score)
        for name, score in dict_score_test.items():
            mlflow.log_metric(f"{name}_test", score)

        tracker.log_model(model_pipeline, "LGBM" if not test_top_features else "LGBM-15features")

        # Predict
        print("Predicting...")
        output = predict_output(test, model_pipeline)

        print(pd.concat([
            output["DECISION"].value_counts(),
            output["DECISION"].value_counts(normalize=True),
        ], axis=1))

        return model_pipeline, scores, output
    except Exception as err:
        print(err)
    finally:
        tracker.end_run()

def select_top_features(df, with_target=True):
    COLS = [
        'EXT_SOURCE_3',
        'OBS_30_CNT_SOCIAL_CIRCLE',
        'OBS_60_CNT_SOCIAL_CIRCLE',
        'EXT_SOURCE_2',
        'AMT_REQ_CREDIT_BUREAU_YEAR',
        'CNT_CHILDREN',
        'CNT_FAM_MEMBERS',
        'CREDIT_TERM',
        'NONLIVINGAREA_MEDI',
        'EXT_SOURCE_1',
        'DAYS_LAST_PHONE_CHANGE',
        'NONLIVINGAREA_MODE',
        'AMT_GOODS_PRICE',
        'REGION_POPULATION_RELATIVE',
        'NONLIVINGAREA_AVG'
    ]

    if with_target:
        COLS.append('TARGET')

    return df[COLS]
