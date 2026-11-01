import numpy as np

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    RANDOM_STATE,
    LIGHTGBM_PARAMS,
    CATBOOST_PARAMS,
    ELASTICNET_PARAMS,
    XGBOOST_PARAMS,
    WEIGHTED_MODEL_WEIGHTS
)


def create_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(
        include=["object", "category", "str"]
    ).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            keep_empty_features=True
        ))
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent",
            keep_empty_features=True
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return preprocessor


def create_pipe(model, preprocessor, scale=False):
    steps = [("preprocessor", preprocessor)]

    if scale:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))

    return Pipeline(steps)


def create_model(model, preprocessor, scale=False):
    return TransformedTargetRegressor(
        regressor=create_pipe(model, preprocessor, scale=scale),
        func=np.log1p,
        inverse_func=np.expm1
    )


def create_ensemble_models(X):
    preprocessor = create_preprocessor(X)

    best_estimators = {
        "LightGBM": create_model(
            LGBMRegressor(**LIGHTGBM_PARAMS),
            preprocessor,
            scale=False
        ),

        "CatBoost": create_model(
            CatBoostRegressor(**CATBOOST_PARAMS),
            preprocessor,
            scale=False
        ),

        "ElasticNet": create_model(
            ElasticNet(**ELASTICNET_PARAMS),
            preprocessor,
            scale=True
        ),

        "XGBoost": create_model(
            XGBRegressor(**XGBOOST_PARAMS),
            preprocessor,
            scale=False
        )
    }

    voting_model = VotingRegressor(
        estimators=[
            ("lgbm", best_estimators["LightGBM"]),
            ("catboost", best_estimators["CatBoost"]),
            ("elasticnet", best_estimators["ElasticNet"]),
            ("xgb", best_estimators["XGBoost"])
        ]
    )

    weighted_model = VotingRegressor(
        estimators=[
            ("lgbm", best_estimators["LightGBM"]),
            ("catboost", best_estimators["CatBoost"]),
            ("xgb", best_estimators["XGBoost"])
        ],
        weights=WEIGHTED_MODEL_WEIGHTS
    )

    return voting_model, weighted_model