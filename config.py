from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

# Reproducibility
RANDOM_STATE = 42

# Target and ID
TARGET = "SalePrice"
ID_COLUMN = "Id"

# Cross-validation
N_SPLITS = 5
SHUFFLE = True

# Outliers
OUTLIER_IDS = [524, 1299]

# LightGBM parameters
LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 3,
    "num_leaves": 7,
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "n_jobs": 1
}

# CatBoost parameters
CATBOOST_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.1,
    "verbose": 0,
    "random_state": RANDOM_STATE
}

# ElasticNet parameters
ELASTICNET_PARAMS = {
    "alpha": 0.01,
    "l1_ratio": 0.5,
    "max_iter": 20000
}

# XGBoost parameters
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.1,
    "max_depth": 3,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_jobs": 1
}

# Weighted Averaging parameters
WEIGHTED_MODEL_WEIGHTS = [2, 3, 2]

# Final Blending weights
VOTING_WEIGHT = 0.75
WEIGHTED_AVERAGING_WEIGHT = 0.25

# Submission
SUBMISSION_FILENAME = "blend_vot_weight.csv"