#Blending weight aver + voting
import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_log_error
import sys
sys.path.append(r'C:\Users\User\house-prices-competitions')
from preprocessing import prepare_features
X, y, X_test, test_ids = prepare_features()

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median', keep_empty_features=True))
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

def create_pipe(model, preprocessor, scale=False):
    steps = [('preprocessor', preprocessor)]

    if scale:
        steps.append(('scaler', StandardScaler()))

    steps.append(('model', model))

    return Pipeline(steps)

def create_model(model, preprocessor, scale=False):
    return TransformedTargetRegressor(
        regressor=create_pipe(model, preprocessor, scale=scale),
        func=np.log1p,
        inverse_func=np.expm1
    )

best_estimators = {
    'LightGBM': create_model(
        LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            random_state=42,
            verbose=-1,
            n_jobs=1
        ),
        preprocessor,
        scale=False
    ),

    'CatBoost': create_model(
        CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            verbose=0,
            random_state=42
        ),
        preprocessor,
        scale=False
    ),

    'ElasticNet': create_model(
        ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=20000
        ),
        preprocessor,
        scale=True
    ),

    'XGBoost': create_model(
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=3,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42,
            objective='reg:squarederror',
            eval_metric='rmse',
            n_jobs=1
        ),
        preprocessor,
        scale=False
    )
}


voting_model = VotingRegressor(
    estimators=[
        ('lgbm', best_estimators['LightGBM']),
        ('catboost', best_estimators['CatBoost']),
        ('elasticnet', best_estimators['ElasticNet']),
        ('xgb', best_estimators['XGBoost'])
    ]
)

weighted_model = VotingRegressor(
    estimators=[
        ('lgbm', best_estimators['LightGBM']),
        ('catboost', best_estimators['CatBoost']),
        ('xgb', best_estimators['XGBoost'])
    ],
    weights=[2, 3, 2]
)

cv = KFold(
    n_splits = 5,
    shuffle = True,
    random_state = 42
)

oof_voting = cross_val_predict(
    voting_model,
    X,
    y,
    cv = cv,
    n_jobs = 1
)

oof_weighted = cross_val_predict(
    weighted_model,
    X,
    y,
    cv = cv,
    n_jobs = 1
)

def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_pred = np.clip(y_pred, 0, None)

    return np.sqrt(
        mean_squared_log_error(y_true, y_pred)
    )

best_voting_weight = 0.75
best_weighted_weight = 0.25
blend_pred = best_voting_weight * oof_voting + best_weighted_weight * oof_weighted
score = rmsle(y, blend_pred)
    
voting_model.fit(X, y)
weighted_model.fit(X, y)
test_voting = voting_model.predict(X_test)
test_weighted = weighted_model.predict(X_test)

final_pred = 0.75 * test_voting + 0.25 * test_weighted
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_pred
})

submission['SalePrice'] = submission['SalePrice'].clip(lower=0)
os.makedirs('submissions', exist_ok=True)
submission.to_csv('submissions/blend_vot_weight.csv', index=False)