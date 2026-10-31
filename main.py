#LGBMBoost (best params)
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
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

model = TransformedTargetRegressor(
    regressor = create_pipe(
        LGBMRegressor(n_estimators = 500, learning_rate = 0.05, max_depth = 3, num_leaves = 7, random_state=42, verbose=-1, n_jobs=1),
        preprocessor,
        scale = False
    ),
    func = np.log1p,
    inverse_func = np.expm1
)

model.fit(X, y)
predictions = model.predict(X_test)

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': predictions
})

submission['SalePrice'] = submission['SalePrice'].clip(lower=0)
submission.to_csv('submission_lgbmboost.csv', index=False)

