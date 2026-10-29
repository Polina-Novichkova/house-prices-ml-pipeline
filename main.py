#XGBoost
from xgboost import XGBRegressor
import pandas as pd
from preprocessing import prepare_features
X, y, X_test, test_ids = prepare_features()

fold_rmse = []
    
model = XGBRegressor(
    n_estimators = 200,
    learning_rate = 0.05,
    max_depth = 4,
    random_state = 42
)

model.fit(X, y)
predictions = model.predict(X_test)

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': predictions
})

submission['SalePrice'] = submission['SalePrice'].clip(lower=0)
submission.to_csv('submission_xgboost.csv', index=False)
