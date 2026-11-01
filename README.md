House Prices — Advanced Regression Techniques
- About the Project
This project is a solution to the House Prices: Advanced Regression Techniques Kaggle competition.
The goal is to predict residential house prices based on various features describing properties, including their size, quality, location, and condition.
This project covers the complete machine learning workflow: Exploratory Data Analysis, data preprocessing, feature engineering, model training, cross-validation, model comparison, and ensemble learning.

- Best Result
The best-performing solution is a blending ensemble combining Voting Regressor and Weighted Averaging.
Model
Weighted Averaging
Kaggle RMSLE
0.12775
Model
Blending (75% Voting + 25% Weighted Averaging)
Kaggle RMSLE
0.12749
Best Kaggle score: 0.12749

- Models
Baseline and Linear Models
Linear Regression
Ridge Regression
Lasso Regression
ElasticNet
Distance-Based and Tree Models
K-Nearest Neighbors
Decision Tree
Random Forest
Gradient Boosting
CatBoost
LightGBM
XGBoost
Deep Learning
Multilayer Perceptron (MLP)
Ensemble Methods
Voting Regressor
Stacking Regressor
Bagging
Weighted Averaging
Blending

- Data Preprocessing
The preprocessing pipeline includes:
Handling missing values using SimpleImputer.
Encoding categorical features using OneHotEncoder.
Scaling numerical features for models sensitive to feature scales.
Applying transformations to highly skewed numerical features.
Feature Engineering to create additional informative features.
Preprocessing operations are integrated into Scikit-learn Pipelines to prevent data leakage during cross-validation.

- Model Validation
Models are evaluated using 5-fold KFold cross-validation with shuffling and a fixed random state.
The main evaluation metric is RMSLE (Root Mean Squared Logarithmic Error).
Out-of-fold predictions are also used to evaluate and compare ensemble combinations.

- Project Structure
house-prices-competitions/
│
├── data/                  # Competition datasets
├── notebooks/             # EDA and model experiments
├── submissions/           # Generated Kaggle submissions
│
├── config.py              # Project configuration
├── preprocessing.py       # Data preprocessing and feature engineering
├── model.py               # Model and ensemble definitions
├── main.py                # Main ML pipeline
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md

- Installation and Usage
1. Clone the repository
git clone https://github.com/Polina-Novichkova/house-prices-competitions.git
cd house-prices-competitions
2. Install dependencies
pip install -r requirements.txt
3. Prepare the data
Download train.csv and test.csv from the Kaggle competition and place them in the data/ directory.
4. Run the pipeline
python main.py
The script runs the machine learning pipeline and generates a submission file in the submissions/ directory.

- Technologies
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
CatBoost
LightGBM
XGBoost
Jupyter Notebook