import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(filename):
    paths = [
        os.path.join(BASE_DIR, '..', 'data', filename),
        os.path.join(BASE_DIR, 'data', filename),
        os.path.join(BASE_DIR, filename),
        filename,
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f'Файл {filename} не найден!')

def prepare_features():
    train = load_data('train.csv')
    test = load_data('test.csv')
    test_ids = test['Id']
    
    #Удаление выбросов только из train
    train = train[~train['Id'].isin([524, 1299])].copy()
    
    dt = pd.concat([train, test], ignore_index = True)
    
    #Обработка пропусков
    no_obj = [
        'PoolQC',
        'MiscFeature',   
        'Alley',           
        'Fence',
        'FireplaceQu'  
    ]
    dt[no_obj] = dt[no_obj].fillna('None')
    
    no_garage = [
        'GarageType',
        'GarageFinish',
        'GarageQual',
        'GarageCond'
    ]
    dt[no_garage] = dt[no_garage].fillna('None')
    
    no_bsmt = [
    'BsmtCond',
    'BsmtExposure',
    'BsmtQual',
    'BsmtFinType2',
    'BsmtFinType1'
    ]
    #числовые фичи
    bsmt_num = [
        'BsmtFinSF1',
        'BsmtFinSF2',
        'BsmtUnfSF',
        'TotalBsmtSF',
        'BsmtFullBath',
        'BsmtHalfBath'
    ]
    mask_no_bsmt = (dt['TotalBsmtSF'].isna() & dt[no_bsmt].isna().all(axis=1))
    dt.loc[mask_no_bsmt, bsmt_num] = 0
    dt.loc[mask_no_bsmt, no_bsmt] = 'None'
    mask1 = dt['TotalBsmtSF'] == 0
    dt.loc[mask1, no_bsmt] = (dt.loc[mask1, no_bsmt].fillna('None'))
    
    dt[['BsmtFullBath', 'BsmtHalfBath']] = (dt[['BsmtFullBath', 'BsmtHalfBath']].fillna(0))

    mask2 = (dt['MasVnrType'].isna() & dt['MasVnrArea'].isna())
    dt.loc[mask2, 'MasVnrType'] = 'None'
    dt.loc[mask2, 'MasVnrArea'] = 0
    dt['HouseAge'] = dt['YrSold'] - dt['YearBuilt']
    dt['YearsSinceRemod'] = dt['YrSold'] - dt['YearRemodAdd']
    
    skewed_cols = [
    'MiscVal',
    'PoolArea',
    'LotArea',
    'LowQualFinSF',
    '3SsnPorch',
    'BsmtFinSF2',
    'EnclosedPorch',
    'ScreenPorch',
    'MasVnrArea',
    'OpenPorchSF'
    ]   
    for col in skewed_cols:
        if (dt[col].dropna() < 0).any():
            continue
        dt[col] = np.log1p(dt[col])
     
    n_train = len(train)
    train_clean = dt.iloc[:n_train].copy()
    test_clean = dt.iloc[n_train:].copy()
    X = train_clean.drop(columns = ['SalePrice', 'Id'])
    y = train_clean['SalePrice']
    X_test = test_clean.drop(columns = ['SalePrice', 'Id'])
    
    return X, y, X_test, test_ids
