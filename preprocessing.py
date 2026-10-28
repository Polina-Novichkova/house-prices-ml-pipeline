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
        'Fence'   
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
    mask2 = dt['TotalBsmtSF'] > 0
    dt.loc[mask2, 'BsmtFinType2'] = dt.loc[mask2, 'BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])
    dt.loc[mask2, 'BsmtExposure'] = dt.loc[mask2, 'BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])
    dt.loc[mask2, 'BsmtCond'] = dt.loc[mask2, 'BsmtCond'].fillna(train['BsmtCond'].mode()[0])
    dt.loc[mask2, 'BsmtQual'] = dt.loc[mask2, 'BsmtQual'].fillna(train['BsmtQual'].mode()[0])
    dt[['BsmtFullBath', 'BsmtHalfBath']] = (dt[['BsmtFullBath', 'BsmtHalfBath']].fillna(0))

    mask3 = (dt['MasVnrType'].isna() & dt['MasVnrArea'].isna())
    dt.loc[mask3, 'MasVnrType'] = 'None'
    dt.loc[mask3, 'MasVnrArea'] = 0
    mask4 = (dt['MasVnrType'].isna() & dt['MasVnrArea'] > 0)
    dt.loc[mask4, 'MasVnrType'] = dt.loc[mask4, 'MasVnrType'].fillna(train['MasVnrType'].mode()[0])
    
    dt['FireplaceQu'] = dt['FireplaceQu'].fillna('None')
    
    mode_cols = [
        'MSZoning',          
        'Utilities',     
        'Functional',   
        'Exterior2nd', 
        'Exterior1st', 
        'Electrical', 
        'KitchenQual',   
        'SaleType'
    ]
    for col in mode_cols:
        dt[col] = dt[col].fillna(train[col].mode()[0])
    
    mask5 = dt['Id'] == 2577
    dt.loc[mask5, ['GarageCars', 'GarageArea']] = 0
    
    neighb_med = train.groupby('Neighborhood')['LotFrontage'].median()
    all_med = train['LotFrontage'].median()
    for neighb, med in neighb_med.items():
        mask6 = (dt['LotFrontage'].isna() & (dt['Neighborhood'] == neighb))
        dt.loc[mask6, 'LotFrontage'] = med
    dt['LotFrontage'] = dt['LotFrontage'].fillna(all_med)
    
    dt['MasVnrType'] = dt['MasVnrType'].fillna('None')
    dt['GarageYrBlt'] = dt['GarageYrBlt'].fillna(0)
    
    cat_cols = dt.select_dtypes(include = ['object', 'str']).columns
    dt = pd.get_dummies(dt, columns = cat_cols, dtype = int)
    
    #Новые признаки
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
