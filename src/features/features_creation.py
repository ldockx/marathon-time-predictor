import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#import logging --> transform print statements to logging


"""
This file transforms the raw data into useable features for the training of the model
"""

#configuering some project specific things

file_path = os.path.join('data', 'raw', 'MarathonData.csv')
df = pd.read_csv(file_path)
config = {
    'columns_to_drop': ['id', 'Marathon', 'Name', 'CATEGORY'],
    'columns_to_encode': ['Category'],
    'columns_to_numeric' : 'CrossTraining',
    'columns_to_ensure_numeric': ['km4week', 'sp4week', 'CrossTraining', 'Wall21'], #should be the same as the ones you are going to scale 
    'columns_to_scale': ['km4week', 'sp4week', 'CrossTraining', 'Wall21'],
    'target_column': 'MarathonTime',
    'test_size': 0.2,
    'random_state': 8
}


def drop_irrelevant_columns(df, columns_to_drop): #['id', 'Marathon', 'Name', 'CATEGORY']
    
    result = df.drop(columns=columns_to_drop, errors='ignore')
    return result

def one_hot_encode(df, columns_to_encode): #['Category']

    result = pd.get_dummies(df, columns=columns_to_encode, dtype=int)

    return result

def extract_numeric_column(df, column_name, fillna_value=0, dtype=float):

    df[column_name] =  (
        df[column_name]
        .str.extract(r"(\d+)")   # extract digits
        .astype(dtype)            # convert type
        .fillna(fillna_value)     # fill missing values
    )

    return df

def ensure_numeric_columns_are_numeric(df, columns_to_check, errors="coerce"):
     
    for col in columns_to_check:
        df[col] = pd.to_numeric(df[col], errors=errors)

    return df

def fill_na_with_median(df):
     
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

#YOU MAY ONLY SCALE FEATURES AFTER THE SPLIT    !!!
#you should fit the scaler only on the x_train data, then transform x_test
def split_data(df, target_column, test_size=0.2, random_state=8):
    
    # seperate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test, columns_to_scale):

    X_train = X_train.copy()
    X_test = X_test.copy()
    
    scaler = StandardScaler()
    X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    return X_train, X_test, scaler


def create_features():#df, config):

    global df, config

    # drop irrelevant columns
    df = drop_irrelevant_columns(df, config.get('columns_to_drop', []))
    
    # one-hot encode categorical features
    df = one_hot_encode(df, config.get('columns_to_encode', []))

    # extract numeric column
    df = extract_numeric_column(df, config.get('columns_to_numeric', []))
    
    # dnsure numeric columns are properly typed
    df = ensure_numeric_columns_are_numeric(
        df, 
        config.get('columns_to_ensure_numeric', [])
    )
    
    # fill missing values
    df = fill_na_with_median(df)
    
    # Split data BEFORE scaling --> import, check scaler logic if forgotten
    X_train, X_test, y_train, y_test = split_data(
        df,
        config['target_column'],
        test_size=config.get('test_size', 0.2),
        random_state=config.get('random_state', 8)
    )
    
    # scale features (fit on train, transform both)
    X_train, X_test, scaler = scale_features(
        X_train, 
        X_test, 
        config.get('columns_to_scale', [])
    )
    
    return X_train, X_test, y_train, y_test, scaler


     
