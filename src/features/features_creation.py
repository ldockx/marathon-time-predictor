import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


"""
This file transforms the raw data into useable features for the training of the model
"""

#define data set here later

def drop_irrelevant_columns(df, columns_to_drop): #['id', 'Marathon', 'Name', 'CATEGORY']
    
    for col in columns_to_drop:
        df_new = df.drop(col, axis=1)

    return df_new

def one_hot_encode(df, columns_to_encode): #['Category']

    return pd.get_dummies(df, columns=columns_to_encode, dtype=int)

def extract_numeric_column(df, column_name, fillna_value=0, dtype=float):

        return (
        df[column_name]
        .str.extract(r"(\d+)")   # extract digits
        .astype(dtype)            # convert type
        .fillna(fillna_value)     # fill missing values
    )

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

def scale_features(df, columns_to_scale): #["km4week", "sp4week", "CrossTraining", "Wall21"]
     
    scaler = StandardScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df, scaler


     
