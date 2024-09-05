import pandas as pd
import config

def datetimes(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X['created_at'] = pd.to_datetime(X['created_at'], format='%Y-%m-%d %H:%M:%S')
    X['created_hours'] = X['created_at'].dt.hour
    X = X.drop('created_at', axis=1)
    if 'actual_delivery_time' in X.columns:
        X = X.drop('actual_delivery_time', axis=1)
    return X

def remove_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    features = [f for f in X.columns if f in config.FEATURES]
    X = X[features]
    return X

def main(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X = datetimes(X)
    X = remove_features(X)
    return X