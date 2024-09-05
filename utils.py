import pandas as pd
import numpy as np
import config
import json

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

class preprocess:
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
    

class imputers:
    class MarketIDImputer(TransformerMixin):
        def __init__(self) -> None:
            self.store_to_market = {}

        def fit(self, X, y=None):
            with open(config.STORE_TO_MARKET) as json_file:
                self.store_to_market = json.load(json_file)
            return self

        def transform(self, X, y=None):
            X = X.copy()
            X['market_id'] = X['store_id'].apply(lambda x: self.store_to_market[x] if x in self.store_to_market else None)
            X['market_id'] = X['market_id'].astype('float64')
            return X
        
    class PrimaryCategoryImputer(TransformerMixin):
        def __init__(self) -> None:
            self.store_to_primary = {}

        def fit(self, X, y=None):
            with open(config.STORE_TO_PRIMARY) as json_file:
                self.store_to_primary = json.load(json_file)
            return self

        def transform(self, X, y=None):
            X = X.copy()
            X['store_primary_category'] = X['store_primary_category'].fillna(X['store_id'].apply(lambda x: self.store_to_primary[x] if x in self.store_to_primary else None))
            return X

class encoders:
    class StorePrimaryEncoder(TransformerMixin):
        def __init__(self):
            self.store_primary_encode = {}
        
        def fit(self, X, y=None):
            with open(config.STORE_PRIMARY_ENCODE) as json_file:
                self.store_primary_encode = json.load(json_file)
            return self
        
        def transform(self, X, y=None):
            X = X.copy()
            unknown_cat = [cat for cat in X['store_primary_category'].unique() if cat not in self.store_primary_encode.keys()]
            encode_dict = self.store_primary_encode
            if len(unknown_cat) > 0:
                encode_dict.update({cat: np.nan for cat in unknown_cat})
            X['store_primary_category'] = X['store_primary_category'].replace(encode_dict)
            return X
            