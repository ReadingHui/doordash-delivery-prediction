import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

class MarketIDImputer(TransformerMixin):
    def __init__(self) -> None:
        self.store_to_market = {}

    def fit(self, X, y=None):
        self.store_to_market = X.sort_values(by='created_at')[['store_id', 'market_id']].dropna(subset='market_id').drop_duplicates(keep='last').set_index('store_id').to_dict()['market_id']
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['market_id'] = X['store_id'].apply(lambda x: self.store_to_market[x] if x in self.store_to_market else None)
        return X
    
class PrimaryCategoryImputer(TransformerMixin):
    def __init__(self) -> None:
        self.store_to_primary = {}

    def fit(self, X, y=None):
        # Some of the store has more than one primary category, choosing the latest one by sorting by `created_at`
        self.store_to_primary = X.sort_values(by='created_at')[['store_id', 'store_primary_category']].dropna(subset='store_primary_category').drop_duplicates(subset='store_id', keep='last').set_index('store_id')['store_primary_category'].to_dict()
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['store_primary_category'] = X['store_primary_category'].fillna(X['store_id'].apply(lambda x: self.store_to_primary[x] if x in self.store_to_primary else None))
        return X
    
class MarketFeatureImputer(TransformerMixin):
    def __init__(self, method='median', market_features=None) -> None:
        self.method = method
        self.market_features = market_features
        self.nearest = {}
        self.market_features_median = {}

    def fit(self, X, y=None):
        # Some of the store has more than one primary category, choosing the latest one by sorting by `created_at`
        if self.method == 'median':
            for feature in self.market_features:
                self.market_features_median[feature] = X.groupby(['store_id', 'created_hours'])[feature].transform('median')
        elif self.method == 'nearest':
            self.nearest = (pd.merge_asof(
                                X[['store_id', 'created_hours'] + self.market_features].sort_values('created_hours').reset_index(),            
                                X[['store_id', 'created_hours'] + self.market_features].sort_values('created_hours').dropna(subset=self.market_features), 
                                by='store_id',                                         
                                on='created_hours', direction='nearest'                   
                                        )
                            .set_index('index')[[feature + '_y' for feature in self.market_features]].rename({feature + '_y': feature for feature in self.market_features}, axis=1)
                            )
        else:
            raise ValueError('No such method.')
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if self.method == 'median':
            for feature in self.market_features:
                X[feature] = X[feature].fillna(self.market_features_median[feature])
        elif self.method == 'nearest':
            for feature in self.market_features:
                X[feature] = X[feature].fillna(self.nearest[feature], downcast='infer')
        else:
            raise ValueError('No such method.')
        return X

class Imputers():
    def __init__(self, market_features):
        self.market_features = market_features
        self.pipeline = Pipeline(
                                    [
                                        ('market_id', MarketIDImputer()),
                                        ('primary_category', PrimaryCategoryImputer()),
                                        ('market_features_median', MarketFeatureImputer(method='median', market_features=self.market_features)),
                                        ('market_features_nearest', MarketFeatureImputer(method='nearest', market_features=self.market_features)),
                                    ]
                                )