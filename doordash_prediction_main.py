import pandas as pd
import numpy as np
import lightgbm as lgb
import config
import utils
import os

CUR_PATH = os.path.dirname(__file__)

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)

def preprocess(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X = utils.preprocess.datetimes(X)
    X = utils.preprocess.remove_features(X)
    return X

def imputes(X: pd.DataFrame) -> pd.DataFrame:
    mid = utils.imputers.MarketIDImputer()
    pci = utils.imputers.PrimaryCategoryImputer()
    X = mid.fit_transform(X)
    X = pci.fit_transform(X)
    return X

if __name__ == '__main__':
    data = load_data(CUR_PATH + config.TEST_FILE)
    data = preprocess(data)
    data = imputes(data)
    model = lgb.Booster(model_file=CUR_PATH + config.MODEL_FILE)
    encoder = utils.encoders.StorePrimaryEncoder()
    data = encoder.fit_transform(data)
    y = model.predict(data)
    print(f'First five predictions: {y[:5]}')
    output_filename = input('Enter your desire output filename:')
    np.savetxt(output_filename + '.csv', y, delimiter=',')
    print(f'{output_filename}.csv sucessfully saved.')