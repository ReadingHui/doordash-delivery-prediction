import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin

def data_summary(df:pd.DataFrame, index_col: list[str] = None, target: str = None, verbose: bool = True) -> dict:
    df = df.copy()
    if index_col:
        df = df.drop(index_col, axis=1)
    if target:
        df = df.drop(target, axis=1)
    num_col = df.select_dtypes(include='number').columns.to_list()
    cat_col = df.select_dtypes(include='category').columns.to_list()
    misc_col = [c for c in df.columns if ((c not in num_col) and (c not in cat_col))]
    col_with_nan = df.columns[df.isnull().any()].to_list()
    info = {
        'num_col': num_col,
        'cat_col': cat_col,
        'misc_col': misc_col,
        'col_with_nan': col_with_nan,
        'na_count': pd.Series({feature: df[feature].isnull().sum() for feature in col_with_nan}).to_frame(name='na_values_count')
    }

    if verbose:    
        if num_col:
            print(f'Potential numerical features are: {num_col}')
        else:
            print('No potential numerical feature')
        if cat_col:
            print(f'Potential categorical features are: {cat_col}')
        else:
            print('No potential categorical feature')
        if misc_col:
            print(f'Miscellenous features are: {misc_col}')
        if col_with_nan:
            print(f'Columns with missing values are: {col_with_nan}')
        else:
            print(f'There is no missing value')
    


    return info

class FeatureTarget:
    def feature_target_split(df: pd.DataFrame, target: str, col_drop: list[str] = []):
        y = df[target]
        mask = col_drop + [target]
        X = df.drop(columns=mask, axis=1)
        return X, y

class Plots:
    def distribution_plots(df:pd.DataFrame, rotate: bool = False) -> None:
        # Set the Seaborn style
        sns.set(style="whitegrid")

        # Define the plot size and the number of rows and columns in the grid
        num_plots = len(df.columns)
        rows = (num_plots + 1) // 2  # Calculate the number of rows needed (two plots per row)
        cols = 2  # Two plots per row
        _, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8 * cols, 6 * rows))

        # Iterate through the numerical features and create the density plots
        for i, feature_name in enumerate(df.columns):
            row_idx, col_idx = divmod(i, cols)  # Calculate the current row and column index
            cur_axes = axes[row_idx, col_idx]
            first_30 = df.dropna(subset=feature_name)[feature_name].value_counts()  # Count the appearances
            first_30 = first_30.iloc[:30].index.sort_values()   # Take the highest 30 and sort by index value
            if df[feature_name].dtype in [np.int32, np.int64, pd.Int64Dtype()] or pd.api.types.is_object_dtype(df[feature_name]):
                if len(df[feature_name].unique()) < 30:
                    sns.countplot(df.dropna(subset=feature_name), x=feature_name, ax=cur_axes, color='darkcyan')
                    cur_axes.set_title(f'Count Plot of {feature_name}')
                else:
                    sns.countplot(df.dropna(subset=feature_name), x=feature_name, ax=cur_axes, color='darkcyan', order=first_30)
                    cur_axes.set_title(f'Count Plot of {feature_name} (Top 30)')
                cur_axes.set_xlabel(feature_name)
                cur_axes.set_ylabel('Count')
                cur_axes.bar_label(cur_axes.containers[0])
            else:
                sns.histplot(df.dropna(subset=feature_name), x=feature_name, kde=True, ax=cur_axes)
                cur_axes.set_title(f'Density Plot of {feature_name}')
                cur_axes.set_xlabel(feature_name)
                cur_axes.set_ylabel('Density')
            if rotate == True or (type(rotate) == list and feature_name in rotate):
                cur_axes.tick_params(axis='x', rotation=270)
        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plots

        plt.show()

class Correlations:
    def features_corr(df: pd.DataFrame, num_features: list[str]) -> None:
        df = df[num_features]
        cmap = sns.color_palette("light:b", as_cmap=True)
        sns.heatmap(df.corr().abs(), cmap=cmap,
                square=True, linewidths=.5, annot=True)
        plt.show()

    def cat_target_corr(data:pd.DataFrame, target: str) -> None:
        df = data.drop(target, axis=1)
        num_features = df.columns[(df.dtypes == 'int64') | (df.dtypes == 'float64')].to_list()
        cat_features = df.columns[df.dtypes == 'object'].to_list()
        features = num_features + cat_features

        # Set the Seaborn style
        sns.set(style="whitegrid")

        # Define the plot size and the number of rows and columns in the grid
        num_plots = len(features)
        rows = (num_plots + 1) // 2  # Calculate the number of rows needed (two plots per row)
        cols = 2  # Two plots per row
        _, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8 * cols, 6 * rows))

        # Iterate through the numerical features and create the density plots
        for i, feature_name in enumerate(features):
            row_idx, col_idx = divmod(i, cols)  # Calculate the current row and column index
            cur_axes = axes[row_idx, col_idx]
            if feature_name in num_features:
                sns.histplot(data=data, x=feature_name, hue=target, kde=True, multiple='stack', ax=cur_axes)
                cur_axes.set_title(f'Distribution Plot of {feature_name} subject to {target}')
                cur_axes.set_xlabel(feature_name)
                cur_axes.set_ylabel('Density')
            elif feature_name in cat_features:
                sns.countplot(data=data, x=feature_name, hue=target, ax=cur_axes)
                cur_axes.set_title(f'Count Plot of {feature_name} subject to {target}')
                cur_axes.set_xlabel(feature_name)
                cur_axes.set_ylabel('Count')
                cur_axes.bar_label(cur_axes.containers[0])
        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plots

        plt.show()

class ColumnTransformers:
    class DatatimeConverter(BaseEstimator, TransformerMixin):
        # Support single format or different format in list/dict type

        def __init__(self, columns=None, format=None):
            self.columns = columns
            self.format = format
        
        def fit(self, X, y=None):
            if type(self.format) == str:
                self.format = {col: self.format for col in self.columns}
            if type(self.format) == list:
                if len(self.columns) != len(self.format):
                    raise IndexError('Length of columns does not match length of format')
                self.format = {self.columns[i]: self.format[i] for i in range(len(self.columns))}
            return self
        
        def transform(self, X, y=None):
            X = X.copy()
            for col in self.columns:
                X[col] = pd.to_datetime(X[col], format=self.format[col])
            return X

    class IntFloatConverter(BaseEstimator, TransformerMixin):    
        # Convert int columns to float columns and vice versa
        def __init__(self, int_to_float=None, float_to_int=None):
            self.int_to_float = int_to_float
            self.float_to_int = float_to_int
        
        def fit(self, X, y=None):
            int_err = []
            float_err = []
            for col in self.int_to_float:
                if X[col].dtype not in ['int32', 'int64', 'Int32', 'Int64']:
                    int_err.append(col)
            for col in self.float_to_int:
                if X[col].dtype not in ['float32', 'float64']:
                    float_err.append(col)
            if len(int_err) > 0:
                raise TypeError(f'Columns {int_err} are not integer dtype.')
            if len(float_err) > 0:
                raise TypeError(f'Columns {float_err} are not float dtype.')
            return self
        
        def transform(self, X, y=None):
            X = X.copy()
            X[self.int_to_float] = X[self.int_to_float].astype('float64')
            X[self.float_to_int] = X[self.float_to_int].astype('Int64')
            return X