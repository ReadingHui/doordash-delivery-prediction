import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class feature_target:
    def feature_target_split(df: pd.DataFrame, target: str, col_drop=[]):
        y = df[target]
        mask = col_drop + [target]
        if col_drop:
            X = df.drop(columns=mask, axis=1)
        else:
            X = df.drop(columns=target, axis=1)
        return X, y

class plots:
    def distributionPlots(df:pd.DataFrame) -> None:
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
            if df[feature_name].dtype in [np.int32, np.int64, pd.Int64Dtype()] or pd.api.types.is_object_dtype(df[feature_name]):
                if len(df[feature_name].unique()) < 30:
                    sns.countplot(df.dropna(subset=feature_name), x=feature_name, ax=axes[row_idx, col_idx], color='darkcyan')
                    axes[row_idx, col_idx].set_title(f'Count Plot of {feature_name}')
                else:
                    sns.countplot(df.dropna(subset=feature_name), x=feature_name, ax=axes[row_idx, col_idx], color='darkcyan', order=df.dropna(subset=feature_name)[feature_name].value_counts().iloc[:30].index)
                    axes[row_idx, col_idx].set_title(f'Count Plot of {feature_name} (Top 30)')
                axes[row_idx, col_idx].set_xlabel(feature_name)
                axes[row_idx, col_idx].set_ylabel('Count')
                axes[row_idx, col_idx].bar_label(axes[row_idx, col_idx].containers[0])
            else:
                sns.histplot(df.dropna(subset=feature_name), x=feature_name, kde=True, ax=axes[row_idx, col_idx])
                axes[row_idx, col_idx].set_title(f'Density Plot of {feature_name}')
                axes[row_idx, col_idx].set_xlabel(feature_name)
                axes[row_idx, col_idx].set_ylabel('Density')
        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plots

        plt.show()

class correlations:
    def featuresCorr(df: pd.DataFrame, num_features: list) -> None:
        df = df[num_features]
        cmap = sns.color_palette("light:b", as_cmap=True)
        sns.heatmap(df.corr().abs(), cmap=cmap,
                square=True, linewidths=.5, annot=True)
        plt.show()

    def catTargetCorr(data:pd.DataFrame, target: str) -> None:
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
            if feature_name in num_features:
                sns.histplot(data=data, x=feature_name, hue=target, kde=True, multiple='stack', ax=axes[row_idx, col_idx])
                axes[row_idx, col_idx].set_title(f'Distribution Plot of {feature_name} subject to {target}')
                axes[row_idx, col_idx].set_xlabel(feature_name)
                axes[row_idx, col_idx].set_ylabel('Density')
            elif feature_name in cat_features:
                sns.countplot(data=data, x=feature_name, hue=target, ax=axes[row_idx, col_idx])
                axes[row_idx, col_idx].set_title(f'Count Plot of {feature_name} subject to {target}')
                axes[row_idx, col_idx].set_xlabel(feature_name)
                axes[row_idx, col_idx].set_ylabel('Count')
                axes[row_idx, col_idx].bar_label(axes[row_idx, col_idx].containers[0])
        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plots

        plt.show()