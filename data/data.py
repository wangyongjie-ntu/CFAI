"""Module pointing to different implementations of Data class

DiCE requires only few parameters about the data such as the range of continuous features and the levels of categorical features. Hence, DiCE can be used for a private data whose meta data are only available (such as the feature names and range/levels of different features) by specifying appropriate parameters.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

class PublicData:
    """A data interface for public data."""

    def __init__(self, **params):

        """Init method

        :param dataframe: Pandas DataFrame.
        :param target: Outcome feature name.
        :param permitted_range (optional): Dictionary with feature names as keys and permitted range in list as values. Defaults to the range inferred from training data.

        """

        if isinstance(params['dataframe'], pd.DataFrame):
            self.data_df = params['dataframe']
        else:
            raise ValueError("should provide a pandas dataframe")

        if type(params['target']) is str:
            self.outcome_name = params['target']
        else:
            raise ValueError("should provide the name of outcome feature")

        self.all_feature_names = self.data_df.columns[0:-1].to_list()
        self.train_df, self.test_df = self.split_data(self.data_df)
        self.features = self.train_df.iloc[:, 0:-1].values
        self.targets = self.train_df.iloc[:, -1:].values

        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.norm_features = self.feature_scaler.fit_transform(self.train_df.iloc[:, 0:-1])
        self.norm_targets = self.target_scaler.fit_transform(self.train_df.iloc[:, -1:])

        if 'permitted_range' in params: 
            self.permitted_range = params['permitted_range']
            if not self.check_features_range():
                raise ValueError(
                    "permitted range of features should be within their original range")

    def check_features_range(self):
        for feature in self.all_feature_names:
            if feature in self.permitted_range:
                min_value = self.train_df[feature].min()
                max_value = self.train_df[feature].max()

                if self.permitted_range[feature][0] < min_value and self.permitted_range[feature][1] > max_value:
                    return False
            else:
                self.permitted_range[feature] = [self.train_df[feature].min(), self.train_df[feature].max()]
        return True

    def normalize_data(self, df):
        """Normalizes continuous features to make them fall in the range [0,1]."""
        return self.feature_scaler.transform(df)

    def de_normalize_data(self, df):
        """De-normalizes continuous features from [0,1] range to original range."""
        return self.feature_scaler.inverse_transform(df)

    def normalize_target(self, target):
        return self.target_scaler.transform(target)

    def de_normalize_target(self, target):
        return self.target_scaler.inverse_transform(target)

    def split_data(self, data):
        train_df, test_df = train_test_split(
            data, train_size = 5760, test_size = 2016, shuffle = False)
        return train_df, test_df

    def get_mads(self, normalized = True):
        """Computes Median Absolute Deviation of features."""

        if not normalized:
            self.mads = np.median(self.features - np.median(self.features, 0), 0)
        else:
            self.mads = np.median(self.norm_features - np.median(self.norm_features, 0), 0)

    def get_stds(self, normalized = True):
        """Computes the standard deviation of features."""

        if not normalized:
            self.stds = np.std(self.features)
        else:
            self.stds = np.std(self.norm_features)

    def get_minx_maxx(self, normalized = True):
        """Computes max/min of features."""
        if not normalized:
            self.minx, self.maxx =  self.features.min(0), self.features.max(0)
        else:
            self.minx, self.maxx =  self.norm_features.min(0), self.norm_features.max(0)

    def get_mask_of_features_to_vary(self, features_to_vary='all'):
        """Gets indexes from feature names of one-hot-encoded data."""

        mask = np.ones(len(self.all_feature_names)) 
        if features_to_vary == "all":
            return mask
        else:
            for i  in range(len(self.all_feature_names)):
                if self.all_feature_names[i] not in features_to_vary:
                    mask[i] = 0
            return mask

    def get_weights_of_features_to_vary(self, feature_weights):
        """Gets predefined weights."""

        weights = np.zeros(len(self.all_feature_names)) 
        for key, value in feature_weights.items():
            idx = self.all_feature_names.index(key)
            weights[idx] = float(value)

        return weights
    def get_quantiles_from_training_data(self, quantile = 0.05, normalized = True):
        
        quantile = np.zeros(len(self.all_feature_names))
        if normalized:
            quantile = [np.quantile(abs(list(set(self.norm_features[i].tolist())) - np.median(list(set(self.norm_features[i].tolist())))), quantile) for i in self.norm_features]
            return quantile

        else:
            quantile = [np.quantile(abs(list(set(self.norm_features[i].tolist())) - np.median(list(set(self.norm_features[i].tolist())))), quantile) for i in self.norm_features]
            return quantile
