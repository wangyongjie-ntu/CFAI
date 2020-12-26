#Filename:	Dataset.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 20 Des 2020 03:36:28  WIB

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats

class Dataset(object):

    """ A data interface for public dataset. """
    def __init__(self, **params):
        """Init Method

        :param dataframe: Pandas Dataframe.
        :param continuous_features: List of names of continous features. The remained features are categorical features.
        :param outcome_name: Outcome feature name.
        :param scaler: The scale type(MinMaxScaler, StandardScaler etc.).
        :param test_size(optional): the proportion of test set split. defaults to 0.2.
        :param random_state: the random state for train_test_split.
        :param custom_preprocessing(optinal): the preprocessing method provided by users.
        :param data_name(optinal): Dataset name.

        """

        if isinstance(params['dataframe'], pd.DataFrame):
            self.data_df = params['dataframe']
        else:
            raise ValueError('The dataframe is not provided')

        if type(params['outcome_name']) is str:
            self.outcome_name = params['outcome_name']
        else:
            raise ValueError('The outcome feature is not provided')

        if 'data_name' in params:
            self.data_name = params['data_name']
        else:
            self.data_name = 'unknown'

        if 'custom_preprocessing' in params:
            self.df = params['custom_preprocessing'](self.data_df.copy())

        if type(params['continuous_features']) is list:
            self.continuous_features_names = params['continuous_features']
        else:
            raise ValueError('The continuous_features is not provided')

        self.categorical_feature_names = [name for name in self.data_df.columns.tolist()
                if name not in self.continuous_features_names + [self.outcome_name]]
        self.feature_names = [name for name in self.data_df.columns.tolist() if name != self.outcome_name]

        self.continuous_features_indices = [self.data_df.columns.get_loc(name) for name in self.continuous_features_names]
        self.categorical_feature_indices = [self.data_df.columns.get_loc(name) for name in self.categorical_feature_names]
        self.outcome_index = [self.data_df.columns.get_loc(self.outcome_name)]

        if 'test_size' in params:
            self.test_size = params['test_size']
        else:
            self.test_size = 0.2

        if 'random_state' in params:
            self.random_state = params['random_state']
        else:
            self.random_state = 0
    
        
        for feature in self.categorical_feature_names:
            self.data_df[feature] = self.data_df[feature].apply(str)
            self.data_df[feature] = self.data_df[feature].astype('category')

        for feature in self.continuous_features_names:
            if self.data_df[feature].dtype == np.float64 or self.data_df[feature].dtype == np.float32:
                self.data_df[feature] = self.data_df[feature].astype(np.float32)
            else:
                self.data_df[feature] = self.data_df[feature].astype(np.int32)

        
        if len(self.categorical_feature_names) > 0:
            one_hot_encoded_data = pd.get_dummies(data = self.data_df, columns = self.categorical_feature_names)
            self.one_hot_encoded_names  = one_hot_encoded_data.columns.tolist()
            self.one_hot_encoded_names.remove(self.outcome_name)
        else:
            one_hot_encoded_data = self.data_df
            self.one_hot_encoded_names = self.feature_names
        
        self.encoded_categorical_feature_indices = self.get_encoded_categorial_feaure_indices()
        # The column name is reordered after one-hot encoding.
        one_hot_x = one_hot_encoded_data[self.one_hot_encoded_names].values
        one_hot_y = one_hot_encoded_data[self.outcome_name].values

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(one_hot_x, one_hot_y, test_size = self.test_size, random_state = self.random_state)
        
         
        # scale the raw data
        if 'scaler' in params:
            self.scaler = params['scaler']
            self.scaler.fit(self.train_x)
            self.train_scaled_x, self.test_scaled_x = self.scaler.transform(self.train_x.astype(np.float32)), self.scaler.transform(self.test_x.astype(np.float32))

        if len(self.continuous_features_names) > 0:
            self.permitted_range_before_scale = [self.train_x[:, list(range(len(self.continuous_features_names)))].min(0),
                    self.train_x[:, list(range(len(self.continuous_features_names)))].max(0)]
            self.permitted_range_after_scale = [self.train_scaled_x[:, list(range(len(self.continuous_features_names)))].min(0),
                    self.train_scaled_x[:, list(range(len(self.continuous_features_names)))].max(0)]

    def normalize_data(self, input_x):
        try:
            return self.scaler.transform(input_x.astype(np.float32)) # the scaler always returns the float64
        except:
            raise ValueError('scaler is not provided in normalization')

    def denormalize_data(self, input_x):
        try:
            return self.scaler.inverse_transform(input_x)
        except:
            raise ValueError('scaler is not provided in denormalization')

    def get_mads(self, normalized = False):

        if not normalized:
            self.mads = np.median(abs(self.train_x - np.median(self.train_x, 0)), 0)
        else:
            self.mads = np.median(abs(self.train_scaled_x - np.median(self.train_scaled_x, 0)), 0)

        # replace the 0 mad to 1. Do we need to compute the mads of normalized data? 
        idx = np.argwhere((self.mads - 0) < 1e-5)
        np.put_along_axis(self.mads[np.newaxis, ], idx, 1, 1)
        return self.mads

    def get_mask_of_features_to_vary(self, features_to_vary = ['all']):

        mask = np.ones(len(self.one_hot_encoded_names))
        if features_to_vary == ['all']:
            return mask
        else:
            for i in range(len(self.one_hot_encoded_names)):
                mask[i] = 0
                for feature in features_to_vary:
                    if self.one_hot_encoded_names[i].startswith(feature):
                        mask[i] = 1
                        break

            return mask

    def get_indices_of_features_to_vary(self, features_to_vary = ['all']):

        indices = []
        if features_to_vary == ['all']:
            indices = list(range(len(self.one_hot_encoded_names)))
        else:
            for i in range(len(self.one_hot_encoded_names)):
                for feature in features_to_vary:
                    if self.one_hot_encoded_names[i].startswith(feature):
                        indices.append(i)
                        break

        return indices

    def get_weight_of_features_to_vary(self, features_weights):
        """
        features_weigths: dictionary 
        """

        weights = np.ones(len(self.one_hot_encoded_names))
        for i in range(len(self.one_hot_encoded_names)):
            for key, value in features_weights.items():
                if self.one_hot_encoded_names[i].startswith(key):
                    weights[i] = float(value)

        return weights
    
    def onehot_decode(self, data, prefix_sep = '_'):
        """
        get the original dataframe from dummy onehot encoded data
        """

        if isinstance(data, np.ndarray):
            index = list(range(len(data)))
            data = pd.DataFrame(data = data, index = index, columns = self.one_hot_encoded_names)
        
        out = data.copy()
        for feat in self.categorical_feature_names:
            cat_col_values = []
            for val in list(self.data_df[feat].unique()):
                cat_col_values.append(feat + prefix_sep + str(val))

            match_cols = [c for c in data.columns if c in cat_col_values]
            cols, cat_values = [[c.replace(x, "") for c in match_cols] for x in ["", feat + prefix_sep]]
            out[feat] = pd.Categorical(np.array(cat_values)[np.argmax(data[cols].values, axis = 1)])
            out.drop(cols, axis = 1, inplace = True)
        
        # The columns are shuffled after one-hot encoding and decoding. Here we re-order the columns as the input dataframe.
        
        pairs = list(zip(self.continuous_features_indices + self.categorical_feature_indices, self.continuous_features_names + self.categorical_feature_names))
        sorted_pairs = sorted(pairs, key = lambda t:t[0])
        sorted_indices, sorted_columns = list(zip(*sorted_pairs))

        return out[list(sorted_columns)]

    def get_encoded_categorial_feaure_indices(self):

        cols = []
        for col_parent in self.categorical_feature_names:
            temp = [self.one_hot_encoded_names.index(col) for col in self.one_hot_encoded_names if col.startswith(col_parent) and col not in self.continuous_features_names]
            cols.append(temp)

        return cols
    
    def get_quantiles_from_data(self, quantile = 0.05, normalized = True):

        quantile = np.zeros(len(self.one_hot_encoded_names))
        if normalized:
            quantile = []
        else:
            quantile = []

        return quantile

    def compute_continuous_percentile_shift(self, source, target, normalized = False, method = 'sum'):

        countinous_shift = np.zeros(len(self.continuous_features_names))
        for i in range(len(self.continuous_features_names)):

            if normalized:
                source_percentile = stats.percentileofscore(self.train_scaled_x[:, i], source[:, i])
                target_percentile = stats.percentileofscore(self.train_scaled_x[:, i], target[:, i])
                countinous_shift[i] = np.abs(source_percentile - target_percentile)
            else:
                source_percentile = stats.percentileofscore(self.train_x[:, i], source[:, i])
                target_percentile = stats.percentileofscore(self.train_x[:, i], target[:, i])
                countinous_shift[i] = np.abs(source_percentile - target_percentile)
        
        if method == "sum":
            score = np.sum(countinous_shift)
        else:
            score = np.max(countinous_shift)

        return score

    def compute_categorical_changes(self, source, target):

        source, target = self.onehot_decode(source), self.onehot_decode(target)
        match = (source[self.categorical_feature_names] != target[self.categorical_feature_names]).values.sum(1)
        categorical_change = np.mean(match)
        return categorical_change

    def prepare_query(self, query_instance, normalized = False):
        
        if isinstance(query_instance, list):
            test = pd.DataFrame(query_instance, orient = 'index', columns = self.feature_names)
        elif isinstance(query_instance, dict):
            test = pd.DataFrame.from_records([query_instance])
        else:
            raise ValueError("unsupported data type of query_instance")
    
        tmp  = np.zeros((1, len(self.one_hot_encoded_names)))
        onehot_test = pd.DataFrame(tmp, columns = self.one_hot_encoded_names)

        for name, content in test.items():
            if content.dtype == np.float64 or content.dtype == np.float32:
                onehot_test[name] = test[name]
            elif content.dtype == np.int64 or content.dtype == np.int32:
                onehot_test[name] = test[name]
            else:
                onehot_name = name + "_" + content[0]
                onehot_test[onehot_name] = 1

        onehot_test = onehot_test.values
        if normalized:
            onehot_test = self.normalize_data(onehot_test)

        return onehot_test

