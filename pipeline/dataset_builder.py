import logging

import pandas as pd
import numpy as np

class DatasetBuilder:
    def __init__(self, data_config=None):
        if data_config is None:
            raise ValueError("DataConfig must be provided.")

        self.data_config = data_config

        self.df = None
        self.X = None
        self.y = None

        self.logger = logging.getLogger(__name__)

    def load_data(self):
        self.df = pd.read_csv(self.data_config.data_path, index_col='time')
        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.df.sort_index()
        self.df = self.df[~self.df.index.duplicated(keep='first')]

        if self.data_config.columns_to_keep:
            self.X = self.df[self.data_config.columns_to_keep].copy()
        else:
            self.X = self.df.copy()

        self.X = self.X.drop(columns=[self.data_config.target_column])
        self.y = self.df[[self.data_config.target_column]].copy()

        self.y = self.y.shift(-self.data_config.lag_config.default_lag_in_minutes * 2)
        self.y = self.y.bfill().ffill()

        if self.X.isna().sum().sum() > 0:
            print(f"NaNs found in X after loading data from {self.data_config.data_path} in columns: {self.X.columns[self.X.isna().any()].tolist()}. Ffilling + bffilling.")
            self.X.ffill(inplace=True)
            self.X.bfill(inplace=True)

        if self.y.isna().sum().sum() > 0:
            raise ValueError(f"NaNs found in y after loading data from {self.data_config.data_path}")

    def preprocess_data(self, train_size,
                        detrend=True, combine_f10_f30=False, use_lagged_inputs=False, input_lags_in_minutes=None, use_time_feature=False, inputs_blacklisted_from_lagging=None):
        if combine_f10_f30:
            if 'F10.7 (LASP)' in self.X.columns and 'F30 (LASP)' in self.X.columns:
                self.logger.info("Combining F10.7 and F30 into a single feature 'F10.7 + F30'")
                self.X['F10.7 + F30'] = self.X['F10.7 (LASP)'] + self.X['F30 (LASP)']
                self.X.drop(columns=['F10.7 (LASP)', 'F30 (LASP)'], inplace=True)

        if detrend:
            for col in self.X.columns:
                self.X[f'{col}_diff'] = self.X[col].diff().rolling(5 * 2).mean().bfill().ffill()
            self.logger.info("detrending complete.")

        train_var = self.X.iloc[:int(train_size)].var()
        cols_to_drop = train_var[train_var == 0].index
        self.X = self.X.drop(columns=cols_to_drop)
        if len(cols_to_drop) > 0:
            self.logger.info(f"Dropped columns with zero variance during training period: {cols_to_drop.tolist()}")

        if use_lagged_inputs:
            lagged_features = []
            if not input_lags_in_minutes:
                self.logger.error("input_lags_in_minutes must be provided if use_lagged_inputs is True.")
                raise ValueError("input_lags_in_minutes must be provided if use_lagged_inputs is True.")

            for lag in input_lags_in_minutes:
                lagged = self.X[self.X.columns].shift(lag * 2).add_suffix(f"_{lag}")

                lagged = lagged[[col for col in lagged.columns if col not in (inputs_blacklisted_from_lagging or [])]]
                lagged_features.append(lagged)

            self.X = pd.concat([self.X] + lagged_features, axis=1).bfill()

        if use_time_feature:
            self.X['t'] = np.arange(len(self.X))
            self.logger.info("Added time feature 't'.")

    def get_data(self):
        return self.X, self.y
