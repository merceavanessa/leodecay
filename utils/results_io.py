import os
import pandas as pd
import numpy as np
import joblib

class ResultsIO:
    def __init__(self, results_base_path, satellite, target, most_recent_timestamp=None, score_type='no_val'):
        self.results_base_path = results_base_path
        self.satellite = satellite
        self.base_data_path_root = os.path.join(results_base_path, 'satellites', satellite, 'data')
        self.score_type = score_type
        self.base_data_path = os.path.join(self.base_data_path_root, score_type)
        self.most_recent_timestamp = most_recent_timestamp
        self.target = target
        model_files = [f for f in os.listdir(self.base_data_path) if f.startswith('model_') and self.most_recent_timestamp in f]

        if not model_files:
            self.model = None
        else:
            self.model = joblib.load(os.path.join(self.base_data_path, model_files[0]))

        shap_files = [f for f in os.listdir(self.base_data_path_root) if f.startswith('shap') and self.most_recent_timestamp in f]
        self.shap_files = {f.split('_')[1]: os.path.join(self.base_data_path_root, f) for f in shap_files}  # e.g., 'train', 'test'

    def _read_file(self, subdir, prefix):
        file_path = os.path.join(self.base_data_path, subdir, f'{prefix}_{self.most_recent_timestamp}.csv')
        df = pd.read_csv(file_path, index_col='time')
        df.index = pd.to_datetime(df.index)
        return df

    def _read_file_by_path(self, file_path):
        df = pd.read_csv(file_path, index_col='time')
        df.index = pd.to_datetime(df.index)
        return df

    def read(self, mode='train'):
        X, y, y_pred = None, None, None

        if mode == 'train':
            X = self._read_file('train', 'X')
            y = self._read_file('train', 'y')
            y_pred = self._read_file('train', 'y_pred')

        if mode == 'test':
            y = self._read_file('test', 'y')
            y_pred = self._read_file('test', 'y_pred')
            X = self._read_file('test', 'X') if os.path.exists(os.path.join(self.base_data_path, 'test', f'X_{self.most_recent_timestamp}.csv')) else None

        if mode == 'val':
            X = self._read_file('val', 'X')
            y = self._read_file('val', 'y')
            y_pred = self._read_file('val', 'y_pred')

        return X, y, y_pred

    def read_all(self):
        (X_train, y_train, y_pred_train) = self.read(mode='train')

        if self.score_type == 'with_val':
            X_val, y_val, y_pred_val = self.read(mode='val')

        X_test, y_test, y_pred_test = self.read(mode='test')

        if self.score_type == 'with_val':
            return (X_train, y_train, y_pred_train), (X_val, y_val, y_pred_val), (X_test, y_test, y_pred_test)
        else:
            return (X_train, y_train, y_pred_train), (X_test, y_test, y_pred_test)

    def get_scaled_X(self, mode='train'):
        data = self.read(mode=mode)
        if self.model is None:
            raise ValueError("No model loaded for scaling. The model must exist in the data folder.")

        transformer = self.model[:-1]
        X, _, _ = data
        X_scaled = pd.DataFrame(transformer.transform(X), index=X.index, columns=X.columns)
        return X_scaled

    def read_shap(self, mode='test'):
        if mode not in self.shap_files:
            raise FileNotFoundError(f"No SHAP file found for mode '{mode}'")
        shap_path = self.shap_files[mode]
        shap_values = np.load(shap_path, allow_pickle=True)
        return shap_values

    def read_results(self):
        summary_path = os.path.join(self.results_base_path, 'summary')
        summary_file = f'all_results_with_importance_{self.most_recent_timestamp}.pkl'
        full_summary_path = os.path.join(summary_path, summary_file)
        if not os.path.exists(full_summary_path):
            raise FileNotFoundError(f"No summary file found at {full_summary_path}")
        all_results = pd.read_pickle(full_summary_path)
        return all_results


