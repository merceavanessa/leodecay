from urllib.error import HTTPError
import pandas as pd
import logging
from datetime import datetime, timedelta
from ..data.data_downloader import QueryBasedAPIDownloader


class LatisLoader(QueryBasedAPIDownloader):
    def __init__(self, folder_path, dataset, start_date, end_date):
        self.dataset = dataset
        self.start_date = start_date
        self.end_date = end_date
        self.folder_path = folder_path
        self.start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        self.logger = logging.getLogger(__name__)

    def build_query_url(self, dataset, suffix="csv", projections=None, min_time="", max_time="", operations=None, is2=False):
        """Build a query URL for the LATIS API."""
        if projections is None:
            projections = []
        if operations is None:
            operations = []
        return self.latis_query_url(dataset=dataset, suffix=suffix, projections=projections, 
                                     min_time=min_time, max_time=max_time, operations=operations, is2=is2)

    def latis_query_url(self, dataset, suffix="csv", projections=None, min_time="", max_time="", operations=None, is2=False):
        """Build a LATIS query URL with the given parameters."""
        if projections is None:
            projections = []
        if operations is None:
            operations = []
        base = f"https://lasp.colorado.edu/space-weather-portal/latis/dap{'2' if is2 else ''}/"
        query = base + dataset + "." + suffix + "?"
        if projections:
            query += ",".join(projections)
        if min_time:
            query += "&time>=" + min_time  # note: could use "time>"
        if max_time:
            query += "&time<" + max_time   # note: could use "time<="
        if operations:
            query += "&" + "&".join(operations)
        return query

    def load_data(self, folderpath=None, dataset=None):
        """Load data from LATIS query-based API."""
        if folderpath:
            self.folder_path = folderpath
        if dataset:
            self.dataset = dataset
            
        solar_datasets = ['penticton_radio_flux', 'cls_radio_flux_f30']
        solar_indices_data = self.fetch_data(
            solar_datasets)
        # log first and last date
        self.logger.info(f"Solar data range: {solar_indices_data.index[0]} to {solar_indices_data.index[-1]}")
        geomagnetic_datasets = ['potsdam_kp', 'potsdam_ap']
        geomagnetic_indices_data = self.fetch_data(
            geomagnetic_datasets, drop_cols=['enum_index'])
        self.logger.info(f"Geomagnetic data range: {geomagnetic_indices_data.index[0]} to {geomagnetic_indices_data.index[-1]}")
        combined_df_name = self.combine_dataframes(
            solar_indices_data, geomagnetic_indices_data)
        return combined_df_name

    def _fetch_datasets(self, datasets, time_format="yyyy-MM-dd'T'HH:mm:ss", drop_cols=None):
        """Fetch data from LATIS API for multiple datasets."""
        if drop_cols is None:
            drop_cols = []
        data = None

        for dataset in datasets:
            try:
                min_time = f"{self.start_date}T00:00:00"
                max_time = f"{(self.end_date_dt + timedelta(days=1)).strftime('%Y-%m-%d')}T00:00"
            
                query = self.latis_query_url(dataset, "csv", min_time=min_time, max_time=max_time, operations=[f'formatTime({time_format})'])
                self.logger.info(f"Query URL: {query}")
                dataset_data = pd.read_csv(query, parse_dates=True, index_col=[0], date_format='%Y-%m-%dT%H:%M:%S')
                dataset_data.index = dataset_data.index.round('h')

                dataset_data = dataset_data.select_dtypes(include=['float64', 'int64'])
                dataset_data = dataset_data.resample("10S").mean()
                self.logger.info(f"Dataset range: {dataset_data.index[0]} to {dataset_data.index[-1]}")
            except HTTPError as e:
                self.logger.error(f'Failed to download data for {dataset}: {e}')
                continue

            data = dataset_data if data is None else data.merge(dataset_data, on=f'time ({time_format})', how='left') 
            self.logger.info(f'Successfully fetched {dataset}')

        for col in drop_cols:
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)

        return data

    def fetch_data(self, datasets=None, time_format="yyyy-MM-dd'T'HH:mm:ss", drop_cols=None):
        """Fetch data from LATIS API for multiple datasets."""
        return self._fetch_datasets(datasets, time_format, drop_cols)
    
    def combine_dataframes(self, solar_indices_data=None, geomagnetic_indices_data=None):
        """Combine solar and geomagnetic data and save to CSV."""
        combined_df = solar_indices_data.join(geomagnetic_indices_data, how='right')
        combined_csv_filename = f"{self.folder_path}/{self.dataset}.csv"
        combined_df.to_csv(combined_csv_filename, index_label = 'time')
        self.logger.info(f"Combined data saved. Date range: {combined_df.index[0]} to {combined_df.index[-1]}")

        return combined_csv_filename