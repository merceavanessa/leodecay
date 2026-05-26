from urllib.error import HTTPError
import pandas as pd
from data.data_downloader import DataDownloader
from datetime import datetime, timedelta

class LatisLoader(DataDownloader):
    def __init__(self, folder_path, dataset, start_date, end_date):
        self.dataset = dataset
        self.start_date = start_date
        self.end_date = end_date
        self.folder_path = folder_path
        self.start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    def latis_query_url(self, dataset, suffix="csv", projections=[], min_time="", max_time="", operations=[], is2=False):
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

    def load_data(self):
        """Load data from Proxies source."""
        solar_datasets = ['penticton_radio_flux', 'cls_radio_flux_f30']
        solar_indices_data = self.fetch_data(
            solar_datasets)
        # print first and last date
        print(solar_indices_data.index[0], solar_indices_data.index[-1])
        geomagnetic_datasets = ['potsdam_kp', 'potsdam_ap']
        geomagnetic_indices_data = self.fetch_data(
            geomagnetic_datasets, drop_cols=['enum_index'])
        print(geomagnetic_indices_data.index[0], geomagnetic_indices_data.index[-1])
        combined_df_name = self.combine_dataframes(
            solar_indices_data, geomagnetic_indices_data)
        return combined_df_name

    def fetch_data(self, datasets, time_format="yyyy-MM-dd'T'HH:mm:ss", drop_cols=[]):
        data = None

        for dataset in datasets:
            try:
                min_time = f"{self.start_date}T00:00:00"
                max_time = f"{(self.end_date_dt + timedelta(days=1)).strftime('%Y-%m-%d')}T00:00"
            
                query = self.latis_query_url(dataset, "csv", min_time=min_time, max_time=max_time, operations=[f'formatTime({time_format})'])
                print(query)
                dataset_data = pd.read_csv(query, parse_dates=True, index_col=[0], date_format='%Y-%m-%dT%H:%M:%S')
                print('----------------')
                dataset_data.index = dataset_data.index.round('h')

                dataset_data = dataset_data.select_dtypes(include=['float64', 'int64'])
                dataset_data = dataset_data.resample("1H").mean()
                print(dataset_data.index[0], dataset_data.index[-1])
            except HTTPError as e:
                print(f'Failed to download data for {dataset}', e)
                continue

            data = dataset_data if data is None else data.merge(dataset_data, on=f'time ({time_format})', how='left') #.join(dataset_data, how='left')
            print(f'Fetched {dataset}')

        for col in drop_cols:
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)

        return data
    
    def combine_dataframes(self, solar_indices_data, geomagnetic_indices_data):
        combined_df = solar_indices_data.join(geomagnetic_indices_data, how='right')
        combined_csv_filename = f"{self.folder_path}/{self.dataset}.csv"
        combined_df.to_csv(combined_csv_filename, index_label = 'time')
        print(combined_df.index[0], combined_df.index[-1])

        return combined_csv_filename