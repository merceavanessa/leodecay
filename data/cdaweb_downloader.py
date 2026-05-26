import pandas as pd
from cdasws import CdasWs
import glob
import os
import logging
from datetime import datetime, timedelta
from data.data_downloader import DataDownloader
import shutil

class CdaWebLoader(DataDownloader):
    def __init__(self, folder_path, dataset, start_date="1995-01-01", end_date="2024-03-03"):
        self.dataset = dataset.removesuffix(".csv")
        self.folder_path = folder_path
        self.start_date = start_date
        self.end_date = end_date
        self.start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    def load_data(self):
        """Load data from OMNI source."""
        if self.dataset:
            self.fetch_data()
            combined_df_name = self.combine_dataframes()
            logging.info(f"Combined data saved to {combined_df_name}")
        else:
            logging.warning("Please provide the dataset name")

    def fetch_data(self):
        cdas = CdasWs()

        # Combine all yearly CSV files into a single DataFrame
        if not os.path.exists(f"{self.folder_path}/{self.dataset}_intermediate"):
            os.mkdir(f"{self.folder_path}/{self.dataset}_intermediate")

        vars = list(map(lambda x: x['Name'], cdas.get_variables(self.dataset)))

        # if multi-year, download separately for each year
        if self.start_date_dt.year != self.end_date_dt.year:
            for year in range(self.start_date_dt.year, self.end_date_dt.year + 1):
                min_time = f"{year}-01-01T00:00:00.000Z"
                max_time = f"{year+1}-01-01T00:00:00.000Z"

                self.fetch_for_date(cdas, vars, year, min_time, max_time)
        else:
            min_time = f"{self.start_date}T00:00:00.000Z"
            max_time = f"{(self.end_date_dt + timedelta(days=1)).strftime('%Y-%m-%d')}T00:00:00.000Z"
            
            self.fetch_for_date(cdas, vars, self.start_date_dt.year, min_time, max_time)


    def fetch_for_date(self, cdas, vars, year, min_time, max_time):
        status, data = cdas.get_data(self.dataset, vars, min_time, max_time)

        if status['http']['status_code'] == 200 and len(status['cdas']['error']) == 0:
            df = pd.DataFrame(data)

            df['Epoch'] = pd.to_datetime(df['Epoch'])

            csv_filename = f"{self.folder_path}/{self.dataset}_intermediate/{self.dataset}_{year}.csv"
            df.to_csv(csv_filename, index=False)

            logging.info(f"Data for {year} saved to {csv_filename}")
        else:
            logging.error(
                        f"Failed to fetch data for {year}. Status: {status}")

    def combine_dataframes(self):
        all_files = glob.glob(os.path.join(
            f"{self.folder_path}/{self.dataset}_intermediate/", "*.csv"))

        dfs = []

        for csv_file in all_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values(by='Epoch', ascending=True)

        combined_df = combined_df[combined_df['Epoch'] <= self.end_date + 'T23:59:59.999Z']

        combined_csv_filename = f"{self.folder_path}/{self.dataset}.csv"
        combined_df.to_csv(combined_csv_filename, index=False)

        shutil.rmtree(f"{self.folder_path}/{self.dataset}_intermediate")

        return combined_csv_filename
