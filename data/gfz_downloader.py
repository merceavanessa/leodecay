import os
import logging
import requests
from tqdm import tqdm
import pandas as pd

class GFZDownloader:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.logger = logging.getLogger(__name__)

    def download_kp(self, start_date, end_date):
        self.logger.info("Downloading Kp data.")
        base_url = "https://datapub.gfz-potsdam.de/download/10.5880.GFZ.2.3.{YEAR}.001/kp_{YEAR}.wdc"
        years = range(start_date.year, end_date.year + 1)
        for year in years:
            url = base_url.format(YEAR=year)
            file_name = f"kp_{year}.wdc"
            self._download_file(url, file_name)

    def download_hp(self, start_date, end_date, index):
        self.logger.info(f"Downloading {index} data.")
        url = f"https://kp.gfz.de/app/json/?start={start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}&end={end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}&index={index}&status=def"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if not data.get(index):
                self.logger.warning(f"No {index} data found for the specified period.")
                return None

            df = pd.DataFrame({
                'datetime': pd.to_datetime(data['datetime']),
                index: data[index]
            })
            df.set_index('datetime', inplace=True)
            return df

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download {index} data. Error: {e}")
        except (ValueError, KeyError) as e:
            self.logger.error(f"Failed to parse {index} data. Error: {e}")
        return None


    def _download_file(self, url, file_name):
        output_path = os.path.join(self.output_folder, file_name)
        if os.path.exists(output_path):
            self.logger.info(f"File {file_name} already exists. Skipping download.")
            return

        self.logger.info(f"Downloading {file_name} from {url}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(output_path, 'wb') as f, tqdm(
                desc=file_name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
            self.logger.info(f"Successfully downloaded {file_name} to {output_path}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download {file_name}. Error: {e}")

    def download_hp_combined(self, start_date, end_date, dataset):
        indices = ['Hp30', 'Hp60']
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        self.logger.info(f"Downloading {indices} data.")
        df_list = []
        for index in indices:
            df = self.download_hp(start_date, end_date, index)
            if df is not None:
                df_list.append(df)
        
        if not df_list:
            self.logger.warning("No data downloaded for any of the indices.")
            return

        combined_df = pd.concat(df_list, axis=1)
        
        output_path = os.path.join(self.output_folder, f"{dataset}.csv")
        combined_df.to_csv(output_path)
        self.logger.info(f"Successfully downloaded and saved combined GFZ data to {output_path}")
