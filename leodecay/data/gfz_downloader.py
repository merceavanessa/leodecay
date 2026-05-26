import os
import logging
import requests
from tqdm import tqdm
import pandas as pd
from ..data.data_downloader import DataDownloader


class GFZDownloader(DataDownloader):
    """Downloads GFZ data including Hp30/Hp60 (JSON API)."""
    
    def __init__(self, folder_path, dataset=None, start_date=None, end_date=None):
        """Initialize GFZ downloader.
        
        Args:
            folder_path: Output folder for downloaded files
            dataset: Dataset name (for filename)
            start_date: Start date (string or datetime)
            end_date: End date (string or datetime)
        """
        self.folder_path = folder_path
        self.dataset = dataset
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logging.getLogger(__name__)
        self._ensure_folder_exists()

    def _ensure_folder_exists(self):
        """Ensure output folder exists."""
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            self.logger.info(f"Created output folder: {self.folder_path}")

    def load_data(self, folderpath=None, dataset=None):
        """Load data from GFZ source."""
        if folderpath:
            self.folder_path = folderpath
            self._ensure_folder_exists()
        if dataset:
            self.dataset = dataset
        
        self.fetch_data()

    def fetch_data(self):
        """Fetch data from GFZ source."""
        self._download_hp_combined(["Hp30", "Hp60"])

    def _download_hp_combined(self, indices):
        """Download Hp30 and Hp60 data using JSONAPIDownloader pattern and combine."""
        self.logger.info(f"Downloading {indices} data from API.")
        df_list = []
        for index in indices:
            df = self._query_hp_api(index)
            if df is not None:
                df_list.append(df)
        
        if not df_list:
            self.logger.warning("No data downloaded for any of the indices.")
            return

        combined_df = pd.concat(df_list, axis=1)
        output_path = os.path.join(self.folder_path, 
                                   f"{self.dataset}.csv")
        combined_df.to_csv(output_path)
        self.logger.info(f"Successfully downloaded and saved combined GFZ data to {output_path}")

    def _download_file(self, url, file_name):
        """Download a file from URL and save to output folder."""
        output_path = os.path.join(self.folder_path, file_name)
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

    def _query_hp_api(self, index):
        """Query the GFZ Hp API for a specific index (Hp30 or Hp60)."""
        url = f"https://kp.gfz.de/app/json/?start={self.start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}&end={self.end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}&index={index}&status=def"
        
        self.logger.info(f"Querying API for {index}: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if not data.get(index):
                self.logger.warning(f"No {index} data found for the specified period.")
                return None

            df = pd.DataFrame({
                'time': pd.to_datetime(data['datetime']),
                index: data[index]
            })
            df.set_index('time', inplace=True)
            self.logger.info(f"Successfully fetched {index} data")
            return df

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to query {index} data. Error: {e}")
        except (ValueError, KeyError) as e:
            self.logger.error(f"Failed to parse {index} data. Error: {e}")
        return None
