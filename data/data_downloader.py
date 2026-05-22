from abc import abstractmethod


class DataDownloader:

    @abstractmethod
    def load_data(self, folderpath, dataset=None):
        pass

    @abstractmethod
    def fetch_data(self):
       pass

    @abstractmethod
    def combine_dataframes(self):
        pass
