from abc import ABC, abstractmethod


class DataDownloader(ABC):
    """Base class for all data downloaders."""

    @abstractmethod
    def load_data(self, folderpath, dataset=None):
        """Load data and save to folderpath."""
        pass

    @abstractmethod
    def fetch_data(self):
        """Fetch data from the source."""
        pass

class QueryBasedAPIDownloader(DataDownloader):
    """Base class for query-based HTTP APIs that accept parameters in URLs."""
    
    @abstractmethod
    def build_query_url(self, **kwargs):
        """Build a query URL with the given parameters."""
        pass


class FileDownloader(DataDownloader):
    """Base class for static file downloads via HTTP."""
    
    @abstractmethod
    def download_file(self, url, file_name):
        """Download a file from URL and save to output folder."""
        pass


class JSONAPIDownloader(DataDownloader):
    """Base class for JSON APIs that return structured data."""
    
    @abstractmethod
    def query_api(self, **kwargs):
        """Query the API and return parsed JSON data."""
        pass
