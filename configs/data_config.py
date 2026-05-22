from dataclasses import dataclass
from typing import List
from .lag_config import LagConfig

@dataclass
class DataConfig:
    data_path: str
    columns_to_keep: List[str]
    target_column: str
    lag_config: LagConfig = None,

    def __post_init__(self):
        if not self.data_path or not self.target_column:
            raise ValueError("DataConfig must have a valid data_path, columns_to_keep, and target_column.")
