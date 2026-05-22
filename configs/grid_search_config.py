from dataclasses import dataclass
from typing import List

import re
import numpy as np

def get_numeric_alphas_instance_from_str(alphas_list, alpha_range_str):
    match = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", alpha_range_str)
    start, stop, num = map(float, match[:3])
    num = int(num)

    target = next(
        a for a in alphas_list
        if np.isclose(a['start'], start) and np.isclose(a['stop'], stop) and a['num'] == num
    )

    alpha_numeric = np.logspace(target['start'], target['stop'], target['num'], base=target['base'])

    return alpha_numeric

@dataclass
class GridSearchConfig:
    alphas: dict[str, List[float]]
    ks: List[int]
    n_splits: List[int]
    tols: List[float]
    train_size: int
    test_size: int
    offset: int
    use_lagged_inputs: bool
    detrend: bool
    use_time_feature: bool
    max_iter: int
    target: str
    cols_train: List[str]
    whitelisted_features: List[str]
    target_lags_in_minutes: List[int]
    input_lags_in_minutes: List[int]
    dataset_path: str = None
    inputs_blacklisted_from_lagging: List[str] = None
    grid_notes: str = ""

    @classmethod
    def from_dict(cls, cfg_dict: dict, dataset_path: str):

        whitelisted_features = []
        if cfg_dict['use_lagged_inputs']:
            whitelisted_features += [f"{f}_{lag}" for f in cfg_dict['whitelisted_features'] for lag in
                                     cfg_dict['input_lags_minutes']]

        alphas = {
            f"np.logspace({r['start']}, {r['stop']}, {r['num']})":
                np.logspace(r['start'], r['stop'], r['num'])
            for r in cfg_dict["alphas"]
        }

        return cls(
            alphas=alphas,
            ks=cfg_dict["ks"],
            n_splits=cfg_dict["n_splits"],
            tols=cfg_dict["tols"],
            train_size=cfg_dict["train_size"],
            test_size=cfg_dict["test_size"],
            offset=cfg_dict["offset"],
            use_lagged_inputs=cfg_dict["use_lagged_inputs"],
            use_time_feature=cfg_dict["use_time_feature"],
            detrend=cfg_dict["detrend"],
            max_iter=cfg_dict["max_iter"],
            target=cfg_dict["target"],
            cols_train=cfg_dict["cols_train"],
            whitelisted_features=whitelisted_features,
            target_lags_in_minutes=cfg_dict["target_lags_minutes"],
            input_lags_in_minutes=cfg_dict["input_lags_minutes"],
            dataset_path=dataset_path
        )