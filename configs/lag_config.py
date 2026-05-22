from dataclasses import dataclass

@dataclass
class LagConfig:
    use_default: bool
    default_lag_in_minutes: int
    target_column: str
    use_immediate_target: bool = False
