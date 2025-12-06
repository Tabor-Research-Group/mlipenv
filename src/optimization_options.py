from dataclasses import dataclass
from typing import List, Union, Dict, Tuple

from src.enums.output_enum import _output_file_registry

@dataclass
class OptimizationConfiguration:
    optimizer: str
    output: Union[str, Tuple[str]] = tuple(_output_file_registry().keys())
    fmax: float = 0.02
    steps: int = 5

@dataclass
class EnergyConfiguration:
    order: int = 1

@dataclass
class BaseConfiguration:
    method: str
    options: Union[Dict, OptimizationConfiguration, EnergyConfiguration]
    atoms: str
    coordinates: str
    charge: Union[float, List[float]]
    spin: float
    output_dir: str
