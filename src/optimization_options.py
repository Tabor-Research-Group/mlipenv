from dataclasses import dataclass
from typing import List, Union, Dict, Tuple

from src.enums.output_enum import _output_file_registry

@dataclass
class OptimizerConfiguration:
    logging: str = ""
    output: Union[str, Tuple[str]] = tuple(_output_file_registry().keys())

@dataclass
class ASEOptimizerConfiguration(OptimizerConfiguration):
    fmax: float = 0.02
    steps: int = 5

@dataclass
class OptimizationConfiguration:
    optimizer: str
    options: Union[Dict, OptimizerConfiguration]
    charge: Union[float, List[float]]
    spin: float

@dataclass
class BaseConfiguration:
    method: str
    options: Union[Dict, OptimizationConfiguration]
    atoms: str
    coordinates: str
    output_dir: str
