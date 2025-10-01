from dataclasses import dataclass
from typing import List, Union, Dict

@dataclass
class OptimizerConfiguration:
    type: str

@dataclass
class ASEOptimizerConfiguration(OptimizerConfiguration):
    fmax: float = 0.02
    steps: int = 5
    logging: str = ""

@dataclass
class OptimizationConfiguration:
    optimizer: str
    options: Union[Dict, OptimizerConfiguration]
    charge: Union[float, List[float]]
    spin: float

@dataclass
class BaseConfiguration:
    method: str
    atoms: str
    coordinates: str
    output_dir: str
    options: Union[Dict, OptimizationConfiguration]
