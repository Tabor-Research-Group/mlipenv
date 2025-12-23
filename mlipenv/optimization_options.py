from dataclasses import dataclass, field
from typing import List, Union, Dict, Tuple

from mlipenv.enums.output_enum import _output_file_registry

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
class CalculatorConfiguration:
    device: str = None

@dataclass
class MACECalculatorConfiguration(CalculatorConfiguration):
    model_path: str = None
    calculator: str = None

@dataclass
class AIMNetCalculatorConfiguration(CalculatorConfiguration):
    model_path: str = None

structure_path_keys = ["structure_path", "xyz_path"]
@dataclass
class BaseConfiguration:
    method: str
    options: Union[Dict, OptimizationConfiguration, EnergyConfiguration]
    atoms: str
    coordinates: str
    charge: Union[float, List[float]]
    spin: float
    output_dir: str
    calculator_options: Union[Dict, CalculatorConfiguration] = field(default_factory=dict)
