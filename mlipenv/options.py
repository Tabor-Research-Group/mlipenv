from dataclasses import dataclass
from typing import List, Union

STRUCTURE_PATH_KEYS=["structures", "structure_path", "xyz_files", "xyz", "system_data"]

CONFIGURATION_REGISTRY={}
def register_configuration(key, config_factory=None):
    if config_factory is None:
        def register(config_factory):
            return register_configuration(key, config_factory)
        return register
    else:
        CONFIGURATION_REGISTRY[key] = config_factory
        return config_factory

@dataclass
@register_configuration("calculator")
class CalculatorConfiguration:
    device: str

@dataclass
@register_configuration("mace")
class MACECalculatorConfiguration(CalculatorConfiguration):
    model_path: str = None
    mace_calculator: str = None

@dataclass
@register_configuration("aimnet")
class AIMNetCalculatorConfiguration(CalculatorConfiguration):
    model_path: str

@dataclass
@register_configuration("base")
class BaseConfiguration:
    method: str
    atoms: str
    coordinates: str
    charge: Union[float, List[float]]
    spin: float
    output_dir: str
    
@dataclass
@register_configuration("optimization")
class OptimizationOptions:
    optimizer: str
    # output: Union[str, Tuple[str]] = tuple(_output_file_registry().keys())
    fmax: float = 0.02
    steps: int = 5

@dataclass
@register_configuration("energy")
class EnergyOptions:
    order: int = 1

def get_configuration(config_type: str):
    return CONFIGURATION_REGISTRY[config_type]
