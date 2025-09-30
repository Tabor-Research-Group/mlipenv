from dataclasses import dataclass

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
    type: str
    charge: float | list[float]
    spin: float
    optimizer: OptimizerConfiguration

@dataclass
class BaseConfiguration:
    method: str
    atoms: str
    coordinates: str
    output_dir: str
    options: OptimizationConfiguration
