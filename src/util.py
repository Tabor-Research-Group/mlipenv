from dataclasses import asdict

from src.optimization_options import *

# def configuration_builder(method, atoms, coordinates, output_dir, **kwargs):
def configuration_builder(method, atoms, coordinates, output_dir=".", **kwargs):
    if method == "optimize":
        options = build_optimization_config(**kwargs)
    elif method == "energy":
        options = build_energy_config(**kwargs)
    else:
        raise NotImplementedError
    return asdict(BaseConfiguration(method, atoms, coordinates, output_dir, options))

def build_optimization_config(optimizer, charge=0, spin=1, **kwargs):
    optimizer = optimizer.lower()
    if optimizer == "ase":
        config = ASEOptimizerConfiguration(**kwargs)
    elif optimizer == "scipy":
        ...
    elif "mark" in optimizer:
        ...
    return OptimizationConfiguration(optimizer, config, charge, spin)

def build_energy_config(**kwargs):
    ...