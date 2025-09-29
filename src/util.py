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

def build_optimization_config(type, charge="0", spin="1.0", **kwargs):
    if type.lower() == "ase":
        optimizer = ASEOptimizerConfiguration(type, **kwargs)
    elif type.lower() == "scipy":
        ...
    elif "mark" in type.lower():
        ...
    return OptimizationConfiguration(type, charge, spin, optimizer)

def build_energy_config(**kwargs):
    ...