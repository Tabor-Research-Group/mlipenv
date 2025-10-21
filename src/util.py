from dataclasses import asdict

from src.optimization_options import *

# def configuration_builder(method, atoms, coordinates, output_dir, **kwargs):
def configuration_builder(method, 
                          atoms, 
                          coordinates, 
                          charge, 
                          spin, 
                          output_dir=".", 
                          **kwargs):
    if method == "optimize":
        options = build_optimization_config(**kwargs)
    elif method == "energy":
        options = build_energy_config(**kwargs)
    else:
        raise NotImplementedError
    return asdict(BaseConfiguration(method, 
                                    options, 
                                    atoms, 
                                    coordinates, 
                                    charge, 
                                    spin, 
                                    output_dir))

def build_optimization_config(optimizer, **kwargs):
    optimizer = optimizer.lower()
    if optimizer == "ase":
        config = ASEOptimizationConfiguration(optimizer, **kwargs)
    elif optimizer == "scipy":
        ...
    elif "mark" in optimizer:
        ...
    return config

def build_energy_config(**kwargs):
    ...
