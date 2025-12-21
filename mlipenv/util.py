import os
from dataclasses import asdict
import logging

from mlipenv.optimization_options import *

logger = logging.getLogger(__name__)

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
        config = OptimizationConfiguration(optimizer, **kwargs)
    elif "better" in optimizer:
        config = OptimizationConfiguration(optimizer, **kwargs)
    return config

def build_energy_config(**kwargs):
    ...

def find_file(root, target):
    for dirpath, dirs, files in os.walk(root):
        if target in files:
            return os.path.join(dirpath, target)
    return None

def convert_from_xyz(file):
    import re
    with open(file, "r") as f:
        _ = int(f.readline().strip())
        f.readline()
        atoms, coordinates = zip(*[
            (m[0], m[1:]) for line in f.readlines() if (m := re.match(r'\s*([A-Za-z]+)\s+(-?\d*\.\d+)\s+(-?\d*\.\d+)\s+(-?\d*\.\d+)\s*', line).groups())
        ])
    return atoms, coordinates

def _convert_to_nparr(file):
    if file.endswith(".xyz"):
        try:
            return convert_from_xyz(file)
        except:
            logger.exception(f"file {file} is called an xyz, but failed parse. continuing without it.")
    # this is where you would add support for other input file types.
    else:
        logger.error(f"could not find a parser for file {file}. continuing without it.")
    return None, None

def convert_to_nparr(fp_like):
    if os.path.isdir(fp_like):
        files = [os.path.join(dp, fn) for dp, dn, fns in os.walk(fp_like) for fn in fns]
    else:
        files = [fp_like]
    atoms_list, coordinates_list = map(list, zip(*[_convert_to_nparr(file) for file in files]))
    return atoms_list, coordinates_list