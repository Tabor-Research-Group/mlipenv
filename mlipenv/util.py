import os
import logging

logger = logging.getLogger(__name__)

# def configuration_builder(method, atoms, coordinates, output_dir, **kwargs):
def configuration_builder(method, 
                          atoms, 
                          coordinates, 
                          charge, 
                          spin, 
                          output_dir=".", 
                          device=None,
                          model_path=None,
                          mace_calculator=None,
                          **kwargs):
    from dataclasses import asdict
    from mlipenv.options import get_config

    if method == "optimize":
        options = build_optimization_config(**kwargs)
    elif method == "energy":
        options = build_energy_config(**kwargs)
    else:
        raise NotImplementedError
    if mace_calculator is not None:
        calculator_options = get_config("mace")(device, model_path, mace_calculator)
    elif model_path is not None:
        calculator_options = get_config("aimnet")(device, model_path)
    else:
        calculator_options = get_config("calculator")(device)
    return asdict(get_config("base")(method, 
                                    options, 
                                    atoms, 
                                    coordinates, 
                                    charge, 
                                    spin, 
                                    output_dir, 
                                    calculator_options))

# NEED TO FIX: I can't tell what the point of this function was.
def build_optimization_config(optimizer, **kwargs):
    # from mlipenv.optimization_options import get_config
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

def load_config(config_bundle):
    import json
    from mlipenv.options import STRUCTURE_PATH_KEYS
    if isinstance(config_bundle, str):
        if os.path.exists(config_bundle):
            with open(config_bundle, "r") as f:
                config = json.load(f)
        else:
            try:
                config = json.loads(config_bundle)
            except Exception as e:
                raise NotImplementedError("Cannot load from string that is neither a valid path to nor formatted JSON itself.") from e
    
    elif isinstance(config_bundle, dict):
        config = config_bundle
    else:
        raise NotImplementedError(f"Intractable input type: {type(config_bundle)}")
    found_structure_path_key = next((s for s in STRUCTURE_PATH_KEYS if s in config), None)
    if found_structure_path_key:
        from mlipenv.util import convert_to_nparr
        atoms, coordinates = convert_to_nparr(config[found_structure_path_key])
        # I give up. these keys are hard-coded.
        config["atoms"] = atoms
        config["coordinates"] = coordinates
        config.pop(found_structure_path_key, None)
    return config

def load_multidim_parameter(parameter_bundle):
    import numpy as np
    if isinstance(parameter_bundle, str):
        if os.path.exists(parameter_bundle):
            data = np.load(parameter_bundle)
            if len(data.files) > 1:
                parameter = [data[key] for key in data.files]
            else:
                parameter = data[data.files[0]]
                # runners enforce one-at-a-time batching, if the input
                # is a multi-dim np array, we splice it up here.
                if parameter.ndim > 2:
                    parameter = [parameter[i] for i in parameter.shape[0]]
                else:
                    parameter = [parameter]
    elif isinstance(parameter_bundle, list):
        parameter = parameter_bundle
    return parameter


def build_calculator_options(cls, **options):
    from dataclasses import fields
    entries = [f.name for f in fields(cls)]
    filtered_options = {k:v for k,v in options.items() if k in entries}
    extra_options = {k:v for k,v in options.items() if k not in entries}
    return cls(**filtered_options), extra_options
