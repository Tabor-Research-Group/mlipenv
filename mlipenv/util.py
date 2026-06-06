import os
import logging

from mlipenv.options import get_configuration

logger = logging.getLogger(__name__)

CONFIG_BUILDER_REGISTRY = {}
def register_config_builder(name, config_factory=None):
    if config_factory is None:
        def register(config_factory):
            return register_config_builder(name, config_factory)
        return register
    else:
        CONFIG_BUILDER_REGISTRY[name] = config_factory
        return config_factory
    
def get_config_builder(name):
    return CONFIG_BUILDER_REGISTRY[name]

@register_config_builder("optimization")
def build_optimization_config(config, optimization_options=None, calculator_options=None):
    config["optimization_options"] = optimization_options
    config["calculator_options"] = calculator_options

@register_config_builder("energy")
def build_energy_config(config, **kwargs):
    config["energy_options"] = kwargs

def configuration_builder(method, 
                          atoms, 
                          coordinates, 
                          charge, 
                          spin, 
                          output_dir=".", 
                          **kwargs):
    from dataclasses import asdict
    config = asdict(get_configuration("base")(method, atoms, coordinates, charge, spin, output_dir))
    get_config_builder(method)(config=config, **kwargs)
    return config

def find_file(root, target):
    for dirpath, dirs, files in os.walk(root):
        if target in files:
            return os.path.join(dirpath, target)
    return None

DEFAULT_CHARGE=0
def convert_from_xyz(file):
    import re
    with open(file, "r") as f:
        _ = int(f.readline().strip())
        optional_info_line = f.readline()
        try:
            charge = int(optional_info_line.strip())
        except:
            charge = DEFAULT_CHARGE
        atoms, coordinates = zip(*[
            (m[0], m[1:]) for line in f.readlines() if (m := re.match(r'\s*([A-Za-z]+)\s+(-?\d*\.\d+|\d+)\s+(-?\d*\.\d+|\d+)\s+(-?\d*\.\d+|\d+)\s*', line).groups())
        ])
    return atoms, coordinates, charge

def _convert_molecules_to_nparr(file):
    if file.endswith(".xyz"):
        try:
            return convert_from_xyz(file)
        except:
            logger.exception(f"file {file} is called an xyz, but failed parse. continuing without it.")
    # this is where you would add support for other input file types.
    else:
        logger.error(f"could not find a parser for file {file}. continuing without it.")
    return None, None, None

def convert_molecules_to_nparr(fp_like):
    if os.path.isdir(fp_like):
        files = [os.path.join(dp, fn) for dp, dn, fns in os.walk(fp_like) for fn in fns]
    else:
        files = [fp_like]
    atoms_list, coordinates_list, charge_list = map(list, zip(*[_convert_molecules_to_nparr(file) for file in files]))
    return atoms_list, coordinates_list, charge_list

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
        atoms, coordinates, charge = convert_molecules_to_nparr(config[found_structure_path_key])
        # I give up. these keys are hard-coded.
        config["atoms"] = atoms
        config["coordinates"] = coordinates
        config["charge"] = charge
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
                if parameter.ndim > 1:
                    parameter = [parameter[i] for i in range(parameter.shape[0])]
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
