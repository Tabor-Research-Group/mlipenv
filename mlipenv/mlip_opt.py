import os
import json
import logging

from mlipenv.managers import OptimizationManager, EnergyManager
from mlipenv.optimization_options import BaseConfiguration, structure_path_keys

def configure_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger(__name__).info("logger configured")

def load_config(config_bundle):
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
    found_structure_path_key = next((s for s in structure_path_keys if s in config), None)
    if found_structure_path_key:
        from mlipenv.util import convert_to_nparr
        atoms, coordinates = convert_to_nparr(config[found_structure_path_key])
        # I give up. these keys are hard-coded.
        config["atoms"] = atoms
        config["coordinates"] = coordinates
        config.pop(found_structure_path_key, None)
    return BaseConfiguration(**config)
    
def get_runner_for_method(config):
    if config.method == "optimize":
        return OptimizationManager(config)
    elif config.method == "energy":
        return EnergyManager(config)
    else:
        raise NotImplementedError(f"Unknown method type: {config.method}")

def call_to_mlip_server(config_bundle):
    configure_logger()
    config = load_config(config_bundle)
    runner = get_runner_for_method(config)
    runner.run()
