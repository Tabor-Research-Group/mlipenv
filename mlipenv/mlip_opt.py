
from mlipenv.options import get_configuration
from mlipenv.runners import get_runner
from mlipenv.util import load_config

DEFAULT_OPTIMIZER="ase"
def spy_optimizer(optimizer=None, **kwargs):
    if not optimizer:
        optimizer = kwargs.get("optimization_options", {}).get("optimizer", DEFAULT_OPTIMIZER)
    return optimizer
    
def get_runner_for_method(base_config, runner_args):
    if base_config.method == "optimize":
        optimizer = spy_optimizer(**runner_args)
        print(optimizer)
        return get_runner(optimizer)(base_config, **runner_args)
    elif base_config.method == "energy":
        return get_runner("energy")(base_config, **runner_args)
    else:
        raise NotImplementedError(f"Unknown method type: {base_config.method}")

def build_base_config(method, 
                      atoms, 
                      coordinates, 
                      charge, 
                      spin, 
                      output_dir=".",
                    **kwargs):
    return get_configuration("base")(method, atoms, coordinates, charge, spin, output_dir), kwargs

def call_to_mlip_server(config_bundle):
    config_args = load_config(config_bundle)
    base_config, runner_args = build_base_config(**config_args)
    runner = get_runner_for_method(base_config, runner_args)
    runner.run()
    runner.export_results()