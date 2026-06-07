
from mlipenv.options import get_configuration
from mlipenv.runners import get_runner
from mlipenv.util import load_config

METHOD_REGISTRY={}
def register_method(key, method=None):
    if method is None:
        def register(method):
            return register_method(key, method)
        return register
    else:
        METHOD_REGISTRY[key] = method
        return method
def resolve_method(method):
    return METHOD_REGISTRY[method]

DEFAULT_OPTIMIZER="ase"
def spy_optimizer(optimizer=None, **kwargs):
    if not optimizer:
        optimizer = kwargs.get("optimization_options", {}).get("optimizer", DEFAULT_OPTIMIZER)
    return optimizer

@register_method("optimization")
def get_optimizer(base_config, **runner_args):
    optimizer = spy_optimizer(**runner_args)
    return get_runner(optimizer)(base_config, **runner_args)
    
def get_runner_for_method(base_config, runner_args):
    base_config.method = runner_args.get("method", base_config.method)
    method_dispatch = METHOD_REGISTRY.get(base_config.method)
    if method_dispatch is None:
        method_dispatch = get_runner(base_config.method)
    return method_dispatch(base_config, **runner_args)

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