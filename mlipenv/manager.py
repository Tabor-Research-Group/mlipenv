import os
import logging

from mlipenv.util import load_config, build_base_config

def configure_logger(base_config):
    log_file = base_config.log_file
    output_dir = base_config.output_dir
    log_fpath = log_file if log_file is None else os.path.join(output_dir, log_file)
    logging.basicConfig(
        filename=log_fpath,
        level=logging.INFO,
        format="%(asctime)s :: %(levelname)s :: %(message)s",
    )
    return logging.getLogger(__name__)

DEFAULT_OPTIMIZER="ase"
def spy_optimizer(optimizer=None, **kwargs):
    if not optimizer:
        optimizer = kwargs.get("optimization_options", {}).get("optimizer", DEFAULT_OPTIMIZER)
    return optimizer

def get_runner_for_method(base_config, runner_args):
    from mlipenv.runners import get_runner
    if base_config.method == "optimization":
        optimizer = spy_optimizer(**runner_args)
        return get_runner(optimizer)(base_config, **runner_args)
    elif base_config.method == "energy":
        return get_runner("energy")(base_config, **runner_args)
    else:
        raise NotImplementedError(f"Unknown method type: {base_config.method}")


def execute_mlip_job(config_bundle):
    config_args = load_config(config_bundle)
    base_config, runner_args = build_base_config(**config_args)
    logger = configure_logger(base_config)
    logger.info("logger configured.")
    try:
        runner = get_runner_for_method(base_config, runner_args)
        runner.run()
        runner.export_results()
    except Exception:
        logger.exception("Job failed.")

if __name__ == "__main__":
    import sys
    execute_mlip_job(sys.argv[1])