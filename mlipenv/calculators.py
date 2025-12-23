import os
import logging
from dataclasses import asdict

from mlipenv.util import find_file
from mlipenv.optimization_options import CalculatorConfiguration, MACECalculatorConfiguration, AIMNetCalculatorConfiguration

logger = logging.getLogger(__name__)

CALCULATOR = os.environ["CALCULATOR"].lower()

def get_calc(calculator_options):
    if CALCULATOR == "fairchem":
        return get_fairchem_calc(**asdict(calculator_options))
    elif CALCULATOR == "aimnet2":
        return get_aimnet_calc(**asdict(calculator_options))
    elif CALCULATOR == "mace":
        return get_mace_calc(**asdict(calculator_options))
    else:
        raise ValueError("be careful messing with the `CALCULATOR` environment variable!")
    
def build_calculator_options(calculator_options):
    import torch.cuda
    detected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not calculator_options["device"] or (calculator_options["device"] == "gpu" and detected_device == "cuda"):
        calculator_options["device"] = detected_device
    elif calculator_options["device"] != detected_device:
        logger.warning(f"detected hardware: {detected_device}. using the device that you requested. ({calculator_options["device"]}.) ...")
    if CALCULATOR == "fairchem":
        return CalculatorConfiguration(**calculator_options)
    elif CALCULATOR == "aimnet2":
        return AIMNetCalculatorConfiguration(**calculator_options)
    elif CALCULATOR == "mace":
        return MACECalculatorConfiguration(**calculator_options)
    else:
        raise ValueError("be careful messing with the `CALCULATOR` environment variable!")
    
MODEL_CACHE_DIR = "MODEL_CACHE_DIR"
DEFAULT_CACHE_DIR = "DEFAULT_MODEL_CACHE_DIR"
def get_fairchem_predict_unit(device, model="uma-s-1p1"):
    cache_locs = []
    if MODEL_CACHE_DIR in os.environ:
        cache_locs.append(os.environ[MODEL_CACHE_DIR])
    if DEFAULT_CACHE_DIR in os.environ:
        cache_locs.append(os.environ[DEFAULT_CACHE_DIR])
    if len(cache_locs) == 0:
        raise ValueError(f"either `{MODEL_CACHE_DIR}` or `{DEFAULT_CACHE_DIR}` must be set at the environment level")

    from fairchem.core.calculate.pretrained_mlip import load_predict_unit
    from omegaconf import OmegaConf
    for cache_dir in cache_locs:
        try:
            model_file = f"{model}.pt" if model[-3:] != ".pt" else model
            model_path = find_file(cache_dir, model_file)
            atom_refs_path = find_file(cache_dir, "iso_atom_elem_refs.yaml")
            atom_refs = OmegaConf.load(atom_refs_path)
            return load_predict_unit(model_path, inference_settings="default", device=device, atom_refs=atom_refs)
        except:
            pass

def get_fairchem_calc(device, model="uma-s-1p1", task_name="omol", **kwargs):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    try:
        predictor = get_fairchem_predict_unit(device, model)
        return FAIRChemCalculator(predictor, task_name=task_name)
    except:
        raise ValueError(f"could not load model files. you should check on `{MODEL_CACHE_DIR}` and/or `{DEFAULT_CACHE_DIR}`")

AIMNET_DEFAULT_CALC="aimnet2"
def get_aimnet_calc(model_path, **kwargs):
    from aimnet2calc import AIMNet2ASE
    if model_path:
        try:
            return AIMNet2ASE(base_calc=model_path)
        except:
            logger.warning(f"could not load the model from path {model_path}. proceeding with default.")
    return AIMNet2ASE(base_calc=AIMNET_DEFAULT_CALC)
    # return AIMNet2ASE(base_calc="aimnet2", charge=0, mult=1)

# def get_mace_model_registry():
#     return {
#     ("mace_omol", "extra_large"): "MACE-omol-0-extra-large-1024.model"
# }

MACE_DEFAULT_MODEL_PATH="/home/models/MACE-omol-0-extra-large-1024.model"
def get_mace_calc(model_path, calculator, device, **kwargs):
    import mace.calculators
    calculator = calculator.lower()
    if calculator == "omol":
        calculator = "mace_omol"
    elif calculator == "off":
        calculator = "mace_off"
    elif calculator == "mp":
        calculator = "mace_mp"
    elif calculator == "anicc":
        calculator = "mace_anicc"
    calc_cls = getattr(mace.calculators, calculator)
    if model_path:
        try:
            return calc_cls(model=model_path, device=device)
        except:
            logger.warning(f"could not load the model from path {model_path}. proceeding with default.")
    model_path = MACE_DEFAULT_MODEL_PATH
    # model_fn = get_mace_model_registry.get((calculator, model), None)
    # if model_fn:
    #     model = os.path.join("/home/models", model_fn)
    return calc_cls(model=model_path, device=device)
