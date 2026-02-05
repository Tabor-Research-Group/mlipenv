import os
import logging

from mlipenv.util import find_file

logger = logging.getLogger(__name__)


CALCULATOR_REGISTRY = {}
def register_calculator(name, calc_factory=None):
    if calc_factory is None:
        # used as a decorator i.e.
        # @register_calculator(name)
        # def calc(...): ...
        def register(calc_factory):
            return register_calculator(name, calc_factory)
        return register
    else:
        CALCULATOR_REGISTRY[name] = calc_factory
        return calc_factory

def get_calc(*, calculator=None, **calculator_options):
    if calculator is None:
        calculator = os.environ.get("CALCULATOR").lower()
    if isinstance(calculator, str):
        calculator = CALCULATOR_REGISTRY[calculator]
    return calculator(**calculator_options)

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

@register_calculator("fairchem")
def get_fairchem_calc(device, model="uma-s-1p1", task_name="omol", **kwargs):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    try:
        predictor = get_fairchem_predict_unit(device, model)
        return FAIRChemCalculator(predictor, task_name=task_name, **kwargs)
    except:
        raise ValueError(f"could not load model files. you should check on `{MODEL_CACHE_DIR}` and/or `{DEFAULT_CACHE_DIR}`")

AIMNET_DEFAULT_CALC="aimnet2"
@register_calculator("aimnet")
@register_calculator("aimnet2")
def get_aimnet_calc(model_path, **kwargs):
    from aimnet2calc import AIMNet2ASE
    if model_path:
        try:
            return AIMNet2ASE(base_calc=model_path, **kwargs)
        except:
            logger.warning(f"could not load the model from path {model_path}. proceeding with default.")
    return AIMNet2ASE(base_calc=AIMNET_DEFAULT_CALC)

MACE_DEFAULT_MODEL_PATH="/home/models/MACE-omol-0-extra-large-1024.model"
MACE_CALCULATOR_TYPES=["mace_omol", "mace_off", "mace_mp", "mace_anicc"]
MACE_CALCULATOR_ALIASES=["omol" "off", "mp", "anicc"]
@register_calculator("mace")
def get_mace_calc(model_path, mace_calculator, device, **kwargs):
    import mace.calculators
    if mace_calculator:
        mace_calculator = mace_calculator.lower()
    for calculator_type, calculator_alias in zip(MACE_CALCULATOR_TYPES, MACE_CALCULATOR_ALIASES):
        if not mace_calculator or mace_calculator == calculator_type or mace_calculator == calculator_alias:
            try:
                calc_cls = getattr(mace.calculators, calculator_type)
                return calc_cls(model=model_path, device=device, **kwargs)
            except:
                logger.warning(f"could not load using MACE calculator class: {calculator_type}.")
    # if model_path:
    #     try:
    #         return calc_cls(model=model_path, device=device)
    #     except:
    #         logger.warning(f"could not load the model from path {model_path}. proceeding with default.")
    logger.info("no valid model found. using default model and type...")
    model_path = MACE_DEFAULT_MODEL_PATH
    return calc_cls(model=model_path, device=device)


