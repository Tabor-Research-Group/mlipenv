import os

from src.util import find_file

def get_calc(**kwargs):
    calculator = os.environ["CALCULATOR"].lower()
    if calculator == "fairchem":
        return get_fairchem_calc(**kwargs)
    elif calculator == "aimnet2":
        return get_aimnet_calc(**kwargs)
    elif calculator == "mace":
        return get_mace_calc(**kwargs)
    else:
        raise ValueError("be careful messing with `CALCULATOR` environment variable!")

MODEL_CACHE_DIR = "MODEL_CACHE_DIR"
DEFAULT_CACHE_DIR = "DEFAULT_MODEL_CACHE_DIR"
def get_fairchem_calc(model="uma-s-1p1", task_name="omol", device="cpu"):
    try:
        if MODEL_CACHE_DIR in os.environ:
            cache_dir = os.environ[MODEL_CACHE_DIR]
        else:
            cache_dir = os.environ[DEFAULT_CACHE_DIR]
    except:
        raise ValueError(f"either `{MODEL_CACHE_DIR}` or `{DEFAULT_CACHE_DIR}` must be set at the environment level")

    from fairchem.core.calculate.pretrained_mlip import load_predict_unit
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from omegaconf import OmegaConf
    model_file = f"{model}.pt" if model[-3:] != ".pt" else model
    model_path = find_file(cache_dir, model_file)
    atom_refs_path = find_file(cache_dir, "iso_atom_elem_refs.yaml")
    atom_refs = OmegaConf.load(atom_refs_path)
    predictor = load_predict_unit(model_path, inference_settings="default", device=device, atom_refs=atom_refs)
    return FAIRChemCalculator(predictor, task_name=task_name)

def get_aimnet_calc(base_calc="aimnet2"):
    from aimnet2calc import AIMNet2ASE
    return AIMNet2ASE(base_calc=base_calc)
    # return AIMNet2ASE(base_calc="aimnet2", charge=0, mult=1)

def get_mace_calc(calculator="mace_omol", model="extra_large", device="cpu"):
    import mace.calculators
    calc_cls = getattr(mace.calculators, calculator)
    return calc_cls(model=model, device=device, default_dtype='float64')
