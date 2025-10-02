import os

def get_calc(**kwargs):
    calculator = os.environ["CALCULATOR"].lower()
    if calculator == "fairchem":
        return get_fairchem_calc(**kwargs)
    elif calculator == "aimnet2":
        return get_aimnet_calc(**kwargs)
    elif calculator == "mace":
        return get_mace_calc(**kwargs)
    else:
        raise NotImplementedError("be careful messing with `CALCULATOR` environment variable!")

def get_fairchem_calc(predict_unit="uma-s-1p1", device="cpu", task_name="omol"):
    from fairchem.core.calculate.pretrained_mlip import get_predict_unit
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    predictor = get_predict_unit(predict_unit, device=device)
    return FAIRChemCalculator(predictor, task_name=task_name)

def get_aimnet_calc(base_calc="aimnet2"):
    from aimnet2calc import AIMNet2ASE
    return AIMNet2ASE(base_calc=base_calc)
    # return AIMNet2ASE(base_calc="aimnet2", charge=0, mult=1)

def get_mace_calc(calculator="mace_omol", model="extra_large", device="cpu"):
    import mace.calculators
    calc_cls = getattr(mace.calculators, calculator)
    return calc_cls(model=model, device=device, default_dtype='float64')