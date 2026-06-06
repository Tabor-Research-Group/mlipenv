import logging

import torch.autograd
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

logger = logging.getLogger(__name__)

def get_higher_derivatives(obj, calculator, device, order):
    derivatives = differentiators[type(calculator)](obj, calculator, device, order)
    return {str(i): derivative for i, derivative in enumerate(derivatives)}

def autograd_derivative(func, pos, order, max_order):
    func = func.reshape(-1)
    derived_func = []
    for i, f_i in enumerate(func):
        df_i = torch.autograd.grad(f_i, 
                                   pos, 
                                   create_graph=order!=max_order,
                                   retain_graph=(order!=max_order or i < func.numel()-1))[0].reshape(-1)
        derived_func.append(df_i)
    dim = pos.numel()
    shape = [dim]*order
    derived_func = torch.stack(derived_func, dim=0).reshape(shape)
    return derived_func

def fairchem_differentiator(obj, calculator, device, order):
    from fairchem.core.datasets.atomic_data import AtomicData

    predictor = calculator.predictor
    data = AtomicData.from_ase(obj, task_name="omol", r_data_keys=["charge", "spin"])
    data = data.to(device)
    data["pos"].requires_grad_(True)
    pos = data["pos"]
    try:
        predictor.model.module.backbone.regress_config.forces = False
        predictor.model.module.backbone.regress_config.stress = False
        if not predictor.lazy_model_intialized:
            predictor._lazy_init(data)
        preds = predictor.model(data)
        
        omol_energy_task_name = "omol_energy"
        omol_energy_task = predictor.model.module.tasks[omol_energy_task_name]
        energy = preds[omol_energy_task_name][omol_energy_task.property]
        
        denormed_energy = omol_energy_task.normalizer.denorm(energy)
        if omol_energy_task.element_references is not None:
            denormed_energy = omol_energy_task.element_references.undo_refs(data, denormed_energy)
        logger.info(f"{energy}, {denormed_energy}")

        outputs = [denormed_energy.detach().cpu().numpy()]
        derived_func = denormed_energy
        for n in range(1, order+1):
            logger.info(n)
            derived_func = autograd_derivative(derived_func, pos, order=n, max_order=order)
            outputs.append(derived_func.detach().cpu().numpy())
    finally:
        predictor.model.module.backbone.regress_config.forces = True
        predictor.model.module.backbone.regress_config.stress = True
    return outputs

differentiators = {
    FAIRChemCalculator: fairchem_differentiator
}

