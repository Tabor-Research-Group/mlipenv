import os
import abc
import logging
import time

import numpy as np
from ase import Atoms
# from Psience.Molecools.Evaluator import PropertyEvaluator

from mlipenv.calculators import get_calc # , build_calculator_options
from mlipenv.options import get_configuration
from mlipenv.enums.output_enum import _output_file_registry
from mlipenv.optimizers import BetterBFGS
from mlipenv.util import load_multidim_parameter, build_calculator_options

logger = logging.getLogger(__name__)

RUNNER_REGISTRY={}
def register_runner(key, runner_factory=None):
    if runner_factory is None:
        def register(runner_factory):
            return register_runner(key, runner_factory)
        return register
    else:
        RUNNER_REGISTRY[key] = runner_factory

def get_runner(key):
    return RUNNER_REGISTRY[key]

class BaseRunner:
    debug = os.environ.get("DEBUG", "").lower() == "true"
    def __init__(self, base_config, **kwargs):
        self.atoms = load_multidim_parameter(base_config.atoms)
        self.coordinates = load_multidim_parameter(base_config.coordinates)
        self.charge = self.load_charge(base_config.charge)
        self.spin = base_config.spin
        self.output_dir = base_config.output_dir

    def load_charge(self, charge):
        if isinstance(charge, int):
            charge = [charge] * len(self.coordinates)
        charge = np.asanyarray(charge)
        # in case you want to template charges in chunks
        charge_template = np.asarray(charge[-1]).flatten()
        charge = charge.flatten()
        if len(self.atoms) < len(self.coordinates):
            self.atoms = self.atoms + [self.atoms[-1]] * (len(self.coordinates) - len(self.atoms))
        while len(charge) < len(self.coordinates):
            charge = np.append(charge, charge_template)
        return charge
    
    def export_results(self):
        self.export_results_subroutine()
        formatted_results = self.format_results()
        output_paths = self.get_output_with_defaults()
        for loc, res in zip(output_paths, formatted_results):
            np.savez(loc, *res)

    def atomize(self, atom_symbols, coordinates, charge):
        atoms = Atoms(symbols=atom_symbols, positions=coordinates)
        atoms.info["charge"] = int(charge)
        atoms.info["spin"] = self.spin
        atoms.calc = self.calc
        return atoms
    
    def format_results(self):
        return [[f(obj) for obj in self.results] for f in self.result_getters()]

    def export_results_subroutine(self):
        pass

    def get_calc_for_runner(self):
        from dataclasses import asdict
        return get_calc(**asdict(self.calculator_options))

    @abc.abstractmethod
    def result_getters():
        ...
    
    @abc.abstractmethod
    def get_output_with_defaults(self):
        ...

    @abc.abstractmethod
    def run(self):
        ...

@register_runner("energy")
class EnergyRunner(BaseRunner):
    def __init__(self, base_config, **kwargs):
        super().__init__(base_config)
        self.load_energy_configs(**kwargs)
        t1=time.time()
        self.calc = self.get_calc_for_runner()
        logger.info(f"loading time for calculator: {time.time()-t1:.3f} seconds.")

    # a lot of the functionality in here overlaps with OptimizationRunner.load_runner_configs.
    def load_energy_configs(self, 
                            energy_options, 
                            calculator_options=None,
                            **kwargs
                            ):
        calc_type = os.environ.get("CALCULATOR", "").lower()
        self.energy_options = get_configuration("energy")(**energy_options)
        try:
            calc_configuration_cls = get_configuration(calc_type)
        except:
            calc_configuration_cls = get_configuration("calculator")
        if not calculator_options:
                calculator_options = dict()
        if "device" not in calculator_options:
            import torch.cuda
            calculator_options["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.calculator_options, self.loose_calc_kwargs = build_calculator_options(calc_configuration_cls, **calculator_options)
    
    def result_getters(self):
        getters = [*self.get_pes_derivatives]
        return getters
    
    def get_output_with_defaults(self):
        derivatives_dname = "derivatives"
        return [os.path.join(self.output_dir, derivatives_dname, f"{n}.npz") for n in range(self.order)]
    
    def run(self):
        self.results = [self.atomize(atoms, coords, charge) 
                           for atoms, coords, charge in zip(self.atoms, self.coordinates, self.charge)]

    def get_pes_derivatives(self, obj):
        if self.energy_options.order == 0:
            derivative_dict = {"0": np.array(obj.get_potential_energy())}
        elif self.energy_options.order == 1:
            derivative_dict["1"] = np.array(obj.get_forces())
        else:
            from mlipenv.differentiation import get_higher_derivatives
            derivative_dict = get_higher_derivatives(obj, calculator=obj.calc, device=self.calculator_options.device, order=self.energy_options.order)
        self.derivative_dict = derivative_dict
        return list(derivative_dict.values())
        

class BaseOptimizationRunner(BaseRunner):
    def __init__(self, base_config, **kwargs):
        super().__init__(base_config, **kwargs)
        self.load_runner_configs(**kwargs)
    
    def result_getters(self):
        return [self.get_atom_symbols, self.get_coordinates, self.get_gradients, self.get_single_point_energy]
    
    def get_output_with_defaults(self):
        output_files = [_output_file_registry()[k].value for k in _output_file_registry()]
        return [os.path.join(self.output_dir, file) for file in output_files]

    def load_runner_configs(self, 
                            optimization_options, 
                            calculator_options=None,
                            **kwargs
                            ):
        config_type = os.environ.get("calculator", "").lower()
        self.optimization_options = get_configuration("optimization")(**optimization_options)
        try:
            configuration_cls = get_configuration(config_type)
        except:
            configuration_cls = get_configuration("calculator")
        if not calculator_options:
                calculator_options = dict()
        if "device" not in calculator_options:
            import torch.cuda
            calculator_options["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.calculator_options, self.loose_calc_kwargs = build_calculator_options(configuration_cls, **calculator_options)

    def get_calc_for_runner(self):
        from dataclasses import asdict
        return get_calc(**asdict(self.calculator_options), **self.loose_calc_kwargs)

    @abc.abstractmethod
    def get_atom_symbols(self, obj):
        ...
    @abc.abstractmethod
    def get_coordinates(self, obj):
        ...
    @abc.abstractmethod
    def get_gradients(self, obj):
        ...
    @abc.abstractmethod
    def get_single_point_energy(self, obj):
        ...

@register_runner("ase")
class ASEOptimizationRunner(BaseOptimizationRunner):
    def __init__(self, base_config, **kwargs):
        super().__init__(base_config, **kwargs)
        self.run_count = 0
        t1=time.time()
        self.calc = self.get_calc_for_runner()
        logger.info(f"loading time for calculator: {time.time()-t1:.3f} seconds.")

    def run(self):
        optimized_atoms = [self.run_opt(atoms, coords, charge) 
                           for atoms, coords, charge in zip(self.atoms, self.coordinates, self.charge)]
        self.results = optimized_atoms
    
    def get_atom_symbols(self, obj):
        return np.array(obj.get_chemical_symbols())
    def get_coordinates(self, obj):
        return np.array(obj.get_positions())
    def get_gradients(self, obj):
        return np.array(obj.get_forces())
    def get_single_point_energy(self, obj):
        return np.array(obj.get_potential_energy())

    def run_opt(self, atom_symbols, coordinates, charge):
        atoms = self.atomize(atom_symbols, coordinates, charge)
        atoms = self._run_opt(atoms)
        return atoms
    
    def _run_opt(self, atoms):
        from ase.optimize import BFGS
        os.makedirs(os.path.join(self.output_dir, "trajectories"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        opt = BFGS(
            atoms, 
            trajectory=os.path.join(self.output_dir, "trajectories", f"{self.run_count}.traj"),
            logfile=os.path.join(self.output_dir, "logs", f"{self.run_count}.log")
            )
        logger.info(f"running opt {self.run_count} for {self.optimization_options.steps} steps...")
        opt.run(fmax=self.optimization_options.fmax, steps=self.optimization_options.steps)
        logger.info(f"done.")
        self.run_count = self.run_count + 1
        return atoms


class SciPyOptimizationRunner(BaseOptimizationRunner):
    def __init__(self, base_config, **kwargs):
        super().__init__(base_config, **kwargs)
    
    def run(self):
        ...


@register_runner("better")
class BetterOptimizationRunner(BaseOptimizationRunner):
    def __init__(self, base_config, **kwargs):
        if not os.environ.get("CALCULATOR", "").lower() == "fairchem":
            raise NotImplementedError(f"BetterOptimizationRunner is not currently written for {os.environ.get('CALCULATOR', '')} calculator.")
        super().__init__(base_config, **kwargs)

    def step_optimization(self, optimizers, predictor):
        import torch
        from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
        atomic_data = [AtomicData.from_ase(optimizer.ase_atoms, task_name="omol", r_data_keys=["charge", "spin"])
                           for optimizer in optimizers]
        batch = atomicdata_list_to_batch(atomic_data)
        t1 = time.time()
        with torch.no_grad():
            preds = predictor.predict(batch)
        torch_time = time.time()-t1
        t2 = time.time()
        for j, optimizer in enumerate(optimizers):
            optimizer.remember_energy(preds["energy"][j])
            optimizer.optimize_and_update(preds["forces"][batch.batch == j], self.optimization_options.fmax)
        bfgs_time = time.time()-t2
        return torch_time, bfgs_time
    
    def run(self):
        t1=time.time()
        from mlipenv.calculators import get_fairchem_predict_unit
        predictor = get_fairchem_predict_unit(self.calculator_options.device)
        logger.info(f"loading time for calculator: {time.time()-t1:.3f} seconds.")

        optimizers = [BetterBFGS(atoms, coords, charge, self.spin, idx)
                      for idx, (atoms, coords, charge) in enumerate(zip(self.atoms, self.coordinates, self.charge))]
        for optimizer in optimizers:
            if optimizer.is_atom():
                torch_time, bfgs_time = self.step_optimization([optimizer], predictor)
                logger.info(f"Single atom times. prediction: {torch_time:.3f} s. bfgs: {bfgs_time:.3f} s.")
        
        torch_times = []
        bfgs_times = []
        unconverged_optimizers = optimizers
        for i in range(self.optimization_options.steps):
            unconverged_optimizers = [optimizer for optimizer in unconverged_optimizers if not optimizer.converged]
            logger.info(f"optimization step: {i}. num unconverged = {len(unconverged_optimizers)}/{len(optimizers)}")
            if not len(unconverged_optimizers):
                break
            torch_time, bfgs_time = self.step_optimization(unconverged_optimizers, predictor)
            torch_times.append(torch_time)
            bfgs_times.append(bfgs_time)
            
        if self.debug:
            logger.info(f"torch times = {torch_times}")
            logger.info(f"bfgs times = {bfgs_times}")
        self.results = optimizers

    def get_atom_symbols(self, obj):
        return np.array(obj.get_atoms())
    def get_coordinates(self, obj):
        return np.array(obj.get_coordinates())
    def get_gradients(self, obj):
        return np.array(obj.get_forces())
    def get_single_point_energy(self, obj):
        return np.array(obj.get_energy())
    
    def export_results_subroutine(self):
        os.makedirs(os.path.join(self.output_dir, "trajectories"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        for optimizer in self.results:
            optimizer.write_trajectory(self.output_dir)
            optimizer.write_log(self.output_dir)
