import os
import abc
import logging

import numpy as np
from ase import Atoms

from mlipenv.calculators import get_calc, build_calculator_options
from mlipenv.optimization_options import OptimizationConfiguration, EnergyConfiguration
from mlipenv.enums.output_enum import _output_file_registry
from mlipenv.optimizers import BetterBFGS

logger = logging.getLogger(__name__)

class BaseRunner:

    debug = os.environ.get("DEBUG")

    def __init__(self, atoms, coordinates, charge, spin, calculator_options, **kwargs):
        self.atoms = atoms
        self.coordinates = coordinates
        self.charge = charge
        self.spin = spin
        self.calculator_options = build_calculator_options(calculator_options)

    def load_config_with_defaults(self, config):
        if isinstance(self.charge, int):
            self.charge = [self.charge] * len(self.coordinates)
        if len(self.atoms) < len(self.coordinates):
            self.atoms = self.atoms + [self.atoms[-1]] * (len(self.coordinates) - len(self.atoms))
        if len(self.charge) < len(self.coordinates):
            self.charge = self.charge + [self.charge[-1]] * (len(self.coordinates) - len(self.charge))
    
    def export_results(self, output_dir):
        self.export_results_subroutine(output_dir)
        formatted_results = self.format_results()
        output_paths = self.get_output_with_defaults(output_dir)
        for loc, res in zip(output_paths, formatted_results):
            np.savez(loc, *res)

    def atomize(self, atom_symbols, coordinates, charge):
        atoms = Atoms(symbols=atom_symbols, positions=coordinates)
        atoms.info["charge"] = charge
        atoms.info["spin"] = self.spin
        calc = get_calc(self.calculator_options)
        atoms.calc = calc
        return atoms
    
    def format_results(self):
        return [[f(obj) for obj in self.results] for f in self.result_getters()]

    def export_results_subroutine(self, *args):
        pass

    @abc.abstractmethod
    def result_getters():
        ...
    
    @abc.abstractmethod
    def get_output_with_defaults(self, output_dir):
        ...

    @abc.abstractmethod
    def run(self):
        ...

class EnergyRunner(BaseRunner):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.load_config_with_defaults(config)

    def load_config_with_defaults(self, config):
        super().load_config_with_defaults(config)
        self.options = EnergyConfiguration(**config)
    
    def result_getters(self):
        getters = [self.get_single_point_energy]
        if self.options.order > 0:
            getters.append(self.get_gradients)
        return getters
    
    def get_output_with_defaults(self, output_dir):
        output_file_registry = _output_file_registry()
        output_files = [output_file_registry["energies"].value]
        if self.options.order > 0:
            output_files.append(output_file_registry["gradients"].value)
        return [os.path.join(output_dir, file) for file in output_files]
    
    def run(self):
        self.results = [self.atomize(atoms, coords, charge) 
                           for atoms, coords, charge in zip(self.atoms, self.coordinates, self.charge)]

    def get_gradients(self, obj):
        return np.array(obj.get_forces())
    def get_single_point_energy(self, obj):
        return np.array(obj.get_potential_energy())

class BaseOptimizationRunner(BaseRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def result_getters(self):
        return [self.get_atom_symbols, self.get_coordinates, self.get_gradients, self.get_single_point_energy]
    
    def get_output_with_defaults(self, output_dir):
        output_files = [_output_file_registry()[k].value for k in self.options.output]
        return [os.path.join(output_dir, file) for file in output_files]

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


class ASEOptimizationRunner(BaseOptimizationRunner):
    def __init__(self, config, output_dir, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.run_count = 0
        self.load_config_with_defaults(config)

    def load_config_with_defaults(self, config):
        super().load_config_with_defaults(config)
        self.options = OptimizationConfiguration(**config)
    
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
        opt.run(fmax=self.options.fmax, steps=self.options.steps)
        self.run_count = self.run_count + 1
        return atoms


class SciPyOptimizationRunner(BaseOptimizationRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def run(self):
        ...


class BetterOptimizationRunner(ASEOptimizationRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def run(self):
        import torch
        import time
        if not os.environ["CALCULATOR"].lower() == "fairchem":
            raise NotImplementedError(f"BetterOptimizationRunner is not currently written for {os.environ["CALCULATOR"]} calculator.")
        from mlipenv.calculators import get_fairchem_predict_unit
        from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
        predictor = get_fairchem_predict_unit(self.calculator_options.device)
        optimizers = [BetterBFGS(atoms, coords, charge, idx)
                      for idx, (atoms, coords, charge) in enumerate(zip(self.atoms, self.coordinates, self.charge))]
        torch_times = []
        bfgs_times = []
        for i in range(self.options.steps):
            unconverged_optimizers = [optimizer for optimizer in optimizers if not optimizer.converged]
            logger.info(f"optimization step: {i}. num unconverged = {len(unconverged_optimizers)}/{len(optimizers)}")
            if not len(unconverged_optimizers):
                break
            # if self.debug and self.debug.lower() == "true":
            #     print(f"step {i}")
            #     print(f"num unconverged = {len(unconverged_optimizers)}")
            atomic_data = [AtomicData.from_ase(optimizer.ase_atoms, task_name="omol")
                           for optimizer in unconverged_optimizers]
            batch = atomicdata_list_to_batch(atomic_data)
            t1 = time.time()
            with torch.no_grad():
                preds = predictor.predict(batch)
            torch_times.append(time.time()-t1)
            t2 = time.time()
            for j, optimizer in enumerate(unconverged_optimizers):
                optimizer.remember_energy(preds["energy"][j])
                optimizer.optimize_and_update(preds["forces"][batch.batch == j], self.options.fmax)
            bfgs_times.append(time.time()-t2)
        if self.debug and self.debug.lower() == "true":
            print(f"torch times = {torch_times}")
            print(f"bfgs times = {bfgs_times}")
        self.results = optimizers

    def get_atom_symbols(self, obj):
        return np.array(obj.get_atoms())
    def get_coordinates(self, obj):
        return np.array(obj.get_coordinates())
    def get_gradients(self, obj):
        return np.array(obj.get_forces())
    def get_single_point_energy(self, obj):
        return np.array(obj.get_energy())
    
    def export_results_subroutine(self, output_dir):
        os.makedirs(os.path.join(output_dir, "trajectories"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        for optimizer in self.results:
            optimizer.write_trajectory(output_dir)
            optimizer.write_log(output_dir)

