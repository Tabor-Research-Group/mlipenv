import os
import abc

import numpy as np
from ase import Atoms

from src.calculators import get_calc
from src.optimization_options import ASEOptimizationConfiguration, EnergyConfiguration
from src.enums.output_enum import _output_file_registry
from src.optimizers import BetterBFGS

class BaseRunner:
    def __init__(self, atoms, coordinates, charge, spin, **kwargs):
        self.atoms = atoms
        self.coordinates = coordinates
        self.charge = charge
        self.spin = spin

    def load_config_with_defaults(self, config):
        if isinstance(self.charge, int):
            self.charge = [self.charge] * len(self.coordinates)
        if len(self.atoms) < len(self.coordinates):
            self.atoms = self.atoms[:-1] + [self.atoms[-1]] * (len(self.coordinates) - len(self.atoms) + 1)
    
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
        calc = get_calc()
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
        self.options = ASEOptimizationConfiguration(**config)
    
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
        if not os.environ["CALCULATOR"].lower() == "fairchem":
            raise NotImplementedError(f"BetterOptimizationRunner is not currently written for {os.environ["CALCULATOR"]} calculator.")
        from src.calculators import get_fairchem_predict_unit
        from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
        predictor = get_fairchem_predict_unit()
        optimizers = [BetterBFGS(atoms, coords, charge, idx)
                      for idx, (atoms, coords, charge) in enumerate(zip(self.atoms, self.coordinates, self.charge))]
        for i in range(self.options.steps):
            print(f"step {i}")
            # if oom is ever encountered, schedule this with a dataloader
            atomic_data = [AtomicData.from_ase(optimizer.ase_atoms, task_name="omol")
                           for optimizer in optimizers if not optimizer.converged]
            batch = atomicdata_list_to_batch(atomic_data)
            with torch.no_grad():
                preds = predictor.predict(batch)
            for j, optimizer in enumerate(optimizers):
                if not optimizer.converged:
                    optimizer.remember_energy(preds["energy"][j])
                    optimizer.optimize_and_update(preds["forces"][batch.batch == j], self.options.fmax)
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
    

    # def run_opt(self, atom_symbols, coordinates, charge):
    #     atoms = self.atomize(atom_symbols, coordinates, charge)
    #     atoms = self._run_opt(atoms)
    #     return atoms
    
    # def _run_opt(self, atoms):
    #     from ase.optimize import BFGS
    #     os.makedirs(os.path.join(self.output_dir, "trajectories"), exist_ok=True)
    #     os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
    #     opt = BFGS(
    #         atoms, 
    #         trajectory=os.path.join(self.output_dir, "trajectories", f"{self.run_count}.traj"),
    #         logfile=os.path.join(self.output_dir, "logs", f"{self.run_count}.log")
    #         )
    #     opt.run(fmax=self.options.fmax, steps=self.options.steps)
    #     self.run_count = self.run_count + 1
    #     return atoms
