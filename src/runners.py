import os
import abc

import numpy as np
from ase import Atoms

from src.calculators import get_calc
from src.optimization_options import ASEOptimizerConfiguration

class BaseOptimizationRunner:
    def __init__(self, atoms, coordinates):
        self.atoms = atoms
        self.coordinates = coordinates

    def export_results(self, output_dir):
        atom_symbols, coordinates, gradients, energies = self.format_results()
        np.savez(os.path.join(output_dir, "atoms.npz"), atom_symbols)
        np.savez(os.path.join(output_dir, "coordinates.npz"), coordinates)
        np.savez(os.path.join(output_dir, "gradients.npz"), gradients)
        np.savez(os.path.join(output_dir, "energies.npz"), energies)

    def format_results(self):
        return [[f(obj) for obj in self.results] for f in self.result_getters()]
    
    def result_getters(self):
        return [self.get_atom_symbols, self.get_coordinates, self.get_gradients, self.get_single_point_energy]
        
    @abc.abstractmethod
    def run(self):
        ...

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
    # option to write .trj and .log files to a specified place


class ASEOptimizationRunner(BaseOptimizationRunner):
    def __init__(self, atoms, coordinates, config):
        super().__init__(atoms, coordinates)
        self.load_config_with_defaults(config)

    def load_config_with_defaults(self, config):
        self.options = ASEOptimizerConfiguration(**config.options)
        if isinstance(self.options.charge, float):
            self.options.charge = [self.options.charge] * len(self.coordinates)
        if len(self.atoms) < len(self.coordinates):
            self.atoms = self.atoms[:-1] + [self.atoms[-1]] * (len(self.coordinates) - len(self.atoms) + 1)
    
    def run(self):
        optimized_atoms = [self.run_opt(atoms, coords, charge) 
                           for atoms, coords, charge in zip(self.atoms, self.coordinates, self.options.charge)]
        self.results = optimized_atoms
    
    def get_atom_symbols(self, obj):
        return np.array(obj.get_chemical_symbols())
    def get_coordinates(self, obj):
        return np.array(obj.get_positions())
    def get_gradients(self, obj):
        return np.array(obj.calc.get_forces())
    def get_single_point_energy(self, obj):
        return np.array(obj.calc.get_potential_energy())

    def run_opt(self, atom_symbols, coordinates, charge):
        atoms = Atoms(symbols=atom_symbols, positions=coordinates)
        atoms.info["charge"] = charge
        atoms.info["spin"] = self.options.spin
        calc = get_calc()
        atoms.calc = calc
        atoms = self._run_opt(atoms)
        return atoms
    
    def _run_opt(self, atoms):
        from ase.optimize import BFGS
        opt = BFGS(atoms)
        opt.run(fmax=self.options.fmax, steps=self.options.steps)
        return atoms


class SciPyOptimizationRunner(BaseOptimizationRunner):
    def __init__(self, atoms, coordinates, config):
        super().__init__(atoms, coordinates)
    
    def run(self):
        ...


class MarksOptimizationRunner(BaseOptimizationRunner):
    def __init__(self, atoms, coordinates, config):
        super().__init__(atoms, coordinates)
    
    def run(self):
        ...
