import os
import abc

import numpy as np
from ase import Atoms

from src.calculators import get_calc
from src.optimization_options import ASEOptimizationConfiguration, EnergyConfiguration
from src.enums.output_enum import _output_file_registry

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
        # not explicitly enforcing alignment here. relying on default ordering given by
        # self.result_getters and src.enums.output_enum._output_file_registry
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
    # option to write .trj and .log files to a specified place


class ASEOptimizationRunner(BaseOptimizationRunner):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
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
        opt = BFGS(atoms)
        opt.run(fmax=self.options.fmax, steps=self.options.steps)
        return atoms


class SciPyOptimizationRunner(BaseOptimizationRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def run(self):
        ...


class MarksOptimizationRunner(BaseOptimizationRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def run(self):
        ...
