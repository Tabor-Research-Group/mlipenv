import os
import abc

import numpy as np

from src.runners import ASEOptimizationRunner, SciPyOptimizationRunner, MarksOptimizationRunner
from src.optimization_options import OptimizationConfiguration

class BaseManager:
    def __init__(self, config):
        self.atoms = self.load_atoms(config.atoms)
        self.coordinates = self.load_coordinates(config.coordinates)
        self.output_dir = config.output_dir

    def _load_parameter(self, parameter_bundle):
        if isinstance(parameter_bundle, str):
            if os.path.exists(parameter_bundle):
                data = np.load(parameter_bundle)
                parameter = [data[key] for key in data.files]
        elif isinstance(parameter_bundle, list):
            parameter = parameter_bundle
        return parameter

    def load_coordinates(self, coordinates_bundle):
        try:
            coordinates = self._load_parameter(coordinates_bundle)
        except Exception as e:
            raise NotImplementedError(f"Could not load coordinates from: {coordinates_bundle}") from e
        return coordinates
    
    def load_atoms(self, atoms_bundle):
        try:
            atoms = self._load_parameter(atoms_bundle)
        except Exception as e:
            raise NotImplementedError(f"Could not load atoms from: {atoms_bundle}") from e
        return atoms
        
    @abc.abstractmethod
    def run(self):
        ...


class EnergyManager(BaseManager):
    def __init__(self, config):
        super().__init__(config)
    
    def run(self):
        self.compute_energy()
    
    def compute_energy(self):
        ...

class OptimizationManager(BaseManager):
    def __init__(self, config):
        super().__init__(config)
        self.config = OptimizationConfiguration(**config.options)

    def get_optimization_scheme(self):
        requested_optimizer = self.config.optimizer.lower()
        if requested_optimizer == "ase":
            return ASEOptimizationRunner(self.atoms, self.coordinates, self.config)
        elif requested_optimizer == "scipy":
            return SciPyOptimizationRunner(self.atoms, self.coordinates, self.config)
        elif "mark" in requested_optimizer:
            return MarksOptimizationRunner(self.atoms, self.coordinates, self.config)
        else:
            raise NotImplementedError(f"Unknown optimizer: {requested_optimizer}")
    
    def run(self):
        optimization_runner = self.get_optimization_scheme()
        optimization_runner.run()
        optimization_runner.export_results(self.output_dir)