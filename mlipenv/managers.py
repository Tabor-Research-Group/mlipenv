import os
import abc
import traceback
import logging

import numpy as np

from mlipenv.runners import ASEOptimizationRunner, SciPyOptimizationRunner, BetterOptimizationRunner, EnergyRunner

logger = logging.getLogger(__name__)

class BaseManager:
    def __init__(self, config):
        self.atoms = self.load_atoms(config.atoms)
        self.coordinates = self.load_coordinates(config.coordinates)
        self.charge = config.charge
        self.spin = config.spin
        self.output_dir = config.output_dir
        self.calculator_options = config.calculator_options

    def _load_parameter(self, parameter_bundle):
        if isinstance(parameter_bundle, str):
            if os.path.exists(parameter_bundle):
                data = np.load(parameter_bundle)
                if len(data.files) > 1:
                    parameter = [data[key] for key in data.files]
                else:
                    parameter = data[data.files[0]]
                    # runners enforce one-at-a-time batching, if the input
                    # is a multi-dim np array, we splice it up here.
                    if parameter.ndim > 2:
                        parameter = [parameter[i] for i in parameter.shape[0]]
                    else:
                        parameter = [parameter]
        elif isinstance(parameter_bundle, list):
            parameter = parameter_bundle
        return parameter

    def load_coordinates(self, coordinates_bundle):
        try:
            coordinates = self._load_parameter(coordinates_bundle)
        except Exception as e:
            traceback.print_exc()
            raise NotImplementedError(f"Could not load coordinates from: {coordinates_bundle}")
        return coordinates
    
    def load_atoms(self, atoms_bundle):
        try:
            atoms = self._load_parameter(atoms_bundle)
        except Exception as e:
            traceback.print_exc()
            raise NotImplementedError(f"Could not load atoms from: {atoms_bundle}")
        return atoms
        
    @abc.abstractmethod
    def run(self):
        ...


class EnergyManager(BaseManager):
    def __init__(self, config):
        super().__init__(config)
        self.config = config.options

    def run(self):
        self.compute_energy()
    
    def compute_energy(self):
        energy_runner = EnergyRunner(**self.__dict__)
        energy_runner.run()
        energy_runner.export_results(self.output_dir)


class OptimizationManager(BaseManager):
    def __init__(self, config):
        super().__init__(config)
        self.config = config.options

    def get_optimization_scheme(self):
        if "optimizer" not in self.config:
            logger.warning("'optimizer' was not included in the 'options' dictionary. using default optimizer.")
            return BetterOptimizationRunner(**self.__dict__)
        requested_optimizer = self.config["optimizer"]
        if "better" in requested_optimizer or "default" in requested_optimizer:
            return BetterOptimizationRunner(**self.__dict__)
        elif requested_optimizer == "ase":
            return ASEOptimizationRunner(**self.__dict__)
        # elif requested_optimizer == "scipy":
        #     return SciPyOptimizationRunner(**self.__dict__)
        else:
            raise NotImplementedError(f"Unknown optimizer: {requested_optimizer}")
    
    def run(self):
        optimization_runner = self.get_optimization_scheme()
        optimization_runner.run()
        optimization_runner.export_results(self.output_dir)
