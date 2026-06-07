import os
import abc
import logging

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
        return runner_factory

def get_runner(key):
    if isinstance(key, str):
        return RUNNER_REGISTRY[key]
    else:
        return RUNNER_REGISTRY.get(key, key)

class BaseRunner(metaclass=abc.ABCMeta):
    def __init__(self, base_config, **kwargs):
        self.config = base_config

    @abc.abstractmethod
    def run(self):
        ...

    @abc.abstractmethod
    def export_results(self):
        ...

class NPZBatchExportRunner(BaseRunner):
    debug = os.environ.get("DEBUG")
    def __init__(self, base_config, **kwargs):
        super().__init__(base_config)
        self.atoms = load_multidim_parameter(base_config.atoms)
        self.coordinates = np.asarray(load_multidim_parameter(base_config.coordinates), dtype=np.float32)
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
    def result_getters(self):
        ...
    
    @abc.abstractmethod
    def get_output_with_defaults(self):
        ...

@register_runner("energy")
class EnergyRunner(NPZBatchExportRunner):
    def __init__(self, base_config, **kwargs):
        super().__init__(base_config)
        self.load_config_with_defaults(**kwargs)

    def load_config_with_defaults(self, order=1, **kwargs):
        self.energy_options = get_configuration("energy")(order)
    
    def result_getters(self):
        getters = [self.get_single_point_energy]
        if self.energy_options.order > 0:
            getters.append(self.get_gradients)
        return getters
    
    def get_output_with_defaults(self):
        output_file_registry = _output_file_registry()
        output_files = [output_file_registry["energies"].value]
        if self.energy_options.order > 0:
            output_files.append(output_file_registry["gradients"].value)
        return [os.path.join(self.output_dir, file) for file in output_files]
    
    def run(self):
        self.results = [self.atomize(atoms, coords, charge) 
                           for atoms, coords, charge in zip(self.atoms, self.coordinates, self.charge)]

    def get_gradients(self, obj):
        return np.array(obj.get_forces())
    def get_single_point_energy(self, obj):
        return np.array(obj.get_potential_energy())

class BaseOptimizationRunner(NPZBatchExportRunner):
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
        config_type = os.environ["CALCULATOR"].lower()
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
        import time
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
        print(f"running opt {self.run_count} for {self.optimization_options.steps} steps...")
        opt.run(fmax=self.optimization_options.fmax, steps=self.optimization_options.steps)
        print(f"done.")
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
        if not os.environ["CALCULATOR"].lower() == "fairchem":
            calc = os.environ["CALCULATOR"]
            raise NotImplementedError(f"BetterOptimizationRunner is not currently written for {calc} calculator.")
        super().__init__(base_config, **kwargs)
    
    def run(self):
        import torch
        import time
        t1=time.time()
        from mlipenv.calculators import get_fairchem_predict_unit
        from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
        predictor = get_fairchem_predict_unit(self.calculator_options.device)
        logger.info(f"loading time for calculator: {time.time()-t1:.3f} seconds.")
        optimizers = [BetterBFGS(atoms, coords, charge, self.spin, idx)
                      for idx, (atoms, coords, charge) in enumerate(zip(self.atoms, self.coordinates, self.charge))]
        torch_times = []
        bfgs_times = []
        for i in range(self.optimization_options.steps):
            unconverged_optimizers = [optimizer for optimizer in optimizers if not optimizer.converged]
            logger.info(f"optimization step: {i}. num unconverged = {len(unconverged_optimizers)}/{len(optimizers)}")
            if not len(unconverged_optimizers):
                break
            # if self.debug and self.debug.lower() == "true":
            #     print(f"step {i}")
            #     print(f"num unconverged = {len(unconverged_optimizers)}")
            atomic_data = [AtomicData.from_ase(optimizer.ase_atoms, task_name="omol", r_data_keys=["charge", "spin"])
                           for optimizer in unconverged_optimizers]
            batch = atomicdata_list_to_batch(atomic_data)
            t1 = time.time()
            with torch.no_grad():
                preds = predictor.predict(batch)
            torch_times.append(time.time()-t1)
            t2 = time.time()
            for j, optimizer in enumerate(unconverged_optimizers):
                optimizer.remember_energy(preds["energy"][j])
                optimizer.optimize_and_update(preds["forces"][batch.batch == j], self.optimization_options.fmax)
            bfgs_times.append(time.time()-t2)
            # if self.debug and self.debug.lower() == "true":
            #     if hasattr(predictor.model.module.backbone.charge_embedding, "charges"):
            #         all_charges = predictor.model.module.backbone.charge_embedding.charges
            #         for atoms, charges in zip([uo.atoms for uo in unconverged_optimizers], all_charges):
            #             print(atoms, charges, len(unconverged_optimizers))
            #             for atom, charge in zip(atoms, charges):
            #                 print(atom, charge)
            #             print()
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
    
    def export_results_subroutine(self):
        os.makedirs(os.path.join(self.output_dir, "trajectories"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        for optimizer in self.results:
            optimizer.write_trajectory(self.output_dir)
            optimizer.write_log(self.output_dir)

@register_runner("psience")
class PsienceRunner(BaseRunner):
    def __init__(self, base_config, *, tasks, output_file, order=None, **kwargs):
        super().__init__(base_config)
        if isinstance(tasks, str):
            tasks = [tasks]
        self.tasks = tasks
        self.mol_kwargs = kwargs
        self._results = []
        self._atoms = base_config.atoms
        self._coords = base_config.coordinates
        self.order = order
        self.output_file = output_file
        self._ref_mol = None
        self._evaluators = {}

    def load_ref_mol(self):
        from Psience.Molecools import Molecule
        from McUtils.Data import UnitsData

        if self._ref_mol is None:
            coords = self._coords * UnitsData.convert("BohrRadius", "Angstroms")
            if coords.ndim > 2:
                coords = coords.reshape((-1,) + coords.shape[-2:])[0]
            self._ref_mol = Molecule(
                self._atoms,
                coords,
                **self.mol_kwargs
            )
        return self._ref_mol

    def load_evaluator(self, key):
        if key == 'energy':
            return self._ref_mol.get_energy_evaluator()
        elif key == 'charge':
            return self._ref_mol.get_charge_evaluator()
        elif key == 'dipole':
            return self._ref_mol.get_dipole_evaluator()
        else:
            raise ValueError(f"Unknown evaluator {key}")

    def dispatch_on_task(self, task):
        if task == 'energy':
            return self.load_evaluator('energy')(self._coords, order=self.order)
        else:
            raise ValueError(f"Unknown task {task}")

    def run(self):
        if len(self._results) == 0:
            for task in self.tasks:
                subres = self.dispatch_on_task(task)
                self._results.append(subres)

    def export_results(self):
        from McUtils.Scaffolding import write_flat_tree
        write_flat_tree(self.output_file, {'results': self._results})
        return {'output_file': self.output_file}
