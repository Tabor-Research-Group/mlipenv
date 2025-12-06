import os

import numpy as np
from ase import Atoms

class BetterBFGS:

    def __init__(self, atoms, coordinates, charge, index):
        self.atoms = atoms
        self.index = index
        self.ase_atoms = Atoms(atoms, coordinates)
        self.ase_atoms.info.update({"charge": charge})
        self.coordinates_history = [coordinates]
        self.forces_history = []
        self.prev_hessian = None
        self.energy_history = []
        self.converged = False

    def remember_energy(self, energy):
        self.energy_history.append(energy)

    def get_atoms(self):
        return self.atoms
    def get_coordinates(self):
        return self.coordinates_history[-1]
    def get_forces(self):
        return self.forces_history[-1]
    def get_energy(self):
        return self.energy_history[-1]
    
    def write_trajectory(self, output_dir):
        from ase import Atoms
        from ase.io import Trajectory
        with Trajectory(os.path.join(output_dir, "trajectories", f"{self.index}.traj"), "w") as traj:
            for coordinates in self.coordinates_history:
                traj.write(Atoms(self.atoms, coordinates))
    
    def write_log(self, output_dir):
        with open(os.path.join(output_dir, "logs", f"{self.index}.log"), "w") as f:
            f.write(f"{'step':^4} {'energy':^24} {'fmax':^11}\n")
            for idx, (energy, forces) in enumerate(zip(self.energy_history, self.forces_history)):
                f.write(f"{idx:^4} {energy:^24} {np.max(forces):^11.5}\n")

    def bfgs_hess_approximation(self):
        dr = (self.coordinates_history[-1] - self.coordinates_history[-2]).flatten()
        dg = - (self.forces_history[-1] - self.forces_history[-2])
        H = self.prev_hessian
        dkg = H @ dr
        alpha = 1 / (dr.T @ dg.flatten())
        beta = -1 / (dr.T @ dkg)
        H_next = H + alpha*np.outer(dg, dg) + beta*np.outer(dkg, dkg)
        self.prev_hessian = H_next

    def compute_steps(self):
        l, Q = np.linalg.eig(self.prev_hessian)
        steps = Q @ (( Q.T @ self.forces_history[-1].flatten()) / l)
        return np.reshape(steps, self.coordinates_history[-1].shape)
    
    def optimize_and_update(self, predicted_forces, fmax):
        self.forces_history.append(np.asanyarray(predicted_forces))
        if np.max(self.forces_history[-1]) < fmax:
            self.converged = True
            return
        if self.prev_hessian is not None:
            self.bfgs_hess_approximation()
        else:
            self.prev_hessian = np.eye(self.coordinates_history[0].size) * 70
        new_coordinates = self.step_gradient_descent()
        self.coordinates_history.append(new_coordinates)
        self.ase_atoms.set_positions(new_coordinates)

    def step_gradient_descent(self, step_length_cap=0.2):
        steps = self.compute_steps()
        step_lengths = [np.linalg.norm(s) for s in steps]
        regularization = min(1, step_length_cap/max(step_lengths))
        return self.coordinates_history[-1] + regularization*steps
