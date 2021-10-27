import numpy as np
from dataclasses import dataclass
from centrex_TlF.states import UncoupledBasisState, State

@dataclass
class Molecule:
    """
    Defines a molecule object
    """
    state: State = 1*UncoupledBasisState(J = 2, mJ = 0, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2= 1/2)# Molecular state assumed for the molecule to calculate trajectory inside electrostatic lens
    x: np.ndarray = np.array((0,0,0))
    v: np.ndarray = np.array((0,0,200))
    a: np.ndarray = np.array((0,-9.81,0))
    t: float = 0.
    alive: bool = True
    mass: float = (204.38+19.00)*1.67e-27 # Molecular mass in kg

    def __init__(self):
        # Check that velocity and position are of correct shape
        assert self.x.shape == (3,), f"Input x0 shape: {self.x.shape}. Needs to be (3,)."
        assert self.v.shape == (3,), f"Input v0 shape: {self.v.shape}. Needs to be (3,)."

        # Initialize containers for molecule trajectory
        self.initialize_trajectory()

    def initialize_trajectory(self):
        """
        Function that initializes a trajectory container for the molecule. Currently a dictionary of lists that
        contain numpy arrays. Consider initializing as ndarray of correct size for speeding up simulation
        """
        self.trajectory = {"x" : [], "v" : [], "a" : [], "t" : []}

    def update_trajectory(self):
        """
        Adds current position, velocity, acceleration and time to molecule trajectory.
        """
        self.trajectory["x"].append(self.x)
        self.trajectory["v"].append(self.v)
        self.trajectory["a"].append(self.a)
        self.trajectory["t"].append(self.t)

    def update_position(self, delta_t):
        """
        Calculates the position of the molecule after time delta_t (s) has passed and updates the attributes
        """
        self.x = self.x + self.v*delta_t + self.a*delta_t**2/2
        self.v = self.v + self.a*delta_t
        self.t += delta_t


    def trajectory_to_arrays(self):
        """
        Converts the trajectory lists into np.arrays
        """
        for key, item in self.trajectory.items():
            self.trajectory[key] = np.array(item)

    def set_aperture_hit(self, aperture_name):
        """
        Sets an attribute that tells what aperture the molecule hit
        """
        self.aperture_hit = aperture_name

    def set_dead(self):
        """
        Sets the molecule to dead (or at least not alive...)
        """
        self.alive = False


def main():
    molecule = Molecule(np.array((0,0,0)), np.array((0,0,0)))
    molecule.update_trajectory()
    molecule.update_trajectory()
    molecule.update_trajectory()
    molecule.update_trajectory()
    molecule.trajectory_to_arrays()
    # print(molecule.trajectory)
    molecule.set_aperture_hit("4K shield")
    print(molecule.aperture_hit)
if __name__ == "__main__":
    main()