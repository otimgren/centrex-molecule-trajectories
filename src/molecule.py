import numpy as np
from dataclasses import dataclass
from centrex_TlF.states import UncoupledBasisState, State

@dataclass
class Molecule:
    """
    Defines a molecule object
    """
    state: State = 1*UncoupledBasisState(J = 2, mJ = 0, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = -1/2, Omega = 0,
                                            P = +1, electronic_state = 'X')# Molecular state assumed for the molecule to calculate trajectory inside electrostatic lens
    x: np.ndarray = np.array((0,0,0))
    v: np.ndarray = np.array((0,0,200))
    a: np.ndarray = np.array((0,-9.81,0))
    t: float = 0.
    alive: bool = True
    mass: float = (204.38+19.00)*1.67e-27 # Molecular mass in kg

    def __post_init__(self):
        # Check that velocity and position are of correct shape
        assert self.x.shape == (3,), f"Input x0 shape: {self.x.shape}. Needs to be (3,)."
        assert self.v.shape == (3,), f"Input v0 shape: {self.v.shape}. Needs to be (3,)."
        assert self.a.shape == (3,), f"Input a0 shape: {self.a.shape}. Needs to be (3,)."

    def init_trajectory(self, beamline):
        """
        Initializes a molecular trajectory object based on a given beamline.
        """
        self.trajectory = Trajectory(beamline)


    def update_trajectory(self):
        """
        Adds current position, velocity, acceleration and time to molecule trajectory.
        """
        self.trajectory.update(self)

    def update_position(self, delta_t):
        """
        Calculates the position of the molecule after time delta_t (s) has passed and updates the attributes
        """
        self.x = self.x + self.v*delta_t + self.a*delta_t**2/2
        self.v = self.v + self.a*delta_t
        self.t += delta_t

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

    def plot_trajectory(self, axes, ls: str = 'k'):
        """
        Plots the trajectory of the molecule
        """
        axes[0].plot(self.trajectory.x[:,2], self.trajectory.x[:,0], ls)
        axes[1].plot(self.trajectory.x[:,2], self.trajectory.x[:,1], ls)

class Trajectory:
    """
    Class that stores information about the trajectory of a molecule flying through the beamline.
    """
    def __init__(self, beamline):
        
        # Calculate number of positions where points in the trajectory will be stored
        N_steps = 1 # Initial position

        # Loop over the beamline elements and determine how many steps are required for each
        for element in beamline.elements:
            N_steps += element.N_steps()

        # Make arrays where the trajectory is stored:
        self.x = np.full((N_steps, 3), np.nan)
        self.v = np.full((N_steps, 3), np.nan)
        self.a = np.full((N_steps, 3), np.nan)
        self.t = np.full((N_steps,), np.nan)

        # Initialize a counter to keep track of which position of the arrays we are in
        self.n = 0

    def update(self, molecule):
        """
        Adds the current position of molecule to trajectory arrays at index n and increments n by one
        """
        # Update arrays
        self.x[self.n,:] = molecule.x
        self.v[self.n,:] = molecule.v
        self.a[self.n,:] = molecule.a
        self.t[self.n] = molecule.t 

        # Increment the index counter
        self.n += 1

    def drop_nans(self):
        """
        Drops nans from the trajectory arrays.
        """
        self.x = self.x[np.all(np.isfinite(self.x), axis = 1),:]
        self.v = self.v[np.all(np.isfinite(self.v), axis = 1),:]
        self.a = self.a[np.all(np.isfinite(self.a), axis = 1),:]
        self.t = self.t[np.isfinite(self.t)]

        