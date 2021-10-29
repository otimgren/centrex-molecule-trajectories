import numpy as np
from dataclasses import dataclass
from centrex_TlF.states import UncoupledBasisState, State
from scipy.constants import g

@dataclass
class Molecule:
    """
    Defines a molecule object
    """
    state: State = 1*UncoupledBasisState(J = 2, mJ = 0, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = -1/2, Omega = 0,
                                            P = +1, electronic_state = 'X')# Molecular state assumed for the molecule to calculate trajectory inside electrostatic lens
    alive: bool = True
    mass: float = (204.38+19.00)*1.67e-27 # Molecular mass in kg

    def init_trajectory(self, beamline, x0 =  np.array((0,0,0)), v0 = np.array((0,0,200)), 
                        a0 = np.array((0,-g,0)), t0 = 0):
        """
        Initializes a molecular trajectory object based on a given beamline.
        """
        # Initialize trajectory
        self.trajectory = Trajectory(beamline)

        # Update the initial state of the molecule to the trajectory
        self.trajectory.update(x0, v0, a0, t0)

    def x(self, delta_t: float = None):
        """
        Returns the current position of the molecule if delta_t == None, or the position of the molecule after
        time delta_t if delta_t is provided.
        """
        if not delta_t:
            return self.trajectory.x[self.trajectory.n,:]
        else:
            return self.x() + self.v()*delta_t + self.a()*delta_t**2/2

    def v(self, delta_t: float = None):
        """
        Returns the current velocity of the molecule if delta_t == None, or the velocity of the molecule after
        time delta_t if delta_t is provided.
        """
        if not delta_t:
            return self.trajectory.v[self.trajectory.n,:]
        else:
            return self.v() + self.a()*delta_t

    def a(self):
        """
        Returns the current acceleration of the molecule
        """
        return self.trajectory.a[self.trajectory.n,:]

    def t(self):
        """
        Returns the current time of the molecule
        """
        return self.trajectory.t[self.trajectory.n]

    def update_trajectory(self, delta_t, a = np.array((0,-g,0))):
        """
        Calculates the position of the molecule after time delta_t (s) has passed and updates the trajectory.
        This function is used in the parts of the beamline where the acceleration is constant (i.e. everywhere
        except the electrostatic lens)
        """
        x = self.x(delta_t)
        v = self.v(delta_t)
        t = self.t() + delta_t

        self.trajectory.update(x,v,a,t)

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

    def update(self, x, v, a, t):
        """
        Adds the current position of molecule to trajectory arrays at index n and increments n by one
        """
        # Update arrays
        self.x[self.n,:] = x
        self.v[self.n,:] = v
        self.a[self.n,:] = a
        self.t[self.n] = t 

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

        