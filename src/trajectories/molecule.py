from pathlib import Path
import h5py
from h5py._hl import group
import numpy as np
from dataclasses import dataclass
from scipy.constants import g

@dataclass
class Molecule:
    """
    Defines a molecule object
    """
    alive: bool = True

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
            return self.trajectory.x[self.trajectory.n-1,:]
        else:
            return self.x() + self.v()*delta_t + self.a()*delta_t**2/2

    def v(self, delta_t: float = None):
        """
        Returns the current velocity of the molecule if delta_t == None, or the velocity of the molecule after
        time delta_t if delta_t is provided.
        """
        if not delta_t:
            return self.trajectory.v[self.trajectory.n-1,:]
        else:
            return self.v() + self.a()*delta_t

    def a(self):
        """
        Returns the current acceleration of the molecule
        """
        return self.trajectory.a[self.trajectory.n-1,:]

    def t(self):
        """
        Returns the current time of the molecule
        """
        return self.trajectory.t[self.trajectory.n-1]

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

    def plot_trajectory(self, axes):
        """
        Plots the trajectory of the molecule
        """
        
        if self.aperture_hit == 'Detected':
            color = 'g'
        elif self.aperture_hit == 'Field plates':
            color = 'r'
        else:
            color = 'k'

        axes[0].plot(self.trajectory.x[:,2], self.trajectory.x[:,0], c = color)
        axes[1].plot(self.trajectory.x[:,2], self.trajectory.x[:,1], c = color)

    def save_to_hdf(self, file: h5py.File, run_name: str, group_name: str):
        """
        Saves the trajectory of the molecule and some info about it to an hdf file
        """
        # Save the trajectory
        self.trajectory.save_to_hdf(file, run_name, group_name)

        # Save info about molecule to attributes
        file[run_name + '/' + group_name].attrs['aperture_hit'] = self.aperture_hit
        file[run_name + '/' + group_name].attrs['alive'] = self.alive




class Trajectory:
    """
    Class that stores information about the trajectory of a molecule flying through the beamline.
    """
    def __init__(self, beamline):
        
        # Calculate number of positions where points in the trajectory will be stored
        N_steps = 10 # Initial position and a few extras

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

    def add_steps(self, beamline):
        """
        Adds a number of steps to the trajectories based on given part of beamline
        """
        N_steps = 0
        # Loop over the beamline elements and determine how many steps are required for each
        for element in beamline.elements:
            N_steps += element.N_steps()

        self.x = np.concatenate((self.x, np.full((N_steps, 3), np.nan)))
        self.v = np.concatenate((self.v, np.full((N_steps, 3), np.nan)))
        self.a = np.concatenate((self.a, np.full((N_steps, 3), np.nan)))
        self.t = np.concatenate((self.t, np.full((N_steps, ), np.nan)))

    def drop_nans(self):
        """
        Drops nans from the trajectory arrays.
        """
        self.x = self.x[np.all(np.isfinite(self.x), axis = 1),:]
        self.v = self.v[np.all(np.isfinite(self.v), axis = 1),:]
        self.a = self.a[np.all(np.isfinite(self.a), axis = 1),:]
        self.t = self.t[np.isfinite(self.t)]

    def save_to_hdf(self, file: h5py.File, run_name: str, group_name: str) -> None:
        """
        Saves the trajectory to an hdf file.
        """
        # Start by getting rid of any nans
        self.drop_nans()

        # Open the hdf file and save the positions, velocities, accelerations and times
        # Create the group
        group_path = run_name + '/' + group_name
        file.create_group(group_path)

        # Add datasets to the group
        file[group_path].create_dataset("x", data = self.x)
        file[group_path].create_dataset("v", data = self.v)
        file[group_path].create_dataset("a", data = self.a)
        file[group_path].create_dataset("t", data = self.t)
    

        