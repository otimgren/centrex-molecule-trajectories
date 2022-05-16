from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import scale
from scipy.stats import norm, uniform


class Distribution(ABC):
    """
    Abstract class that defines methods required for distributions (e.g. initial velocity and position 
    distributions)
    """

    @abstractmethod
    def draw(self, n: int) -> np.ndarray:
        """
        Draws n samples from the distribution
        """
        ...

    @abstractmethod
    def save_to_hdf(self, filepath: Path, run_name: str, group_name:str):
        """
        Save the distribution to an hdf file
        """
        ...

@dataclass
class GaussianDistribution(Distribution):
    mean: float  # Mean of gaussian
    sigma: float  # Spread of gaussian

    def __post_init__(self):
        self.dist = norm

    def draw(self, n: int) -> np.ndarray:
        return self.dist.rvs(loc = self.mean, scale = self.sigma, size = n)

    def plot(self):
        fig, ax = plt.subplots()
        x = self.mean + np.linspace(-5,5,1000)*self.sigma
        ax.plot(x, self.dist.pdf(x, loc = self.mean, scale = self.sigma))

    def save_to_hdf(self, filepath: Path, run_name: str, group_name: str):
        """
        Method needs to be defined but is not implemented
        """
        raise NotImplementedError("Saving GaussianDistribution to hdf is not implemented. Save child class instead.") 

@dataclass
class CeNTREXVelocityDistribution(Distribution):
    """
    This is the "standard" initial velocity distribution that I expect will be used in most simulations.
    Gaussian velocity distributions for each direction, but along Z we have a non-zero mean and smaller
    spread. The values are based on the initial beamsource characterization results (contact Oskari Timgren
    if you need the details).
    """
    vx: float = 0. # velocity along X in m/s
    sigmax: float = 39.5 # Spread in velocity along X in m/s
    vy: float = 0. # velocity along y in m/s
    sigmay: float = 39.5 # Spread in velocity along Y in m/s
    vz: float = 184.0 # velocity along Z in m/s
    sigmaz: float = 16.0  # Spread in velocity along Z in m/s

    def draw(self, n: int) -> np.ndarray:
        """
        Draws n samples from the distribution
        """
        return np.array((GaussianDistribution(self.vx, self.sigmax).draw(n),
                         GaussianDistribution(self.vy, self.sigmay).draw(n),
                         GaussianDistribution(self.vz, self.sigmaz).draw(n)
                        ))
    
    def save_to_hdf(self, filepath: Path, run_name: str):
        """
        Saves the distribution to an hdf file
        """
        # Open the hdf file
        with h5py.File(filepath, 'a') as f:
            try:
                # Create a group for the velocity distribution
                group_path = run_name + "/velocity_distribution" 
                f.create_group(group_path)

                # Write the name of the distribution class into file
                f[group_path].attrs['class'] = type(self).__name__

                # Loop over the attributes of the velocity distribution and save them to the attributes
                # of the group
                for key, value in vars(self).items():
                    f[group_path].attrs[key] = value
            
            except ValueError:
                raise ValueError("Can't save velocity distribution. Group already exists!") 
    

@dataclass
class CeNTREXPositionDistribution(Distribution):
    """
    The standard initial position distribution that I expect will be used for most simulations. Initial positions
    are assumed uniformly distributed on a disc at the "zone of freezing" where collisions between molecules are
    assumed to have ceased. This dsitribution has not been experimentally verified in any way for CeNTREX, it is
    based on some results from ACME.
    """
    d: float = 0.02 # Diameter of zone of freezing in m
    z: float = 0.25*0.0254 # Z-position of zone of freezing

    def draw(self, n: int) -> np.ndarray:
        """
        Draws n samples from the distribution
        """
        theta = uniform.rvs(size = n)*2*np.pi
        r = np.sqrt(uniform.rvs(size = n)) * self.d/2

        return np.array((r*np.cos(theta), r*np.sin(theta), np.full(n,self.z)))


    def save_to_hdf(self, filepath: Path, run_name: str):
        """
        Saves the distribution to an hdf file
        """
        # Open the hdf file
        with h5py.File(filepath, 'a') as f:
            try:
                # Create a group for the position distribution
                group_path = run_name + "/position_distribution" 
                f.create_group(group_path)

                # Write the name of the distribution class into file
                f[group_path].attrs['class'] = type(self).__name__

                # Loop over the attributes of the position distribution and save them to the attributes
                # of the group
                for key, value in vars(self).items():
                    f[group_path].attrs[key] = value
            
            except ValueError:
                raise ValueError("Can't save position distribution. Group already exists!")

@dataclass
class GaussianPositionDistribution(Distribution):
    """
    Based on some results from ACME, the initial x and y position distribution
    of the molecules is better described by a Gaussian. I got the sigmax and
    y values from Olivier.
    """
    sigmax: float = 0.25*25.4/5*3.8e-3 # Sigma for the gaussian position distribution
    sigmay: float = 0.25*25.4/5*3.8e-3 # Sigma for the gaussian position distribution
    z: float = 0.25*0.0254 # Z-position of zone of freezing

    def draw(self, n: int) -> np.ndarray:
        """
        Draws n samples from the distribution
        """
        x = GaussianDistribution(0, self.sigmax).draw(n),          
        y = GaussianDistribution(0, self.sigmay).draw(n)

        return np.vstack((x, y, np.full(n,self.z)))


    def save_to_hdf(self, filepath: Path, run_name: str):
        """
        Saves the distribution to an hdf file
        """
        # Open the hdf file
        with h5py.File(filepath, 'a') as f:
            try:
                # Create a group for the position distribution
                group_path = run_name + "/position_distribution" 
                f.create_group(group_path)

                # Write the name of the distribution class into file
                f[group_path].attrs['class'] = type(self).__name__

                # Loop over the attributes of the position distribution and save them to the attributes
                # of the group
                for key, value in vars(self).items():
                    f[group_path].attrs[key] = value
            
            except ValueError:
                raise ValueError("Can't save position distribution. Group already exists!")  




    