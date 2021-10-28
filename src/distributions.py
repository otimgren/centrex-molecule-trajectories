from abc import ABC, abstractmethod
from dataclasses import dataclass
from matplotlib import scale
import numpy as np
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt
from functools import partial

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

@dataclass
class StandardVelocityDistribution(Distribution):
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
        return np.array((GaussianDistribution(self.vx, self.sigmax).draw(n),
                         GaussianDistribution(self.vy, self.sigmay).draw(n),
                         GaussianDistribution(self.vz, self.sigmaz).draw(n)
                        ))

@dataclass
class StandardPositionDistribution(Distribution):
    """
    The standard initial position distribution that I expect will be used for most simulations. Initial positions
    are assumed uniformly distributed on a disc at the "zone of freezing" where collisions between molecules are
    assumed to have ceased. This dsitribution has not been experimentally verified in any way for CeNTREX, it is
    based on some results from ACME.
    """
    d: float = 0.02 # Diameter of zone of freezing in m
    z: float = 0.25*0.0254 # Z-position of zone of freezing

    def draw(self, n: int) -> np.ndarray:
        theta = uniform.rvs(size = n)*2*np.pi
        r = np.sqrt(uniform.rvs(size = n)) * self.d

        return np.array((r*np.cos(theta), r*np.sin(theta), np.full(n,self.z)))






    