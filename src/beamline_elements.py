# Import abstract classes
from abc import ABC, abstractmethod

class BeamlineElement(ABC):
    """
    Abstract class that defines the methods that must be implemented for all beamline elements
    that are used in the simulations.
    """

    @abstractmethod
    def propagate_through(self, molecule):
        """
        Propagates a molecule through the bea
        """
        pass

class CircularAperture(BeamlineElement):
    """
    Class used for beamline elements that have a circular aperture through which the molecules
    are supposed to pass.
    """

