import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os.path import exists
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from centrex_TlF.states import State, UncoupledBasisState
from matplotlib.patches import Rectangle
from scipy.constants import g
from scipy.interpolate import interp1d

from ..beamline import Beamline
from ..molecule import Molecule
from ..stark_potential import stark_potential


@dataclass
class BeamlineElement(ABC):
    """
    Abstract dataclass that defines the methods that must be implemented for all beamline elements
    that are used in the simulations.
    """

    name: str  # Name of beamline element
    z0: float  # Z-position where the beamline element starts in meters
    L: float  # Length or thickness of element along Z
    x0: float = 0.0  # X-coordinate of center of the element (0 corresponds to being on straight line from cold cell)
    y0: float = 0.0  # Y-coordinate of center of element (0 corresponds to being on straight line from cold cell)

    def __post_init__(self):
        self.z1 = self.z0 + self.L

    @abstractmethod
    def propagate_through(self, molecule: Molecule):
        """
        Propagates a molecule through the beamline element
        """

    @abstractmethod
    def N_steps(self):
        """
        Calculates the number of timesteps that are used when propagating through the element.
        """

    @abstractmethod
    def plot(self, axes):
        """
        Plots the beamline element on the provided axes (axes[0] should be XZ plane and axes[1] YZ plane)
        """

    def save_to_hdf(self, filepath: Path, parent_group_path: str) -> None:
        """
        Saves the beamline element to an hdf file 
        """
        # Open the hdf file
        with h5py.File(filepath, "a") as f:
            try:
                # Create a group for the beamline element
                group_path = parent_group_path + "/" + self.name
                f.create_group(group_path)

                # Write the name of the beamline element class into file
                f[group_path].attrs["class"] = type(self).__name__

                # Loop over the attributes of the beamline element and save them to the attributes
                # of the group
                for key, value in vars(self).items():
                    f[group_path].attrs[key] = value

            except ValueError:
                print("Can't save beamline element. Group already exists!")


@dataclass
class CircularAperture(BeamlineElement):
    """
    Class used for beamline elements that have a circular aperture through which the molecules
    are supposed to pass.
    """

    d: float = 0.0254  # Diameter of aperture in meters

    def propagate_through(self, molecule: Molecule):
        """
        Function that checks if the molecule makes it through the aperture without hitting it. I don't really care
        where exactly the molecule hits the aperture, so I'm not calculating the position. Rather just check
        if the molecule is within the diameter of the aperture when it is entering and  exiting the aperture 
        (this ignores the possible edge case of a parabolic trajectory that goes beyond the diameter of the aperture 
        but returns to below the diameter before the end of the aperture - I think the effect of this is negligible).
        """
        # Loop over start of element and end of element
        for z in [self.z0, self.z1]:
            # Calculate the time taken to reach start of element from initial position
            delta_t = (z - molecule.x()[2]) / molecule.v()[2]

            # Calculate the position and velocity of the molecule at start of beamline element
            molecule.update_trajectory(delta_t)

            # Determine if molecule is now within the clear part of the aperture. If not, the molecule is
            # considered dead
            rho = np.sqrt(np.sum(molecule.x()[:2] ** 2))
            if rho > self.d / 2:
                molecule.set_dead()
                molecule.set_aperture_hit(self.name)

                return

    def N_steps(self):
        """
        Max number of steps that are taken when passing through a simple element that doesn't modify the trajectory
        of an incoming molecule is two (entering and exiting)
        """
        return 2

    def plot(self, axes):
        """
        Plot the aperture on the provided axes
        """
        rect1 = Rectangle(
            (self.z0, self.d / 2), self.z1 - self.z0, 1, color=(0.5, 0.5, 0.5)
        )
        rect2 = Rectangle(
            (self.z0, -self.d / 2 - 1), self.z1 - self.z0, 1, color=(0.5, 0.5, 0.5)
        )
        axes[0].add_patch(rect1)
        axes[0].add_patch(rect2)

        rect3 = Rectangle(
            (self.z0, self.d / 2), self.z1 - self.z0, 1, color=(0.5, 0.5, 0.5)
        )
        rect4 = Rectangle(
            (self.z0, -self.d / 2 - 1), self.z1 - self.z0, 1, color=(0.5, 0.5, 0.5)
        )
        axes[1].add_patch(rect3)
        axes[1].add_patch(rect4)


@dataclass
class RectangularAperture(BeamlineElement):
    """
    Class used for beamline elements that have a rectangular aperture through which the molecules
    are supposed to pass.
    """

    w: float = 0.02  # Height of aperture along Y in meters
    h: float = 0.02  # Width of aperture along X in meters

    def __post_init__(self):
        super().__post_init__()
        # Calculate the coordinates of the edges of the aperture
        self.x1 = self.x0 - self.w / 2
        self.x2 = self.x0 + self.w / 2
        self.y1 = self.y0 - self.h / 2
        self.y2 = self.y0 + self.h / 2

    def propagate_through(self, molecule: Molecule):
        """
        Function that checks if the molecule makes it through the aperture without hitting it. I don't really care
        where exactly the molecule hits the aperture, so I'm not calculating the position. Rather just check
        if the molecule is within the diameter of the aperture when it is entering and  exiting the aperture 
        (this ignores the possible edge case of a parabolic trajectory that goes beyond the diameter of the aperture 
        but returns to below the diameter before the end of the aperture - I think the effect of this is negligible).
        """
        # Loop over start of element and end of element
        for z in [self.z0, self.z1]:
            # Calculate the time taken to reach start of element from initial position
            delta_t = (z - molecule.x()[2]) / molecule.v()[2]

            # Calculate the position and velocity of the molecule at start of beamline element
            molecule.update_trajectory(delta_t)

            # Determine if molecule is now within the clear part of the aperture. If not, the molecule is
            # considered dead
            if not (
                (self.x1 < molecule.x()[0] < self.x2)
                and (self.y1 < molecule.x()[1] < self.y2)
            ):
                molecule.set_dead()
                molecule.set_aperture_hit(self.name)
                return

    def N_steps(self):
        """
        Max number of steps that are taken when passing through a simple element that doesn't modify the trajectory
        of an incoming molecule is two (entering and exiting)
        """
        return 2

    def plot(self, axes):
        """
        Plot aperture on the provided axes
        """
        rect1 = Rectangle((self.z0, self.x2), self.z1 - self.z0, 0.05, color="k")
        rect2 = Rectangle((self.z0, self.x1 - 0.05), self.z1 - self.z0, 0.05, color="k")
        axes[0].add_patch(rect1)
        axes[0].add_patch(rect2)

        rect3 = Rectangle((self.z0, self.y2), self.z1 - self.z0, 0.05, color="k")
        rect4 = Rectangle((self.z0, self.y1 - 0.05), self.z1 - self.z0, 0.05, color="k")
        axes[1].add_patch(rect3)
        axes[1].add_patch(rect4)


@dataclass
class FieldPlates(BeamlineElement):
    """
    Class for the field plates of the main interaction region of CeNTREX
    """

    w: float = 0.02  # Height of aperture along Y in meters

    def __post_init__(self):
        super().__post_init__()
        # Calculate the coordinates of the edges of the aperture
        self.x1 = self.x0 - self.w / 2
        self.x2 = self.x0 + self.w / 2

    def propagate_through(self, molecule: Molecule):
        """
        Function that checks if the molecule makes it through the field plates without hitting them. If the molecule
        hits the field plates, the position where it hits them is calculated
        """
        # Calculate the time taken to reach start of element from initial position
        delta_t = (self.z0 - molecule.x()[2]) / molecule.v()[2]

        # Calculate the position and velocity of the molecule at start of beamline element
        molecule.update_trajectory(delta_t)

        # Determine if molecule is now within the clear part of the aperture. If not, the molecule is
        # considered dead
        if not (self.x1 < molecule.x()[0] < self.x2):
            molecule.set_dead()
            molecule.set_aperture_hit(self.name)
            return

        # Next check if the molecule makes it through the field plates to the end
        # Calculate the time taken to reach end of the field plates from initial position
        delta_t = (self.z1 - molecule.x()[2]) / molecule.v()[2]

        # Calculate the position of the molecule at the end of the field plates
        x = molecule.x(delta_t)

        # Check if molecule is within bounds
        if not (self.x1 < x[0] < self.x2):
            # Calculate time taken to hit field plate if molecule is moving in -ve x-direction
            if molecule.v()[0] < 0:
                delta_t = (self.x1 - molecule.x()[0]) / molecule.v()[0]
            # Calculate time taken to hit field plate if molecule is moving in +ve x-direction
            elif molecule.v()[0] > 0:
                delta_t = (self.x2 - molecule.x()[0]) / molecule.v()[0]

            # Calculate final position of molecule
            molecule.update_trajectory(delta_t)
            molecule.set_dead()
            molecule.set_aperture_hit(self.name)
            return

        else:
            # If molecule made it through, update trajectory
            molecule.update_trajectory(delta_t)
            return

    def N_steps(self):
        """
        Max number of steps that are taken when passing through a simple element that doesn't modify the trajectory
        of an incoming molecule is two (entering and exiting)
        """
        return 2

    def plot(self, axes):
        """
        Plot field plates on the provided axes
        """
        rect1 = Rectangle((self.z0, self.x2), self.z1 - self.z0, 0.02, color="y")
        rect2 = Rectangle((self.z0, self.x1 - 0.02), self.z1 - self.z0, 0.02, color="y")
        axes[0].add_patch(rect1)
        axes[0].add_patch(rect2)


def main():
    aperture = CircularAperture(name="40K Shield", z0=0.99, L=0.01, d=0)
    rect_aperture = RectangularAperture(
        name="Detection aperture", z0=100, L=0.01, w=0.00001, h=1e-6
    )
    beamline = Beamline([aperture, rect_aperture])
    molecule = Molecule()
    molecule.init_trajectory(beamline)
    beamline.propagate_through(molecule)
    print(molecule)

    file_path = Path("./saved_data/test.hdf")
    run_name = "test"
    aperture.save_to_hdf(filepath=file_path, parent_group_path=run_name)


if __name__ == "__main__":
    main()

