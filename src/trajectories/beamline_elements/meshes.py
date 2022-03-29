import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil
from os.path import exists
from pathlib import Path
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np
from centrex_TlF.states import State, UncoupledBasisState
from hexalattice import hexalattice
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, RegularPolygon
from scipy.constants import g
from scipy.interpolate import interp1d

from ..beamline import Beamline
from ..molecule import Molecule
from .apertures import BeamlineElement

__all__ = ["Honeycomb"]


@dataclass
class Honeycomb(BeamlineElement):
    """
    Class to represent a honeycomb structure.
    """

    width: float = 2 * 25.4e-3
    height: float = 2 * 25.4e-3
    cell_wall_thickness: float = 1e-4
    cell_wall_length: float = 25.4e-3 / 16

    def __post_init__(self) -> None:
        """
        Initialize the honeycomb structure.
        """
        super().__post_init__()

        # Calculate the coordinates of the edges of the aperture
        self.x1 = self.x0 - self.width / 2
        self.x2 = self.x0 + self.width / 2
        self.y1 = self.y0 - self.height / 2
        self.y2 = self.y0 + self.height / 2

        # Calculate the number of polygons required in the x- and y-direction
        self.nx = ceil(self.width / (self.cell_wall_length * np.sqrt(3)))
        self.ny = ceil(self.height / (self.cell_wall_length * 3 / 2))

        # Make grid of polygon centers
        self.xcoords, self.ycoords = hexalattice.make_grid(
            nx=self.nx,
            ny=self.ny,
            n=0,
            min_diam=self.cell_wall_length * np.sqrt(3),
            align_to_origin=True,
            crop_circ=0,
            rotate_deg=0,
        )

        # Make list of polygons as patches
        self.patches = self.make_patches()

    def make_patches(self) -> List:
        """
        Makes list of patches of polygons.
        """
        patches = []
        for x, y in zip(self.xcoords, self.ycoords):
            patch = RegularPolygon(
                (x, y),
                6,
                radius=(
                    self.cell_wall_length * np.sqrt(3) - self.cell_wall_thickness / 2
                )
                / 2,
            )
            patches.append(patch)

        return patches

    def propagate_through(self, molecule: Molecule):
        """
        Propagate molecule through mesh
        """
        idx = None
        # Loop over start of element and end of element
        for z in [self.z0, self.z1]:
            # Calculate the time taken to reach start of element from initial position
            delta_t = (z - molecule.x()[2]) / molecule.v()[2]

            # Calculate the position and velocity of the molecule at start of beamline element
            molecule.update_trajectory(delta_t)

            # Determine the position of the molecule in the xy-plane
            point = (molecule.x()[0], molecule.x()[1])

            # Determine which hexagon the position corresponds to
            # (only at start of element, if molecule switched to different
            # hexagon, it must have hit a wall)
            if not idx:
                rhos = np.sqrt(
                    (point[0] - self.xcoords) ** 2 + (point[1] - self.ycoords) ** 2
                )
                idx = np.argmin(rhos)

            patch = self.patches[idx]

            if not patch.contains_point(point):
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
        rect1 = Rectangle(
            (self.z0, self.x1), self.z1 - self.z0, self.x2 - self.x1, color="k"
        )
        axes[0].add_patch(rect1)

        rect3 = Rectangle(
            (self.z0, self.y1), self.z1 - self.z0, self.y2 - self.y1, color="k"
        )
        axes[1].add_patch(rect3)

    def plot_mesh(self, ax: plt.Axes = None):
        """
        Plots the hexagon mesh on the provided axes
        """
        if ax is None:
            fig, ax = plt.subplots()

        patch_collection = PatchCollection(self.patches)
        ax.add_collection(patch_collection)
        ax.set_title("Hexagonal mesh")
        ax.set_xlabel("X-position / m")
        ax.set_ylabel("Y-position / m")
        ax.set_xlim([-self.width / 2, self.width / 2])
        ax.set_ylim([-self.height / 2, self.height / 2])

        return ax

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
                vars_dict = vars(self).copy()
                vars_dict.pop("patches")
                for key, value in vars_dict.items():
                    f[group_path].attrs[key] = value

            except ValueError:
                print("Can't save beamline element. Group already exists!")
