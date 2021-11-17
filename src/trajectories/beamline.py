from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import h5py

@dataclass
class Beamline:
    """
    Class for beamlines which are collections of beamline elements.
    """
    elements: List

    def __post_init__(self):
        self.sort_elements()

    def propagate_through(self, molecule):
        """
        Propagates a molecule through the beamline by propagating a molecule through each of the beamline elements.
        """
        # Loop over elements and propagate molecule through them
        for element in self.elements:
            # Propagate the molecule through the current element
            element.propagate_through(molecule)

            # If molecule hit the element, stop looping
            if not molecule.alive:
                break

        # If molecule is still alive, count is as detected
        if molecule.alive:
            molecule.set_aperture_hit("Detected")

        # Drop any nans in molecule trajectory
        molecule.trajectory.drop_nans()

    def sort_elements(self):
        """
        Sorts the beamline elements in the order that molecules fly through them
        """
        get_z0 = lambda x: x.z0
        self.sort_elements = self.elements.sort(key = get_z0)

    def find_element(self, name):
        """
        Returns the element of the beamline with the provided name
        """
        for i, element in enumerate(self.elements):
            if element.name == name:
                return element

        print(f"Element with name '{name}' not found in beamline")

    def plot(self):
        """
        Plots the beamline
        """
        fig, axes = plt.subplots(2,1,figsize = (16,9))
        axes[0].set_ylabel("X-position / m")
        axes[0].set_ylim([-0.06, 0.06])
        axes[1].set_xlabel("Z-position / m")
        axes[1].set_ylabel("Y-position / m")
        axes[1].set_ylim([-0.06, 0.06])
        axes[0].set_xlim([0, self.elements[-1].z1 + 0.1])
        axes[1].set_xlim([0, self.elements[-1].z1 + 0.1])

        # Loop over elements and plot them
        for element in self.elements:
            element.plot(axes)

        return axes

    def save_to_hdf(self, filepath: Path, run_name: str) -> None:
        """
        Saves the beamline to an hdf file
        """
        # Open the hdf file
        with h5py.File(filepath, 'a') as f:
            # Create a group for the beamline 
            group_path = run_name + "/beamline/"
            f.create_group(group_path)

            # Loop over the beamline elements and save them to the file
            for element in self.elements:
                element.save_to_hdf(filepath, group_path)



