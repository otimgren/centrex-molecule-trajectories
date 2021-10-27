from dataclasses import dataclass
from typing import List
from molecule import Molecule
import matplotlib.pyplot as plt

@dataclass
class Beamline:
    """
    Class for beamlines which are collections of beamline elements.
    """
    elements: List

    def propagate_through(self, molecule: Molecule) -> None:
        """
        Propagates a molecule through the beamline by propagating a molecule through each of the beamline elements.
        """
        # Sort the elements
        self.sort_elements()
        
        # Loop over elements and propagate molecule through them
        for element in self.elements:
            element.propagate_through(molecule)

    def sort_elements(self):
        """
        Sorts the beamline elements in the order that molecules fly through them
        """
        get_z0 = lambda x: x.z0
        self.sort_elements = self.elements.sort(key = get_z0)