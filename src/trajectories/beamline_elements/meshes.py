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
from .apertures import BeamlineElement


@dataclass
class Honeycomb(BeamlineElement):
    """
    Class to represent a honeycomb structure.
    """

    cell_wall_thickness: float = 1e-4
    cell_wall_length: float = 25.4e-3 / 16

