from pathlib import Path
import importlib
import inspect

import h5py

from .beamline import Beamline
from .beamline_elements import BeamlineElement

def import_element_from_hdf(element_name: str, filepath: Path, run_name: str) -> BeamlineElement:
    """
    Imports a beamline element from an hdf file
    """
    with h5py.File(filepath, 'r') as f:
        # Read the attributes which define the beamline element
        attributes = dict(f[run_name + '/beamline/' + element_name].attrs.items())

    # Figure out the class of the beamline element
    class_name = attributes.pop('class')

    # Get the class of the beamline element
    module = importlib.import_module('trajectories.beamline_elements')
    class_ = getattr(module, class_name)
    
    # Figure out the arguments needed for instantiation of class
    init_arg_names = list(inspect.signature(class_.__init__).parameters)[1:]
    
    # Dictionary of of arguments
    init_args = {key:attributes[key] for key in init_arg_names}

    # Initialize the beamline element and return it
    return class_(**init_args)

def import_beamline_from_hdf(filepath: Path, run_name: str):
    """
    Imports a beamline from an hdf file for the specified run.
    """
    # Open the hdf file
    with h5py.File(filepath, 'r') as f:
        # Find the paths to the beamline elements stored for the specified run
        element_names = list(f[run_name + '/beamline'].keys())

    # Loop over element names and generate each element
    elements = []
    for element_name in element_names:
        elements.append(import_element_from_hdf(element_name, filepath, run_name))

    # Make the beamline based on the elements and return it
    return Beamline(elements)

def import_trajectories_from_hdf(filepath: Path, run_name: str):
    """
    Imports molecular trajectories from an hdf file
    """
    #to do
    ...

def import_distribution_from_hdf(filepath: Path, run_name: str):
   """
   Imports a position or velocity distribution from an hdf file 
   """ 
    # to do
   ...

def import_counter_from_hdf():
    """
    Imports a counter from an hdf file
    """
    # to do
    ...