from pathlib import Path
import importlib
import inspect
from typing import List

import h5py


from .beamline import Beamline
from .beamline_elements import BeamlineElement
from .molecule import Molecule, Trajectory
from .trajectory_simulator import Counter, SimulationResult

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

def import_trajectories_from_hdf(filepath: Path, run_name: str, group_name: str = 'trajectories')-> List[Molecule]:
    """
    Imports molecular trajectories from an hdf file
    """
    # Open the hdf file
    with h5py.File(filepath, 'r') as f:
        # Find the names of the molecule trajectory datasets
        dataset_names =  list(f[run_name + '/' + group_name].keys())

        # Loop over the datasets and make a list of molecules
        molecules = []
        for dataset_name in dataset_names:
            # Get trajectory data for the molecule
            x = f[run_name + '/' + group_name + '/' + dataset_name]['x'][()]
            v = f[run_name + '/' + group_name + '/' + dataset_name]['v'][()]
            a = f[run_name + '/' + group_name + '/' + dataset_name]['a'][()]
            t = f[run_name + '/' + group_name + '/' + dataset_name]['t'][()]


            # Get some data about the molecule
            attributes = dict(f[run_name + '/' + group_name + '/' + dataset_name].attrs)
            aperture_hit = attributes['aperture_hit']
            alive = attributes['alive']

            # Initialize trajectory
            trajectory = Trajectory(Beamline([]))
            trajectory.x = x
            trajectory.v = v
            trajectory.a = a
            trajectory.t = t

            # Initialize a molecule
            molecule = Molecule()
            molecule.trajectory = trajectory
            molecule.set_aperture_hit(aperture_hit)
            molecule.alive = alive

            # Add molecule to list
            molecules.append(molecule)

    return molecules

def import_distribution_from_hdf(dist_name: str, filepath: Path, run_name: str) -> BeamlineElement:
    """
    Imports a distribution from an hdf file
    """
    with h5py.File(filepath, 'r') as f:
        # Read the attributes which define the distribution
        attributes = dict(f[run_name + '/' + dist_name].attrs.items())

    # Figure out the class of the distribution
    class_name = attributes.pop('class')

    # Get the class of the beamline element
    module = importlib.import_module('trajectories.distributions')
    class_ = getattr(module, class_name)
    
    # Figure out the arguments needed for instantiation of class
    init_arg_names = list(inspect.signature(class_.__init__).parameters)[1:]
    
    # Dictionary of of arguments
    init_args = {key:attributes[key] for key in init_arg_names}

    # Initialize the distribution and return it
    return class_(**init_args)

def import_counter_from_hdf(filepath: Path, run_name: str):
    """
    Imports a counter from an hdf file
    """
    with h5py.File(filepath, 'r') as f:
        # Read the attributes of the counter group
        attributes = dict(f[run_name + '/counter'].attrs.items())

    # Make a counter object and return it
    counter = Counter()
    counter.counter_dict = attributes
    return counter

def import_sim_result_from_hdf(filepath: Path, run_name: str):
    """
    Imports a SimulationResult from an hdf file
    """
    beamline = import_beamline_from_hdf(filepath, run_name)
    xdist = import_distribution_from_hdf('position_distribution', filepath, run_name)
    vdist = import_distribution_from_hdf('velocity_distribution', filepath, run_name)
    counter = import_counter_from_hdf(filepath, run_name)
    molecules = import_trajectories_from_hdf(filepath, run_name)

    return SimulationResult(counter, beamline, xdist, vdist, molecules)