from copy import copy
from dataclasses import dataclass
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from os import path
from pathlib import Path
from typing import List

import h5py
from tqdm import tqdm

from .beamline import Beamline
from .distributions import CeNTREXPositionDistribution, CeNTREXVelocityDistribution, Distribution
from .molecule import Molecule

class TrajectorySimulator:
    """
    Class that is used to run trajectory simulations and to store the results.
    """
    def __init__(self) -> None:
        # Initialize a counter that keeps track of how many molecules hit each beamline element
        self.counter = Counter()

        # Initialize a dictionary that will store results
        self.results = {}

    def run_simulation(self, beamline: Beamline, run_name: str, 
                       vdist = CeNTREXVelocityDistribution(), xdist = CeNTREXPositionDistribution(),  
                       N_traj: int = 1000, apertures_of_interest = [], n_jobs = 1) -> List:
        """
        Simulates N_traj trajectories of molecules flying through the beamline using parallel processing.
        """
        # To conserve memory, doing the trajectories in loops
        N_loops = 100*n_jobs
        N = int(N_traj/N_loops)
        
        
        # Define function that will be run in parallel
        def parallel_func():
            # Initialize a counter
            counter = Counter()

            # Generate desired number of initial positions and velocities
            vs = vdist.draw(N)
            xs = xdist.draw(N)

            # Loop over number of desired trajectories
            molecules = []
            for i in range(N):
                # Initialize a molecule
                molecule = Molecule()
                # beamline_pre_lens = Beamline(beamline.elements[:beamline.find_element(name).index])
                molecule.init_trajectory(beamline, xs[:,i], vs[:,i]) 

                # Propagate molecule through beamline
                beamline.propagate_through(molecule)

                # Increment counter
                counter.increment_counter(molecule.aperture_hit)
                
                # If molecule hit one of the apertures of interest, save it to list
                if molecule.aperture_hit in apertures_of_interest:
                    molecules.append(molecule)

            return molecules, counter

        # results is a list of tuples List[(List of molecules, Counter)]
        results = Parallel(n_jobs = n_jobs, verbose = 1)(delayed(parallel_func)() for _ in range(N_loops))
        
        # Make lists of molecules and counters
        molecules = []
        counters = []
        for item in results:
            molecules.append(item[0])
            counters.append(item[1])
        molecules = [item for sublist in molecules for item in sublist]

        # Re-initialize the counter so only results for this run are stored in it
        self.counter = Counter()

        # Each parallel thread has a separate counter so need to merge them
        self.counter.merge_counters(counters)

        # Add molecules and counter to results dictionary
        self.result = SimulationResult(self.counter, beamline, xdist, vdist, molecules)
        self.results[run_name] = SimulationResult(self.counter, beamline, xdist, vdist,  molecules)

class Counter:
    """
    Counter is used to keep track of how many molecules hit each aperture during a simulation run.
    """
    def __init__(self) -> None:
        """
        Initializes counter dictionary
        """
        self.counter_dict = {}

    def increment_counter(self, aperture_hit: str) -> None:
        """
        Increments the counter for given aperture.
        """
        if aperture_hit in self.counter_dict.keys():
            self.counter_dict[aperture_hit] += 1
        else:
            self.counter_dict[aperture_hit] = 1
            
    def print(self) -> None:
        """
        Prints the number of molecules that hit each beamline element.
        """
        print("Number of molecules that hit each element:")
        for key, value in self.counter_dict.items():
            print(f"{key} : {value}")

    def calculate_efficiency(self) -> float:
        """
        Returns the efficiency, i.e. proportion of molecules that make it to detection
        """
        total = 0
        for key, value in self.counter_dict.items():
            total += value

        if "Detected" in self.counter_dict.keys():
            return self.counter_dict["Detected"]/total
        else:
            return 0

    def merge_counters(self, others: List) -> None:
        """
        Merges self with other counters contained in a list.
        """
        # Loop over the other counters
        for other in others:
            # Check if keys of other are found in self
            for key, value in other.counter_dict.items():
                if key in self.counter_dict.keys():
                    self.counter_dict[key] += value
                else:
                    self.counter_dict[key] = value

    def save_to_hdf(self, filepath: Path, run_name: str) -> None:
        """
        Saves the counter information into an hdf file
        """ 
        # Open the hdf file
        with h5py.File(filepath, 'a') as f:
            try:
                # Create a group for the counter
                group_path = run_name + "/counter"
                f.create_group(group_path)

                # Loop over the items in the counter and save them to the attributes
                # of the group
                for key, value in self.counter_dict.items():
                    f[group_path].attrs[key] = value
            
            except ValueError:
                raise ValueError("Can't save counter. Group already exists!") 


@dataclass
class SimulationResult:
    """
    Class used for storing results from a trajectory simulation, plotting the results and saving
    them to hdf files
    """
    counter: Counter
    beamline: Beamline
    xdist: Distribution
    vdist: Distribution
    molecules: List[Molecule]

    def plot(self, N_max: int = 10000) -> None:
        """
        Plots the simulation result
        """
        # Plot the beamline
        axes = self.beamline.plot()

        # Plot molecule trajectories (but not too many)
        n_max = np.min((N_max, len(self.molecules)))
        n = 0
        while n < n_max:
            self.molecules[n].plot_trajectory(axes)
            n += 1
            
        plt.show()

    def save_to_hdf(self, filepath: Path, run_name: str) -> None:
        """
        Saves the simulation result to an hdf file
        """
        # Create a group for the run in the hdf file
        with h5py.File(filepath, 'a') as f:
            try:
                f.create_group(run_name)

            except ValueError:
                # If run name already exists, ask user if it should be overwritten
                val = input("Run name already exists. Overwrite? y/n")
                
                if val == 'y':
                    del f[run_name]   
                    f.create_group(run_name)
                else: 
                    return

        # Loop over the attributes of the object and save them to hdf files
        attributes_dict = copy(vars(self))
        molecules = attributes_dict.pop('molecules')
        for _, value in attributes_dict.items():
            value.save_to_hdf(filepath, run_name)

        # Save the molecule trajectories to file
        self.save_molecules_to_hdf(filepath, run_name)

    def save_molecules_to_hdf(self, filepath: Path, run_name: str) -> None:
        """
        Saves a list of molecule trajectories to an hdf file
        """
        print("Saving trajectories...")
        for i, molecule in enumerate(tqdm(self.molecules)):
            group_name = f"trajectories/molecule_{i}"
            molecule.save_to_hdf(filepath, run_name, group_name)




        



