from dataclasses import dataclass
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import List

from centrex_TlF.states import State

from .beamline import Beamline
from .distributions import StandardPositionDistribution, StandardVelocityDistribution
from .molecule import Molecule

class TrajectorySimulator:
    """
    Class that is used to run trajectory simulations and to store the results.
    """
    def __init__(self, filename: str = None) -> None:
        self.counter = Counter()

    def run_simulation(self, beamline: Beamline, name:str = "ES lens",
                       vdist = StandardVelocityDistribution(), xdist = StandardPositionDistribution(),  
                       N_traj: int = 1000, apertures_of_interest = []) -> List:
        """
        Simulates N trajectories of molecules flying through the beamline
        """
        # To conserve memory, doing the trajectories in loops
        N_loops = 100
        N = int(N_traj/N_loops)
        
        # Plot beamline
        axes = beamline.plot()
        
        for _ in tqdm(range(N_loops)):
            # Generate desired number of initial positions and velocities
            vs = vdist.draw(N)
            xs = xdist.draw(N)

            # Loop over number of desired trajectories
            molecules = []
            for i in range(N):
                # Initialize a molecule
                molecule = Molecule()
                beamline_pre_lens = Beamline(beamline.elements[:beamline.find_element(name).index])
                molecule.init_trajectory(beamline_pre_lens, xs[:,i], vs[:,i]) 

                # Propagate molecule through beamline
                beamline.propagate_through(molecule)

                # Increment counter
                self.counter.increment_counter(molecule.aperture_hit)
                
                if molecule.aperture_hit in apertures_of_interest:
                    molecule.plot_trajectory(axes)
                    molecules.append(molecule)

        plt.show()

        return molecules

    def run_simulation_parallel(self, beamline: Beamline, name:str = "ES lens",
                       vdist = StandardVelocityDistribution(), xdist = StandardPositionDistribution(),  
                       N_traj: int = 1000, apertures_of_interest = [], n_jobs = 10) -> List:
        """
        Simulates N_traj trajectories of molecules flying through the beamline using parallel processing.
        """
        # To conserve memory, doing the trajectories in loops
        N_loops = 100*n_jobs
        N = int(N_traj/N_loops)
        
        # Plot beamline
        axes = beamline.plot()
        
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
                beamline_pre_lens = Beamline(beamline.elements[:beamline.find_element(name).index])
                molecule.init_trajectory(beamline_pre_lens, xs[:,i], vs[:,i]) 

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

        # Merge counters
        self.counter.merge_counters(counters)

        # Plot molecule trajectories
        for molecule in molecules:
            molecule.plot_trajectory(axes)
            
        plt.show()

        return molecules

class Counter:
    """
    Counter is used to keep track of how many molecules hit each aperture during a simulation run
    """
    def __init__(self) -> None:
        """
        Initializes counter dictionary
        """
        self.counter_dict = {}

    def increment_counter(self, aperture_hit: str) -> None:
        """
        Increments the counter for given aperture
        """
        if aperture_hit in self.counter_dict.keys():
            self.counter_dict[aperture_hit] += 1
        else:
            self.counter_dict[aperture_hit] = 1
            
    def print(self) -> None:
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

    def merge_counters(self, others: List):
        """
        Merges self with other counters contained in a list.
        """
        # Loop over the other countets
        for other in others:
            # Check if keys of other are found in self
            for key, value in other.counter_dict.items():
                if key in self.counter_dict.keys():
                    self.counter_dict[key] += value
                else:
                    self.counter_dict[key] = value

