from dataclasses import dataclass
import numpy as np
from beamline import Beamline
from centrex_TlF.states import UncoupledBasisState, State
from beamline_elements import CircularAperture, ElectrostaticLens, FieldPlates, RectangularAperture
from distributions import StandardPositionDistribution, StandardVelocityDistribution
from typing import List
from molecule import Molecule, Trajectory
import timeit
from tqdm import tqdm
import matplotlib.pyplot as plt

class TrajectorySimulator:
    """
    Class that is used to run trajectory simulations and to store the results in an hdf file.
    """

    def __init__(self, filename: str = None) -> None:
        pass

    def generate_molecules(self, beamline: Beamline, state: State, vdist = StandardVelocityDistribution(), 
                            xdist = StandardPositionDistribution(),  N_molecules = 1000) -> np.ndarray:
        """
        Generates a list of molecules in specified states with initial velocities and positions based on given distributions.
        """
        # Get initial velocities and positions
        vs = vdist.draw(N_molecules)
        xs = xdist.draw(N_molecules)

        # Make molecules list
        molecules = np.empty(N_molecules, dtype = 'object')
        for i in range(N_molecules):
            molecule = Molecule(state, xs[:,i], vs[:,i])
            molecule.init_trajectory(beamline)
            molecules[i] = molecule

        return molecules

    def run_simulation(self, beamline: Beamline, state: State = Molecule().state, 
                       vdist = StandardVelocityDistribution(), xdist = StandardPositionDistribution(),  
                       N_traj: int = 1000, apertures_of_interest = []) -> List:
        """
        Simulates N trajectories of molecules flying through the beamline
        """
        # Generate desired number of molecules
        molecules = self.generate_molecules(beamline, state, N_molecules = N_traj)

        # Plot beamline
        axes = beamline.plot()

        # Loop over molecules and propagate them through the beamline
        for molecule in tqdm(molecules):
            beamline.propagate_through(molecule)
            
            if molecule.aperture_hit in apertures_of_interest:
                molecule.plot_trajectory(axes)

        plt.show()

        return molecules
    

def main():
    # Define the beamline elements
    m_per_in = 0.0254 # Constant for converting meters to inches
    fourK_shield = CircularAperture(z0 = 0.7*m_per_in, L = 0.25*m_per_in, d = 1*m_per_in, name = '4K shield')
    fortyK_shield = CircularAperture(z0 = fourK_shield.z1 + 1.25*m_per_in, L = 0.25*m_per_in, d = 1*m_per_in,
                                     name = '40K shield')
    bb_exit = CircularAperture(z0 = fortyK_shield.z1 + 2.5*m_per_in, L = 0.75*m_per_in, name = 'BB exit')
    lens = ElectrostaticLens(z0 = bb_exit.z1+33*m_per_in, L = 0.6, name = "ES lens")
    field_plates = FieldPlates(z0 = 2.43, L = 3.0, w = 0.02, name = "Field plates")
    dr_aperture = RectangularAperture(z0 = field_plates.z1 + 39.9*m_per_in, L = 0.25*m_per_in, name = "DR aperture",
                                      w = 0.018, h = 0.03)

    beamline_elements = [fourK_shield, fortyK_shield, bb_exit, lens, field_plates, dr_aperture]

    # Define beamline object
    beamline = Beamline(beamline_elements)

    # Define a simulator object
    simulator = TrajectorySimulator()
    state = Molecule().state
    # print(timeit.timeit("simulator.generate_molecules(state, N_molecules = 10000)", number = 1000, globals=locals()))

    # Run simulator
    apertures_of_interest = ["ES lens", "Field plates", "DR aperture", "Detected"]
    molecules = simulator.run_simulation(beamline, N_traj=int(1e5), apertures_of_interest = apertures_of_interest)

    # molecule = Molecule()
    # molecule.init_trajectory(beamline)
    # print(molecule.trajectory.x.shape)
    # molecule.update_trajectory()
    # print(molecule.trajectory.x.shape)
    # molecule.update_trajectory()
    # molecule.update_trajectory()
    # molecule.update_trajectory()
    # molecule.trajectory.drop_nans()
    # print(molecule.trajectory.x.shape)
    

if __name__ == "__main__":
    main()