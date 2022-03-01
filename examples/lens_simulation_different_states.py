"""
I'm simulating the efficiency of the beamline used in 'lens_simulation_beamline.py' for different
molecular states. Different state are affected differently by the electrostatic lens, so will have
different probabilities of making it to detection.
"""

from os import stat
from pathlib import Path
from centrex_TlF.states import UncoupledBasisState
from trajectories.beamline_elements import CircularAperture, RectangularAperture, ElectrostaticLens, FieldPlates
from trajectories.beamline import Beamline
from trajectories.trajectory_simulator import TrajectorySimulator



def main():
    # Define the beamline elements
    m_per_in = 0.0254 # Constant for converting meters to inches
    fourK_shield = CircularAperture(z0 = 1.7*m_per_in, L = 0.25*m_per_in, d = 1*m_per_in, name = '4K shield')
    fortyK_shield = CircularAperture(z0 = fourK_shield.z1 + 1.25*m_per_in, L = 0.25*m_per_in, d = 1*m_per_in,
                                     name = '40K shield')
    bb_exit = CircularAperture(z0 = fortyK_shield.z1 + 2.5*m_per_in, L = 0.75*m_per_in, d = 4*m_per_in,
                               name = 'BB exit')
    lens = ElectrostaticLens(z0 = bb_exit.z1+33*m_per_in, L = 0.6, name = "ES lens")
    field_plates = FieldPlates(z0 = 2.43, L = 3.0, w = 0.02, name = "Field plates")
    dr_aperture = RectangularAperture(z0 = field_plates.z1 + 39.9*m_per_in, L = 0.25*m_per_in, name = "DR aperture",
                                      w = 0.018, h = 0.03)

    # Collect beamline elements into a list
    beamline_elements = [fourK_shield, fortyK_shield, bb_exit, lens, field_plates, dr_aperture]

    # Define beamline object
    beamline = Beamline(beamline_elements)

    # Define a simulator object
    simulator = TrajectorySimulator()

    # Define the apertures of interest (only interested in detected molecules)
    aoi = ["Detected"]

    # Define name for savefile
    filepath = Path("./saved_data/efficiency_for_different_states.hdf")

    # Define the states whose trajectories will be simulated (the nuclear spin part doesn't matter)
    states = [
        # 1*UncoupledBasisState(J = 0, mJ = 0, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = 1/2, Omega=0,
        #                         P = +1, electronic_state= 'X'),
        # 1*UncoupledBasisState(J = 1, mJ = 0, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = 1/2, Omega=0,
        #                         P = -1, electronic_state= 'X'),
        # 1*UncoupledBasisState(J = 1, mJ = 1, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = 1/2, Omega=0,
        #                         P = -1, electronic_state= 'X'),
        # 1*UncoupledBasisState(J = 2, mJ = 0, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = 1/2, Omega=0,
        #                         P = +1, electronic_state= 'X'),
        # 1*UncoupledBasisState(J = 2, mJ = 1, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = 1/2, Omega=0,
        #                         P = +1, electronic_state= 'X'),
        # 1*UncoupledBasisState(J = 2, mJ = 2, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = 1/2, Omega=0,
        #                         P = +1, electronic_state= 'X'),
        1*UncoupledBasisState(J = 3, mJ = 0, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = 1/2, Omega=0,
                                P = -1, electronic_state= 'X'),
        1*UncoupledBasisState(J = 3, mJ = 1, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = 1/2, Omega=0,
                                P = -1, electronic_state= 'X'),
        1*UncoupledBasisState(J = 3, mJ = 2, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = 1/2, Omega=0,
                                P = -1, electronic_state= 'X'),
        1*UncoupledBasisState(J = 3, mJ = 3, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = 1/2, Omega=0,
                                P = -1, electronic_state= 'X')
    ]

    # Loop over states and run the simulation
    for state in states:
        # Figure out a run name based on the quantum numbers
        J = state.find_largest_component().J
        mJ = state.find_largest_component().mJ
        run_name = f"J = {J}, mJ = {mJ}, N = 1e8"

        # Modify the state assumed for the electrostatic lens
        beamline.find_element("ES lens").state = state
        beamline.find_element("ES lens").a_interp = None

        # Run the simulation
        simulator.run_simulation(beamline, run_name, N_traj = int(1e8), apertures_of_interest=aoi,
                                 n_jobs = 10)

        # Print state and result
        efficiency = simulator.counter.calculate_efficiency()*100
        print(f"Beamline efficiency for {state}: {efficiency:.4f}%")

        # Save result
        simulator.result.save_to_hdf(filepath, run_name)

if __name__  == "__main__":
    main()