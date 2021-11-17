from pathlib import Path

import matplotlib.pyplot as plt

from trajectories.beamline_elements import BeamlineElement, CircularAperture, ElectrostaticLens, FieldPlates, RectangularAperture
from trajectories.beamline import Beamline
from trajectories.post_processing import find_radial_pos_dist
from trajectories.trajectory_simulator import TrajectorySimulator
from trajectories.utils import import_sim_result_from_hdf



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

    # # Define beamline object
    beamline = Beamline(beamline_elements)
    # print(beamline)

    # Define a simulator object
    simulator = TrajectorySimulator()

    # Run simulator
    # aoi = ["DR aperture", "Detected", "Field plates", "Inside lens"]
    # simulator.run_simulation(beamline, 'test', N_traj=int(1e4), apertures_of_interest = aoi,
    #                          n_jobs = 10)
    # simulator.counter.print()
    # print(f"Beamline efficiency: {simulator.counter.calculate_efficiency()*100:.4f}%")

    filepath = Path("./saved_data/lens_simulation_beamline.hdf")
    run_name = 'Electrostatic lens simulation 11/15/2021 - 1e6'
    # simulator.result.plot()
    # simulator.result.save_to_hdf(filepath, run_name)

    # Test importing simulation result from file
    result2 = import_sim_result_from_hdf(filepath, run_name) 
    # result2.plot()
    result2.counter.print()

    # Test getting the radial positions at a given position
    z = dr_aperture.z0
    rho = find_radial_pos_dist(result2, z)
    print(rho.shape)
    print(rho)

    # Plot a 2D histogram of the position distribution
    fig, ax = plt.subplots(figsize = (16,9))
    ax.hist2d(rho[:,0], rho[:,1], bins = 25)
    ax.set_xlabel("X-position / m")
    ax.set_ylabel("Y-position / m")
    plt.show()

    # Plot 1D histograms for each axis
    # X-position
    fig, ax = plt.subplots(figsize = (16,9))
    ax.hist(rho[:,0], bins = 25)
    ax.set_xlabel("X-position / m")
    plt.show()

    # Y-position
    fig, ax = plt.subplots(figsize = (16,9))
    ax.hist(rho[:,1], bins = 25)
    ax.set_xlabel("Y-position / m")
    plt.show()

if __name__ == "__main__":
    main()