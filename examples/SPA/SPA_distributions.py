"""
Script for determining the position and velocity distributions of molecules
passing through the SPA region and ending up detected by the laser.
"""

from pathlib import Path

from trajectories.beamline import Beamline
from trajectories.beamline_elements.apertures import (CircularAperture,
                                                      FieldPlates,
                                                      RectangularAperture)
from trajectories.beamline_elements.electrostatic_lens import ElectrostaticLens
from trajectories.distributions import GaussianPositionDistribution
from trajectories.trajectory_simulator import TrajectorySimulator


def main():
    # Define the beamline elements
    m_per_in = 0.0254  # Constant for converting meters to inches

    fourK_shield = CircularAperture(
        z0=1.7 * m_per_in, L=0.25 * m_per_in, d=1 * m_per_in, name="4K shield"
    )

    fortyK_shield = CircularAperture(
        z0=fourK_shield.z1 + 1.25 * m_per_in,
        L=0.25 * m_per_in,
        d=1 * m_per_in,
        name="40K shield",
    )

    bb_exit = CircularAperture(
        z0=fortyK_shield.z1 + 2.5 * m_per_in,
        L=0.75 * m_per_in,
        d=4 * m_per_in,
        name="BB exit",
    )

    RC_entrance_aperture = CircularAperture(
        z0=17.36 * m_per_in, L=0.125 * m_per_in, d=8e-3, name="RC entrance",
    )

    RC_exit_aperture = CircularAperture(
        z0=(17.36+9) * m_per_in, L=0.125 * m_per_in, d=8e-3, name="RC exit",
    )

    SPA_entrance = CircularAperture(
        z0=bb_exit.z1 + 20.5 * m_per_in,
        L=0.375 * m_per_in,
        d=1.75 * m_per_in,
        name="SPA entrance",
    )

    SPA_exit = CircularAperture(
        z0=SPA_entrance.z1 + 9.625 * m_per_in,
        L=0.375 * m_per_in,
        d=1.75 * m_per_in,
        name="SPA exit",
    )

    DR_entrance_aperture = CircularAperture(
        z0=(35.37 + 11) * m_per_in, L=0.125 * m_per_in, d=150e-3, name="DR entrance",
    )

    laser = RectangularAperture(
        z0=DR_entrance_aperture.z1 + 3.02 * m_per_in,
        L=2e-3,
        name="laser",
        w=0.05,
        h=0.05,
    )

    # Collect beamline elements into a list
    beamline_elements = [
        fourK_shield,
        fortyK_shield,
        bb_exit,
        RC_entrance_aperture,
        RC_exit_aperture,
        SPA_entrance,
        SPA_exit,
        DR_entrance_aperture,
        laser,
    ]

    # Define beamline object
    beamline = Beamline(beamline_elements)
    beamline.plot()

    # Define a simulator object
    simulator = TrajectorySimulator()

    # Define apertures of interest
    aoi = [
        "Detected",
    ]

    # Run simulator
    run_name = "SPA position distributions - apertures - ACME ini pos - 5-16-2022 - 1e9"
    simulator.run_simulation(
        beamline, run_name, N_traj=int(1e9), apertures_of_interest=aoi, n_jobs=9,
         xdist= GaussianPositionDistribution(),
    )

    # Print how many molecules hit each aperture what percentage of molecules make
    # it to detection
    simulator.counter.print()
    print(f"Beamline efficiency: {simulator.counter.calculate_efficiency()*100:.4f}%")

    # Save results to file
    filepath = Path("./saved_data/SPA_pos_vel_distr.hdf")
    simulator.result.save_to_hdf(filepath, run_name)


if __name__ == "__main__":
    main()
