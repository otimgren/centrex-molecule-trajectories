"""
Post-processing the results for the electrostatic lens simulation beamline.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from trajectories.post_processing import find_radial_pos_dist
from trajectories.utils import import_sim_result_from_hdf


def main():

    # Get the result from file
    filepath = Path("./saved_data/SPA_pos_vel_distr.hdf")
    run_name = "SPA position distributions - with apertures - 3-29-2022"
    result = import_sim_result_from_hdf(filepath, run_name)

    # Plot some trajectories
    result.plot()

    # Plot the radial position distribution of molecules when at center of microwaves
    z = 32.5 * 0.0254
    rho = find_radial_pos_dist(result, z)

    print(rho.shape)

    # Plot a 2D histogram of the position distribution
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.hist2d(rho[:, 0], rho[:, 1], bins=25)
    ax.set_xlabel("X-position / m")
    ax.set_ylabel("Y-position / m")
    plt.show()

    # Plot 1D histograms for each axis
    # X-position
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.hist(rho[:, 0], bins=25)
    ax.set_xlabel("X-position / m")
    plt.show()

    # Y-position
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.hist(rho[:, 1], bins=25)
    ax.set_xlabel("Y-position / m")
    plt.show()


if __name__ == "__main__":
    main()
