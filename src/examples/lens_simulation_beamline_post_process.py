"""
Post-processing the results for the electrostatic lens simulation beamline.
"""
from pathlib import Path

from trajectories.utils import import_sim_result_from_hdf

def main():

    # Get the result from file
    filepath = Path("./saved_data/lens_simulation_beamline.hdf")
    run_name = 'Electrostatic lens simulation 11/15/2021 - 1e6'
    result = import_sim_result_from_hdf(filepath, run_name)

    # Plot some trajectories
    result.plot()

    # Plot the radial position distribution of molecules about to enter the detection region





if __name__ == "__main__":
    main()