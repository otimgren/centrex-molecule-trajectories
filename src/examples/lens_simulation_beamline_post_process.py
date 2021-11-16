"""
Post-processing the results for the electrostatic lens simulation beamline.
"""
from trajectories.utils import import_sim_result_from_hdf

def main():
    result = import_sim_result_from_hdf("./saved_results/")

if __name__ == "__main__":
    main()