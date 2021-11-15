from pathlib import Path
import importlib

from trajectories.beamline_elements import BeamlineElement, CircularAperture, ElectrostaticLens, FieldPlates, RectangularAperture
from trajectories.beamline import Beamline
from trajectories.trajectory_simulator import TrajectorySimulator
from trajectories.utils import import_beamline_from_hdf

from pathlib import Path

import h5py

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
    print(beamline)

    # Define a simulator object
    simulator = TrajectorySimulator()

    # Run simulator
    aoi = ["DR aperture", "Detected", "Field plates", "Inside lens"]
    simulator.run_simulation(beamline, 'test', N_traj=int(1e4), apertures_of_interest = aoi,
                             n_jobs = 10)
    simulator.counter.print()
    print(f"Beamline efficiency: {simulator.counter.calculate_efficiency()*100:.4f}%")

    filepath = Path("./saved_data/test.hdf")
    run_name = 'run10'
    print(len(simulator.result.molecules))
    # simulator.result.plot()
    simulator.result.save_to_hdf(filepath, run_name)

    # Test saving beamline to file
     ## beamline.save_to_hdf(filepath=file_path, parent_group_path=run_name)

    # # Test importing beamline from file
    # beamline2 = import_beamline_from_hdf(file_path, run_name) 
    # print(beamline2)

if __name__ == "__main__":
    main()