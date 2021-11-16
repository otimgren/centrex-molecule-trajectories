"""
Defines the beamline used for simulating the effect of the electrostatic lens back when it was designed. The 
results here are the same as using the old code.
"""

from pathlib import Path
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

    # Run simulator
    aoi = ["Detected", "DR aperture", "Field plates", "Inside lens"] # Define apertures of interest so 
    run_name = 'Electrostatic lens simulation 11/15/2021 - 1e6'
    simulator.run_simulation(beamline, run_name, N_traj=int(1e6), apertures_of_interest = aoi, n_jobs = 10)
    simulator.counter.print()
    print(f"Beamline efficiency: {simulator.counter.calculate_efficiency()*100:.4f}%")

    # Save results to file
    filepath = Path("./saved_data/lens_simulation_beamline.hdf")
    simulator.result.save_to_hdf(filepath, run_name)

    
if __name__ == "__main__":
    main()