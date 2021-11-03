# centrex-molecule-trajectories
This is a package for simulating the trajectories molecules/atoms in molecular/atomic beam experiments, specfically developed for the needs of the CeNTREX experiment but hopefully generalizable enough for other experiments also. The package implements a Monte Carlo simulation of trajectories by generating molecules based on given initial position and velocity distributions. The simulation propagates the molecules through a set of beamline elements (e.g. apertures, guiding or slowing elements) and tracks the trajectories, how many molecules hit each beamline element and how many make it through the experiment to detection.

## Installation
Copy the source code using git `git clone https://github.com/otimgren/centrex-molecule-trajectories.git` or by downloading and extracting the zip file from GitHub. Then run `python setup.py install` in the root folder (the one containing setup.py).

## Description
The basic structure of the simulation code is as follows:
1. Define a beamline i.e. the set of beamline elements that the molecular beam experiment is made of and that the molecules are supposed to fly through.
2. Generate a molecule based on an assumed initial position and velocity distribution.
3. Calculate the trajectory of the molecule through the beamline.
4. Record the trajectory, whether or not the molecule made it through the experiment to detection, or if it hit one the beamline elements

### Coordinate system
The software assigns the Z-axis to be along the long axis of the molecular beam experiment, Y to be the up-down direction (so $\vec{g}$ is pointing in the negative Y-direction) and X is then the left-right axis.

### Beamline element

Molecular beam experiments are usually made out of distinct modules which can have different effects on the trajectories of the molecules flying through them (e.g. an aperture might restrict the velocities of molecules that make it through an experiment, while an electrostatic lens would actually guide the trajectories). The package implements an abstract class `trajectories.beamline_elements.BeamlineElement` to reflect this. All concrete implementations of beamline elements (e.g. CircularAperture, ElectrostaticLens) should inherit from this class and implement the following methods:
- `propagate_through(self, molecule: Molecule)`: This method propagates a molecule through the beamline element by calculating its trajectory and checking if the molecule will hit the beamline element. Could be as simple as a ballistic trajectory under gravity or a more complicated like a molecule propagating through the electrostatic lens.
- `N_steps(self)`: Calculates the number of timesteps needed to propagate through the beamline element
- `plot(self)`: For plotting the beamline element on XZ and YZ planes

Each beamline element also must have the following attributes:
- `name: str`: Name to keep track of which beamline element molecules are hitting
- `z0`: The Z-coordinate of the start of the beamline element
- `L: float`: length of the beamline element along Z

### Beamline
A collection of beamline elements forms a beamline and we have the `trajectories.beamline.Beamline` class for that. When initializing it takes a list of `BeamlineElement`s. The methods are:
- `propagate_through(self, molecule, name: str = "ES lens")`: Propagates a molecule through the beamline using the `propagate_through` methods of each beamline element. `name` is a clunky way of specifying an element of the beamline where more entries are added to the molecular trajectory (will be fixed at some point).
- `sort_elements(self)`: Sorts the list of beamline elements by their `z0` values (i.e. order in which the molecule flies through them (assuming molecule is flying along positive Z-direction))
- `find_element(self, name)`: This returns the element with the given name
- `plot(self)`: Plots the whole beamline using the `plot` method of the beamline elements. Returns a pair of axes on which molecule trajectories can be plotted.

### Distributions
All distributions should inherit from the `trajectories.distributions.Distribution` abstract class to ensure that they have the `draw` method:
- `draw(self, n)`: Draws n samples from the distribution

`StandardVelocityDistribution` and `StandardPositionDistribution` are the velocity and position distributions based on latest data (as of 11/2/2021) for CeNTREX

### Molecule
A class `trajectories.molecule.Molecule` is implemented for molecules. Currently this just keeps track of the trajectory of the molecule using an instance of the `trajectories.molecule.Trajectory` class.

### Trajectory simulations
Use an instance of the `trajectories.trajectrory_simulator.TrajectorySimulator` to actually run the simulations. Methods:
- `run_simulation`: Takes as inputs the beamline, initial position and velocity distributions, how many molecules should be simulated, and a list of beamline elements of interest (molecules that hit any of these will have their trajectories saved).
- `run_simulation_parallel`: Same as `run_simulation` but runs in parallel. Use `n_jobs` to control number of processes used

The `TrajectorySimulator` also initializes an instance of the `Counter` class which is used to keep track of how many molecules hit each beamline element, even if the trajectories of molecules hitting some of these elements are not saved to conserve memory.
