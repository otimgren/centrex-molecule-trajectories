import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os.path import exists
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from centrex_TlF.states import State, UncoupledBasisState
from matplotlib.patches import Rectangle
from scipy.constants import g
from scipy.interpolate import interp1d

from ..beamline import Beamline
from ..molecule import Molecule
from ..stark_potential import stark_potential
from .apertures import BeamlineElement

__all__ = ["ElectrostaticLens"]


@dataclass
class ElectrostaticLens(BeamlineElement):
    """
    Class used for propagating molecules through the electrostatic lens.
    """

    d: float = 1.75 * 0.0254  # Bore diameter of lens in m
    dz: float = 1e-3  # Spatial size of integration step that is taken inside the lens
    V: float = 27e3  # Voltage on lens electrodes
    a_interp: interp1d = None
    state: State = 1 * UncoupledBasisState(
        J=2,
        mJ=0,
        I1=1 / 2,
        m1=1 / 2,
        I2=1 / 2,
        m2=-1 / 2,
        Omega=0,
        P=+1,
        electronic_state="X",
    )  # Molecular state assumed for the molecule to calculate trajectory inside electrostatic lens
    mass: float = (
        204.38 + 19.00
    ) * 1.67e-27  # Molecular mass in kg for molecule that will be propagating through lens

    def propagate_through(self, molecule):
        """
        Propagates a molecule through the electrostatic lens.
        """
        # Calculate the time taken to reach start of lens from initial position
        delta_t = (self.z0 - molecule.x()[2]) / molecule.v()[2]

        # Calculate the position and velocity of the molecule at start of beamline element
        molecule.update_trajectory(delta_t)

        # Determine if molecule is now within the lens bore. If not, the molecule is
        # considered dead
        rho = np.sqrt(np.sum(molecule.x()[:2] ** 2))
        if rho > self.d / 2:
            molecule.set_dead()
            molecule.set_aperture_hit("Lens entrance")
            return

        # Now propagate the molecule inside the lens. RK4 is used to integrate the time-evolution of the trajectory
        # here
        self.propagate_inside_lens(molecule)
        if not molecule.alive:
            return

        # Make sure that molecule is now at the end of the lens:
        # Calculate the time taken to reach start of lens from initial position
        delta_t = (self.z1 - molecule.x()[2]) / molecule.v()[2]

        # Calculate the position and velocity of the molecule at start of beamline element
        molecule.update_trajectory(delta_t)

    def propagate_inside_lens(self, molecule):
        """
        Function that calculates and updates the trajectory of a molecule while it is inside the electrostatic lens.
        """
        # Get value of counter that keeps track of trajectory indices
        n = molecule.trajectory.n

        # Calculate number of integration steps to take inside the lens and the timestep
        N_steps = int(np.rint(self.L / self.dz))
        dt = self.dz / molecule.v()[2]

        # Loop over timesteps
        for i in range(N_steps):
            x = molecule.x()
            k1 = molecule.v()
            l1 = self.lens_acceleration(x)

            k2 = k1 + dt * l1 / 2
            l2 = self.lens_acceleration(x + dt * k1)

            k3 = k1 + dt * l2 / 2
            l3 = self.lens_acceleration(x + dt * k2 / 2)

            k4 = k1 + dt * l3
            l4 = self.lens_acceleration(x + dt * k3)

            # Update the molecule trajectory
            molecule.trajectory.x[n, :] = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            molecule.trajectory.v[n, :] = k1 + dt * (l1 + 2 * l2 + 2 * l3 + l4) / 6
            molecule.trajectory.t[n] = molecule.trajectory.t[n - 1] + dt
            molecule.trajectory.a[n, :] = l1
            n += 1
            molecule.trajectory.n = n

            # Cut molecules outside the allowed region
            rho = np.sqrt(np.sum(molecule.x()[:2] ** 2))
            if rho > self.d / 2:
                molecule.set_dead()
                molecule.set_aperture_hit("Inside lens")
                return

    def N_steps(self):
        """
        Number of steps when going through an electrostatic lens is 1 for getting to the entrance of the lens
        plus int(np.rint(self.L/self.dz)) for propagating inside and exiting the lens
        """
        return 1 + int(np.rint(self.L / self.dz))

    def plot(self, axes):
        """
        Plot lens on the provided axes
        """
        rect1 = Rectangle((self.z0, self.d / 2), self.z1 - self.z0, 0.02, color="b")
        rect2 = Rectangle(
            (self.z0, -self.d / 2 - 0.02), self.z1 - self.z0, 0.02, color="b"
        )
        axes[0].add_patch(rect1)
        axes[0].add_patch(rect2)

        rect3 = Rectangle((self.z0, self.d / 2), self.z1 - self.z0, 0.02, color="b")
        rect4 = Rectangle(
            (self.z0, -self.d / 2 - 0.02), self.z1 - self.z0, 0.02, color="b"
        )
        axes[1].add_patch(rect3)
        axes[1].add_patch(rect4)

    def save_to_hdf(self, filepath: Path, parent_group_path: str) -> None:
        """
        Saves the beamline element to an hdf file 
        """
        # Open the hdf file
        with h5py.File(filepath, "a") as f:
            # Create a group for the beamline element
            group_path = parent_group_path + "/" + self.name
            f.create_group(group_path)

            # Write the name of the beamline element class into file
            f[group_path].attrs["class"] = type(self).__name__

            # Loop over the attributes of the beamline element and save them to the attributes
            # of the group
            for key, value in vars(self).items():
                if key not in ["state", "a_interp"] and value:
                    f[group_path].attrs[key] = value
                elif key == "a_interp":
                    pass
                else:
                    f[group_path].attrs[key] = value.__repr__()

    def lens_acceleration(self, x):
        """
        Calculates the acceleration (in m/s^2) for a molecule inside the electrostatic lens. To speed this up, an
        interpolation function that gives the acceleration as a function of radial position is saved the first time
        this function is run for a given lens configuration and molecular state.
        """
        if not self.a_interp:
            # Find the relevant quantum numbers for calculating the acceleration
            J = self.state.find_largest_component().J
            mJ = self.state.find_largest_component().mJ

            # Check if the interpolation function has already been saved to file
            filename = (
                f"acceleration_interp_d={self.d:.4f}m_V={self.V:.1f}V_J={J}_mJ={mJ}.pkl"
            )
            INTERP_DIR = "./interpolation_functions/"
            if exists(INTERP_DIR + filename):
                with open(INTERP_DIR + filename, "rb") as f:
                    self.a_interp = pickle.load(f)

            # If not, calculate the acceleration as function of position inside the lens
            else:
                print(
                    "Interpolation function for lens acceleration not found, making new one"
                )
                # Make an array of radius values for which to calculate the Stark shift
                dr = 1e-4
                r_values = np.linspace(
                    0, self.d / 2 * 1.01, int(np.round(self.d / 2 / dr))
                )

                # Convert radius values into electric field values (assuming E = 2*V/R^2 * r within the lens radius)
                E_values = 2 * self.V / ((self.d / 2) ** 2) * r_values  # E is in V/m

                # Calculate the Stark shift at each of these positions
                V_stark = stark_potential(self.state, E_values / 100)

                # Calculate radial acceleration at each radius based on dV_stark/dr
                a_values = -np.gradient(V_stark, dr) / self.mass

                # Make an interpolating function based on the radial positions and calculated accelerations
                self.a_interp = interp1d(r_values, a_values)

                # Save the interpolation function to file
                with open(INTERP_DIR + filename, "wb+") as f:
                    pickle.dump(self.a_interp, f)

        # Calculate the acceleration at the position of the molecule using the interpolation function
        r = np.sqrt(np.sum(x[:2] ** 2))
        a_r = self.a_interp(r)

        a = np.zeros((3,))
        if r != 0:
            # Resolve acceleration into components
            a[0] = a_r * x[0] / r
            a[1] = a_r * x[1] / r
            a[2] = 0

        a[1] -= g

        return a
