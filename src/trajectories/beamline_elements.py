from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Value
import pickle
from os.path import exists
from pathlib import Path

import h5py
from h5py._hl import group
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import g

from .beamline import Beamline
from centrex_TlF.states import UncoupledBasisState, State
from .molecule import Molecule
from .stark_potential import stark_potential

@dataclass
class BeamlineElement(ABC):
    """
    Abstract dataclass that defines the methods that must be implemented for all beamline elements
    that are used in the simulations.
    """
    name: str # Name of beamline element
    z0: float # Z-position where the beamline element starts in meters
    L: float # Length or thickness of element along Z
    x0: float = 0. # X-coordinate of center of the element (0 corresponds to being on straight line from cold cell)
    y0: float = 0. # Y-coordinate of center of element (0 corresponds to being on straight line from cold cell)

    def __post_init__(self):
        self.z1 = self.z0 + self.L

    @abstractmethod
    def propagate_through(self, molecule: Molecule):
        """
        Propagates a molecule through the beamline element
        """
    
    @abstractmethod
    def N_steps(self):
        """
        Calculates the number of timesteps that are used when propagating through the element.
        """

    @abstractmethod
    def plot(self, axes):
        """
        Plots the beamline element on the provided axes (axes[0] should be XZ plane and axes[1] YZ plane)
        """

    def save_to_hdf(self, filepath: Path, parent_group_path: str) -> None:
        """
        Saves the beamline element to an hdf file 
        """
        # Open the hdf file
        with h5py.File(filepath, 'a') as f:
            try:
                # Create a group for the beamline element
                group_path = parent_group_path + "/" + self.name
                f.create_group(group_path)

                # Write the name of the beamline element class into file
                f[group_path].attrs['class'] = type(self).__name__

                # Loop over the attributes of the beamline element and save them to the attributes
                # of the group
                for key, value in vars(self).items():
                    f[group_path].attrs[key] = value
            
            except ValueError:
                "Group already exists!" 
        
@dataclass
class CircularAperture(BeamlineElement):
    """
    Class used for beamline elements that have a circular aperture through which the molecules
    are supposed to pass.
    """
    d: float = 0.0254 # Diameter of aperture in meters
    
    def propagate_through(self, molecule: Molecule):
        """
        Function that checks if the molecule makes it through the aperture without hitting it. I don't really care
        where exactly the molecule hits the aperture, so I'm not calculating the position. Rather just check
        if the molecule is within the diameter of the aperture when it is entering and  exiting the aperture 
        (this ignores the possible edge case of a parabolic trajectory that goes beyond the diameter of the aperture 
        but returns to below the diameter before the end of the aperture - I think the effect of this is negligible).
        """
        # Loop over start of element and end of element
        for z in [self.z0, self.z1]:
            #Calculate the time taken to reach start of element from initial position
            delta_t = (z - molecule.x()[2])/molecule.v()[2]

            # Calculate the position and velocity of the molecule at start of beamline element
            molecule.update_trajectory(delta_t)

            # Determine if molecule is now within the clear part of the aperture. If not, the molecule is 
            # considered dead
            rho = np.sqrt(np.sum(molecule.x()[:2]**2))
            if rho > self.d/2:
                molecule.set_dead()
                molecule.set_aperture_hit(self.name)

                return

    def N_steps(self):
        """
        Max number of steps that are taken when passing through a simple element that doesn't modify the trajectory
        of an incoming molecule is two (entering and exiting)
        """
        return 2
    

    def plot(self, axes):
        """
        Plot the aperture on the provided axes
        """
        rect1 = Rectangle((self.z0, self.d/2), self.z1-self.z0, 1, color = (.5, .5,.5))
        rect2 = Rectangle((self.z0, -self.d/2-1), self.z1-self.z0, 1,color = (0.5, 0.5, 0.5))
        axes[0].add_patch(rect1)
        axes[0].add_patch(rect2)


        rect3 = Rectangle((self.z0, self.d/2), self.z1-self.z0, 1, color = (.5, .5,.5))
        rect4 = Rectangle((self.z0, -self.d/2-1), self.z1-self.z0, 1,color = (0.5, 0.5, 0.5))
        axes[1].add_patch(rect3)
        axes[1].add_patch(rect4)


@dataclass
class RectangularAperture(BeamlineElement):
    """
    Class used for beamline elements that have a rectangular aperture through which the molecules
    are supposed to pass.
    """
    w: float = 0.02 # Height of aperture along Y in meters
    h: float = 0.02 # Width of aperture along X in meters

    def __post_init__(self):
        super().__post_init__()
        # Calculate the coordinates of the edges of the aperture
        self.x1 = self.x0 - self.w/2
        self.x2 = self.x0 + self.w/2
        self.y1 = self.y0 - self.h/2
        self.y2 = self.y0 + self.h/2

    
    def propagate_through(self, molecule: Molecule):
        """
        Function that checks if the molecule makes it through the aperture without hitting it. I don't really care
        where exactly the molecule hits the aperture, so I'm not calculating the position. Rather just check
        if the molecule is within the diameter of the aperture when it is entering and  exiting the aperture 
        (this ignores the possible edge case of a parabolic trajectory that goes beyond the diameter of the aperture 
        but returns to below the diameter before the end of the aperture - I think the effect of this is negligible).
        """
        # Loop over start of element and end of element
        for z in [self.z0, self.z1]:
            #Calculate the time taken to reach start of element from initial position
            delta_t = (z - molecule.x()[2])/molecule.v()[2]

            # Calculate the position and velocity of the molecule at start of beamline element
            molecule.update_trajectory(delta_t)

            # Determine if molecule is now within the clear part of the aperture. If not, the molecule is 
            # considered dead
            if not ((self.x1 < molecule.x()[0] < self.x2) and (self.y1 < molecule.x()[1] < self.y2)):
                molecule.set_dead()
                molecule.set_aperture_hit(self.name)
                return

    def N_steps(self):
        """
        Max number of steps that are taken when passing through a simple element that doesn't modify the trajectory
        of an incoming molecule is two (entering and exiting)
        """
        return 2

    def plot(self, axes):
        """
        Plot aperture on the provided axes
        """
        rect1 = Rectangle((self.z0, self.x2), self.z1-self.z0, 0.05, color = 'k')
        rect2 = Rectangle((self.z0, self.x1-0.05), self.z1-self.z0, 0.05, color = 'k')
        axes[0].add_patch(rect1)
        axes[0].add_patch(rect2)

        rect3 = Rectangle((self.z0, self.y2), self.z1-self.z0, 0.05, color = 'k')
        rect4 = Rectangle((self.z0, self.y1-0.05), self.z1-self.z0, 0.05, color = 'k')
        axes[1].add_patch(rect3)
        axes[1].add_patch(rect4)

@dataclass
class FieldPlates(BeamlineElement):
    """
    Class for the field plates of the main interaction region of CeNTREX
    """
    w: float = 0.02 # Height of aperture along Y in meters

    def __post_init__(self):
        super().__post_init__()
        # Calculate the coordinates of the edges of the aperture
        self.x1 = self.x0 - self.w/2
        self.x2 = self.x0 + self.w/2

    def propagate_through(self, molecule: Molecule):
        """
        Function that checks if the molecule makes it through the field plates without hitting them. If the molecule
        hits the field plates, the position where it hits them is calculated
        """
        #Calculate the time taken to reach start of element from initial position
        delta_t = (self.z0 - molecule.x()[2])/molecule.v()[2]

        # Calculate the position and velocity of the molecule at start of beamline element
        molecule.update_trajectory(delta_t)

        # Determine if molecule is now within the clear part of the aperture. If not, the molecule is 
        # considered dead
        if not (self.x1 < molecule.x()[0] < self.x2):
            molecule.set_dead()
            molecule.set_aperture_hit(self.name)
            return

        # Next check if the molecule makes it through the field plates to the end
        #Calculate the time taken to reach end of the field plates from initial position
        delta_t = (self.z1 - molecule.x()[2])/molecule.v()[2]

        # Calculate the position of the molecule at the end of the field plates
        x = molecule.x(delta_t)

        # Check if molecule is within bounds
        if not (self.x1 < x[0] < self.x2):
            # Calculate time taken to hit field plate if molecule is moving in -ve x-direction
            if molecule.v()[0] < 0:
                delta_t = (self.x1 - molecule.x()[0])/molecule.v()[0]
            #Calculate time taken to hit field plate if molecule is moving in +ve x-direction
            elif molecule.v()[0] > 0:
                delta_t = (self.x2 - molecule.x()[0])/molecule.v()[0]

            # Calculate final position of molecule
            molecule.update_trajectory(delta_t)
            molecule.set_dead()
            molecule.set_aperture_hit(self.name)
            return
        
        else:
            # If molecule made it through, update trajectory
            molecule.update_trajectory(delta_t)
            return

    def N_steps(self):
        """
        Max number of steps that are taken when passing through a simple element that doesn't modify the trajectory
        of an incoming molecule is two (entering and exiting)
        """
        return 2

    def plot(self, axes):
        """
        Plot field plates on the provided axes
        """
        rect1 = Rectangle((self.z0, self.x2), self.z1-self.z0, 0.02, color = 'y')
        rect2 = Rectangle((self.z0, self.x1-0.02), self.z1-self.z0, 0.02, color = 'y')
        axes[0].add_patch(rect1)
        axes[0].add_patch(rect2)

@dataclass
class ElectrostaticLens(BeamlineElement):
    """
    Class used for propagating molecules through the electrostatic lens.
    """
    d: float = 1.75*0.0254 # Bore diameter of lens in m
    dz: float = 1e-3 # Spatial size of integration step that is taken inside the lens
    V: float = 27e3 # Voltage on lens electrodes
    a_interp: interp1d = None
    state: State = 1*UncoupledBasisState(J = 2, mJ = 0, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = -1/2, Omega = 0,
                                            P = +1, electronic_state = 'X')# Molecular state assumed for the molecule to calculate trajectory inside electrostatic lens
    mass: float = (204.38+19.00)*1.67e-27 # Molecular mass in kg for molecule that will be propagating through lens

    def propagate_through(self, molecule):
        """
        Propagates a molecule through the electrostatic lens.
        """        
        #Calculate the time taken to reach start of lens from initial position
        delta_t = (self.z0 - molecule.x()[2])/molecule.v()[2]

        # Calculate the position and velocity of the molecule at start of beamline element
        molecule.update_trajectory(delta_t)

        # Determine if molecule is now within the lens bore. If not, the molecule is 
        # considered dead
        rho = np.sqrt(np.sum(molecule.x()[:2]**2))
        if rho > self.d/2:
            molecule.set_dead()
            molecule.set_aperture_hit("Lens entrance")
            return

        # Now propagate the molecule inside the lens. RK4 is used to integrate the time-evolution of the trajectory
        # here
        self.propagate_inside_lens(molecule)
        if not molecule.alive:
            return

        # Make sure that molecule is now at the end of the lens:
        #Calculate the time taken to reach start of lens from initial position
        delta_t = (self.z1 - molecule.x()[2])/molecule.v()[2]
    
        # Calculate the position and velocity of the molecule at start of beamline element
        molecule.update_trajectory(delta_t)

    def propagate_inside_lens(self, molecule):
        """
        Function that calculates and updates the trajectory of a molecule while it is inside the electrostatic lens.
        """
        # Get value of counter that keeps track of trajectory indices
        n = molecule.trajectory.n

        # Calculate number of integration steps to take inside the lens and the timestep
        N_steps = int(np.rint(self.L/self.dz))
        dt = self.dz/molecule.v()[2]

        # Loop over timesteps
        for i in range(N_steps):
            x = molecule.x()
            k1 = molecule.v()
            l1 = self.lens_acceleration(x)

            k2 = k1+dt*l1/2
            l2 = self.lens_acceleration(x + dt*k1)

            k3 = k1+dt*l2/2
            l3 = self.lens_acceleration(x + dt*k2/2)

            k4 = k1+dt*l3
            l4 = self.lens_acceleration(x + dt*k3)

            # Update the molecule trajectory
            molecule.trajectory.x[n, :] = x + dt*(k1+2*k2+2*k3+k4)/6
            molecule.trajectory.v[n, :] = k1 + dt*(l1+2*l2+2*l3+l4)/6
            molecule.trajectory.t[n] = molecule.trajectory.t[n-1] + dt
            molecule.trajectory.a[n, :] = l1
            n += 1
            molecule.trajectory.n = n

            #Cut molecules outside the allowed region
            rho = np.sqrt(np.sum(molecule.x()[:2]**2))
            if rho > self.d/2:
                molecule.set_dead()
                molecule.set_aperture_hit("Inside lens")
                return

    def N_steps(self):
        """
        Number of steps when going through an electrostatic lens is 1 for getting to the entrance of the lens
        plus int(np.rint(self.L/self.dz)) for propagating inside and exiting the lens
        """
        return 1 + int(np.rint(self.L/self.dz))

    def plot(self, axes):
        """
        Plot lens on the provided axes
        """
        rect1 = Rectangle((self.z0, self.d/2), self.z1-self.z0, 0.02, color = 'b')
        rect2 = Rectangle((self.z0, -self.d/2-0.02), self.z1-self.z0, 0.02, color = 'b')
        axes[0].add_patch(rect1)
        axes[0].add_patch(rect2)

        rect3 = Rectangle((self.z0, self.d/2), self.z1-self.z0, 0.02, color = 'b')
        rect4 = Rectangle((self.z0, -self.d/2-0.02), self.z1-self.z0, 0.02, color = 'b')
        axes[1].add_patch(rect3)
        axes[1].add_patch(rect4)

    def save_to_hdf(self, filepath: Path, parent_group_path: str) -> None:
        """
        Saves the beamline element to an hdf file 
        """
        # Open the hdf file
        with h5py.File(filepath, 'a') as f:
            # Create a group for the beamline element
            group_path = parent_group_path + "/" + self.name
            f.create_group(group_path)

            # Loop over the attributes of the beamline element and save them to the attributes
            # of the group
            for key, value in vars(self):
                if key != 'state':
                    f[group_path].attrs[key] = value

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
            filename = f"acceleration_interp_d={self.d:.4f}m_V={self.V:.1f}V_J={J}_mJ={mJ}.pkl"
            INTERP_DIR = "./interpolation_functions/"
            if exists(INTERP_DIR+filename):
                with open(INTERP_DIR+filename, 'rb') as f:
                    self.a_interp = pickle.load(f)

            # If not, calculate the acceleration as function of position inside the lens
            else:
                print("Interpolation function for lens acceleration not found, making new one")
                # Make an array of radius values for which to calculate the Stark shift
                dr = 1e-4
                r_values = np.linspace(0,self.d/2*1.01,int(np.round(self.d/2/dr)))
                
                # Convert radius values into electric field values (assuming E = 2*V/R^2 * r within the lens radius)
                E_values = 2*self.V/((self.d/2)**2) * r_values # E is in V/m

                # Calculate the Stark shift at each of these positions
                V_stark = stark_potential(self.state, E_values/100)

                # Calculate radial acceleration at each radius based on dV_stark/dr
                a_values = -np.gradient(V_stark, dr)/self.mass

                # Make an interpolating function based on the radial positions and calculated accelerations
                self.a_interp = interp1d(r_values, a_values)

                # Save the interpolation function to file
                with open(INTERP_DIR+filename, 'wb+') as f:
                    pickle.dump(self.a_interp, f)

        # Calculate the acceleration at the position of the molecule using the interpolation function
        r = np.sqrt(np.sum(x[:2]**2))
        a_r = self.a_interp(r)

        a = np.zeros((3,))
        if r != 0:
            #Resolve acceleration into components
            a[0] = a_r*x[0]/r
            a[1] = a_r*x[1]/r
            a[2] = 0

        a[1] -= g
        
        return a


def main():
    aperture = CircularAperture(name='40K Shield', z0 = .99, L = 0.01, d = 0)
    rect_aperture = RectangularAperture(name = 'Detection aperture', z0 = 100, L = .01, w = 0.00001, h = 1e-6)
    beamline = Beamline([aperture, rect_aperture])
    molecule = Molecule()
    molecule.init_trajectory(beamline)
    beamline.propagate_through(molecule)
    print(molecule)

    lens = ElectrostaticLens(name = "Lens", z0 = 0.3, L =  0.6, V = 27e3)
    lens.lens_acceleration(np.array((0,0,0.01)))

    fig, ax = plt.subplots()
    rs = np.linspace(0, lens.d/2,1000)
    accs  = np.array([np.array(lens.lens_acceleration(np.array((r,0,0)))[0]) for r in rs])
    ax.plot(rs, accs)
    plt.show()

    file_path = Path("./saved_data/test.hdf")
    run_name = 'test'
    aperture.save_to_hdf(filepath=file_path, parent_group_path=run_name)




if __name__ == '__main__':
    main()

