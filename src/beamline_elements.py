import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from molecule import Molecule
import pickle
from scipy.interpolate import interp1d
from os.path import exists
from stark_potential import stark_potential

@dataclass
class BeamlineElement(ABC):
    """
    Abstract dataclass that defines the methods that must be implemented for all beamline elements
    that are used in the simulations.
    """
    name: str # Name of beamline element
    z0: float # Z-position where the beamline element starts in meters
    z1: float # Z-position where the beamline element ends in meters
    x0: float = 0. # X-coordinate of center of the element (0 corresponds to being on straight line from cold cell)
    y0: float = 0. # Y-coordinate of center of element (0 corresponds to being on straight line from cold cell)

    @abstractmethod
    def propagate_through(self, molecule: Molecule):
        """
        Propagates a molecule through the beamline element
        """
        ...
    
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
            t1 = (z - molecule.x[2])/molecule.v[2]

            # Calculate the position and velocity of the molecule at start of beamline element
            molecule.update_position(t1)
            molecule.update_trajectory()

            # Determine if molecule is now within the clear part of the aperture. If not, the molecule is 
            # considered dead
            rho = np.sqrt(np.sum(molecule.x[:2]**2))
            if rho > self.d/2:
                molecule.set_dead()
                molecule.set_aperture_hit(self.name)

                return

@dataclass
class RectangularAperture(BeamlineElement):
    """
    Class used for beamline elements that have a rectangular aperture through which the molecules
    are supposed to pass.
    """
    w: float = 0.02 # Height of aperture along Y in meters
    h: float = 0.02 # Width of aperture along X in meters
    
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
            t1 = (z - molecule.x[2])/molecule.v[2]

            # Calculate the position and velocity of the molecule at start of beamline element
            molecule.update_position(t1)
            molecule.update_trajectory()

            # Determine if molecule is now within the clear part of the aperture. If not, the molecule is 
            # considered dead
            x1, x2 = self.x0 - self.w/2, self.x0 + self.w/2
            y1, y2 = self.y0 - self.h/2, self.y0 + self.h/2
            if (x1 < molecule.x[0] < x2) and (y1 < molecule.x[1] < y2):
                molecule.set_dead()
                molecule.set_aperture_hit(self.name)


@dataclass
class ElectrostaticLens(BeamlineElement):
    """
    Class used for propagating molecules through the electrostatic lens.
    """
    d: float = 1.75*0.0254 # Bore diameter of lens in m
    L: float = 0.6 # Length of lens in m
    dz: float = 1e-3 # Spatial size of integration step that is taken inside the lens
    V: float = 27e3 # Voltage on lens electrodes

    def propagate_through(self, molecule):
        """
        Propagates a molecule through the electrostatic lens.
        """
        #Calculate the time taken to reach start of lens from initial position
        t1 = (self.z0 - molecule.x[2])/molecule.v[2]

        # Calculate the position and velocity of the molecule at start of beamline element
        molecule.update_position(t1)
        molecule.update_trajectory()

        # Determine if molecule is now within the lens bore. If not, the molecule is 
        # considered dead
        rho = np.sqrt(np.sum(molecule.x[:2]**2))
        if rho > self.d/2:
            molecule.set_dead()
            molecule.set_aperture_hit(self.name)

            return

        # Now propagate the molecule inside the lens. RK4 is used to integrate the time-evolution of the trajectory
        # here
        
        # Calculate number of integration steps to take inside the lens and the timestep
        N_steps = int(np.rint(self.L/self.dz))
        dt = self.dz/molecule.vz

        # Loop over timesteps
        for i in range(N_steps):
            k1 = molecule.v
            l1 = self.lens_acceleration(molecule.x, molecule)

            k2 = molecule.v+dt*l1/2
            l2 = self.lens_acceleration(molecule.x + dt*k1, molecule)

            k3 = molecule.v+dt*l2/2
            l3 = self.lens_acceleration(molecule.x + dt*k2/2, molecule)

            k4 = molecule.v+dt*l3
            l4 = self.lens_acceleration(molecule.x + dt*k3, molecule)

            molecule.x += dt*(k1+2*k2+2*k3+k4)/6
            molecule.v += dt*(l1+2*l2+2*l3+l4)/6
            molecule.t += dt
            molecule.a = l1
            molecule.update_trajectory()

            #Cut molecules outside the allowed region
            rho = np.sqrt(molecule.x[0]**2 + molecule.x[1]**2)
            if rho > self.d/2:
                molecule.set_dead()
                molecule.set_aperture_hit(self.name)
                return


    def lens_acceleration(self, x, molecule):
        """
        Calculates the acceleration (in m/s^2) for a molecule inside the electrostatic lens.
        """

        # Find the relevant quantum numbers for calculating the acceleration
        J = molecule.state.find_largest_component().J
        mJ = molecule.state.find_largest_component().mJ

        # Check if the interpolation function has already been saved to file
        filename = f"acceleration_interp_d={self.d:.4f}m_V={self.V:.1f}V_J={J}_mJ={mJ}.pkl"
        INTERP_DIR = "../interpolation_functions/"
        if exists(INTERP_DIR+filename):
            with open(INTERP_DIR+filename, 'rb') as f:
                a_interp = pickle.load(f)

        # If not, calculate the acceleration as function of position inside the lens
        else:
            # Make an array of radius values for which to calculate the Stark shift
            dr = 1e-4
            r_values = np.linspace(0,self.d/2*1.01,int(np.round(self.d/2/dr)))
            
            # Convert radius values into electric field values (assuming E = 2*V/R^2 * r within the lens radius)
            E_values = 2*self.V/((self.d/2)**2) * r_values # E is in V/m

            # Calculate the Stark shift at each of these positions
            V_stark = stark_potential(molecule.state, E_values/100)

            # Calculate radial acceleration at each radius based on dV_stark/dr
            a_values = -np.gradient(V_stark,dr)/molecule.mass

            # Make an interpolating function based on the radial positions and calculated accelerations
            a_interp = interp1d(r_values, a_values)

        # Calculate the acceleration at the position of the molecule using the interpolation function
        r = np.sqrt(np.sum(x[:2]**2))
        a_r = a_interp(r)

        a = np.zeros((3,))
        if r != 0:
            #Resolve acceleration into components
            a[0] = a_r*x[0]/r
            a[1] = a_r*x[1]/r - g
            a[2] = 0

        return a

def main():
    aperture =  CircularAperture(name='40K Shield', z0 = .99, z1 = 1, d = 0)
    rect_aperture = RectangularAperture(name = 'Detection aperture', z0 = 100, z1 = 100.01, w = 0.00001, h = 1e-6)
    molecule = Molecule()
    print(aperture)
    aperture.propagate_through(molecule)
    print(molecule)
    rect_aperture.propagate_through(molecule)
    print(molecule.state.find_largest_component().J)

if __name__ == '__main__':
    main()

