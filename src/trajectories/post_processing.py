from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from .trajectory_simulator import SimulationResult


def take_timestep(x0: np.ndarray, v0: np.ndarray, a0: np.ndarray, dt: float):
    """
    With initial position, velocity and acceleration calculates position and velocity after 
    timestep dt
    """
    x = x0 + v0 * dt + a0 * dt ** 2 / 2
    v = v0 + a0 * dt

    return x, v


def find_radial_pos_dist(
    result: SimulationResult, z: float, elements: List[str] = None
) -> np.ndarray:
    """
    Returns the radial positions of the molecules stored in result at the given z position

    parameters:
    -----------
    result  : SimulationResult that contains the trajectories of the molecules etc.
    z:      : The z-position where the radial position distribution is to be found
    elements: The beamline elements that we are interested in (molcules that didn't hit
                one of these will be ignored)

    returns:
    numpy array of shape (2, len(results.molecules)) which has the x and y positions
    for each molecule at the specified z-position
    """
    # Loop over the molecules stored in result
    print("Determining radial positions...")
    rho_list = []
    for molecule in tqdm(result.molecules):
        # Check if the molecule made it to specified z along beamline, if not go next
        if molecule.x()[2] < z:
            continue

        # Check if molecule hit one of the apertures of interest
        if elements is not None:
            if molecule.aperture_hit not in elements:
                continue

        # If the exact value of z happens to be in the trajectory, return the radial position at
        # that point
        if z in molecule.trajectory.x[:, 2]:
            n = molecule.trajectory.x[:, 2].tolist().index(z)

            # Append x- and y-position to list
            rho_list.append(molecule.trajectory.x[n, :2])

        else:
            # Find the point in the molecule trajectory before z that is closest to the
            # specified z-position
            n = (molecule.trajectory.x[:, 2] < z).tolist().index(False) - 1

            # Find position, velocity and acceleration at this point
            x = molecule.trajectory.x[n, :]
            v = molecule.trajectory.v[n, :]
            a = molecule.trajectory.a[n, :]

            # Calculate the time the molecule will take to reach z
            dt = (z - x[2]) / v[2]

            # Calculate position and velocity after dt
            x, v = take_timestep(x, v, a, dt)

            # Append to x- and y- position list
            rho_list.append(x[:2])

    # Convert list to array and return it
    return np.array(rho_list)


def find_vel_dist(
    result: SimulationResult, z: float, elements: List[str] = None
) -> np.ndarray:
    """
    Returns the velocities of the molecules stored in result at the given z position


    parameters:
    -----------
    result  : SimulationResult that contains the trajectories of the molecules etc.
    z:      : The z-position where the velocity distribution is to be found
    elements: The beamline elements that we are interested in (molcules that didn't hit
                one of these will be ignored)

    returns:
    numpy array of shape (3, len(results.molecules)) which has the xyz velocities
    for each molecule at the specified z-position
    """
    # Loop over the molecules stored in result
    print("Determining radial velocities...")
    v_list = []
    for molecule in tqdm(result.molecules):
        # Check if the molecule made it to specified z along beamline, if not go next
        if molecule.x()[2] < z:
            continue

        # Check if molecule hit one of the apertures of interest
        if elements is not None:
            if molecule.aperture_hit not in elements:
                continue

        # If the exact value of z happens to be in the trajectory, return the radial position at
        # that point
        if z in molecule.trajectory.x[:, 2]:
            n = molecule.trajectory.x[:, 2].tolist().index(z)

            # Append x- and y-velocity to list
            v_list.append(molecule.trajectory.v[n, :])

        else:
            # Find the point in the molecule trajectory before z that is closest to the
            # specified z-position
            n = (molecule.trajectory.x[:, 2] < z).tolist().index(False) - 1

            # Find position, velocity and acceleration at this point
            x = molecule.trajectory.x[n, :]
            v = molecule.trajectory.v[n, :]
            a = molecule.trajectory.a[n, :]

            # Calculate the time the molecule will take to reach z
            dt = (z - x[2]) / v[2]

            # Calculate position and velocity after dt
            x, v = take_timestep(x, v, a, dt)

            # Append to x- and y- velocity list
            v_list.append(v)

    # Convert list to array and return it
    return np.array(v_list)


def plot_radial_pos_dist(result: SimulationResult, z: float):
    """
    Plots the radial position distribution of the molecules at a given z-position along the 
    beamline.
    """
    # to do
    pass


def plot_radial_vel_dist(result: SimulationResult, z: float):
    """
    Plots the radial velocity distribution of the molecules at a given Z-position along the
    beamline.
    """
    # to do
    pass
