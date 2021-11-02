import numpy as np
from centrex_TlF.states import generate_uncoupled_states_ground, UncoupledBasisState, State
from centrex_TlF.states.utils import find_closest_vector_idx
from centrex_TlF.hamiltonian import generate_uncoupled_hamiltonian_X, generate_uncoupled_hamiltonian_X_function
import matplotlib.pyplot as plt
from scipy.constants import hbar

def stark_potential(state, Ezs):
    """
    Function that finds the Stark shift as a function of electric field for a molecule in any X-state of TlF

    inputs:
    state = Molecular state that is propagating through the electrostatic lens
    H = Hamiltonian on of TlF as function of electric field
    QN = list of states that defines the basis for the Hamiltonian
    Ezs = Array with magnitudes of electric field where stark shift needs to be calculated (V/cm)

    outputs:
    V_array = Stark potential at various electric field values
    """
    #Get Hamiltonian from file
    #Make list of quantum numbers that defines the basis for the matrices
    QN = generate_uncoupled_states_ground([0,1,2,3,4,5,6])

    #Get H_0 from file (H_0 should be in rad/s)
    H = generate_uncoupled_hamiltonian_X_function(generate_uncoupled_hamiltonian_X(QN))

    #Electric and magnetic field values for intializing reference matrix of eigenvectors and eigenvalues 
    E = np.array((0,0,100))
    B = np.array((0,0,0.0001))

    #Initialize the reference matrix of eigenvectors
    H_0 = H(E,B)
    E_ref, V_ref = np.linalg.eigh(H_0)
    V_ref_0 = V_ref
    state_energies = np.zeros((len(Ezs),len(E_ref)))

    for i, Ez in enumerate(Ezs):
        E = np.array((0,0,Ez))    
        H_i = H(E,B)
        
        #Diagonalize H
        D_0, V_0 = np.linalg.eigh(H_i)
        
        #Diagonalize the Hamiltonian
        energies, evecs = D_0, V_0
        
        #Reorder the eigenstates so that a given index corresponds to the same "adiabatic" state
        energies, evecs = reorder_evecs(evecs,energies,V_ref)

        #Store the calculated energies
        state_energies[i,:] = energies
        
        V_ref = evecs

    #Find index of the state we want
    state_i = find_closest_vector_idx(state.state_vector(QN), V_ref_0)

    #Find the energies of the desired state at the various electric field values (in J)
    V_array = state_energies[:,state_i]*hbar

    fig, ax = plt.subplots()
    ax.plot(Ezs, V_array)
    plt.show()

    return V_array


""" 
Function to reshuffle the eigenvectors and eigenenergies based on a reference
V_in = eigenvector matrix to be reorganized
E_in = energy vector to be reorganized
V_ref = reference eigenvector matrix to be reorganized
V_out = reorganized version of V_in
E_out = reorganized version of E_in
"""
def reorder_evecs(V_in,E_in,V_ref):
    #Take dot product between each eigenvector in V and state_vec
    overlap_vectors = np.absolute(np.matmul(np.conj(V_in.T),V_ref))
    
    #Find which state has the largest overlap:
    index = np.argsort(np.argmax(overlap_vectors,axis = 1))
    #Store energy and state
    E_out = E_in[index]
    V_out = V_in[:,index]   
    
    return E_out, V_out


def main():
    state = 1*UncoupledBasisState(J = 3, mJ = 0, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = -1/2, Omega = 0,
                                  P = -1, electronic_state = 'X')

    Ezs = np.linspace(0,70000, 1000)
    V_starkJ3 = stark_potential(state, Ezs)

    state = 1*UncoupledBasisState(J = 2, mJ = 0, I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = -1/2, Omega = 0,
                                  P = 1, electronic_state = 'X')
    V_starkJ2 = stark_potential(state, Ezs)


    fig, ax = plt.subplots()
    ax.plot(Ezs, (V_starkJ3-V_starkJ3[0])/hbar/(2*np.pi)/1e6)
    ax.plot(Ezs, (V_starkJ2-V_starkJ2[0])/hbar/(2*np.pi)/1e6)

    ax.set_xlabel("E-field magnitude / V/cm")
    ax.set_ylabel("Stark shift / MHz")
    plt.show()

if __name__ == "__main__":
    main()