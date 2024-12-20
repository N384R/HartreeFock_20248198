from itertools import product, combinations
import numpy as np
from scipy.special import erf
from .basis_set.basis_set import BasisSet


class HartreeFock:
    '''
    Class to perform Hartree-Fock calculations.

    Attributes:
    molecule (str): Molecule string.
    basis_set (BasisSet): Basis set object.
    overlap (np.ndarray): Overlap matrix.
    one_electron (np.ndarray): One-electron matrix.
    two_electron (np.ndarray): Two-electron matrix.
    energy (float): Energy of the molecule.
    eigenvectors (np.ndarray): Eigenvectors of the molecule.
    '''

    def __init__(self, molecule, basis_set_name, spin=0):
        self.molecule = molecule
        self.basis_set = BasisSet(basis_set_name)
        self.spin = spin
        self.energy_elec = None
        self.energy_total = None
        self.eigenvectors = None

        self.coordinates = self.set_coordinates(molecule)
        self.basis_functions = self.set_basis_function()
        self.overlap = self.calculate_overlap()
        self.kinetic = self.calculate_kinetic()
        self.nuclear_attraction = self.calculate_nuclear_attraction()
        self.two_electron = self.calculate_electron_repulsion()
        self.nuclear_repulsion = self.calculate_nuclear_repulsion()

    def set_coordinates(self, molecule):
        '''Parse the molecule string and return the coordinates of the atoms.'''

        coordinates = []
        for atom in molecule.split(';'):
            symbol = atom.strip().split()[0]
            x, y, z = map(float, atom.split()[1:])
            coordinates.append({'symbol': symbol, 'coord': (x, y, z)})
        return coordinates

    def set_basis_function(self):
        '''Set the contracted Gaussian basis functions for an atom.'''

        symbols = [coord['symbol'] for coord in self.coordinates]
        alpha, d, indices = [], [], []
        for idx, symbol in enumerate(symbols):
            for primitive in self.basis_set[symbol]['basis'].values():
                self.coordinates[idx]['charge'] = self.basis_set[symbol]['charge']
                alpha.append(primitive[:, 0])
                d.append(primitive[:, 1])
                indices.append(idx)

        contracted_gaussians = {
            'alpha': [np.array(a) for a in alpha],
            'd': [np.array(coeff) for coeff in d],
            'indices': indices
        }
        return contracted_gaussians

    def calculate_overlap(self):
        '''Calculate the overlap matrix.'''

        n = len(self.basis_functions['alpha'])
        overlap = np.zeros((n, n))
        for i, j in product(range(n), repeat=2):
            overlap[i, j] = self.__overlap_ij(i, j)
        return overlap

    def __overlap_ij(self, i, j):
        a = self.basis_functions['alpha']
        d = self.basis_functions['d']
        idx = self.basis_functions['indices']
        ri = self.coordinates[idx[i]]['coord']
        rj = self.coordinates[idx[j]]['coord']
        rij = np.linalg.norm(np.array(ri) - np.array(rj))
        overlap_ij = 0.0
        for k, alpha in enumerate(a[i]):
            for l, beta in enumerate(a[j]):
                p = alpha + beta
                gaussian_factor = ((np.pi / p)**(3/2)
                                   * (4 * alpha * beta / np.pi**2)**(3/4))
                exponential = np.exp(-alpha * beta * rij**2 / p)
                overlap_ij += d[i][k] * d[j][l] * gaussian_factor * exponential
        return overlap_ij.round(6)

    def calculate_kinetic(self):
        '''Calculate the kinetic energy matrix.'''

        n = len(self.basis_functions['alpha'])
        kinetic = np.zeros((n, n))
        for i, j in product(range(n), repeat=2):
            kinetic[i, j] = self.__kinetic_ij(i, j)
        return kinetic

    def __kinetic_ij(self, i, j):
        a = self.basis_functions['alpha']
        d = self.basis_functions['d']
        idx = self.basis_functions['indices']
        ri = self.coordinates[idx[i]]['coord']
        rj = self.coordinates[idx[j]]['coord']
        rij = np.linalg.norm(np.array(ri) - np.array(rj))
        kinetic_ij = 0.0
        for k, alpha in enumerate(a[i]):
            for l, beta in enumerate(a[j]):
                p = alpha + beta
                gaussian_factor = ((np.pi / p)**(3/2)
                                   * (alpha * beta / p)
                                   * (4 * alpha * beta / np.pi**2)**(3/4)
                                   * (3 - 2 * alpha * beta * rij**2 / p))
                exponential = np.exp(-a[i][k] * a[j][l] * rij**2 / p)
                kinetic_ij += d[i][k] * d[j][l] * gaussian_factor * exponential
        return kinetic_ij.round(6)

    def calculate_nuclear_attraction(self):
        '''Calculate the nuclear attraction matrix.'''

        n = len(self.basis_functions['alpha'])
        nuclear_attraction = np.zeros((n, n))
        for i, j in product(range(n), repeat=2):
            nuclear_attraction[i, j] = self.__nuclear_attraction_ij(i, j)
        return nuclear_attraction

    def __nuclear_attraction_ij(self, i, j):
        def f0(t):
            if t == 0:
                return 1.0
            return 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))
        a = self.basis_functions['alpha']
        d = self.basis_functions['d']
        idx = self.basis_functions['indices']
        ri = self.coordinates[idx[i]]['coord']
        rj = self.coordinates[idx[j]]['coord']
        rij = np.linalg.norm(np.array(ri) - np.array(rj))
        nuclear_attraction_ij = 0.0
        for k, alpha in enumerate(a[i]):
            for l, beta in enumerate(a[j]):
                p = alpha + beta
                rp = (alpha * np.array(ri) + beta * np.array(rj)) / p

                # Nuclear attraction for each nucleus
                for nucleus in self.coordinates:
                    rc = np.array(nucleus['coord'])
                    zc = nucleus['charge']
                    rpc = np.linalg.norm(rp - rc)

                    gaussian_term = ((2 * np.pi / p)
                                     * (4 * alpha * beta / np.pi**2)**(3/4)
                                     * np.exp(-alpha * beta * rij**2 / p))
                    erf_term = f0(p * rpc**2)
                    nuclear_attraction_ij += (-zc * d[i][k] * d[j][l]
                                              * gaussian_term * erf_term)
        return nuclear_attraction_ij.round(6)

    def calculate_electron_repulsion(self):
        '''Calculate the electron repulsion matrix.'''

        n = len(self.basis_functions['alpha'])
        electron_repulsion = np.zeros((n, n, n, n))
        for i, j, k, l in product(range(n), repeat=4):
            electron_repulsion[i, j, k, l] = self.__electron_repulsion_ijkl(i, j,
                                                                            k, l)
        return electron_repulsion

    def __electron_repulsion_ijkl(self, i, j, k, l):
        def f0(t):
            if t == 0:
                return 1.0
            return 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))
        a = self.basis_functions['alpha']
        d = self.basis_functions['d']
        idx = self.basis_functions['indices']
        ri = self.coordinates[idx[i]]['coord']
        rj = self.coordinates[idx[j]]['coord']
        rk = self.coordinates[idx[k]]['coord']
        rl = self.coordinates[idx[l]]['coord']
        rij = np.linalg.norm(np.array(ri) - np.array(rj))
        rkl = np.linalg.norm(np.array(rk) - np.array(rl))
        electron_repulsion_ijkl = 0.0
        for r, alpha in enumerate(a[i]):
            for s, beta in enumerate(a[j]):
                for t, gamma in enumerate(a[k]):
                    for u, delta in enumerate(a[l]):
                        p = alpha + beta
                        q = gamma + delta
                        rp = (alpha * np.array(ri) + beta * np.array(rj)) / p
                        rq = (gamma * np.array(rk) + delta * np.array(rl)) / q
                        rpq = np.linalg.norm(rp - rq)
                        gaussian_term = ((2 * np.pi**(5/2)) / (p * q * np.sqrt(p + q))
                                         * (16 * alpha * beta * gamma * delta / np.pi**4)**(3/4)
                                         * np.exp(-alpha * beta * rij**2 / p)
                                         * np.exp(-gamma * delta * rkl**2 / q))
                        erf_term = f0(p * q * rpq**2 / (p + q))
                        electron_repulsion_ijkl += (d[i][r] * d[j][s] * d[k][t] * d[l][u]
                                                    * gaussian_term * erf_term)
        return electron_repulsion_ijkl.round(6)

    def calculate_nuclear_repulsion(self):
        '''Calculate the nuclear repulsion energy.'''

        nuclear_repulsion = 0.0
        n = len(self.coordinates)
        for i, j in combinations(range(n), 2):
            zi = self.coordinates[i]['charge']
            zj = self.coordinates[j]['charge']
            ri = np.array(self.coordinates[i]['coord'])
            rj = np.array(self.coordinates[j]['coord'])
            rij = np.linalg.norm(ri - rj)
            nuclear_repulsion += zi * zj / rij
        return nuclear_repulsion.round(6)

    def __rhf(self):
        # Initialize the parameters
        n = len(self.basis_functions['alpha'])  # Number of basis functions
        max_iter = 100
        convergence_threshold = 1e-6

        # Retrieve matrices
        S = self.overlap
        T = self.kinetic
        V = self.nuclear_attraction
        G = self.two_electron
        nucl_rep = self.nuclear_repulsion

        # Core Hamiltonian
        H_core = T + V

        # Orthogonalization matrix (S^-1/2)
        S_eigvals, S_eigvecs = np.linalg.eigh(S)
        S_inv_sqrt = S_eigvecs @ np.diag(1 / np.sqrt(S_eigvals)) @ S_eigvecs.T

        # Initialize the density matrix
        P = np.zeros((n, n))

        # SCF iterations
        energy = 0.0
        for iteration in range(max_iter):
            # Build the Fock matrix
            F = H_core.copy()
            for i, j, k, l in product(range(n), repeat=4):
                F[i, j] += P[k, l] * (G[i, j, k, l] - G[i, l, k, j]/2)

            # Diagonalize the Fock matrix
            F_prime = S_inv_sqrt @ F @ S_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(F_prime)

            # eigenvectors to the original basis
            C = S_inv_sqrt @ eigvecs

            # update the density matrix
            P_new = np.zeros_like(P)
            num_electrons = sum(coord['charge']
                                for coord in self.coordinates) // 2
            for i in range(num_electrons):
                P_new += 2 * np.outer(C[:, i], C[:, i])

            # Calculate the electronic energy
            electronic_energy = 0.5 * np.sum(P_new * (H_core + F))

            # Check convergence
            if np.abs(electronic_energy - energy) < convergence_threshold:
                break

            # Update the density matrix and energy
            P = P_new
            energy = electronic_energy

        else:
            print("SCF did not converge within the maximum number of iterations.")

        # Total energy
        total_energy = energy + nucl_rep
        self.energy_elec = energy
        self.energy_total = total_energy
        self.eigenvectors = C

        print(f"SCF converged energy: {total_energy}")

        return total_energy, C

    def rhf(self):
        '''Perform Restricted Hartree-Fock calculation.'''
        return self.__rhf()

    def __rohf(self):
        # Initialize the parameters
        n = len(self.basis_functions['alpha'])  # Number of basis functions
        max_iter = 100
        convergence_threshold = 1e-6

        # Retrieve matrices
        S = self.overlap
        T = self.kinetic
        V = self.nuclear_attraction
        G = self.two_electron
        nucl_rep = self.nuclear_repulsion

        # Core Hamiltonian
        H_core = T + V

        # Orthogonalization matrix (S^-1/2)
        S_eigvals, S_eigvecs = np.linalg.eigh(S)
        S_inv_sqrt = S_eigvecs @ np.diag(1 / np.sqrt(S_eigvals)) @ S_eigvecs.T

        # Initialize the density matrix
        P_closed = np.zeros((n, n))
        P_open = np.zeros((n, n))

        num_electrons = sum(coord['charge'] for coord in self.coordinates)
        num_open = self.spin * 2
        num_closed = (num_electrons - num_open) // 2

        # SCF iterations
        energy = 0.0
        for iteration in range(max_iter):
            # Build the Fock matrix
            F = H_core.copy()
            for i, j, k, l in product(range(n), repeat=4):
                F[i, j] += P_closed[k, l] * (G[i, j, k, l] - G[i, l, k, j]/2)
                F[i, j] += P_open[k, l] * (G[i, j, k, l] - G[i, l, k, j])

            # Diagonalize the Fock matrix
            F_prime = S_inv_sqrt @ F @ S_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(F_prime)

            # eigenvectors to the original basis
            C = S_inv_sqrt @ eigvecs
            # update the density matrix
            P_new_closed = np.zeros_like(P_closed)
            P_new_open = np.zeros_like(P_open)

            for i in range(num_closed):
                P_new_closed += 2 * np.outer(C[:, i], C[:, i])

            for i in range(num_closed, num_closed + num_open):
                P_new_open += np.outer(C[:, i], C[:, i])

            P_total_new = P_new_closed + P_new_open
            # Calculate the electronic energy
            electronic_energy = 0.5 * np.sum(P_total_new * (H_core + F))

            # Check convergence
            if np.abs(electronic_energy - energy) < convergence_threshold:
                break

            # Update the density matrix and energy
            P_closed = P_new_closed
            P_open = P_new_open
            energy = electronic_energy

        else:
            print("SCF did not converge within the maximum number of iterations.")

        # Total energy
        total_energy = energy + nucl_rep
        self.energy_elec = energy
        self.energy_total = total_energy
        self.eigenvectors = C

        print(f"SCF converged energy: {total_energy}")

        return total_energy, C

    def rohf(self):
        '''Perform Restricted Open-Shell Hartree-Fock calculation.'''
        return self.__rohf()


if __name__ == "__main__":
    H2 = 'H 0.0 0.0 0.0; H 0.0 0.0 1.4'
    hf = HartreeFock(H2, 'sto-3g')
    print(hf.overlap)
