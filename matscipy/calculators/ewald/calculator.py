#
# Copyright 2018-2019, 2021 Jan Griesser (U. Freiburg)
#           2021 Lars Pastewka (U. Freiburg)
#
# matscipy - Materials science with Python at the atomic-scale
# https://github.com/libAtoms/matscipy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""
Pair potential + Ewald summation
"""

#
# Coding convention
# * All numpy arrays are suffixed with the array dimensions
# * The suffix stands for a certain type of dimension:
#   - n: Atomic index, i.e. array dimension of length nb_atoms
#   - p: Pair index, i.e. array dimension of length nb_pairs
#   - c: Cartesian index, array dimension of length 3
#   - l: Wave vector index, i.e. array of dimension length nb_kvectors

import numpy as np

from scipy.special import erfc

from scipy.sparse.linalg import cg

import ase

from ase.calculators.calculator import Calculator

from itertools import product

from ...neighbours import neighbour_list, first_neighbours, mic

from ...numpy_tricks import mabincount

from ...elasticity import Voigt_6_to_full_3x3_stress

###

# We express charges q as multiples of the elementary charge e: q = x*e
# Prefactor and charge of Coulomb potential:  e^2/(4*pi*epsilon0) = 14.399645 eV * Angström 
conversion_prefactor = 14.399645
#conversion_prefactor = 1

###

class BKS_ewald():
    """
    Beest, Kramer, van Santen (BKS) potential.
    Functional form is Buckingham + Coulomb potential

    Buckingham:  
        Energy is shifted to zero at the cutoff.
    Coulomb:   
        Electrostatic interaction is treated using the traditional Ewald summation.
                     
    References
    ----------
        B. W. Van Beest, G. J. Kramer and R. A. Van Santen, Phys. Rev. Lett. 64.16 (1990)
    """

    def __init__(self, A, B, C, alp, cutoff_r, cutoff_k, nb_kspace, accuracy):
        self.A = A
        self.B = B 
        self.C = C
        self.alp = alp
        self.cutoff_r = cutoff_r
        self.cutoff_k = cutoff_k
        self.nb_kspace = nb_kspace
        self.accuracy = accuracy

        # Expression for shifting energy
        self.buck_offset_energy = A * np.exp(-B*cutoff_r) - C/cutoff_r**6

    def get_nb_kspace(self):
        """
        Return triplet of integers defining maximal number of reciprocal points
        """
        return self.nb_kspace

    def get_alpha(self):
        return self.alp

    def get_cutoff_real(self):
        """
        Cutoff radius for the short range interaction
        """
        return self.cutoff_r

    def get_cutoff_kspace(self):
        """
        Cutoff wave vector in reciprocal space
        """
        return self.cutoff_k

    def get_accuracy(self):
        """
        Accuracy in reciprocal space 
        """
        return self.accuracy
        
    def energy_rspace(self, r, pair_charge, a):
        """
        Potential of short range Buckingham and Coulomb
        """
        E_buck = self.A * np.exp(-self.B*r) - self.C / r**6 - self.buck_offset_energy 
        E_coul = conversion_prefactor * pair_charge * erfc(a*r) / r

        return E_buck + E_coul

    def first_derivative_rspace(self, r, pair_charge, a):
        """
        First derivative of short range Buckingham and Coulomb part
        """
        f_buck = -self.A * self.B * np.exp(-self.B*r) + 6 * self.C / r**7 
        f_coul = -conversion_prefactor * pair_charge * (erfc(a*r) / r**2 +
                   2 * a * np.exp(-(a*r)**2) / (np.sqrt(np.pi)*r))

        return f_buck + f_coul

    def second_derivative_rspace(self, r, pair_charge, a):
        """
        Second derivative of short range Buckingham and Coulomb part
        """
        k_buck = self.A * self.B**2 * np.exp(-self.B*r) - 42 * self.C / r**8
        k_coul = conversion_prefactor * pair_charge * (2 * erfc(a * r) / r**3
            + 4 * a * np.exp(-(a*r)**2) / np.sqrt(np.pi) * (1 / r**2 + a**2))

        return k_buck + k_coul

###

class Ewald(Calculator):
    implemented_properties = ['energy', 'free_energy', 'stress', 'forces', 'hessian']
    default_parameters = {}
    name = 'Ewald'

    def __init__(self, f, cutoff=None):
        Calculator.__init__(self)
        self.f = f
        self.dict = {x: obj.get_cutoff_real() for x, obj in f.items()}

    def determine_alpha(self, charge, acc, cutoff, n, cell):
        """
        Determine an estimate for alpha on the basis of the cell, cutoff and desired accuracy
        (Adopted from LAMMPS)
        """

        # The kspace rms error is computed relative to the force that two unit point
        # charges exert on each other at a distance of 1 Angström
        accuracy_relative = acc * conversion_prefactor

        qsqsum = conversion_prefactor * np.sum(charge**2)

        a = accuracy_relative * np.sqrt(n * cutoff * cell[0,0] * cell[1, 1] * cell[2, 2]) / (2 * qsqsum)
        
        if a >= 1.0:
            return (1.35 - 0.15*np.log(accuracy_relative)) / cutoff

        else:
            return np.sqrt(-np.log(a)) / cutoff

    def determine_nk(self, charge, c, cell, acc, a, natoms):
        """
        Determine the maximal number of points in reciprocal space for each direction,
        scale according to skew of cell and determine the cutoff in reciprocal space 
        """

        # The kspace rms error is computed relative to the force that two unit point
        # charges exert on each other at a distance of 1 Angström
        accuracy_relative = acc * conversion_prefactor

        nxmax = 1
        nymax = 1
        nzmax = 1
     
        # 
        qsqsum = conversion_prefactor * np.sum(charge**2)

        error = c.rms_kspace(nxmax, cell[0, 0], natoms, a, qsqsum)
        while error > (accuracy_relative):
            nxmax += 1
            error = c.rms_kspace(nxmax, cell[0, 0], natoms, a, qsqsum)

        error = c.rms_kspace(nymax, cell[1, 1], natoms, a, qsqsum)
        while error > (accuracy_relative):
            nymax += 1
            error = c.rms_kspace(nymax, cell[1, 1], natoms, a, qsqsum)

        error = c.rms_kspace(nzmax, cell[2, 2], natoms, a, qsqsum)
        while error > (accuracy_relative):
            nzmax += 1
            error = c.rms_kspace(nzmax, cell[2, 2], natoms, a, qsqsum)

        kxmax = 2*np.pi / cell[0, 0] * nxmax
        kymax = 2*np.pi / cell[1, 1] * nymax
        kzmax = 2*np.pi / cell[2, 2] * nzmax
        
        kmax = max(kxmax, kymax, kzmax)

        # Check if box is triclinic --> Scale lattice vectors for triclinic skew
        if np.count_nonzero(cell - np.diag(np.diagonal(cell))) != 9:
            vector = np.array([nxmax/cell[0, 0], nymax/cell[1, 1], nzmax/cell[2, 2]])
            scaled_nbk = np.dot(np.array(np.abs(cell)), vector)
            nxmax = max(1, np.int(scaled_nbk[0]))
            nymax = max(1, np.int(scaled_nbk[1]))
            nzmax = max(1, np.int(scaled_nbk[2]))

        return kmax, np.array([nxmax, nymax, nzmax])

    def determine_kc(self, cell, nk):
        """
        Determine maximal wave vector based in a given integer triplet
        """
        kxmax = 2*np.pi / cell[0, 0] * nk[0]
        kymax = 2*np.pi / cell[1, 1] * nk[1]
        kzmax = 2*np.pi / cell[2, 2] * nk[2]

        return max(kxmax, kymax, kzmax)

    def rms_kspace(self, km, l, n, alp, q2):
        """
        Compute the root mean square error of the force in reciprocal space
        
        Reference
        ------------------
        Henrik G. Petersen, The Journal of chemical physics 103.9 (1995)
        """

        return 2 * q2 * alp / l * np.sqrt(1/(np.pi*km*n)) * np.exp(-(np.pi*km/(alp*l))**2) 

    def rms_rspace(self, charge, cell, a, rc):
        """
        Compute the root mean square error of the force in real space
        
        Reference
        ------------------
        Henrik G. Petersen, The Journal of chemical physics 103.9 (1995)
        """

        return 2 * np.sum(charge**2) * np.exp(-(a*rc)**2) / np.sqrt(rc * len(charge) * cell[0,0] * cell[1, 1] * cell[2, 2])

    def allowed_wave_vectors(self, cell, km, a, nk):
        """
        Compute allowed wave vectors and the prefactor I 
        """
        nx = nk[0]
        ny = nk[1]
        nz = nk[2]    
   
        vector_n = np.array(list(product(range(-nx, nx+1), range(-ny, ny+1), range(-nz, nz+1))))
        
        k_lc = 2 * np.pi * np.dot(np.linalg.inv(np.array(cell)), vector_n.T).T
        
        k = np.linalg.norm(k_lc, axis=1)
        
        mask = np.logical_and(k <= km, k != 0)

        return np.exp(-(k[mask]/(2*a))**2) / k[mask]**2, k_lc[mask]

    def self_energy(self, charge, a):
        """
        Return the self energy
        """
        return - conversion_prefactor * a * np.sum(charge**2) / np.sqrt(np.pi)

    def kspace_energy(self, charge, pos, vol, I, k):
        """
        Return the energy from the reciprocal space contribution
        """

        struc = np.sum(charge * np.exp(1j*np.tensordot(k, pos, axes=((1),(1)))), axis=1)

        return conversion_prefactor * 2 * np.pi * np.sum(I.T * np.absolute(struc)**2) / vol

    def first_derivative_kspace(self, charge, natoms, vol, i, j, r, I, k):
        """
        Return the kspace part of the force 
        """
        charge_cos_ln = (I * mabincount(i, charge[j] * np.sin(np.tensordot(k, r, axes=((1),(1)))), natoms, axis=1).T).T

        f = np.sum(charge_cos_ln.reshape(len(k), natoms, 1) * k.reshape(len(k), 1, 3), axis=0)

        return -conversion_prefactor * 4 * np.pi * (charge * f.T).T / vol

    def stress_kspace(self, charge, pos, vol, a, I, k):
        """
        Return the stress contribution of the long-range Coulomb part
        """
        struc_l = np.sum(charge * np.exp(1j*np.tensordot(k, pos, axes=((1),(1)))), axis=1)

        wave_vectors_cc = (k.reshape(-1, 3, 1) * k.reshape(-1, 1, 3)) * (1/(2*a**2) + 2/np.linalg.norm(k, axis=1)**2).reshape(-1,1,1) - np.identity(3) 
        
        stress_cc = np.sum((I * np.absolute(struc_l)**2).reshape(len(I), 1, 1) * wave_vectors_cc, axis=0)
        
        stress_cc *= (conversion_prefactor * 2 * np.pi / vol)

        return np.array([stress_cc[0, 0],        # xx
                         stress_cc[1, 1],        # yy
                         stress_cc[2, 2],        # zz
                         stress_cc[1, 2],        # yz
                         stress_cc[0, 2],        # xz
                         stress_cc[0, 1]])       # xy


    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        nb_atoms = len(atoms)
        atnums = atoms.numbers
        atnums_in_system = set(atnums)

        # Check some properties of input data
        if atoms.has("charge"):
            charge_n = self.atoms.get_array("charge")
        else:
            raise AttributeError(
                "Attribute error: Unable to load atom charges from atoms object!")

        if np.sum(charge_n) > 1e-3:
            print("Net charge in system: ", np.sum(charge_n))
            raise AttributeError(
                "Attribute error: System is not charge neutral!")      

        if not any(self.atoms.get_pbc()):
            raise AttributeError(
                "Attribute error: This code only works for 3D systems with periodic boundaries!")    

        for index, pairs in enumerate(self.f.values()):
            if index == 0:
                alpha = pairs.get_alpha()
                rc = pairs.get_cutoff_real()
                kc = pairs.get_cutoff_kspace()
                nbk_c = pairs.get_nb_kspace()
                accuracy = pairs.get_accuracy()
            else:
                if (rc != pairs.get_cutoff_real()) or (kc != pairs.get_cutoff_kspace()) or (alpha != pairs.get_alpha()) or (accuracy != pairs.get_accuracy()):
                    raise AttributeError(
                        "Attribute error: Cannot use different rc, kc, number of wave vectors or accuracy!")     

        # Check if alpha is given otherwise estimate it
        if alpha == None:
            alpha = self.determine_alpha(charge_n, accuracy, rc, nb_atoms, atoms.get_cell())

        # Check if nx, ny and nz are given, otherwise estimate values
        if np.any(nbk_c) == None:
            kc, nbk_c = self.determine_nk(charge_n, atoms.get_calculator(), atoms.get_cell(), accuracy, alpha, nb_atoms)    

        # If nx, ny and nz are given but cutoff in reciprocal space not, compute 
        if np.all(nbk_c) != None and kc == None:
            kc = self.determine_kc(atoms.get_cell(), nbk_c)

        # Compute and print error estimates
        rms_real_space = self.rms_rspace(charge_n, atoms.get_cell(), alpha, rc)
        rms_kspace_x = self.rms_kspace(nbk_c[0], atoms.get_cell()[0, 0], nb_atoms, alpha, conversion_prefactor*np.sum(charge_n**2))
        rms_kspace_y = self.rms_kspace(nbk_c[1], atoms.get_cell()[1, 1], nb_atoms, alpha, conversion_prefactor*np.sum(charge_n**2))
        rms_kspace_z = self.rms_kspace(nbk_c[2], atoms.get_cell()[2, 2], nb_atoms, alpha, conversion_prefactor*np.sum(charge_n**2))

        """
        print("Estimated alpha: ", alpha)
        print("Estimated absolute RMS force accuracy (Real space): ", np.absolute(rms_real_space))
        print("Cutoff for kspace vectors: ", kc)
        print("Estimated kspace triplets nx/ny/nx: ", nbk_c[0], "/", nbk_c[1], "/", nbk_c[2]) 
        print("Estimated absolute RMS force accuracy (Kspace): ", np.sqrt(rms_kspace_x**2 + rms_kspace_y**2 + rms_kspace_z**2))
        """

        # Neighbor list for short range interaction
        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', atoms=atoms, cutoff=self.dict)
        chargeij = charge_n[i_p] * charge_n[j_p]

        if np.sum(i_p == j_p) > 1e-5:
            print("Atoms can see themselves!")

        mask = np.logical_and(r_p < rc+0.001, r_p > rc-0.001)
        if np.sum(mask) > 1e-5:
            print("Atoms sits at cutoff!")
            print("r_p: \n", r_p[mask])

        # List of all atoms and distances
        ij_n = np.array(list(product(range(0, nb_atoms), range(0, nb_atoms))))
        i_n = ij_n[:,0]
        j_n = ij_n[:,1]
        r_nc = mic(atoms.get_positions()[j_n,:] - atoms.get_positions()[i_n,:], cell=atoms.get_cell())
        r_n = np.linalg.norm(r_nc, axis=1)

        mask = np.logical_and(r_p < rc+0.001, r_p > rc-0.001)
        if np.sum(mask) > 1e-5:
            print("Atoms sits at cutoff!")
            print("r_p: \n", r_p[mask])

        # Prefactor and wave vectors for reciprocal space 
        I_l, k_lc = self.allowed_wave_vectors(atoms.get_cell(), kc, alpha, nbk_c)

        # Short-range interaction of Buckingham and Ewald
        e_p = np.zeros_like(r_p)
        de_p = np.zeros_like(r_p)
        for params, pair in enumerate(self.dict):
            if pair[0] == pair[1]:
                mask1 = atnums[i_p] == pair[0]
                mask2 = atnums[j_p] == pair[0]
                mask = np.logical_and(mask1, mask2)

                e_p[mask] = self.f[pair].energy_rspace(r_p[mask], chargeij[mask], alpha)
                de_p[mask] = self.f[pair].first_derivative_rspace(r_p[mask], chargeij[mask], alpha)

            if pair[0] != pair[1]:
                mask1 = np.logical_and(
                    atnums[i_p] == pair[0], atnums[j_p] == pair[1])
                mask2 = np.logical_and(
                    atnums[i_p] == pair[1], atnums[j_p] == pair[0])
                mask = np.logical_or(mask1, mask2)

                e_p[mask] = self.f[pair].energy_rspace(r_p[mask], chargeij[mask], alpha)
                de_p[mask] = self.f[pair].first_derivative_rspace(r_p[mask], chargeij[mask], alpha)


        # Energy 
        e_self = self.self_energy(charge_n, alpha)

        e_long = self.kspace_energy(charge_n, atoms.get_positions(), atoms.get_volume(), I_l, k_lc)

        epot = 0.5*np.sum(e_p) + e_self  + e_long 

        # Forces
        df_pc = -0.5*de_p.reshape(-1, 1)*r_pc/r_p.reshape(-1, 1) 

        f_nc = mabincount(j_p, df_pc, nb_atoms) - mabincount(i_p, df_pc, nb_atoms)

        f_nc += self.first_derivative_kspace(charge_n, nb_atoms, self.atoms.get_volume(), i_n, j_n, r_nc, I_l, k_lc)

        # Virial
        virial_v = -np.array([r_pc[:, 0] * df_pc[:, 0],               # xx
                              r_pc[:, 1] * df_pc[:, 1],               # yy
                              r_pc[:, 2] * df_pc[:, 2],               # zz
                              r_pc[:, 1] * df_pc[:, 2],               # yz
                              r_pc[:, 0] * df_pc[:, 2],               # xz
                              r_pc[:, 0] * df_pc[:, 1]]).sum(axis=1)  # xy

        virial_v += self.stress_kspace(charge_n, atoms.get_positions(), self.atoms.get_volume(), alpha, I_l, k_lc) 

        self.results = {'energy': epot,
                        'free_energy': epot,
                        'stress': virial_v/self.atoms.get_volume(),
                        'forces': f_nc}

    ###

    def hessian_rspace(self, atoms, format='dense', divide_by_masses=False):
        """
        Calculate the Hessian matrix for the short range part.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        format: "sparse" or "neighbour-list"
            Output format of the hessian matrix.

        divide_by_masses: bool
            if true return the dynamic matrix else hessian matrix 

        Restrictions
        ----------
        This method is currently only implemented for three dimensional systems
        """

        f = self.f
        nb_atoms = len(atoms)
        atnums = atoms.numbers

        # Check some properties of input data
        if atoms.has("charge"):
            charge_n = atoms.get_array("charge")
        else:
            raise AttributeError(
                "Attribute error: Unable to load atom charges from atoms object!")

        if np.sum(charge_n) > 1e-3:
            print("Net charge in system: ", np.sum(charge_n))
            raise AttributeError(
                "Attribute error: System is not charge neutral!")    

        if not any(atoms.get_pbc()):
            raise AttributeError(
                "Attribute error: This code only works for 3D systems with periodic boundaries in all directions!")  

        for index, pairs in enumerate(f.values()):
            if index == 0:
                alpha = pairs.get_alpha()
                rc = pairs.get_cutoff_real()
                accuracy = pairs.get_accuracy()
            else:
                if (rc != pairs.get_cutoff_real()) or (alpha != pairs.get_alpha()) or (accuracy != pairs.get_accuracy()):
                    raise AttributeError(
                        "Attribute error: Cannot use different rc, alpha or accuracy!") 

        # Check if alpha is given otherwise guess a value 
        if alpha == None:
            alpha = self.determine_alpha(charge_n, accuracy, rc, nb_atoms, atoms.get_cell())

        print("Real space")
        print("------------")
        print("Estimated alpha: ", alpha)

        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', atoms, self.dict)
        chargeij = charge_n[i_p] * charge_n[j_p]
        first_i = first_neighbours(nb_atoms, i_p)

        if np.sum(i_p == j_p) > 1e-5:
            print("Atoms can see themselves!")

        mask = np.logical_and(r_p < rc+0.001, r_p > rc-0.001)
        if np.sum(mask) > 1e-5:
            print("Atoms sits at cutoff!")
            print("r_p: \n", r_p[mask])

        de_p = np.zeros_like(r_p)
        dde_p = np.zeros_like(r_p)
        for params, pair in enumerate(self.dict):
            if pair[0] == pair[1]:
                mask1 = atnums[i_p] == pair[0]
                mask2 = atnums[j_p] == pair[0]
                mask = np.logical_and(mask1, mask2)

                de_p[mask] = f[pair].first_derivative_rspace(r_p[mask], chargeij[mask], alpha)
                dde_p[mask] = f[pair].second_derivative_rspace(r_p[mask], chargeij[mask], alpha)

            if pair[0] != pair[1]:
                mask1 = np.logical_and(
                    atnums[i_p] == pair[0], atnums[j_p] == pair[1])
                mask2 = np.logical_and(
                    atnums[i_p] == pair[1], atnums[j_p] == pair[0])
                mask = np.logical_or(mask1, mask2)

                de_p[mask] = f[pair].first_derivative_rspace(r_p[mask], chargeij[mask], alpha)
                dde_p[mask] = f[pair].second_derivative_rspace(r_p[mask], chargeij[mask], alpha)
        
        n_pc = (r_pc.T/r_p).T
        H_pcc = -(dde_p * (n_pc.reshape(-1, 3, 1)
                           * n_pc.reshape(-1, 1, 3)).T).T
        H_pcc += -(de_p/r_p * (np.eye(3, dtype=n_pc.dtype)
                                    - (n_pc.reshape(-1, 3, 1) * n_pc.reshape(-1, 1, 3))).T).T

        if format == "sparse":
            if divide_by_masses:
                H = bsr_matrix(((H_pcc.T/geom_mean_mass_p).T,
                                j_p, first_i), shape=(3*nb_atoms, 3*nb_atoms))

            else:
                H = bsr_matrix((H_pcc, j_p, first_i), shape=(3*nb_atoms, 3*nb_atoms))

            Hdiag_icc = np.empty((nb_atoms, 3, 3))
            for x in range(3):
                for y in range(3):
                    Hdiag_icc[:, x, y] = - \
                        np.bincount(i_p, weights=H_pcc[:, x, y])

            if divide_by_masses:
                H += bsr_matrix(((Hdiag_icc.T/mass_n).T, np.arange(nb_atoms),
                        np.arange(nb_atoms+1)), shape=(3*nb_atoms, 3*nb_atoms))         

            else:
                H += bsr_matrix((Hdiag_icc, np.arange(nb_atoms),
                         np.arange(nb_atoms+1)), shape=(3*nb_atoms, 3*nb_atoms))

            return H

        elif format == "dense":
            H = np.zeros((3*nb_atoms, 3*nb_atoms))
            for atom in range(len(i_p)):
                H[3*i_p[atom]:3*i_p[atom]+3,
                  3*j_p[atom]:3*j_p[atom]+3] += H_pcc[atom]

            Hdiag_icc = np.empty((nb_atoms, 3, 3))
            for x in range(3):
                for y in range(3):
                    Hdiag_icc[:, x, y] = - \
                        np.bincount(i_p, weights=H_pcc[:, x, y])

            Hdiag_ncc = np.zeros((3*nb_atoms, 3*nb_atoms))
            for atom in range(nb_atoms):
                Hdiag_ncc[3*atom:3*atom+3,
                          3*atom:3*atom+3] += Hdiag_icc[atom]

            H += Hdiag_ncc

            if divide_by_masses:
                masses_p = (atoms.get_masses()).repeat(3)
                H /= np.sqrt(masses_p.reshape(-1,1)*masses_p.reshape(1,-1))
                return H

            else:
                return H

        # Neighbour list format
        elif format == "neighbour-list":
            return H_pcc, i_p, j_p, r_pc, r_p

        else:
           raise AttributeError(
                "Attribute error: Can not return a Hessian matrix in the given format!")                

    ###

    def kspace_properties(self, atoms, prop="Hessian", divide_by_masses=False):
        """
        Calculate the recirprocal contributiom to the Hessian, the non-affine forces and the Born elastic constants

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        choice: "Hessian", "Born" or "NAForces"
            Compute either the Hessian/Dynamical matrix, the Born constants 
            or the non-affine forces.

        divide_by_masses: bool
            if true return the dynamic matrix else Hessian matrix 

        Restrictions
        ----------
        This method is currently only implemented for three dimensional systems
        """
        f = self.f
        nb_atoms = len(atoms)

        # Check some properties of input data
        if atoms.has("charge"):
            charge_n = atoms.get_array("charge")
        else:
            raise AttributeError(
                "Attribute error: Unable to load atom charges from atoms object!")

        if np.sum(charge_n) > 1e-3:
            print("Net charge in system: ", np.sum(charge_n))
            raise AttributeError(
                "Attribute error: System is not charge neutral!")     

        if not any(atoms.get_pbc()):
            raise AttributeError(
                "Attribute error: This code only works for 3D systems with periodic boundaries in all directions!")   

        for index, pairs in enumerate(f.values()):
            if index == 0:
                alpha = pairs.get_alpha()
                rc = pairs.get_cutoff_real()
                kc = pairs.get_cutoff_kspace()
                nbk_c = pairs.get_nb_kspace()
                accuracy = pairs.get_accuracy()
            else:
                if (rc != pairs.get_cutoff_real()) or (kc != pairs.get_cutoff_kspace()) or (alpha != pairs.get_alpha()) or (accuracy != pairs.get_accuracy()):
                    raise AttributeError(
                        "Attribute error: Cannot use different rc, kc, number of wave vectors or accuracy!")     

        # Check if alpha is given otherwise estimate it
        if alpha == None:
            alpha = self.determine_alpha(charge_n, accuracy, rc, nb_atoms, atoms.get_cell())

        # Check if nx, ny and nz are given, otherwise estimate values
        if np.any(nbk_c) == None:
            kc, nbk_c = self.determine_nk(charge_n, atoms.get_calculator(), atoms.get_cell(), accuracy, alpha, nb_atoms)    

        # If nx, ny and nz are given but cutoff in reciprocal space not, compute 
        if np.all(nbk_c) != None and kc == None:
            kc = self.determine_kc(atoms.get_cell(), nbk_c)

        # Compute and print error estimates
        rms_real_space = self.rms_rspace(charge_n, atoms.get_cell(), alpha, rc)
        rms_kspace_x = self.rms_kspace(nbk_c[0], atoms.get_cell()[0, 0], nb_atoms, alpha, conversion_prefactor*np.sum(charge_n**2))
        rms_kspace_y = self.rms_kspace(nbk_c[1], atoms.get_cell()[1, 1], nb_atoms, alpha, conversion_prefactor*np.sum(charge_n**2))
        rms_kspace_z = self.rms_kspace(nbk_c[2], atoms.get_cell()[2, 2], nb_atoms, alpha, conversion_prefactor*np.sum(charge_n**2))

        print("Kspace properties")
        print("----------------------------")
        print("Estimated alpha: ", alpha)
        print("Estimated absolute RMS force accuracy (Real space): ", np.absolute(rms_real_space))
        print("Cutoff for kspace vectors: ", kc)
        print("Estimated kspace vectors nx/ny/nx: ", nbk_c[0], "/", nbk_c[1], "/", nbk_c[2]) 
        print("Estimated absolute RMS force accuracy (Kspace): ", np.sqrt(rms_kspace_x**2 + rms_kspace_y**2 + rms_kspace_z**2))

        # List of distances for all atoms
        ij_n = np.array(list(product(range(0, nb_atoms), range(0, nb_atoms))))
        i_n = ij_n[:,0]
        j_n = ij_n[:,1]
        r_nc = mic(atoms.get_positions()[j_n,:] - atoms.get_positions()[i_n,:], cell=atoms.get_cell())
        r_n = np.linalg.norm(r_nc, axis=1)

        mask = np.logical_and(r_n < rc+0.001, r_n > rc-0.001)
        if np.sum(mask) > 1e-5:
            print("Atoms sits at cutoff!")
            print("r_p: \n", r_p[mask])

        # Prefactor and wave vectors for reciprocal space 
        I_l, k_lc = self.allowed_wave_vectors(atoms.get_cell(), kc, alpha, nbk_c)

        # 
        chargeij = charge_n[i_n] * charge_n[j_n]

        if prop == "Born":
            C = np.zeros((3, 3, 3, 3))
            delta_ab = np.identity(3)

            for index, wavevector in enumerate(k_lc):
                structure_factor = np.sum(charge_n * np.exp(1j*np.sum(wavevector*atoms.get_positions(), axis=1)))
                prefactor = I_l[index] * np.absolute(structure_factor)**2
                sqk = np.sum(wavevector*wavevector)

                # First 
                first = 2 * delta_ab.reshape(3,3,1,1) * delta_ab.reshape(1,1,3,3)

                # Second 
                prefactor_second = 1/(2*alpha**2) + 2/sqk
                second = (wavevector.reshape(1,1,3,1)*wavevector.reshape(1,1,1,3)*delta_ab.reshape(3,3,1,1) +
                          wavevector.reshape(1,3,1,1)*wavevector.reshape(3,1,1,1)*delta_ab.reshape(1,1,3,3))

                # Third
                prefactor_third = 1/(4*alpha**4) + 2/(alpha**2 * sqk) + 8/sqk**2
                third = wavevector.reshape(3,1,1,1)*wavevector.reshape(1,3,1,1)*wavevector.reshape(1,1,3,1)*wavevector.reshape(1,1,1,3)
                
                # Fourth 
                prefactor_fourth = 1/(2*alpha**2) + 2/sqk
                fourth = wavevector.reshape(1,3,1,1) * wavevector.reshape(1,1,1,3) * delta_ab.reshape(3,1,3,1)

                C += prefactor * (first - second*prefactor_second + third*prefactor_third - prefactor_fourth*fourth)
            
            C *= conversion_prefactor * 2 * np.pi / atoms.get_volume()**2 

            return C


            """
            # Derivative with respect to I
            prefactor1_I = (wavevector.reshape(3, 1, 1, 1) * wavevector.reshape(1, 3, 1, 1)) * (1/(2*alpha**2) + 2/np.sum(wavevector * wavevector)) - np.identity(3).reshape(3, 3, 1, 1) 
            prefactor2_I = I_l[index] * (1/(2*alpha**2) + 2/np.sum(wavevector*wavevector))
            first = prefactor1_I * prefactor2_I * wavevector.reshape(1,1,3,1)*wavevector.reshape(1,1,1,3)

            # Derivative with respect to V
            second = I_l[index] * (1/(2*alpha**2) + 2/np.sum(wavevector * wavevector)) * wavevector.reshape(3, 1, 1, 1) * wavevector.reshape(1, 3, 1, 1) * np.identity(3).reshape(1, 1, 3, 3)

            # Derivative with respect to K_l
            third = I_l[index] * 4 * wavevector.reshape(3,1,1,1)*wavevector.reshape(1,3,1,1)*wavevector.reshape(1,1,3,1)*wavevector.reshape(1,1,1,3) / np.sum(wavevector*wavevector)**2

            # Derivative with respect to V^2
            fourth = I_l[index] * 2 * delta_ab.reshape(3,3,1,1) * delta_ab.reshape(1,1,3,3)

            # Sum 
            C = C + np.absolute(structure_factor)**2 * (first + second + third + fourth)
            """

            """
            structure_factor_l = np.sum(charge_n * np.exp(1j*np.tensordot(k_lc, pos_nc, axes=((1),(1)))), axis=1)
            prefactor_l = I_l * np.absolute(structure_factor_l)**2

            # First expression
            first_abab = 2 * np.identity(3).reshape(-1, 3, 3, 1, 1) * np.identity(3).reshape(-1, 1, 1, 3, 3)

            # Second expression 
            second_abab = np.identity(3).reshape(1, 3, 3, 1, 1) * k_lc.reshape(-1, 1, 1, 3, 1) * k_lc.reshape(-1, 1, 1, 1, 3) + \
                          np.identity(3).reshape(1, 1, 1, 3, 3) * k_lc.reshape(-1, 3, 1, 1, 1) * k_lc.reshape(-1, 1, 3, 1, 1)
            second_abab *= -(1/(2*alpha**2) + 2/np.linalg.norm(k_lc, axis=1)**2).reshape(-1, 1, 1, 1, 1)

            # Third expression
            third_abab = k_lc.reshape(-1, 3, 1, 1, 1) * k_lc.reshape(-1, 1, 3, 1, 1) * k_lc.reshape(-1, 1, 1, 3, 1) * k_lc.reshape(-1 ,1, 1, 1, 3)
            third_abab *= (1/(4*alpha**4) + 2/(alpha * np.linalg.norm(k_lc, axis=1))**2 + 8/np.linalg.norm(k_lc, axis=1)**4).reshape(-1, 1, 1, 1, 1)

            C_abab = np.sum(prefactor_l.reshape(-1, 1, 1, 1, 1) * (first_abab + second_abab + third_abab), axis=0)

            return conversion_prefactor * 2 * np.pi * C_abab / atoms.get_volume()**2

            """

        """
        if prop == "Hessian":
            mask = i_n != j_n

            prefactor_ln = np.zeros((len(I_l), len(i_n)))
            prefactor_ln[:,mask] += np.cos(np.tensordot(k_lc, r_nc, axes=((1), (1))))[:, mask]
            prefactor_ln = (I_l * prefactor_ln.T).T
            H_ncc = np.sum((k_lc.reshape(-1, 1, 3, 1) * k_lc.reshape(-1, 1, 1, 3)) * prefactor_ln.reshape(-1, len(i_n), 1, 1), axis=0)
            H_ncc *= (conversion_prefactor * 4 * np.pi * chargeij / atoms.get_volume()).reshape(-1,1,1)

            # Hessian matrix: Off-diagonal elements
            H = np.zeros((3*nb_atoms, 3*nb_atoms))
            for atom in range(len(i_n)):
                H[3*i_n[atom]:3*i_n[atom]+3,
                  3*j_n[atom]:3*j_n[atom]+3] += H_ncc[atom]

            # Hessian matrix: Diagonal elements
            Hdiag_icc = np.empty((nb_atoms, 3, 3))
            for x in range(3):
                for y in range(3):
                    Hdiag_icc[:, x, y] = - \
                        np.bincount(i_n, weights=H_ncc[:, x, y])

            Hdiag_ncc = np.zeros((3*nb_atoms, 3*nb_atoms))
            for atom in range(nb_atoms):
                Hdiag_ncc[3*atom:3*atom+3,
                          3*atom:3*atom+3] += Hdiag_icc[atom]

            H += Hdiag_ncc

            if divide_by_masses:
                masses_p = (atoms.get_masses()).repeat(3)
                H /= np.sqrt(masses_p.reshape(-1,1)*masses_p.reshape(1,-1))
            
            return H
        """
        """
        elif prop == "NAForces":

            structure_ln = (I_l * mabincount(i_n, charge_n[j_n] * np.sin(np.tensordot(k_lc, r_nc, axes=((1),(1)))), nb_atoms, 1).T).T

            prefactor_l = (1/(2*alpha**2) + 2/np.linalg.norm(k_lc, axis=1)**2).reshape(-1, 1, 1)

            prefactor_lcc = (k_lc.reshape(-1, 3, 1) * k_lc.reshape(-1, 1, 3)) * prefactor_l - np.identity(3)

            prefactor_lccc = k_lc.reshape(-1, 3, 1, 1) * prefactor_lcc.reshape(-1, 1, 3, 3)

            naForces_nccc = np.sum(structure_ln.reshape(-1, nb_atoms, 1, 1, 1) * prefactor_lccc.reshape(-1, 1, 3, 3, 3), axis=0 )

            return -conversion_prefactor * 4 * np.pi * (charge_n * naForces_nccc.T).T / atoms.get_volume()
        """

    ###

    def get_hessian(self, atoms):
        """
        Compute the real space + kspace Hessian
        """
        H = self.hessian_rspace(atoms, format='dense')
        H += self.kspace_properties(atoms, prop="Hessian")

        return H

    ###

    def get_nonaffine_forces(self, atoms):
        """
        Compute the non-affine forces which result from an affine deformation of atoms.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """

        nat = len(atoms)

        # Real space 
        H_pcc, i_p, j_p, dr_pc, abs_dr_p = self.hessian_rspace(atoms, 'neighbour-list')
        naF_pcab = -0.5 * H_pcc.reshape(-1, 3, 3, 1) * dr_pc.reshape(-1, 1, 1, 3)
        Snaforces_icab = mabincount(i_p, naF_pcab, nat) - mabincount(j_p, naF_pcab, nat)

        # Reciprocal space
        Lnaforces_icab = self.kspace_properties(atoms, prop="NAForces")

        return Snaforces_icab, Lnaforces_icab 

    ###

    def get_born_elastic_constants(self, atoms):
        """
        Compute the Born elastic constants. 

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """
        # Contribution from real space 
        H_pcc, i_p, j_p, dr_pc, abs_dr_p = self.hessian_rspace(atoms, 'neighbour-list')

        # Second derivative
        C_pabab = H_pcc.reshape(-1, 3, 1, 3, 1) * dr_pc.reshape(-1, 1, 3, 1, 1) * dr_pc.reshape(-1, 1, 1, 1, 3)
        SC_abab = -C_pabab.sum(axis=0) / (2*atoms.get_volume())
        SC_abab = (SC_abab + SC_abab.swapaxes(0, 1) + SC_abab.swapaxes(2, 3) + SC_abab.swapaxes(0, 1).swapaxes(2, 3)) / 4

        # Contribution from reciprocal space 
        BC_abab = self.kspace_properties(atoms, prop="Born")

        # Symmetrize elastic constant tensor
        BC_abab = (BC_abab + BC_abab.swapaxes(0, 1) + BC_abab.swapaxes(2, 3) + BC_abab.swapaxes(0, 1).swapaxes(2, 3)) / 4

        return SC_abab, BC_abab

    ###

    def get_stress_contribution_to_elastic_constants(self, atoms):
        """
        Compute the correction to the elastic constants due to non-zero stress in the configuration.
        Stress term  results from working with the Cauchy stress.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """
        
        stress_ab = Voigt_6_to_full_3x3_stress(atoms.get_stress())
        delta_ab = np.identity(3)

        # Term 1
        C1_abab = -stress_ab.reshape(3, 3, 1, 1) * delta_ab.reshape(1, 1, 3, 3)
        C1_abab = (C1_abab + C1_abab.swapaxes(0, 1) + C1_abab.swapaxes(2, 3) + C1_abab.swapaxes(0, 1).swapaxes(2, 3)) / 4

        # Term 2
        C2_abab = (stress_ab.reshape(3, 1, 1, 3) * delta_ab.reshape(1, 3, 3, 1) + \
                   stress_ab.reshape(3, 1, 3, 1) * delta_ab.reshape(1, 3, 1, 3) + \
                   stress_ab.reshape(1, 3, 1, 3) * delta_ab.reshape(3, 1, 3, 1) + \
                   stress_ab.reshape(1, 3, 3, 1) * delta_ab.reshape(3, 1, 1, 3))/4

        return C1_abab + C2_abab
    
    ###

    def get_birch_coefficients(self, atoms):
        """
        Compute the Birch coefficients (Effective elastic constants at non-zero stress). 
        
        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """
        if self.atoms is None:
            self.atoms = atoms

        # Born (affine) elastic constants
        calculator = atoms.get_calculator()
        bornC_abab = calculator.get_born_elastic_constants(atoms)

        # Stress contribution to elastic constants
        stressC_abab = calculator.get_stress_contribution_to_elastic_constants(atoms)

        return bornC_abab + stressC_abab

    ###

    def get_non_affine_contribution_to_elastic_constants(self, atoms, eigenvalues=None, eigenvectors=None, tol=1e-5):
        """
        Compute the correction of non-affine displacements to the elasticity tensor.
        The computation of the occuring inverse of the Hessian matrix is bypassed by using a cg solver.

        If eigenvalues and and eigenvectors are given the inverse of the Hessian can be easily computed.


        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        eigenvalues: array
            Eigenvalues in ascending order obtained by diagonalization of Hessian matrix.
            If given 

        eigenvectors: array
            Eigenvectors corresponding to eigenvalues.

        tol: float
            Tolerance for the conjugate-gradient solver. 

        """

        nat = len(atoms)

        calc = atoms.get_calculator()

        C_abab = np.zeros((3,3,3,3))        

        if (eigenvalues is not None) and (eigenvectors is not None):
            naforces_icab = calc.get_nonaffine_forces(atoms)

            G_incc = (eigenvectors.T).reshape(-1, 3*nat, 1, 1) * naforces_icab.reshape(1, 3*nat, 3, 3)
            G_incc = (G_incc.T/np.sqrt(eigenvalues)).T
            G_icc  = np.sum(G_incc, axis=1)
            C_abab = np.sum(G_icc.reshape(-1,3,3,1,1) * G_icc.reshape(-1,1,1,3,3), axis=0)

        else:
            H_nn = calc.get_hessian(atoms)
            naforces_icab = calc.get_nonaffine_forces(atoms)

            D_iab = np.zeros((3*nat, 3, 3))
            for i in range(3):
                for j in range(3):
                    x, info = cg(H_nn, naforces_icab[:, :, i, j].flatten(), atol=tol)
                    if info != 0:
                        raise RuntimeError(" info > 0: CG tolerance not achieved, info < 0: Exceeded number of iterations.")
                    D_iab[:,i,j] = x

            C_abab = np.sum(naforces_icab.reshape(3*nat, 3, 3, 1, 1) * D_iab.reshape(3*nat, 1, 1, 3, 3), axis=0)
        
        # Symmetrize 
        C_abab = (C_abab + C_abab.swapaxes(0, 1) + C_abab.swapaxes(2, 3) + C_abab.swapaxes(0, 1).swapaxes(2, 3)) / 4             

        return -C_abab/atoms.get_volume()

    ###

    def get_derivative_volume(self, atoms, d=1e-6):
        """
        Calculate the change of volume with strain using central differences
        """
        nat = len(atoms)
        cell = atoms.cell.copy()
        vol = np.zeros((3,3))

        for i in range(3):
            # Diagonal 
            x = np.eye(3)
            x[i, i] += d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            Vplus = atoms.get_volume()

            x[i, i] -= 2 * d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            Vminus = atoms.get_volume()

            derivative_volume = (Vplus - Vminus) / (2 * d)

            vol[i, i] = derivative_volume

            # Off diagonal
            j = i - 2
            x[i, j] = d
            x[j, i] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            Vplus = atoms.get_volume()

            x[i, j] = -d
            x[j, i] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            Vminus = atoms.get_volume()

            derivative_volume = (Vplus - Vminus) / (4 * d)
            vol[i, j] = derivative_volume
            vol[j, i] = derivative_volume


        return vol

    ###

    def get_derivative_wave_vector(self, atoms, d=1e-6):
        """
        Calculate the change of volume with strain using central differences
        """
        nat = len(atoms)
        cell = atoms.cell.copy()
        
        # Compute smallest wave vector in x 
        print("initial_cell: ", cell)
        initial_k = 2 * np.pi * np.dot(np.linalg.inv(cell), np.array([1,1,1]))
        print("Wave vector for n=(1,1,1): ", initial_k)

        def_k = np.zeros((3,3,3))

        for i in range(3):
            # Diagonal 
            x = np.eye(3)
            x[i, i] += d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            k_pos = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), np.array([1,1,1]))

            x[i, i] -= 2 * d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            k_minus = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), np.array([1,1,1]))
            derivative_k = (k_pos - k_minus) / (2 * d)
            def_k[:, i, i] = derivative_k

            # Off diagonal --> xy, xz, yz
            j = i - 2
            x[i, j] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            scaled_nbk = np.dot(np.array(np.abs(cell)), np.array([1,1,1]))
            k_pos = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), np.array([1,1,1]))

            x[i, j] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            k_minus = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), np.array([1,1,1]))

            derivative_k = (k_pos - k_minus) / (2 * d)
            def_k[:, i, j] = derivative_k

            # Odd diagonal --> yx, zx, zy
            x[j, i] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            k_pos = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), np.array([1,1,1]))

            x[j, i] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            k_minus = 2 * np.pi * np.dot(np.linalg.inv(atoms.get_cell()), np.array([1,1,1]))

            derivative_k = (k_pos - k_minus) / (2 * d)
            def_k[:, j, i] = derivative_k

        return def_k


    ###

    def get_numerical_non_affine_forces(self, atoms, d=1e-6):
        """

        Calculate numerical non-affine forces using central finite differences.
        This is done by deforming the box, rescaling atoms and measure the force.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """

        nat = len(atoms)
        cell = atoms.cell.copy()
        fna_ncc = np.zeros((nat, 3, 3, 3))

        for i in range(3):
            # Diagonal 
            x = np.eye(3)
            x[i, i] += d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            fplus = atoms.get_forces()

            x[i, i] -= 2 * d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            fminus = atoms.get_forces()

            naForces_ncc = (fplus - fminus) / (2 * d)
            fna_ncc[:, 0, i, i] = naForces_ncc[:, 0]
            fna_ncc[:, 1, i, i] = naForces_ncc[:, 1]
            fna_ncc[:, 2, i, i] = naForces_ncc[:, 2]

            # Off diagonal
            j = i - 2
            x[i, j] = d
            x[j, i] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            fplus = atoms.get_forces()

            x[i, j] = -d
            x[j, i] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            fminus = atoms.get_forces()

            naForces_ncc = (fplus - fminus) / (4 * d)
            fna_ncc[:, 0, i, j] = naForces_ncc[:, 0]
            fna_ncc[:, 0, j, i] = naForces_ncc[:, 0]
            fna_ncc[:, 1, i, j] = naForces_ncc[:, 1]
            fna_ncc[:, 1, j, i] = naForces_ncc[:, 1]
            fna_ncc[:, 2, i, j] = naForces_ncc[:, 2]
            fna_ncc[:, 2, j, i] = naForces_ncc[:, 2]

            # Off diagonal --> xy, xz
            """
            j = i - 2
            x[i, j] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            fplus = atoms.get_forces()

            x[i, j] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            fminus = atoms.get_forces()

            naForces_ncc = (fplus - fminus) / (2 * d)
            fna_ncc[:, 0, i, j] = naForces_ncc[:, 0]
            fna_ncc[:, 1, i, j] = naForces_ncc[:, 1]
            fna_ncc[:, 2, i, j] = naForces_ncc[:, 2]

            # Odd diagonal --> yx, 
            x[j, i] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            fplus = atoms.get_forces()

            x[j, i] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            fminus = atoms.get_forces()

            naForces_ncc = (fplus - fminus) / (2 * d)
            fna_ncc[:, 0, j, i] = naForces_ncc[:, 0]
            fna_ncc[:, 1, j, i] = naForces_ncc[:, 1]
            fna_ncc[:, 2, j, i] = naForces_ncc[:, 2]
            """

        return fna_ncc
