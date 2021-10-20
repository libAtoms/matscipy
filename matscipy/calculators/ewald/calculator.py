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
#

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

###

class BKS_ewald():
    """
    Functional Form of the Beest, Kramer, van Santen (BKS) potential.
    Functional form: Buckingham + Coulomb potential

    Buckingham:  
        Energy is shifted to zero at the cutoff.
    Coulomb:   
        Electrostatic interaction is treated using the traditional Ewald summation.
                     
    References
    ----------
        B. W. Van Beest, G. J. Kramer and R. A. Van Santen, Phys. Rev. Lett. 64.16 (1990)
    """

    def __init__(self, A, B, C, alpha, cutoff_r, max_k, nk, accuracy):
        self.A = A
        self.B = B 
        self.C = C
        self.alpha = alpha
        self.cutoff_r = cutoff_r
        self.max_k = max_k
        self.nk = nk
        self.accuracy = accuracy

        # Conversion factor to be consistent with LAMMPS metal units
        conversion_factor = 14.399645
        self.conversion_factor = conversion_factor 

        # Expression for shifting energy/force
        self.buck_offset_energy = A * np.exp(-B*cutoff_r) - C/cutoff_r**6

    def get_nk(self):
        return self.nk

    def get_alpha(self):
        return self.alpha

    def get_cutoff_real(self):
        return self.cutoff_r

    def get_max_k(self):
        return self.max_k

    def get_accuracy(self):
        return self.accuracy
        
    def energy_short(self, r, pair_charge, a):
        """
        Return the energy from Buckingham part and short range Coulomb part.
        """
        E_buck = self.A * np.exp(-self.B*r) - self.C / r**6 - self.buck_offset_energy
        E_coul = conversion_prefactor * pair_charge * erfc(a*r) / r

        return E_buck + E_coul

    def first_derivative_short(self, r, pair_charge, a):
        """
        Return the force from Buckingham part and short range Coulomb part.
        """
        f_buck = -self.A * self.B * np.exp(-self.B*r) + 6 * self.C / r**7 
        f_coul = -conversion_prefactor* pair_charge * (erfc(a*r) / r**2 +
                   2 * a * np.exp(-(a*r)**2) / (np.sqrt(np.pi)*r))

        return f_buck + f_coul

    def second_derivative_short(self, r, pair_charge, a):
        """
        Return the stiffness from Buckingham part and short range Coulomb part.
        """
        k_buck = self.A * self.B**2 * np.exp(-self.B*r) - 42 * self.C / r**8
        k_coul = conversion_prefactor * pair_charge * (2 * erfc(a * r) / r**3
            + 4 * a * np.exp(-(a*r)**2) / np.sqrt(np.pi) * (1 / r**2 + a**2))

        return k_buck + k_coul

    def derivative(self, n=1):
        if n == 1:
            return self.first_derivative
        elif n == 2:
            return self.second_derivative
        else:
            raise ValueError(
                "Don't know how to compute {}-th derivative.".format(n))

###

class Ewald(Calculator):
    implemented_properties = ['energy', 'free_energy', 'stress', 'forces', 'hessian']
    default_parameters = {}
    name = 'PairPotential'

    def __init__(self, f, cutoff=None):
        Calculator.__init__(self)
        self.f = f
        self.dict = {x: obj.get_cutoff_real() for x, obj in f.items()}

    def determine_nk(self, charge, c, cell, acc, a, natoms):
        """
        Determine the maximal number of wave vectors in each direction
        and the absolute value of the maximal wave vector.
        """
        # Accuracy for rms force
        # Force is relative to two point charges with q=1 and distance 1A
        # All in LAMMPS metal units
        #accuracy = acc * 14.399645

        nxmax = 1
        nymax = 1
        nzmax = 1
     
        # 
        qsqsum = 14.399645 * np.sum(charge**2)

        error = c.rms(nxmax, cell[0, 0], natoms, a, qsqsum)
        while error > (acc * 14.399645):
            nxmax += 1
            error = c.rms(nxmax, cell[0, 0], natoms, a, qsqsum)

        error = c.rms(nymax, cell[1, 1], natoms, a, qsqsum)
        while error > (acc*14.399645):
            nymax += 1
            error = c.rms(nymax, cell[1, 1], natoms, a, qsqsum)

        error = c.rms(nzmax, cell[2, 2], natoms, a, qsqsum)
        while error > (acc*14.399645):
            nzmax += 1
            error = c.rms(nzmax, cell[2, 2], natoms, a, qsqsum)

        kxmax = 2*np.pi / cell[0, 0] * nxmax
        kymax = 2*np.pi / cell[1, 1] * nymax
        kzmax = 2*np.pi / cell[2, 2] * nzmax
        
        gs = max(kxmax, kymax, kzmax)

        # Check if box is triclinic --> If yes, scale maximal n 
        if np.count_nonzero(cell - np.diag(np.diagonal(cell))) != 9:
            vector = np.array([nxmax/cell[0, 0], nymax/cell[1, 1], nzmax/cell[2, 2]])
            vec = np.dot(np.array(np.abs(cell)), vector)
            nxmax = max(1, np.int(vec[0]))
            nymax = max(1, np.int(vec[1]))
            nzmax = max(1, np.int(vec[2]))

        return gs, np.array([nxmax, nymax, nzmax])

    def determine_kmax(self, cell, nk):
        """
        Determine maximal wave vector from given nk
        """
        kxmax = 2*np.pi / cell[0, 0] * nk[0]
        kymax = 2*np.pi / cell[1, 1] * nk[1]
        kzmax = 2*np.pi / cell[2, 2] * nk[2]

        return max(kxmax, kymax, kzmax)


    def rms(self, km, l, n, alp, q2):
        """
        Compute the root mean square error of the force for a given kmax
        
        Reference
        ------------------
        Henrik G. Petersen, The Journal of chemical physics 103.9 (1995)
        """

        return 2 * q2 * alp / l * np.sqrt(1/(np.pi*km*n)) * np.exp(-(np.pi*km/(alp*l))**2) 

    def rms_real_space(self, charge, cell, a, rc):
        """
        Compute the root mean square error of the real space Ewald
        
        Reference
        ------------------
        Henrik G. Petersen, The Journal of chemical physics 103.9 (1995)
        """

        return 2 * np.sum(charge**2) * np.exp(-(a*rc)**2) / np.sqrt(rc * len(charge) * cell[0,0] * cell[1, 1] * cell[2, 2])

    def determine_alpha(self, charge, acc, cutoff, n, cell):
        """
        Determine a guess for alpha
        (Same code as in LAMMPS)
        """
        a = acc * np.sqrt(n * cutoff * cell[0,0] * cell[1, 1] * cell[2, 2]) / (2 * np.sum(charge**2))
        
        if a >= 1.0:
            return (1.35 - 0.15*np.log(acc)) / cutoff

        else:
            return np.sqrt(-np.log(a)) / cutoff

    def wave_vectors_rec_triclinic(self, cell, km, a, nk):
        """
        Compute the wave vectors and one often used prefactor for a non-orthogonal box
        """
        nx = nk[0]
        ny = nk[1]
        nz = nk[2]    
   
        vector_n = np.array(list(product(range(-nx, nx+1), range(-ny, ny+1), range(-nz, nz+1))))
        k_lc = 2 * np.pi * np.dot(np.linalg.inv(np.array(cell)), vector_n.T).T
        k = np.linalg.norm(k_lc, axis=1)
        mask = np.logical_and(k <= km, k != 0)

        return np.exp(-(k[mask]/(2*a))**2) / k[mask]**2, k_lc[mask]

    def energy_self(self, charge, a):
        """
        Return the self energy term of the Ewald summation
        """
        return - conversion_prefactor * a * np.sum(charge**2) / np.sqrt(np.pi)

    def energy_long(self, charge, pos, vol, ik, k):
        """
        Return the energy from the reciprocal space contribution
        """

        structure_factor = np.sum(charge * np.exp(1j*np.tensordot(k, pos, axes=((1),(1)))), axis=1)

        return conversion_prefactor * 2 * np.pi * np.sum(ik.T * np.absolute(structure_factor)**2) / vol


    def first_derivative_long(self, charge, natoms, vol, i, j, r, ik, k):
        """
        Return the kspace part of the force 
        """
        im_structure = (ik * mabincount(i, charge[j] * np.sin(np.tensordot(k, r, axes=((1),(1)))), natoms, axis=1).T).T

        f = np.sum(im_structure.reshape(len(k), natoms, 1) * k.reshape(len(k), 1, 3), axis=0)

        return conversion_prefactor * 4 * np.pi * (charge * f.T).T / vol

    def stress_long(self, charge, pos, vol, a, ik, k):
        """
        Return the stress contribution of the long-range Coulomb part
        """
        structure_factor_m = np.sum(charge * np.exp(1j*np.tensordot(k, pos, axes=((1),(1)))), axis=1)

        wave_vetors = (k.reshape(-1, 3, 1) * k.reshape(-1, 1, 3)) * (1/(2*a**2) + 2/np.linalg.norm(k, axis=1)**2).reshape(-1,1,1) - np.identity(3) 
        
        stress_cc = np.sum((ik * np.absolute(structure_factor_m)**2).reshape(len(ik), 1, 1) * wave_vetors, axis=0)
        
        stress_cc *= conversion_prefactor * 2 * np.pi / vol

        return np.array([stress_cc[0, 0],        # xx
                         stress_cc[1, 1],        # yy
                         stress_cc[2, 2],        # zz
                         stress_cc[1, 2],        # yz
                         stress_cc[0, 2],        # xz
                         stress_cc[0, 1]])       # xy


    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        f = self.f
        nb_atoms = len(self.atoms)
        atnums = self.atoms.numbers
        atnums_in_system = set(atnums)
        calc = atoms.get_calculator()
        pos_nc = self.atoms.get_positions()
        cell = self.atoms.get_cell()

        # Check some properties of input data
        if atoms.has("charge"):
            charge_p = self.atoms.get_array("charge")
        else:
            raise AttributeError(
                "Attribute error: Unable to load atom charges from atoms object!")

        if np.sum(charge_p) > 1e-3:
            print("Charge: ", np.sum(charge_p))
            raise AttributeError(
                "Attribute error: We require charge balance!")      

        if not any(self.atoms.get_pbc()):
            raise AttributeError(
                "Attribute error: This code only works for 3D systems with periodic boundaries in all directions!")    

        for index, pairs in enumerate(f.values()):
            if index == 0:
                alpha = pairs.get_alpha()
                rc = pairs.get_cutoff_real()
                kmax = pairs.get_max_k()
                nk = pairs.get_nk()
                accuracy = pairs.get_accuracy()
            else:
                if (rc != pairs.get_cutoff_real()) or (kmax != pairs.get_max_k()) or (alpha != pairs.get_alpha()) or (np.array_equal(nk, pairs.get_nk) or (accuracy != pairs.get_accuracy())):
                    raise AttributeError(
                        "Attribute error: Cannot use different rc, Kmax, number of wave vectors or accuracy!")     

        # Check if alpha is given otherwise guess a value 
        if alpha == None:
            alpha = calc.determine_alpha(charge_p, accuracy, rc, nb_atoms, cell)

        # Check if nx, ny and nz are given, otherwise compute valid values
        if np.any(nk) == None:
            kmax, nk = calc.determine_nk(charge_p, calc, cell, accuracy, alpha, nb_atoms)    

        # Check if nx,ny and nz are given but Kmax not
        if np.all(nk) != None and kmax == None:
            kmax = calc.determine_kmax(cell, nk)

        # Compute and print error estimates
        rms_real_space = calc.rms_real_space(charge_p, cell, alpha, rc)
        rms_kspace_x = calc.rms(nk[0], cell[0, 0], nb_atoms, alpha, np.sum(charge_p**2))
        rms_kspace_y = calc.rms(nk[1], cell[1, 1], nb_atoms, alpha, np.sum(charge_p**2))
        rms_kspace_z = calc.rms(nk[2], cell[2, 2], nb_atoms, alpha, np.sum(charge_p**2))

        
        print("Estimated alpha: ", alpha)
        print("Estimated absolute RMS force accuracy (Real space): ", np.absolute(rms_real_space))
        print("Cutoff for kspace vectors: ", kmax)
        print("Estimated kspace vectors nx/ny/nx: ", nk[0], "/", nk[1], "/", nk[2]) 
        print("Estimated absolute RMS force accuracy (Kspace): ", np.sqrt(rms_kspace_x**2 + rms_kspace_y**2 + rms_kspace_z**2))

        # Neighbor list for short range interaction
        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', self.atoms, self.dict)
        chargeij = charge_p[i_p] * charge_p[j_p]

        if np.sum(i_p == j_p) > 1e-5:
            print("Atoms can see themselves!")

        # List of all atoms
        ij_n = np.array(list(product(range(0, nb_atoms), range(0, nb_atoms))))
        i_n = ij_n[:,0]
        j_n = ij_n[:,1]
        r_nc = mic(pos_nc[i_n,:] - pos_nc[j_n,:], cell=cell)

        # Prefactor and wave vectors for reciprocal space 
        Ik, k_lc = calc.wave_vectors_rec_triclinic(atoms.get_cell(), kmax, alpha, nk)

        # Short-range interaction of Buckingham and Ewald
        e_p = np.zeros_like(r_p)
        de_p = np.zeros_like(r_p)
        for params, pair in enumerate(self.dict):
            if pair[0] == pair[1]:
                mask1 = atnums[i_p] == pair[0]
                mask2 = atnums[j_p] == pair[0]
                mask = np.logical_and(mask1, mask2)

                e_p[mask] = f[pair].energy_short(r_p[mask], chargeij[mask], alpha)
                de_p[mask] = f[pair].first_derivative_short(r_p[mask], chargeij[mask], alpha)

            if pair[0] != pair[1]:
                mask1 = np.logical_and(
                    atnums[i_p] == pair[0], atnums[j_p] == pair[1])
                mask2 = np.logical_and(
                    atnums[i_p] == pair[1], atnums[j_p] == pair[0])
                mask = np.logical_or(mask1, mask2)

                e_p[mask] = f[pair].energy_short(r_p[mask], chargeij[mask], alpha)
                de_p[mask] = f[pair].first_derivative_short(r_p[mask], chargeij[mask], alpha)

        # Energy 
        e_self = calc.energy_self(charge_p, alpha)

        e_long = calc.energy_long(charge_p, atoms.get_positions(), atoms.get_volume(), Ik, k_lc)

        epot = 0.5*np.sum(e_p) + e_self + e_long

        # Forces
        df_pc = -0.5*de_p.reshape(-1, 1)*r_pc/r_p.reshape(-1, 1) 

        f_nc = mabincount(j_p, df_pc, nb_atoms) - mabincount(i_p, df_pc, nb_atoms)

        f_nc += calc.first_derivative_long(charge_p, nb_atoms, atoms.get_volume(), i_n, j_n, r_nc, Ik, k_lc)

        # Virial
        virial_v = -np.array([r_pc[:, 0] * df_pc[:, 0],               # xx
                              r_pc[:, 1] * df_pc[:, 1],               # yy
                              r_pc[:, 2] * df_pc[:, 2],               # zz
                              r_pc[:, 1] * df_pc[:, 2],               # yz
                              r_pc[:, 0] * df_pc[:, 2],               # xz
                              r_pc[:, 0] * df_pc[:, 1]]).sum(axis=1)  # xy

        virial_v += calc.stress_long(charge_p, atoms.get_positions(), atoms.get_volume(), alpha, Ik, k_lc) 

        self.results = {'energy': epot,
                        'free_energy': epot,
                        'stress': virial_v/self.atoms.get_volume(),
                        'forces': f_nc}

    ###

    def get_hessian_short(self, atoms, format='dense', divide_by_masses=False):
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
        if self.atoms is None:
            self.atoms = atoms

        f = self.f
        nb_atoms = len(atoms)
        atnums = atoms.numbers
        calc = atoms.get_calculator()
        cell = atoms.get_cell()

        # Check some properties of input data
        if atoms.has("charge"):
            charge_p = self.atoms.get_array("charge")
        else:
            raise AttributeError(
                "Attribute error: Unable to load atom charges from atoms object!")

        if np.sum(charge_p) >= 1e-3:
            raise AttributeError(
                "Attribute error: We require charge balance!")      

        if not any(self.atoms.get_pbc()):
            raise AttributeError(
                "Attribute error: This code only works for 3D systems with periodic boundaries in all directions!")  

        for index, pairs in enumerate(f.values()):
            if index == 0:
                alpha = pairs.get_alpha()
                rc = pairs.get_cutoff_real()
                kmax = pairs.get_max_k()
                nk = pairs.get_nk()
                accuracy = pairs.get_accuracy()
            else:
                if (rc != pairs.get_cutoff_real()) or (kmax != pairs.get_max_k()) or (alpha != pairs.get_alpha()) or (np.array_equal(nk, pairs.get_nk) or (accuracy != pairs.get_accuracy())):
                    raise AttributeError(
                        "Attribute error: Cannot use different rc, Kmax, number of wave vectors or accuracy!") 

        # Check if alpha is given otherwise guess a value 
        if alpha == None:
            alpha = calc.determine_alpha(charge_p, accuracy, rc, nb_atoms, cell)

        i_p, j_p,  r_p, r_pc = neighbour_list('ijdD', self.atoms, self.dict)
        chargeij = charge_p[i_p] * charge_p[j_p]
        first_i = first_neighbours(nb_atoms, i_p)

        de_p = np.zeros_like(r_p)
        dde_p = np.zeros_like(r_p)
        for params, pair in enumerate(self.dict):
            if pair[0] == pair[1]:
                mask1 = atnums[i_p] == pair[0]
                mask2 = atnums[j_p] == pair[0]
                mask = np.logical_and(mask1, mask2)

                de_p[mask] = f[pair].first_derivative_short(r_p[mask], chargeij[mask], alpha)
                dde_p[mask] = f[pair].second_derivative_short(r_p[mask], chargeij[mask], alpha)

            if pair[0] != pair[1]:
                mask1 = np.logical_and(
                    atnums[i_p] == pair[0], atnums[j_p] == pair[1])
                mask2 = np.logical_and(
                    atnums[i_p] == pair[1], atnums[j_p] == pair[0])
                mask = np.logical_or(mask1, mask2)

                de_p[mask] = f[pair].first_derivative_short(r_p[mask], chargeij[mask], alpha)
                dde_p[mask] = f[pair].second_derivative_short(r_p[mask], chargeij[mask], alpha)
        
        n_pc = (r_pc.T/r_p).T
        H_pcc = -(dde_p * (n_pc.reshape(-1, 3, 1)
                           * n_pc.reshape(-1, 1, 3)).T).T
        H_pcc += -(de_p/r_p * (np.eye(3, dtype=n_pc.dtype)
                                    - (n_pc.reshape(-1, 3, 1) * n_pc.reshape(-1, 1, 3))).T).T

        if format == "dense":
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
                "Attribute error: Can not return a sparse matrix for potentials with long-range interactions")                

    ###

    def get_properties_long(self, atoms, prop="Hessian", format='dense', divide_by_masses=False):
        """
        Calculate the long-range contributiom to the Hessian, the non-affine forces and the born elastic constants

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        choice: "Hessian", "Born" or "NAForces"
            Compute either the Hessian/Dynamical matrix, the Born constants 
            or the non-affine forces.

        format: "sparse" or "neighbour-list"
            Output format of the hessian matrix.

        divide_by_masses: bool
            if true return the dynamic matrix else hessian matrix 

        Restrictions
        ----------
        This method is currently only implemented for three dimensional systems
        """

        if self.atoms is None:
            self.atoms = atoms

        f = self.f
        nb_atoms = len(atoms)
        atnums = atoms.numbers
        calc = atoms.get_calculator()
        cell = atoms.get_cell()
        pos_nc = atoms.get_positions()

        # Check some properties of input data
        if atoms.has("charge"):
            charge_p = self.atoms.get_array("charge")
        else:
            raise AttributeError(
                "Attribute error: Unable to load atom charges from atoms object!")

        if np.sum(charge_p) >= 1e-3:
            raise AttributeError(
                "Attribute error: We require charge balance!")      

        print(self.atoms.get_pbc())
        if not any(self.atoms.get_pbc()):
            raise AttributeError(
                "Attribute error: This code only works for 3D systems with periodic boundaries in all directions!")   

        for index, pairs in enumerate(f.values()):
            if index == 0:
                alpha = pairs.get_alpha()
                rc = pairs.get_cutoff_real()
                kmax = pairs.get_max_k()
                nk = pairs.get_nk()
                accuracy = pairs.get_accuracy()
            else:
                if (rc != pairs.get_cutoff_real()) or (kmax != pairs.get_max_k()) or (alpha != pairs.get_alpha()) or (np.array_equal(nk, pairs.get_nk) or (accuracy != pairs.get_accuracy())):
                    raise AttributeError(
                        "Attribute error: Cannot use different rc, Kmax, number of wave vectors or accuracy!")     

        # Check if alpha is given otherwise guess a value 
        if alpha == None:
            alpha = calc.determine_alpha(charge_p, accuracy, rc, nb_atoms, cell)

        # Check if nx, ny and nz are given, otherwise compute valid values
        if np.any(nk) == None:
            kmax, nk = calc.determine_nk(charge_p, calc, cell, accuracy, alpha, nb_atoms)    

        # Check if nx,ny and nz are given but Kmax not
        if np.all(nk) != None and kmax == None:
            kmax = calc.determine_kmax(cell, nk)

        # Compute and print error estimates
        rms_real_space = calc.rms_real_space(charge_p, cell, alpha, rc)
        rms_kspace_x = calc.rms(nk[0], cell[0, 0], nb_atoms, alpha, np.sum(charge_p**2))
        rms_kspace_y = calc.rms(nk[1], cell[1, 1], nb_atoms, alpha, np.sum(charge_p**2))
        rms_kspace_z = calc.rms(nk[2], cell[2, 2], nb_atoms, alpha, np.sum(charge_p**2))

        print("Estimated alpha: ", alpha)
        print("Estimated absolute RMS force accuracy (Real space): ", np.absolute(rms_real_space))
        print("Cutoff for kspace vectors: ", kmax)
        print("Estimated kspace vectors nx/ny/nx: ", nk[0], "/", nk[1], "/", nk[2]) 
        print("Estimated absolute RMS force accuracy (Kspace): ", np.sqrt(rms_kspace_x**2 + rms_kspace_y**2 + rms_kspace_z**2))

        # List of distances for all atoms
        ij_n = np.array(list(product(range(0, nb_atoms), range(0, nb_atoms))))
        i_n = ij_n[:,0]
        j_n = ij_n[:,1]
        r_nc = mic(pos_nc[i_n,:] - pos_nc[j_n,:], cell=cell)

        # Prefactor and wave vectors for reciprocal space 
        Ik, k_lc = calc.wave_vectors_rec_triclinic(atoms.get_cell(), kmax, alpha, nk)

        # 
        chargeij = charge_p[i_n] * charge_p[j_n]

        if prop == "Hessian":
            mask = i_n != j_n

            prefactor_kn = np.zeros((len(Ik), len(i_n)))
            prefactor_kn[:,mask] += np.cos(np.tensordot(k_lc, r_nc, axes=((1), (1))))[:,mask]
            prefactor_kn = (Ik * prefactor_kn.T).T
            H_ncc = np.sum((k_lc.reshape(-1, 1, 3, 1) * k_lc.reshape(-1, 1, 1, 3)) * prefactor_kn.reshape(-1, len(i_n), 1, 1), axis=0)
            H_ncc *= (conversion_prefactor * 4 * np.pi * chargeij / atoms.get_volume()).reshape(-1,1,1)

            # Build Hessian matrix 
            if format == "dense":
                H = np.zeros((3*nb_atoms, 3*nb_atoms))
                for atom in range(len(i_n)):
                    H[3*i_n[atom]:3*i_n[atom]+3,
                      3*j_n[atom]:3*j_n[atom]+3] += H_ncc[atom]

                # Diagonal elems
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

            else:
               raise AttributeError(
                    "Attribute error: Can not return a sparse matrix for long-range interactions") 

        elif prop == "Born":
            C_abmn = np.zeros((3, 3, 3, 3))

            structure_factor = np.sum(charge_p * np.exp(1j*np.tensordot(k_lc, pos_nc, axes=((1),(1)))), axis=1)
            prefactor = Ik * np.absolute(structure_factor)**2

            # First expression
            first_abab = 2 * np.identity(3).reshape(-1, 3, 3, 1, 1) * np.identity(3).reshape(-1, 1, 1, 3, 3)

            # Second expression 
            second_abab = np.identity(3).reshape(1, 3, 3, 1, 1) * k_lc.reshape(-1, 1, 1, 3, 1) * k_lc.reshape(-1, 1, 1, 1, 3) + \
                          np.identity(3).reshape(1, 1, 1, 3, 3) * k_lc.reshape(-1, 3, 1, 1, 1) * k_lc.reshape(-1, 1, 3, 1, 1)
            second_abab *= -(1/(2*alpha**2) + 2/np.linalg.norm(k_lc, axis=1)**2).reshape(-1, 1, 1, 1, 1)

            # Third expression
            third_abab = k_lc.reshape(-1, 3, 1, 1, 1) * k_lc.reshape(-1, 1, 3, 1, 1) * k_lc.reshape(-1, 1, 1, 3, 1) * k_lc.reshape(-1 ,1, 1, 1, 3)
            third_abab *= (1/(4*alpha**4) + 2/(alpha * np.linalg.norm(k_lc, axis=1))**2 + 8/np.linalg.norm(k_lc, axis=1)**4).reshape(-1, 1, 1, 1, 1)

            C_abab = np.sum(prefactor.reshape(-1, 1, 1, 1, 1) * (first_abab + second_abab + third_abab), axis=0)

            return conversion_prefactor * 2 * np.pi * C_abab / atoms.get_volume()**2

        elif prop == "NAForces":

            naForces_ncc = np.zeros((nb_atoms, 3, 3, 3))

            for index, wavevector in enumerate(k_lc):
                structure_factor = charge_p * Ik[index] * np.bincount(i_n, weights=charge_p[j_n]*np.sin(np.sum(wavevector*r_nc, axis=1)))

                prefactor = (wavevector.reshape(3, 1) * wavevector.reshape(1, 3)) * (1/(2*alpha**2) + 2/np.linalg.norm(wavevector)**2) - np.identity(3) 

                vectorlike = wavevector.reshape(1, 3, 1, 1) * prefactor.reshape(1, 1, 3, 3)

                naForces_ncc += structure_factor.reshape(nb_atoms, 1, 1, 1) * np.repeat(vectorlike, nb_atoms, axis=0)
                

            naForces_ncc *= (-conversion_prefactor * 4 * np.pi / atoms.get_volume())

            return naForces_ncc 

    ###

    def get_hessian(self, atoms):
        """
        Compute the Hessian
        """
        H = self.get_hessian_short(atoms, format='dense')
        H += self.get_properties_long(atoms, prop="Hessian", format='dense')

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
        H_pcc, i_p, j_p, dr_pc, abs_dr_p = self.get_hessian_short(atoms, 'neighbour-list')
        naF_pcab = -0.5 * H_pcc.reshape(-1, 3, 3, 1) * dr_pc.reshape(-1, 1, 1, 3)
        naforces_icab = mabincount(i_p, naF_pcab, nat) - mabincount(j_p, naF_pcab, nat)

        # Reciprocal space
        naforces_icab -= self.get_properties_long(atoms, prop="NAForces")

        return naforces_icab

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
        H_pcc, i_p, j_p, dr_pc, abs_dr_p = self.get_hessian_short(atoms, 'neighbour-list')

        # Second derivative
        C_pabab = H_pcc.reshape(-1, 3, 1, 3, 1) * dr_pc.reshape(-1, 1, 3, 1, 1) * dr_pc.reshape(-1, 1, 1, 1, 3)
        C_abab = -C_pabab.sum(axis=0) / (2*atoms.get_volume())

        # Contribution from reciprocal space 
        C_abab += self.get_properties_long(atoms, prop="Born")

        # Symmetrize elastic constant tensor
        Crec_abab = (C_abab + C_abab.swapaxes(0, 1) + C_abab.swapaxes(2, 3) + C_abab.swapaxes(0, 1).swapaxes(2, 3)) / 4

        return C_abab

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

        return fna_ncc
