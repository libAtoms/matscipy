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
Simple pair potential.
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

from scipy.sparse import bsr_matrix, vstack, hstack

from scipy.special import erfc

import ase

from itertools import product

from ...neighbours import neighbour_list, first_neighbours
from ..calculator import MatscipyCalculator
from ...numpy_tricks import mabincount


###

class BKS_ewald():
    """
    Functional Form of the Beest, Kramer, van Santen (BKS) potential.
    The potential consits of a short range Buckingham potential and a long range Coulomb potential.

    Buckingham part 
        Energy is shifted to zero at the cutoff.
    Coulomb part   
        Electrostatic interaction is treated using the traditional Ewald summation.
                     
    References:
        B. W. Van Beest, G. J. Kramer and R. A. Van Santen, Phys. Rev. Lett. 64.16 (1990)
    """

    def __init__(self, A, B, C, alpha, cutoff_r, cutoff_g, nk):
        self.A = A
        self.B = B 
        self.C = C
        self.alpha = alpha
        self.cutoff_r = cutoff_r
        self.cutoff_g = cutoff_g
        self.nk = nk

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

    def get_cutoff_reciprocal(self):
        return self.cutoff_g

    def get_energy_self(self, charge):
        return - self.conversion_factor * self.alpha * charge**2 / np.sqrt(np.pi)

    def get_energy_sr(self, r, pair_charge):
        """
        Return the energy from Buckingham part and short range Coulomb part.
        """
        E_buck = self.A * np.exp(-self.B*r) - self.C / r**6 - self.buck_offset_energy
        E_coul = self.conversion_factor * pair_charge * erfc(self.alpha*r) / r

        return E_buck + E_coul

    def get_energy_rec(self, charge, pos, vol, iu, k):
        """
        Return the energy from the reciprocal space contribution
        """
        structure_factor = np.zeros(len(iu))
        for index, vector in enumerate(k):
            structure_factor[index] = np.absolute(np.sum(charge * np.exp(1j*np.sum(vector*pos, axis=1)) ))**2

        return self.conversion_factor * 2 * np.pi * np.sum(iu * structure_factor) / vol  

    def first_derivative_sr(self, r, pair_charge):
        """
        Return the force from Buckingham part and short range Coulomb part.
        """
        f_buck = -self.A * self.B * np.exp(-self.B*r) + 6 * self.C / r**7 
        f_coul = -self.conversion_factor * pair_charge * (erfc(self.alpha*r) / r**2
         + 2 * self.alpha * np.exp(-(self.alpha*r)**2) / (np.sqrt(np.pi)*r))

        return f_buck + f_coul

    def first_derivative_reciprocal(self, charge, pos, cell, natoms, vol, iu, k):
        """
        Return the force long range fourier part
        """
        # Find a better solution for this "find all neighbors loop"
        from matscipy.neighbours import mic
        i_n = np.zeros((natoms*(natoms)), dtype=int)
        j_n = np.zeros((natoms*(natoms)), dtype=int)
        r_nc = np.zeros((natoms*(natoms),3))
        # Find all pairs of distances 
        for atomiD1 in range(natoms):
            for atomiD2 in range(natoms):
                if atomiD1 != atomiD2:
                    i_n[atomiD1*(natoms) + atomiD2] = np.int(atomiD1) 
                    j_n[atomiD1*(natoms) + atomiD2] = np.int(atomiD2) 
                    r_nc[atomiD1*(natoms) + atomiD2] = mic(pos[atomiD1,:] - pos[atomiD2,:] ,cell=cell)
        
        #
        f = np.zeros((natoms, 3))
        for index, vector in enumerate(k):
            term = iu[index] * np.bincount(i_n, weights=charge[j_n]*np.sin(np.sum(vector*r_nc, axis=1)))   
            f += (term * np.repeat(vector.reshape(-1, 3), natoms, axis=0).T).T

        return self.conversion_factor * 4 * np.pi * (charge * f.T).T / vol

    def second_derivative_sr(self, r, pair_charge):
        """
        Return the stiffness from Buckingham part and short range Coulomb part.
        """
        k_buck = self.A * self.B**2 * np.exp(-self.B*r) - 42 * self.C / r**8
        k_coul = self.conversion_factor * pair_charge * (2 * erfc(self.alpha * r) / r**3
            + 4 * self.alpha * np.exp(-(self.alpha*r)**2) / np.sqrt(np.pi) * (1 / r**2 + self.alpha**2))

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

class Ewald(MatscipyCalculator):
    implemented_properties = ['energy', 'free_energy', 'stress', 'forces', 'hessian']
    default_parameters = {}
    name = 'PairPotential'

    def __init__(self, f, cutoff=None):
        MatscipyCalculator.__init__(self)
        self.f = f

        self.dict = {x: obj.get_cutoff_real() for x, obj in f.items()}

    def wave_vectors_rec(self, gmax, cell, alph, nk):
        nx = nk[0]
        ny = nk[1]
        nz = nk[2]
        if nx == None and ny == None and nz == None: 
            nx = np.int(cell[0, 0] * gmax / (2*np.pi)) 
            ny = np.int(cell[1, 1] * gmax / (2*np.pi)) 
            nz = np.int(cell[2, 2] * gmax / (2*np.pi)) 
            print("nx/ny/nx", nx, "/", ny, "/", nz)

            G_lc = 2 * np.pi * np.array(list(product(range(-nx, nx+1), range(-ny, ny+1), range(-nz, nz+1)))) / np.array([cell[0, 0], cell[1, 1], cell[2, 2]])
            G = np.sqrt(np.sum(G_lc*G_lc, axis=1))
            mask = np.logical_and(G <= gmax, G != 0)
            G = G[mask]
            G_lc = G_lc[mask]

        else: 
            G_lc = 2 * np.pi * np.array(list(product(range(-nx, nx+1), range(-ny, ny+1), range(-nz, nz+1)))) / np.array([cell[0, 0], cell[1, 1], cell[2, 2]])
            G = np.sqrt(np.sum(G_lc*G_lc, axis=1))            
            mask = G != 0
            G = G[mask]
            G_lc = G_lc[mask]

        print("Number of wave vectors: ", G_lc.shape[0])
        return np.exp(-(G/(2*alph))**2) / G**2, G_lc

    def calculate(self, atoms, properties, system_changes):
        MatscipyCalculator.calculate(self, atoms, properties, system_changes)

        f = self.f
        nb_atoms = len(self.atoms)
        atnums = self.atoms.numbers
        atnums_in_system = set(atnums)
        calc = atoms.get_calculator()

        # Check some properties of input data
        if atoms.has("charge"):
            charge_p = self.atoms.get_array("charge")
        else:
            raise AttributeError(
                "Attribute error: Unable to load atom charges from atoms object!")

        if np.sum(charge_p) != 0:
            raise AttributeError(
                "Attribute error: We require charge balance!")      

        print(self.atoms.get_pbc())
        if not any(self.atoms.get_pbc()):
            raise AttributeError(
                "Attribute error: Thiss code only works for 3D systems with periodic boundaries in all directions!")    

        for index, pairs in enumerate(f.values()):
            if index == 0:
                alpha = pairs.get_alpha()
                rc = pairs.get_cutoff_real()
                rg = pairs.get_cutoff_reciprocal()
                nk = pairs.get_nk()
            else:
                if (rc != pairs.get_cutoff_real()) or (rg != pairs.get_cutoff_reciprocal()) or (alpha != pairs.get_alpha()) or (np.array_equal(nk, pairs.get_nk)):
                    raise AttributeError(
                        "Attribute error: Cannot use different rc, Gmax or number of wave vectors!")                        

        # 
        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', self.atoms, self.dict)
        chargeij = charge_p[i_p] * charge_p[j_p]

        mask = i_p == j_p
        if np.sum(mask) > 0:
            print("Atom can see itself!")

        # Prefactor and wave vectors for reciprocal space 
        Iu, k_lc = calc.wave_vectors_rec(rg, atoms.get_cell(), alpha, nk)

        # Short-range interaction of Buckingham and Ewald
        e_p = np.zeros_like(r_p)
        de_p = np.zeros_like(r_p)
        for params, pair in enumerate(self.dict):
            if pair[0] == pair[1]:
                mask1 = atnums[i_p] == pair[0]
                mask2 = atnums[j_p] == pair[0]
                mask = np.logical_and(mask1, mask2)

                e_p[mask] = f[pair].get_energy_sr(r_p[mask], chargeij[mask])
                de_p[mask] = f[pair].first_derivative_sr(r_p[mask], chargeij[mask])

            if pair[0] != pair[1]:
                mask1 = np.logical_and(
                    atnums[i_p] == pair[0], atnums[j_p] == pair[1])
                mask2 = np.logical_and(
                    atnums[i_p] == pair[1], atnums[j_p] == pair[0])
                mask = np.logical_or(mask1, mask2)

                e_p[mask] = f[pair].get_energy_sr(r_p[mask], chargeij[mask])
                de_p[mask] = f[pair].first_derivative_sr(r_p[mask], chargeij[mask])

        # Energy 
        # Self energy 
        e_self = list(f.values())[0].get_energy_self(charge_p)

        # Long-range reciprocal space 
        e_long = list(f.values())[0].get_energy_rec(charge_p, atoms.get_positions(), atoms.get_volume(), Iu, k_lc)

        epot = 0.5*np.sum(e_p) + np.sum(e_self) + e_long

        # Forces
        df_pc = -0.5*de_p.reshape(-1, 1)*r_pc/r_p.reshape(-1, 1) 
        f_nc = mabincount(j_p, df_pc, nb_atoms) - mabincount(i_p, df_pc, nb_atoms)

        f_nc += list(f.values())[0].first_derivative_reciprocal(charge_p, atoms.get_positions(), atoms.get_cell(), nb_atoms, atoms.get_volume(), Iu, k_lc)

        # Virial
        virial_v = -np.array([r_pc[:, 0] * df_pc[:, 0],               # xx
                              r_pc[:, 1] * df_pc[:, 1],               # yy
                              r_pc[:, 2] * df_pc[:, 2],               # zz
                              r_pc[:, 1] * df_pc[:, 2],               # yz
                              r_pc[:, 0] * df_pc[:, 2],               # xz
                              r_pc[:, 0] * df_pc[:, 1]]).sum(axis=1)  # xy

        self.results = {'energy': epot,
                        'free_energy': epot,
                        'stress': virial_v/self.atoms.get_volume(),
                        'forces': f_nc}

    ###

    def get_hessian(self, atoms, format='dense', divide_by_masses=False):
        """
        Calculate the Hessian matrix for a pair potential.
        For an atomic configuration with N atoms in d dimensions the hessian matrix is a symmetric, hermitian matrix
        with a shape of (d*N,d*N). The matrix is in general a sparse matrix, which consists of dense blocks of
        shape (d,d), which are the mixed second derivatives. The result of the derivation for a pair potential can be
        found e.g. in:
        L. Pastewka et. al. "Seamless elastic boundaries for atomistic calculations", Phys. Rev. B 86, 075459 (2012).

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

        # Check some properties of input data
        if atoms.has("charge"):
            charge_p = self.atoms.get_array("charge")
        else:
            raise AttributeError(
                "Attribute error: Unable to load atom charges from atoms object!")

        if np.sum(charge_p) != 0:
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
                rg = pairs.get_cutoff_reciprocal()
                nk = pairs.get_nk()
            else:
                if (rc != pairs.get_cutoff_real()) or (rg != pairs.get_cutoff_reciprocal()) or (alpha != pairs.get_alpha()) or (np.array_equal(nk, pairs.get_nk)):
                    raise AttributeError(
                        "Attribute error: Cannot use different rc, Gmax or number of wave vectors!")   


        i_p, j_p,  r_p, r_pc = neighbour_list('ijdD', self.atoms, self.dict)
        chargeij = charge_p[i_p] * charge_p[j_p]
        first_i = first_neighbours(nb_atoms, i_p)

        # Prefactor and wave vectors for reciprocal space 
        Iu, k_lc = calc.wave_vectors_rec(rg, atoms.get_cell(), alpha, nk)

        # Real space short range part
        de_p = np.zeros_like(r_p)
        dde_p = np.zeros_like(r_p)
        for params, pair in enumerate(dict):
            if pair[0] == pair[1]:
                mask1 = atnums[i_p] == pair[0]
                mask2 = atnums[j_p] == pair[0]
                mask = np.logical_and(mask1, mask2)

                de_p[mask] = f[pair].first_derivative_sr(r_p[mask], chargeij[mask])
                dde_p[mask] = f2[pair].second_derivative_sr(r_p[mask], chargeij[mask])

            if pair[0] != pair[1]:
                mask1 = np.logical_and(
                    atnums[i_p] == pair[0], atnums[j_p] == pair[1])
                mask2 = np.logical_and(
                    atnums[i_p] == pair[1], atnums[j_p] == pair[0])
                mask = np.logical_or(mask1, mask2)

                de_p[mask] = f[pair].first_derivative_sr(r_p[mask], chargeij[mask])
                dde_p[mask] = df2[pair].second_derivative_sr(r_p[mask], chargeij[mask])
        

        # 
        n_pc = (r_pc.T/r_p).T
        H_pcc = -(dde_p * (n_pc.reshape(-1, 3, 1)
                           * n_pc.reshape(-1, 1, 3)).T).T
        H_pcc += -(de_p/r_p * (np.eye(3, dtype=n_pc.dtype)
                                    - (n_pc.reshape(-1, 3, 1) * n_pc.reshape(-1, 1, 3))).T).T

        # Sparse BSR-matrix
        if format == "sparse":
            if divide_by_masses:
                masses_n = atoms.get_masses()
                geom_mean_mass_p = np.sqrt(masses_n[i_p]*masses_n[j_p])

            if divide_by_masses:
                H = bsr_matrix(((H_pcc.T/geom_mean_mass_p).T, j_p, first_i), shape=(3*nb_atoms, 3*nb_atoms))

            else: 
                H = bsr_matrix((H_pcc, j_p, first_i), shape=(3*nb_atoms, 3*nb_atoms))

            Hdiag_icc = np.empty((nb_atoms, 3, 3))
            for x in range(3):
                for y in range(3):
                    Hdiag_icc[:, x, y] = - \
                        np.bincount(i_p, weights=H_pcc[:, x, y])

            if divide_by_masses:
                H += bsr_matrix(((Hdiag_icc.T/masses_n).T, np.arange(nb_atoms),
                             np.arange(nb_atoms+1)), shape=(3*nb_atoms, 3*nb_atoms))

            else:
                H += bsr_matrix((Hdiag_icc, np.arange(nb_atoms),
                             np.arange(nb_atoms+1)), shape=(3*nb_atoms, 3*nb_atoms))

            return H

        # Dense matrix format
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

    ### 
    def get_stress(self, atoms):
        """
        Compute the analytic expression of stress

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """


    ###

    def get_nonaffine_forces(self, atoms):
        """
        Compute the non-affine forces which result from an affine deformation of atoms.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """