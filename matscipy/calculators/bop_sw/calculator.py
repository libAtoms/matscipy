# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014-2018) James Kermode, King's College London
#                       Lars Pastewka, University of Freiburg
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
# ======================================================================

"""
Bond Order Potential.
"""
import sys

import numpy as np

import ase

from scipy.sparse import bsr_matrix

from ase.atoms import Atoms

from ase.calculators.calculator import Calculator

from ...neighbours import find_indices_of_reversed_pairs, first_neighbours, \
    neighbour_list, triplet_list


from ...elasticity import Voigt_6_to_full_3x3_stress


class AbellTersoffBrennerStillingerWeber(Calculator):
    implemented_properties = ['free_energy', 'energy', 'stress', 'forces']
    default_parameters = {}
    name = 'ThreeBodyPotential'

    def __init__(self, F, G,
                 d1F, d2F, d11F, d22F, d12F,
                 d1G,
                 d11G,
                 d2G,
                 d22G,
                 d12G,
                 cutoff):
        Calculator.__init__(self)
        self.F = F
        self.G = G
        self.d1F = d1F
        self.d2F = d2F
        self.d11F = d11F
        self.d22F = d22F
        self.d12F = d12F
        self.d2G = d2G
        self.d1G = d1G
        self.d22G = d22G
        self.d11G = d11G
        self.d12G = d12G

        self.cutoff = cutoff

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # construct neighbor list
        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', atoms=atoms,
                                             cutoff=self.cutoff)

        nb_atoms = len(self.atoms)
        nb_pairs = len(i_p)

        # normal vectors
        n_pc = (r_pc.T / r_p).T

        # construct triplet list
        first_i = first_neighbours(nb_atoms, i_p)
        ij_t, ik_t = triplet_list(first_i)

        # calculate energy
        G_t = self.G(r_pc[ij_t], r_pc[ik_t])
        xi_p = np.bincount(ij_t, weights=G_t, minlength=nb_pairs)
        F_p = self.F(r_p, xi_p)
        epot = 0.5 * np.sum(F_p)

        d1G_t = self.d1G(r_pc[ij_t], r_pc[ik_t])
        d2F_d2G_t = (self.d2F(r_p[ij_t], xi_p[ij_t]) * self.d2G(r_pc[ij_t], r_pc[ik_t]).T).T
        # calculate forces (per pair)
        fx_p = \
            self.d1F(r_p, xi_p) * n_pc[:, 0] + \
            self.d2F(r_p, xi_p) * np.bincount(ij_t, d1G_t[:, 0], minlength=nb_pairs) + \
            np.bincount(ik_t, d2F_d2G_t[:, 0], minlength=nb_pairs)
        fy_p = \
            self.d1F(r_p, xi_p) * n_pc[:, 1] + \
            self.d2F(r_p, xi_p) * np.bincount(ij_t, d1G_t[:, 1], minlength=nb_pairs) + \
            np.bincount(ik_t, d2F_d2G_t[:, 1], minlength=nb_pairs)
        fz_p = \
            self.d1F(r_p, xi_p) * n_pc[:, 2] + \
            self.d2F(r_p, xi_p) * np.bincount(ij_t, d1G_t[:, 2], minlength=nb_pairs) + \
            np.bincount(ik_t, d2F_d2G_t[:, 2], minlength=nb_pairs)

        # collect atomic forces
        fx_n = 0.5 * (np.bincount(i_p, weights=fx_p) -
                      np.bincount(j_p, weights=fx_p))
        fy_n = 0.5 * (np.bincount(i_p, weights=fy_p) -
                      np.bincount(j_p, weights=fy_p))
        fz_n = 0.5 * (np.bincount(i_p, weights=fz_p) -
                      np.bincount(j_p, weights=fz_p))

        f_nc = np.transpose([fx_n, fy_n, fz_n])

        # Virial 
        virial_v = 0.5 * np.array([r_pc[:, 0] * fx_p,  # xx
                                   r_pc[:, 1] * fy_p,  # yy
                                   r_pc[:, 2] * fz_p,  # zz
                                   r_pc[:, 1] * fz_p,  # yz
                                   r_pc[:, 0] * fz_p,  # xz
                                   r_pc[:, 0] * fy_p]).sum(axis=1)  # xy

        self.results = {'free_energy': epot, 'energy': epot, 'stress': virial_v / self.atoms.get_volume(), 'forces': f_nc}

    def get_hessian(self, atoms, format='sparse', divide_by_masses=False):
        """
        Calculate the Hessian matrix for a bond order potential.
        For an atomic configuration with N atoms in d dimensions the hessian matrix is a symmetric, hermitian matrix
        with a shape of (d*N,d*N). The matrix is in general a sparse matrix, which consists of dense blocks of shape (d,d), which
        are the mixed second derivatives.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        format: "sparse" or "neighbour-list"
            Output format of the hessian matrix.

        divide_by_masses: bool
        	if true return the dynamic matrix else hessian matrix 

		Returns
		-------
		bsr_matrix
			either hessian or dynamic matrix

        Restrictions
        ----------
        This method is currently only implemented for three dimensional systems
        """
        if self.atoms is None:
            self.atoms = atoms

        # construct neighbor list
        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', atoms=atoms,
                                             cutoff=2 * self.cutoff)

        mask_p = r_p > self.cutoff

        nb_atoms = len(self.atoms)
        nb_pairs = len(i_p)

        # reverse pairs
        tr_p = find_indices_of_reversed_pairs(i_p, j_p, r_p)

        # normal vectors
        n_pc = (r_pc.T / r_p).T

        # construct triplet list
        first_i = first_neighbours(nb_atoms, i_p)
        ij_t, ik_t, jk_t = triplet_list(first_i, r_p, self.cutoff, i_p, j_p)
        first_ij = first_neighbours(len(i_p), ij_t)

        nb_triplets = len(ij_t)

        # basic triplet and pair terms
        G_t = self.G(r_pc[ij_t], r_pc[ik_t])
        xi_p = np.bincount(ij_t, weights=G_t, minlength=nb_pairs)
        F_p = self.F(r_p, xi_p)
        
        # Output
        #H_pcc = np.zeros((nb_pairs, 3, 3))

        # Hessian term #4
        d1F_p = self.d1F(r_p, xi_p)
        d1F_p[mask_p] = 0.0  # we need to explicitly exclude everything with r > cutoff

        H_pcc = -(d1F_p * (np.eye(3) - (n_pc.reshape(-1, 3, 1) * n_pc.reshape(-1, 1, 3))).T / r_p).T

        # Hessian term #1
        d11F_p = self.d11F(r_p, xi_p)
        d11F_p[mask_p] = 0.0

        H_pcc -= (d11F_p * (n_pc.reshape(-1, 3, 1) * n_pc.reshape(-1, 1, 3)).T).T

        # Hessian term #2
        d12F_p = self.d12F(r_p, xi_p)
        d12F_p[mask_p] = 0.0

        d2G_tc = self.d2G(r_pc[ij_t], r_pc[ik_t])
        H_temp3_t = (d12F_p[ij_t] * (d2G_tc.reshape(-1, 3, 1) * n_pc[ij_t].reshape(-1, 1, 3)).T).T

        d1G_tc = self.d1G(r_pc[ij_t], r_pc[ik_t])
        H_temp4_t = (d12F_p[ij_t] * (d1G_tc.reshape(-1, 3, 1) * n_pc[ij_t].reshape(-1, 1, 3)).T).T

        # Hessian term #5
        d2F_p = self.d2F(r_p, xi_p)
        d2F_p[mask_p] = 0.0

        d22G_tcc = self.d22G(r_pc[ij_t], r_pc[ik_t])
        H_temp2_t = (d2F_p[ij_t] * d22G_tcc.T).T

        d11G_tcc = self.d11G(r_pc[ij_t], r_pc[ik_t])
        H_temp_t = (d2F_p[ij_t] * d11G_tcc.T).T

        d12G_tcc = self.d12G(r_pc[ij_t], r_pc[ik_t])
        H_temp1_t = (d2F_p[ij_t] * d12G_tcc.T).T
    
        # Hessian term #3

        ## Terms involving D_1 * D_1
        d1G_t = self.d1G(r_pc[ij_t], r_pc[ik_t])

        d22F_p = self.d22F(r_p, xi_p)
        d22F_p[mask_p] = 0.0

        # TODO: bincount multiaxis
        d1xG_p = np.bincount(ij_t, weights=d1G_t[:, 0], minlength=nb_pairs)
        d1yG_p = np.bincount(ij_t, weights=d1G_t[:, 1], minlength=nb_pairs)
        d1zG_p = np.bincount(ij_t, weights=d1G_t[:, 2], minlength=nb_pairs)

        d1G_p = np.transpose([d1xG_p, d1yG_p, d1zG_p])
        H_pcc -= (d22F_p * (d1G_p.reshape(-1, 3, 1) * d1G_p.reshape(-1, 1, 3)).T).T

        ## Terms involving D_2 * D_2
        d2G_t = self.d2G(r_pc[ij_t], r_pc[ik_t])

        # TODO: bincount multiaxis
        d2xG_p = np.bincount(ij_t, weights=d2G_t[:, 0], minlength=nb_pairs)
        d2yG_p = np.bincount(ij_t, weights=d2G_t[:, 1], minlength=nb_pairs)
        d2zG_p = np.bincount(ij_t, weights=d2G_t[:, 2], minlength=nb_pairs)

        d2G_p = np.transpose([d2xG_p, d2yG_p, d2zG_p])

        Q1 = ((d22F_p * d2G_p.T).T[ij_t].reshape(-1, 3, 1) * d2G_t.reshape(-1, 1, 3))

        ## Terms involving D_1 * D_2
        Q2 = ((d22F_p * d1G_p.T).T[ij_t].reshape(-1, 3, 1) * d2G_t.reshape(-1, 1, 3))

        H_pcc -= (d22F_p * (d2G_p.reshape(-1, 3, 1) * d1G_p.reshape(-1, 1, 3)).T).T
        
        # TODO: bincount multiaxis
        for x in range(3):
            for y in range(3):
                H_pcc[:, x, y] -= np.bincount(ij_t, weights=H_temp_t[:, x, y], minlength=nb_pairs)
                H_pcc[:, x, y] += np.bincount(jk_t, weights=H_temp1_t[:, x, y], minlength=nb_pairs) - np.bincount(
                    tr_p[ij_t], weights=H_temp1_t[:, x, y], minlength=nb_pairs) - np.bincount(ik_t, weights=H_temp1_t[:, x, y], minlength=nb_pairs)
                H_pcc[:, x, y] -= np.bincount(ik_t, weights=H_temp2_t[:, x, y], minlength=nb_pairs)
                H_pcc[:, x, y] += np.bincount(tr_p[jk_t], weights=H_temp3_t[:, x, y], minlength=nb_pairs) - np.bincount(
                    ij_t, weights=H_temp3_t[:, x, y], minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=H_temp3_t[:, x, y], minlength=nb_pairs)
                H_pcc[:, x, y] -= np.bincount(ij_t, weights=H_temp4_t[:, x, y], minlength=nb_pairs) + np.bincount(tr_p[ij_t], weights=H_temp4_t[:, x, y], minlength=nb_pairs)
                H_pcc[:, x, y] -= np.bincount(ik_t, weights=Q1[:, x, y], minlength=nb_pairs)
                H_pcc[:, x, y] += np.bincount(jk_t, weights=Q2[:, x, y], minlength=nb_pairs) - np.bincount(ik_t, weights=Q2[:, x, y], minlength=nb_pairs)

        for il_im in range(nb_triplets):
            il = ij_t[il_im]
            im = ik_t[il_im]
            lm = jk_t[il_im]
            for t in range(first_ij[il], first_ij[il + 1]):
                ij = ik_t[t]
                if ij != il and ij != im:
                    r_p_ij = np.array([r_pc[ij]])
                    r_p_il = np.array([r_pc[il]])
                    r_p_im = np.array([r_pc[im]])
                    H_pcc[lm, :, :] += (0.5 * d22F_p[ij] * (
                            self.d2G(r_p_ij, r_p_il).reshape(-1, 3, 1) * self.d2G(r_p_ij, r_p_im).reshape(-1, 1, 3)).T).T.squeeze()
        
        # Add the conjugate terms (symmetrize Hessian)
        H_pcc += H_pcc.transpose(0, 2, 1)[tr_p, :, :]

        if format == "sparse":
            # Construct full diagonal terms from off-diagonal terms
            H_acc = np.zeros([nb_atoms, 3, 3])
            for x in range(3):
                for y in range(3):
                    H_acc[:, x, y] = -np.bincount(i_p, weights=H_pcc[:, x, y])

            if divide_by_masses:
                mass_nat = atoms.get_masses()
                geom_mean_mass_n = np.sqrt(mass_nat[i_p] * mass_nat[j_p])
                return \
                    bsr_matrix(((H_pcc.T / (2 * geom_mean_mass_n)).T, j_p, first_i), shape=(3 * nb_atoms, 3 * nb_atoms)) \
                    + bsr_matrix(((H_acc.T / (2 * mass_nat)).T, np.arange(nb_atoms), np.arange(nb_atoms + 1)),
                                 shape=(3 * nb_atoms, 3 * nb_atoms))
            else:
                return \
                    bsr_matrix((H_pcc / 2, j_p, first_i), shape=(3 * nb_atoms, 3 * nb_atoms)) \
                    + bsr_matrix((H_acc / 2, np.arange(nb_atoms), np.arange(nb_atoms + 1)),
                                 shape=(3 * nb_atoms, 3 * nb_atoms))

        # Neighbour list format
        elif format == "neighbour-list":
            return H_pcc / 2, i_p, j_p, r_pc, r_p

    def get_second_derivative(self, atoms, drda_pc, drdb_pc, i_p=None, j_p=None, r_p=None, r_pc=None):
        """
        Calculate the second derivative of the energy with respect to arbitrary variables a and b.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        drda_pc/drdb_pc: array_like
            Derivative of atom positions with respect to variable a/b.

        i_p: array
            First atom index

        j_p: array
            Second atom index

        r_p: array
            Absolute distance 

        r_pc: array 
            Distance vector

        """
        if self.atoms is None:
            self.atoms = atoms

        if i_p is None or j_p is None or r_p is None or r_pc is None:
            # We need to construct the neighbor list ourselves
            i_p, j_p, r_p, r_pc = neighbour_list('ijdD', atoms=atoms, cutoff=2 * self.cutoff)

        mask_p = r_p > self.cutoff

        nb_atoms = len(self.atoms)
        nb_pairs = len(i_p)

        # normal vectors
        n_pc = (r_pc.T / r_p).T

        # derivative of the lengths of distance vectors
        drda_p = (n_pc * drda_pc).sum(axis=1)
        drdb_p = (n_pc * drdb_pc).sum(axis=1)

        # construct triplet list
        first_i = first_neighbours(nb_atoms, i_p)
        ij_t, ik_t, jk_t = triplet_list(first_i, r_p, self.cutoff, i_p, j_p)
        first_ij = first_neighbours(len(i_p), ij_t)

        nb_triplets = len(ij_t)

        # basic triplet and pair terms
        G_t = self.G(r_pc[ij_t], r_pc[ik_t])
        xi_p = np.bincount(ij_t, weights=G_t, minlength=nb_pairs)
        F_p = self.F(r_p, xi_p)

        # Term 1
        d11F_p = self.d11F(r_p, xi_p)
        d11F_p[mask_p] = 0.0
        T1 = (d11F_p * drda_p * drdb_p).sum()
        
        # Term 2
        d12F_p = self.d12F(r_p, xi_p)
        d12F_p[mask_p] = 0.0

        d2G_tc = self.d2G(r_pc[ij_t], r_pc[ik_t])
        T2 = (d12F_p[ij_t] * (d2G_tc * drda_pc[ik_t]).sum(axis=1) * drdb_p[ij_t]).sum()
        #T2 += (d12F_p[ij_t] * (d2G_tc * drdb_pc[ik_t]).sum(axis=1) * drda_p[ij_t]).sum()

        d1G_tc = self.d1G(r_pc[ij_t], r_pc[ik_t])
        #T2 += (d12F_p[ij_t] * (d1G_tc * drda_pc[ij_t]).sum(axis=1) * drdb_p[ij_t]).sum()
        #T2 += (d12F_p[ij_t] * (d1G_tc * drdb_pc[ij_t]).sum(axis=1) * drda_p[ij_t]).sum()

        # Term 3
        d1G_tc = self.d1G(r_pc[ij_t], r_pc[ik_t])
        d2G_tc = self.d2G(r_pc[ij_t], r_pc[ik_t])
        d22F_p = self.d22F(r_p, xi_p)
        d22F_p[mask_p] = 0.0

        dxida_t = (d1G_tc * drda_pc[ij_t]).sum(axis=1) + (d2G_tc * drda_pc[ik_t]).sum(axis=1)
        dxidb_t = (d1G_tc * drdb_pc[ij_t]).sum(axis=1) + (d2G_tc * drdb_pc[ik_t]).sum(axis=1)
        T3 = (d22F_p *
              np.bincount(ij_t, weights=dxida_t, minlength=nb_pairs) *
              np.bincount(ij_t, weights=dxidb_t, minlength=nb_pairs)).sum()

        # Term 4
        d1F_p = self.d1F(r_p, xi_p)
        d1F_p[mask_p] = 0.0

        Q_pcc = ((np.eye(3) - (n_pc.reshape(-1, 3, 1) * n_pc.reshape(-1, 1, 3))).T / r_p).T

        T4 = (d1F_p * ((Q_pcc * drda_pc.reshape(-1, 3, 1)).sum(axis=1) * drdb_pc).sum(axis=1)).sum()

        # Term 5
        d2F_p = self.d2F(r_p, xi_p)
        d2F_p[mask_p] = 0.0

        d11G_tcc = self.d11G(r_pc[ij_t], r_pc[ik_t])
        d12G_tcc = self.d12G(r_pc[ij_t], r_pc[ik_t])
        d22G_tcc = self.d22G(r_pc[ij_t], r_pc[ik_t])

        #T5_t =  ((d11G_tcc * drdb_pc[ij_t].reshape(-1, 3, 1)).sum(axis=1) * drda_pc[ij_t]).sum(axis=1)
        #T5_t = ((drdb_pc[ik_t].reshape(-1, 1, 3) * d12G_tcc).sum(axis=2) * drda_pc[ij_t]).sum(axis=1)
        T5_t = ((drdb_pc[ij_t].reshape(-1, 3, 1) * d12G_tcc).sum(axis=1) * drda_pc[ik_t]).sum(axis=1)
        #T5_t += ((d22G_tcc * drdb_pc[ik_t].reshape(-1, 3, 1)).sum(axis=1) * drda_pc[ik_t]).sum(axis=1)
        T5 = (d2F_p * np.bincount(ij_t, weights=T5_t, minlength=nb_pairs)).sum()

        #return T1 + T2 + T3 + T4 + T5
        return T2

    def get_hessian_from_second_derivative(self, atoms):
        """
        Compute the Hessian matrix from second derivatives.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """

        if self.atoms is None:
            self.atoms = atoms

        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', atoms=atoms, cutoff=2 * self.cutoff)

        nb_atoms = len(self.atoms)
        nb_pairs = len(i_p)

        H_ab = np.zeros((3 * nb_atoms, 3 * nb_atoms))

        for m in range(0, nb_atoms):
            for cm in range(3):
                drda_pc = np.zeros((nb_pairs, 3))
                drda_pc[i_p == m, cm] = 1
                drda_pc[j_p == m, cm] = -1
                for l in range(0, nb_atoms):
                    for cl in range(3):
                        drdb_pc = np.zeros((nb_pairs, 3))
                        drdb_pc[i_p == l, cl] = 1
                        drdb_pc[j_p == l, cl] = -1
                        H_ab[3 * m + cm, 3 * l + cl] = self.get_second_derivative(atoms, drda_pc, drdb_pc,
                                                                                  i_p=i_p, j_p=j_p, r_p=r_p, r_pc=r_pc)

        return H_ab / 2

    def get_non_affine_forces_from_second_derivative(self, atoms):
        """
        Compute the analytical non-affine forces.  

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """

        if self.atoms is None:
            self.atoms = atoms

        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', atoms=atoms, cutoff=2 * self.cutoff)

        nb_atoms = len(self.atoms)
        nb_pairs = len(i_p)

        naF_ncab = np.zeros((nb_atoms, 3, 3, 3))

        for m in range(0, nb_atoms):
            for cm in range(3):
                drdb_pc = np.zeros((nb_pairs, 3))
                drdb_pc[i_p == m, cm] = 1
                drdb_pc[j_p == m, cm] = -1
                for alpha in range(3):
                    for beta in range(3):       
                        drda_pc = np.zeros((nb_pairs, 3))
                        drda_pc[:, alpha] = r_pc[:, beta]
                        naF_ncab[m, cm, alpha, beta] = self.get_second_derivative(atoms, drda_pc, drdb_pc,
                                                                                  i_p=i_p, j_p=j_p, r_p=r_p, r_pc=r_pc) 
        return naF_ncab / 2

    def get_born_elastic_constants_from_second_derivative(self, atoms):
        """
        Compute the Born elastic constants. 

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """

        if self.atoms is None:
            self.atoms = atoms 

        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', atoms=atoms, cutoff=2 * self.cutoff)

        nb_atoms = len(self.atoms)
        nb_pairs = len(i_p)

        C_cccc = np.zeros((3, 3, 3, 3))
        
        for alpha in range(3):
            for beta in range(3):
                drda_pc = np.zeros((nb_pairs, 3))
                drda_pc[:, alpha] = r_pc[:, beta]/2
                drda_pc[:, beta] += r_pc[:, alpha]/2
                for nu in range(3):
                    for mu in range(3):
                        drdb_pc = np.zeros((nb_pairs, 3))
                        drdb_pc[:, nu] = r_pc[:, mu]/2
                        drdb_pc[:, mu] += r_pc[:, nu]/2
                        C_cccc[alpha, beta, nu, mu] = self.get_second_derivative(atoms, drda_pc, drdb_pc,
                                                                                 i_p=i_p, j_p=j_p, r_p=r_p, r_pc=r_pc)
        
        C_cccc /= (2 * atoms.get_volume())

        # Symmetrize elastic constant tensor
        #C_cccc = (C_cccc + C_cccc.swapaxes(0, 1) + C_cccc.swapaxes(2, 3) + C_cccc.swapaxes(0, 1).swapaxes(2, 3)) / 4

        return C_cccc

    def get_stress_contribution_to_elastic_constants(self, atoms):
        """
        Compute the correction to the elastic constants due to non-zero stress in the configuration.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """
        
        if self.atoms is None:
            self.atoms = atoms

        # Add stress term that comes from working with the Cauchy stress
        stress_cc = Voigt_6_to_full_3x3_stress(self.get_stress())
        delta_cc = np.identity(3)
        C_cccc = delta_cc.reshape(3, 1, 3, 1) * stress_cc.reshape(1, 3, 1, 3) - \
                  (delta_cc.reshape(3, 3, 1, 1) * stress_cc.reshape(1, 1, 3, 3) + \
                   delta_cc.reshape(1, 1, 3, 3) * stress_cc.reshape(3, 3, 1, 1)) / 2

        C_cccc = (C_cccc + C_cccc.swapaxes(0, 1) + C_cccc.swapaxes(2, 3) + C_cccc.swapaxes(0, 1).swapaxes(2, 3)) / 4

        return C_cccc

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
        bornC_cccc = calculator.get_born_elastic_constants_from_second_derivative(atoms)

        # Stress contribution to elastic constants
        stressC_cccc = calculator.get_stress_contribution_to_elastic_constants(atoms)

        return bornC_cccc + stressC_cccc

    def get_non_affine_contribution_to_elastic_constants(self, atoms, eigenvalues=None, eigenvectors=None):
        """
        Compute the correction of non-affine displacements to the elasticity tensor.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """

        try:
            from scipy import linalg
        except ImportError:
            raise ImportError(
                "Import error: Can not compute non-affine elastic constants! Scipy is needed!")

        nat = len(atoms)
        calculator = atoms.get_calculator()

        # Non-affine forces
        forces_natcab = calculator.get_non_affine_forces_from_second_derivative(atoms)

        if eigenvalues is None or eigenvectors is None:  
            H_nn = calculator.get_hessian(atoms, "sparse").todense()
            eigenvalues, eigenvectors = linalg.eigh(H_nn, b=None, subset_by_index=[3, 3*nat-1])
         
        #B_ncc = np.sum(eigvecs_nn.reshape(3*nat-3, 3*nat, 1, 1) * forces_natcab.reshape(1, 3*nat, 3, 3), axis=1)
        #print(B_ncc.shape)
        #B_ncc /= np.sqrt(eigvalues_n.reshape(3*nat-3, 1, 1))
        #B_ncc = B_ncc.reshape(3*nat-3, 3, 3, 1, 1) * B_ncc.reshape(3*nat-3, 1, 1, 3, 3)
        #C_cccc2 = np.sum(B_ncc, axis=0)

        # Compute non-affine contribution 
        C_cccc = np.empty((3,3,3,3))
        for index in range(0, 3*nat -3):
            first_con = np.sum((eigenvectors[:,index]).reshape(3*nat, 1, 1) * forces_natcab.reshape(3*nat, 3, 3), axis=0)
            C_cccc += (first_con.reshape(3,3,1,1) * first_con.reshape(1,1,3,3))/eigenvalues[index]

        return - C_cccc/atoms.get_volume()

    def get_non_affine_forces(self, atoms):
        if self.atoms is None:
            self.atoms = atoms

        # construct neighbor list
        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', atoms=atoms,
                                             cutoff=2*self.cutoff)

        mask_p = r_p > self.cutoff

        nb_atoms = len(self.atoms)
        nb_pairs = len(i_p)

        # reverse pairs
        tr_p = find_indices_of_reversed_pairs(i_p, j_p, r_p)

        # normal vectors
        n_pc = (r_pc.T / r_p).T
        dn_pcc = ((np.eye(3) - (n_pc.reshape(-1, 3, 1) * n_pc.reshape(-1, 1, 3))).T / r_p).T

        # construct triplet list
        first_i = first_neighbours(nb_atoms, i_p)
        ij_t, ik_t, jk_t = triplet_list(first_i, r_p, self.cutoff, i_p, j_p)
        first_ij = first_neighbours(len(i_p), ij_t)

        nb_triplets = len(ij_t)

        # basic triplet and pair terms
        G_t = self.G(r_pc[ij_t], r_pc[ik_t])
        xi_p = np.bincount(ij_t, weights=G_t, minlength=nb_pairs)
        F_p = self.F(r_p, xi_p)


        # Term 1
        d11F_p = self.d11F(r_p, xi_p)
        d11F_p[mask_p] = 0.0

        naF1_nccc =  d11F_p.reshape(-1, 1, 1, 1) * (n_pc.reshape(-1, 3, 1, 1) * n_pc.reshape(-1, 1, 3, 1) * r_pc.reshape(-1, 1, 1, 3))

        # Term 2
        d12F_p = self.d12F(r_p, xi_p)
        d12F_p[mask_p] = 0.0

        naF21_tcab = (d12F_p[ij_t] * (0*n_pc[ij_t].reshape(-1, 3, 1, 1) * self.d1G(r_pc[ij_t], r_pc[ik_t]).reshape(-1, 1, 3, 1) * r_pc[ij_t].reshape(-1, 1, 1, 3) \
                                   + n_pc[ij_t].reshape(-1, 3, 1, 1) * self.d2G(r_pc[ij_t], r_pc[ik_t]).reshape(-1, 1, 3, 1) * r_pc[ik_t].reshape(-1, 1, 1, 3) \
                                   + 0*self.d1G(r_pc[ij_t], r_pc[ik_t]).reshape(-1, 3, 1, 1) * n_pc[ij_t].reshape(-1, 1, 3, 1) * r_pc[ij_t].reshape(-1, 1, 1, 3) \
                                   + 0*self.d2G(r_pc[ij_t], r_pc[ik_t]).reshape(-1, 3, 1, 1) * n_pc[ij_t].reshape(-1, 1, 3, 1) * r_pc[ij_t].reshape(-1, 1, 1, 3)).T).T

        naF22_tcab = -(d12F_p[tr_p[ij_t]] * (0*n_pc[tr_p[ij_t]].reshape(-1, 3, 1, 1) * self.d1G(r_pc[tr_p[ij_t]], r_pc[jk_t]).reshape(-1, 1, 3, 1) * r_pc[tr_p[ij_t]].reshape(-1, 1, 1, 3) \
                                           + n_pc[tr_p[ij_t]].reshape(-1, 3, 1, 1) * self.d2G(r_pc[tr_p[ij_t]], r_pc[jk_t]).reshape(-1, 1, 3, 1) * r_pc[jk_t].reshape(-1, 1, 1, 3) \
                                           + 0*self.d1G(r_pc[tr_p[ij_t]], r_pc[jk_t]).reshape(-1, 3, 1, 1) * n_pc[tr_p[ij_t]].reshape(-1, 1, 3, 1) * r_pc[tr_p[ij_t]].reshape(-1, 1, 1, 3)).T).T
       
        naF23_tcab = -(d12F_p[jk_t] * (0*self.d2G(r_pc[jk_t], r_pc[tr_p[ij_t]]).reshape(-1, 3, 1, 1) * n_pc[jk_t].reshape(-1, 1, 3, 1) * r_pc[jk_t].reshape(-1, 1, 1, 3)).T).T

        # Term 3
        d1G_tc = self.d1G(r_pc[ij_t], r_pc[ik_t])
        d2G_tc = self.d2G(r_pc[ij_t], r_pc[ik_t])
        d22F_p = self.d22F(r_p, xi_p)
        d22F_p[mask_p] = 0.0

        naF31_pccc = np.zeros((nb_pairs, 3, 3, 3))
        # Multiaxis bincount 
        #for i in range(0, 3):
        #    naF31_pccc[:,  ] = d22F_p * (np.bincount(ij_t, weights=d1G_tc[:, i], minlength=nb_atoms).reshape(-1, 3, 1, 1) )

        # Term 4
        d1F_p = self.d1F(r_p, xi_p)
        d1F_p[mask_p] = 0.0

        naF4_nccc = (d1F_p * (dn_pcc.reshape(-1, 3, 3, 1) * r_pc.reshape(-1, 1, 1, 3)).T).T

        # Term 5
        d2F_p = self.d2F(r_p, xi_p)
        d2F_p[mask_p] = 0.0

        d11G_tcc = self.d11G(r_pc[ij_t], r_pc[ik_t])
        d12G_tcc = self.d12G(r_pc[ij_t], r_pc[ik_t])
        d22G_tcc = self.d22G(r_pc[ij_t], r_pc[ik_t])

        #naF51_tcab = (d2F_p[ij_t] * (d11G_tcc.reshape(-1, 3, 3, 1) * r_pc[ij_t].reshape(-1, 1, 1, 3) + d12G_tcc.reshape(-1, 3, 3, 1) * r_pc[ik_t].reshape(-1, 1, 1, 3) \
        #                            d12G_tcc.reshape(-1, )            ).T)

        naF51_tcab = (d2F_p[ij_t] * (d11G_tcc.reshape(-1,3,3,1) * r_pc[ij_t].reshape(-1,1,1,3)).T).T
        naF52_tcab = (d2F_p[ij_t] * (d22G_tcc.reshape(-1,3,3,1) * r_pc[ik_t].reshape(-1,1,1,3)).T).T
        naF53_tcab = (d2F_p[ij_t] * (self.d12G(r_pc[ij_t], r_pc[ik_t]).reshape(-1,3,3,1) * r_pc[ij_t].reshape(-1,1,1,3)).T).T
        naF54_tcab = (d2F_p[jk_t] * (self.d12G(r_pc[jk_t], r_pc[tr_p[ij_t]]).reshape(-1,3,3,1) * r_pc[jk_t].reshape(-1,1,1,3)).T).T

        naForces_natcab = np.zeros((nb_atoms, 3, 3, 3))

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    # Term 1 --> Works!
                    #naForces_natcab[:, i, j, k] += np.bincount(i_p, weights=naF1_nccc[:, i, j, k], minlength=nb_atoms) - \
                    #    np.bincount(j_p, weights=naF1_nccc[:, i, j, k], minlength=nb_atoms)

                    # Term 2
                    naForces_natcab[:, i, j, k] += np.bincount(i_p[ij_t], weights=naF21_tcab[:, i, j, k], minlength=nb_atoms)
                    naForces_natcab[:, i, j, k] += np.bincount(j_p[ij_t], weights=naF22_tcab[:, i, j, k], minlength=nb_atoms)
                    naForces_natcab[:, i, j, k] += np.bincount(i_p[ij_t], weights=naF23_tcab[:, i, j, k], minlength=nb_atoms)

                    # Term 3
                   
                    # Term 4 --> Works! 
                    #naForces_natcab[:, i, j, k] += np.bincount(i_p, weights=naF4_nccc[:, i, j, k], minlength=nb_atoms) - \
                    #    np.bincount(j_p, weights=naF4_nccc[:, i, j, k], minlength=nb_atoms)

                    # Term 5 
                    # Delta11 --> Works 
                    #naForces_natcab[:, i, j, k] += np.bincount(i_p[ij_t], weights=naF51_tcab[:, i, j, k], minlength=nb_atoms)
                    #naForces_natcab[:, i, j, k] -= np.bincount(j_p[ij_t], weights=naF51_tcab[:, i, j, k], minlength=nb_atoms)
                    
                    # Delta22 --> Works
                    #naForces_natcab[:, i, j, k] += np.bincount(i_p[ik_t], weights=naF52_tcab[:, i, j, k], minlength=nb_atoms)
                    #naForces_natcab[:, i, j, k] -= np.bincount(j_p[ik_t], weights=naF52_tcab[:, i, j, k], minlength=nb_atoms)
                    
                    # Delta12 --- > ? 
                    #naForces_natcab[:, i, j, k] += np.bincount(i_p[ij_t], weights=naF53_tcab[:, i, j, k], minlength=nb_atoms)
                    #naForces_natcab[:, i, j, k] -= np.bincount(i_p[ij_t], weights=naF54_tcab[:, i, j, k], minlength=nb_atoms)

                    # Delta21 

        return naForces_natcab / 2
