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
import numpy as np
import sys
import ase
from scipy.sparse import bsr_matrix

from ase.atoms import Atoms
from ase.calculators.calculator import Calculator

from matscipy.neighbours import find_indices_of_reversed_pairs, first_neighbours, \
    neighbour_list, triplet_list

class AbellTersoffBrenner(Calculator):
    implemented_properties = ['energy', 'forces']
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
        nx_p, ny_p, nz_p = n_pc.T

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
        fx_n = 0.5*(np.bincount(i_p, weights=fx_p) -
                    np.bincount(j_p, weights=fx_p))
        fy_n = 0.5*(np.bincount(i_p, weights=fy_p) -
                    np.bincount(j_p, weights=fy_p))
        fz_n = 0.5*(np.bincount(i_p, weights=fz_p) -
                    np.bincount(j_p, weights=fz_p))

        f_n = np.transpose([fx_n, fy_n, fz_n])

        self.results = {'energy': epot, 'forces': f_n}

    def calculate_hessian_matrix(self, atoms, divide_by_masses=False):
        """
        Calculate the Hessian matrix for a bond order potential.
        For an atomic configuration with N atoms in d dimensions the hessian matrix is a symmetric, hermitian matrix
        with a shape of (d*N,d*N). The matrix is in general a sparse matrix, which consists of dense blocks of shape (d,d), which
        are the mixed second derivatives.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

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
        nx_p, ny_p, nz_p = n_pc.T

        # construct triplet list
        first_i = first_neighbours(nb_atoms, i_p)
        ij_t, ik_t, jk_t = triplet_list(first_i, r_p, self.cutoff, i_p, j_p)
        first_ij = first_neighbours(len(i_p), ij_t)

        nb_triplets = len(ij_t)

        # basic triplet and pair terms
        G_t = self.G(r_pc[ij_t], r_pc[ik_t])
        xi_p = np.bincount(ij_t, weights=G_t, minlength=nb_pairs)
        F_p = self.F(r_p, xi_p)

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
        
        # TODO: bincount multiaxis
        for x in range(3):
            for y in range(3):
                H_pcc[:, x, y] -= np.bincount(ij_t, weights=H_temp_t[:, x, y], minlength=nb_pairs)
                # here
                H_pcc[:, x, y] += np.bincount(jk_t, weights=H_temp1_t[:, x, y], minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=H_temp1_t[:, x, y], minlength=nb_pairs) - np.bincount(ik_t, weights=H_temp1_t[:, x, y], minlength=nb_pairs)
                H_pcc[:, x, y] -= np.bincount(ik_t, weights=H_temp2_t[:, x, y], minlength=nb_pairs)
                H_pcc[:, x, y] += np.bincount(tr_p[jk_t], weights=H_temp3_t[:, x, y], minlength=nb_pairs) - np.bincount(ij_t, weights=H_temp3_t[:, x, y], minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=H_temp3_t[:, x, y], minlength=nb_pairs)
                H_pcc[:, x, y] -= np.bincount(ij_t, weights=H_temp4_t[:, x, y], minlength=nb_pairs) + np.bincount(tr_p[ij_t], weights=H_temp4_t[:, x, y], minlength=nb_pairs)
                H_pcc[:, x, y] -= np.bincount(ik_t, weights=Q1[:, x, y], minlength=nb_pairs)
                H_pcc[:, x, y] += np.bincount(jk_t, weights=Q2[:, x, y], minlength=nb_pairs) - np.bincount(ik_t, weights=Q2[:, x, y], minlength=nb_pairs)
        
        
        H_pcc -= (d22F_p * (d2G_p.reshape(-1, 3, 1) * d1G_p.reshape(-1, 1, 3)).T).T
        
        
        for il_im in range(nb_triplets):
            il = ij_t[il_im]
            im = ik_t[il_im]
            lm = jk_t[il_im]
            for t in range(first_ij[il], first_ij[il+1]):
                ij = ik_t[t]
                if ij != il and ij != im:
                    r_p_ij = np.array([r_pc[ij]])
                    r_p_il = np.array([r_pc[il]])
                    r_p_im = np.array([r_pc[im]])
                    H_pcc[lm, :, :] += (0.5 * d22F_p[ij] * (self.d2G(r_p_ij, r_p_il).reshape(-1, 3, 1) * self.d2G(r_p_ij, r_p_im).reshape(-1, 1, 3)).T).T.squeeze()


        # Add the conjugate terms (symmetrize Hessian)
        H_pcc += H_pcc.transpose(0, 2, 1)[tr_p, :, :]

        # Construct full diagonal terms from off-diagonal terms
        H_acc = np.zeros([nb_atoms, 3, 3])
        for x in range(3):
            for y in range(3):
                H_acc[:, x, y] = -np.bincount(i_p, weights=H_pcc[:, x, y])

        if divide_by_masses:
            mass_nat = atoms.get_masses()
            geom_mean_mass_n = np.sqrt(mass_nat[i_p]*mass_nat[j_p])
            return \
                bsr_matrix(((H_pcc.T/(2 * geom_mean_mass_n)).T, j_p, first_i), shape=(3*nb_atoms, 3*nb_atoms)) \
                + bsr_matrix(((H_acc.T/(2 * mass_nat)).T, np.arange(nb_atoms), np.arange(nb_atoms+1)),
                     shape=(3*nb_atoms, 3*nb_atoms))
        else:
            return \
                bsr_matrix((H_pcc/2, j_p, first_i), shape=(3*nb_atoms, 3*nb_atoms)) \
                + bsr_matrix((H_acc/2, np.arange(nb_atoms), np.arange(nb_atoms+1)),
                     shape=(3*nb_atoms, 3*nb_atoms))

