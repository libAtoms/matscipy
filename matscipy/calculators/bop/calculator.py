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
                 d1x2xG, d1y2yG, d1z2zG, d1y2zG, d1x2zG, d1x2yG, d1z2yG, d1z2xG, d1y2xG, d12G,
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
        
        self.d1x2xG = d1x2xG
        self.d1y2yG = d1y2yG
        self.d1z2zG = d1z2zG
        self.d1y2zG = d1y2zG
        self.d1x2zG = d1x2zG
        self.d1x2yG = d1x2yG
        self.d1z2yG = d1z2yG
        self.d1z2xG = d1z2xG
        self.d1y2xG = d1y2xG

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
        H_temp_pcc = (d1F_p * (np.eye(3) - (n_pc.reshape(-1, 3, 1) * n_pc.reshape(-1, 1, 3))).T / r_p).T
        H_pcc = - H_temp_pcc
       
        # Hessian term #1
        d11F_p = self.d11F(r_p, xi_p)
        d11F_p[mask_p] = 0.0
        H_temp_pcc = (d11F_p * (n_pc.reshape(-1, 3, 1) * n_pc.reshape(-1, 1, 3)).T).T
        H_pcc -= H_temp_pcc 
        
        # Hessian term #2
        d12F_p = self.d12F(r_p, xi_p)
        d12F_p[mask_p] = 0.0
        
        
        d2G_tc = self.d2G(r_pc[ij_t], r_pc[ik_t])
        H_temp_t = (d12F_p[ij_t] * (d2G_tc.reshape(-1, 3, 1) * n_pc[ij_t].reshape(-1, 1, 3)).T).T
        
        H_temp_pcc = np.empty_like(H_temp_pcc)
        for x in range(3):
            for y in range(3):
                H_temp_pcc[:, x, y] = np.bincount(tr_p[jk_t], weights=H_temp_t[:, x, y], minlength=nb_pairs) - np.bincount(ij_t, weights=H_temp_t[:, x, y], minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=H_temp_t[:, x, y], minlength=nb_pairs)
        H_pcc += H_temp_pcc
        
        d1G_tc = self.d1G(r_pc[ij_t], r_pc[ik_t])

        H_temp_t = (d12F_p[ij_t] * (d1G_tc.reshape(-1, 3, 1) * n_pc[ij_t].reshape(-1, 1, 3)).T).T      

        for x in range(3):
            for y in range(3):
                H_temp_pcc[:, x, y] = - np.bincount(ij_t, weights=H_temp_t[:, x, y], minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=H_temp_t[:, x, y], minlength=nb_pairs)
        H_pcc += H_temp_pcc

        # Hessian term #5
        d2F_p = self.d2F(r_p, xi_p)
        d2F_p[mask_p] = 0.0

        d22G_tcc = self.d22G(r_pc[ij_t], r_pc[ik_t])

        H_temp_t = (d2F_p[ij_t] * d22G_tcc.T).T

        for x in range(3):
            for y in range(3):
                H_temp_pcc[:, x, y] = - np.bincount(ik_t, weights=H_temp_t[:, x, y], minlength=nb_pairs)
                # H_temp_pcc[:, x, y] = np.bincount(tr_p[jk_t], weights=H_temp_t[:, x, y], minlength=nb_pairs) - np.bincount(ij_t, weights=H_temp_t[:, x, y], minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=H_temp_t[:, x, y], minlength=nb_pairs)
        H_pcc += H_temp_pcc
        
        
        d11G_tcc = self.d11G(r_pc[ij_t], r_pc[ik_t])

        H_temp_t = (d2F_p[ij_t] * d11G_tcc.T).T 

        for x in range(3):
            for y in range(3):
                H_temp_pcc[:, x, y] = - np.bincount(ij_t, weights=H_temp_t[:, x, y], minlength=nb_pairs)
                # H_temp_pcc[:, x, y] = np.bincount(tr_p[jk_t], weights=H_temp_t[:, x, y], minlength=nb_pairs) - np.bincount(ij_t, weights=H_temp_t[:, x, y], minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=H_temp_t[:, x, y], minlength=nb_pairs)
        H_pcc += H_temp_pcc


        Hxx_p = + H_pcc[:,0,0]
        Hyy_p = + H_pcc[:,1,1]
        Hzz_p = + H_pcc[:,2,2]
        Hyz_p = + H_pcc[:,1,2]
        Hxz_p = + H_pcc[:,0,2]
        Hxy_p = + H_pcc[:,0,1]
        Hzy_p = + H_pcc[:,2,1]
        Hzx_p = + H_pcc[:,2,0]
        Hyx_p = + H_pcc[:,1,0]
        
        d1x2xG_t = self.d1x2xG(r_pc[ij_t], r_pc[ik_t])
        d1y2yG_t = self.d1y2yG(r_pc[ij_t], r_pc[ik_t])
        d1z2zG_t = self.d1z2zG(r_pc[ij_t], r_pc[ik_t])
        d1y2zG_t = self.d1y2zG(r_pc[ij_t], r_pc[ik_t])
        d1x2zG_t = self.d1x2zG(r_pc[ij_t], r_pc[ik_t])
        d1x2yG_t = self.d1x2yG(r_pc[ij_t], r_pc[ik_t])
        d1z2yG_t = self.d1z2yG(r_pc[ij_t], r_pc[ik_t])
        d1z2xG_t = self.d1z2xG(r_pc[ij_t], r_pc[ik_t])
        d1y2xG_t = self.d1y2xG(r_pc[ij_t], r_pc[ik_t])
        Hxx_t = d2F_p[ij_t] * d1x2xG_t
        Hyy_t = d2F_p[ij_t] * d1y2yG_t
        Hzz_t = d2F_p[ij_t] * d1z2zG_t
        Hyz_t = d2F_p[ij_t] * d1y2zG_t
        Hxz_t = d2F_p[ij_t] * d1x2zG_t
        Hxy_t = d2F_p[ij_t] * d1x2yG_t
        Hzy_t = d2F_p[ij_t] * d1z2yG_t
        Hzx_t = d2F_p[ij_t] * d1z2xG_t
        Hyx_t = d2F_p[ij_t] * d1y2xG_t
        Hxx_p += np.bincount(jk_t, weights=Hxx_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hxx_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hxx_t, minlength=nb_pairs)
        Hyy_p += np.bincount(jk_t, weights=Hyy_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hyy_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hyy_t, minlength=nb_pairs)
        Hzz_p += np.bincount(jk_t, weights=Hzz_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hzz_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hzz_t, minlength=nb_pairs)
        Hyz_p += np.bincount(jk_t, weights=Hyz_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hyz_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hyz_t, minlength=nb_pairs)
        Hxz_p += np.bincount(jk_t, weights=Hxz_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hxz_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hxz_t, minlength=nb_pairs)
        Hxy_p += np.bincount(jk_t, weights=Hxy_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hxy_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hxy_t, minlength=nb_pairs)
        Hzy_p += np.bincount(jk_t, weights=Hzy_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hzy_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hzy_t, minlength=nb_pairs)
        Hzx_p += np.bincount(jk_t, weights=Hzx_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hzx_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hzx_t, minlength=nb_pairs)
        Hyx_p += np.bincount(jk_t, weights=Hyx_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hyx_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hyx_t, minlength=nb_pairs)

        # Hessian term #3

        ## Terms involving D_1 * D_1
        d1G_t = self.d1G(r_pc[ij_t], r_pc[ik_t])

        d22F_p = self.d22F(r_p, xi_p)
        d22F_p[mask_p] = 0.0
        

        d1xG_p = np.bincount(ij_t, weights=d1G_t[:, 0], minlength=nb_pairs)
        d1yG_p = np.bincount(ij_t, weights=d1G_t[:, 1], minlength=nb_pairs)
        d1zG_p = np.bincount(ij_t, weights=d1G_t[:, 2], minlength=nb_pairs)

        d1G_p = np.transpose([d1xG_p, d1yG_p, d1zG_p])
        H_pcc = - (d22F_p * (d1G_p.reshape(-1, 3, 1) * d1G_p.reshape(-1, 1, 3)).T).T

        
        ## Terms involving D_2 * D_2
        d2G_t = self.d2G(r_pc[ij_t], r_pc[ik_t])


        d2xG_p = np.bincount(ij_t, weights=d2G_t[:, 0], minlength=nb_pairs)
        d2yG_p = np.bincount(ij_t, weights=d2G_t[:, 1], minlength=nb_pairs)
        d2zG_p = np.bincount(ij_t, weights=d2G_t[:, 2], minlength=nb_pairs)

        d2G_p = np.transpose([d2xG_p, d2yG_p, d2zG_p])
        
        Q = ((d22F_p * d2G_p.T).T[ij_t].reshape(-1, 3, 1) * d2G_t.reshape(-1, 1, 3))
        H_temp_pcc = np.zeros_like(H_temp_pcc)
        for x in range(3):
            for y in range(3):
                H_temp_pcc[:, x, y] = - np.bincount(ik_t, weights=Q[:, x, y], minlength=nb_pairs)
                # H_temp_pcc[:, x, y] = np.bincount(tr_p[jk_t], weights=H_temp_t[:, x, y], minlength=nb_pairs) - np.bincount(ij_t, weights=H_temp_t[:, x, y], minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=H_temp_t[:, x, y], minlength=nb_pairs)
        H_pcc += H_temp_pcc

        Hxx_p += H_pcc[:,0,0]
        Hyy_p += H_pcc[:,1,1]
        Hzz_p += H_pcc[:,2,2]
        Hyz_p += H_pcc[:,1,2]
        Hxz_p += H_pcc[:,0,2]
        Hxy_p += H_pcc[:,0,1]
        Hzy_p += H_pcc[:,2,1]
        Hzx_p += H_pcc[:,2,0]
        Hyx_p += H_pcc[:,1,0]

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
                    H_pcc = (0.5 * d22F_p[ij] * (self.d2G(r_p_ij, r_p_il).reshape(-1, 3, 1) * self.d2G(r_p_ij, r_p_im).reshape(-1, 1, 3)).T).T
                    Hxx_p[lm] += H_pcc[:, 0, 0]
                    Hyy_p[lm] += H_pcc[:, 1, 1]
                    Hzz_p[lm] += H_pcc[:, 2, 2]
                    Hyz_p[lm] += H_pcc[:, 1, 2]
                    Hxz_p[lm] += H_pcc[:, 0, 2]
                    Hxy_p[lm] += H_pcc[:, 0, 1]
                    Hzy_p[lm] += H_pcc[:, 2, 1]
                    Hzx_p[lm] += H_pcc[:, 2, 0]
                    Hyx_p[lm] += H_pcc[:, 1, 0]

        ## Terms involving D_1 * D_2
        # Was d2*d1
        Q = ((d22F_p * d1G_p.T).T[ij_t].reshape(-1, 3, 1) * d2G_t.reshape(-1, 1, 3))
        
        H_temp_pcc = np.zeros_like(H_temp_pcc)
        for x in range(3):
            for y in range(3):
                H_temp_pcc[:, x, y] = np.bincount(jk_t, weights=Q[:, x, y], minlength=nb_pairs) - np.bincount(ik_t, weights=Q[:, x, y], minlength=nb_pairs)
                # H_temp_pcc[:, x, y] = np.bincount(tr_p[jk_t], weights=H_temp_t[:, x, y], minlength=nb_pairs) - np.bincount(ij_t, weights=H_temp_t[:, x, y], minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=H_temp_t[:, x, y], minlength=nb_pairs)
        H_pcc = H_temp_pcc

        H_pcc -= (d22F_p * (d2G_p.reshape(-1, 3, 1) * d1G_p.reshape(-1, 1, 3)).T).T

        Hxx_p += H_pcc[:,0,0]
        Hyy_p += H_pcc[:,1,1]
        Hzz_p += H_pcc[:,2,2]
        Hyz_p += H_pcc[:,1,2]
        Hxz_p += H_pcc[:,0,2]
        Hxy_p += H_pcc[:,0,1]
        Hzy_p += H_pcc[:,2,1]
        Hzx_p += H_pcc[:,2,0]
        Hyx_p += H_pcc[:,1,0]

        # Add the conjugate terms (symmetrize Hessian)
        Hxx_p += Hxx_p[tr_p]
        Hyy_p += Hyy_p[tr_p]
        Hzz_p += Hzz_p[tr_p]
        tmp = Hyz_p.copy()
        Hyz_p += Hzy_p[tr_p]
        Hzy_p += tmp[tr_p]
        tmp = Hxz_p.copy()
        Hxz_p += Hzx_p[tr_p]
        Hzx_p += tmp[tr_p]
        tmp = Hxy_p.copy()
        Hxy_p += Hyx_p[tr_p]
        Hyx_p += tmp[tr_p]

        
        # Construct diagonal terms from off-diagonal terms
        Hxx_a = -np.bincount(i_p, weights=Hxx_p)
        Hyy_a = -np.bincount(i_p, weights=Hyy_p)
        Hzz_a = -np.bincount(i_p, weights=Hzz_p)
        Hyz_a = -np.bincount(i_p, weights=Hyz_p)
        Hxz_a = -np.bincount(i_p, weights=Hxz_p)
        Hxy_a = -np.bincount(i_p, weights=Hxy_p)
        Hzy_a = -np.bincount(i_p, weights=Hzy_p)
        Hzx_a = -np.bincount(i_p, weights=Hzx_p)
        Hyx_a = -np.bincount(i_p, weights=Hyx_p)

        # Construct full off-diagonal term
        H_pcc = np.transpose([[Hxx_p, Hyx_p, Hzx_p],
                              [Hxy_p, Hyy_p, Hzy_p],
                              [Hxz_p, Hyz_p, Hzz_p]])

        # Construct full diagonal term
        H_acc = np.transpose([[Hxx_a, Hxy_a, Hxz_a],
                              [Hyx_a, Hyy_a, Hyz_a],
                              [Hzx_a, Hzy_a, Hzz_a]])

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

