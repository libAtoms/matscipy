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
Simple pair potential.
"""

from __future__ import division

import os

import sys

from scipy.sparse import csr_matrix

import numpy as np

import ase
from ase.calculators.calculator import Calculator

from matscipy.neighbours import neighbour_list

###

def get_dynamical_matrix(f, atoms):
    """
    Calculate the dynamical matrix for a pair potential
    """

    dict = {x: obj.get_cutoff() for x,obj in f.items()}
    df = {x: obj.derivative(1) for x,obj in f.items()}
    df2 = {x: obj.derivative(2) for x,obj in f.items()}

    nat = len(atoms)
    atnums = atoms.numbers
    atnums_in_system = set(atnums)

    i_n, j_n, dr_nc, abs_dr_n = neighbour_list('ijDd', atoms, dict)

    e_n = np.zeros_like(abs_dr_n)
    de_n = np.zeros_like(abs_dr_n)
    dde_n = np.zeros_like(abs_dr_n)
    for params, pair in enumerate(dict):
        if pair[0] == pair[1]:
            mask1 = atnums[i_n] == pair[0]
            mask2 = atnums[j_n] == pair[0]
            mask = np.logical_and(mask1, mask2)

            e_n[mask] = f[pair](abs_dr_n[mask])
            de_n[mask] = df[pair](abs_dr_n[mask])
            dde_n[mask] = df2[pair](abs_dr_n[mask])

        if pair[0] != pair[1]:
            mask1 = np.logical_and(atnums[i_n] == pair[0], atnums[j_n] == pair[1])
            mask2 = np.logical_and(atnums[i_n] == pair[1], atnums[j_n] == pair[0])
            mask = np.logical_or(mask1, mask2)

            e_n[mask] = f[pair](abs_dr_n[mask])
            de_n[mask] = df[pair](abs_dr_n[mask]) 
            dde_n[mask] = df2[pair](abs_dr_n[mask])

    de_n = de_n.reshape(-1,1)
    dde_n = dde_n.reshape(-1,1)

    D_mn = csr_matrix((3*nat,3*nat),dtype=np.float64)

    # Off diagonal
    for index, i in enumerate(i_n):
        eij_c = dr_nc[index]/abs_dr_n[index]
        dpara_nn = np.array(-dde_n[index] * np.outer(eij_c, eij_c))
        dortho_nn = np.array(-de_n[index]/(abs_dr_n[index]) * (np.eye(3, dtype=float) - np.outer(eij_c, eij_c)))

        row_n = np.array([3*i, 3*i, 3*i, 3*i+1, 3*i+1, 3*i+1, 3*i+2, 3*i+2, 3*i+2])
        col_n = np.array([3*j_n[index], 3*j_n[index]+1, 3*j_n[index]+2, 3*j_n[index], 3*j_n[index]+1, 3*j_n[index]+2, 3*j_n[index], 3*j_n[index]+1, 3*j_n[index]+2])
        D_mn = D_mn + csr_matrix(((dpara_nn + dortho_nn).flatten(),(row_n,col_n)),shape=(3*nat,3*nat))

    #Main diagonal
    for i in range(len(set(i_n))):
        mask = i_n == i
        e_nc = dr_nc[mask]/abs_dr_n[mask][:,None]
        k_nc = dde_n[mask]
        f_n = de_n[mask]/abs_dr_n[mask][:,None]
        curdata_nn = np.zeros((3,3))

        for j in range(len(e_nc)):
            dpara_nn = k_nc[j] * np.outer(e_nc[j], e_nc[j])
            dortho_nn = np.array(-f_n[j] * (np.eye(3, dtype=float) - np.outer(e_nc[j], e_nc[j])))
            curdata_nn += dpara_nn - dortho_nn
            
        row_n = np.array([3*i, 3*i, 3*i, 3*i+1, 3*i+1, 3*i+1, 3*i+2, 3*i+2, 3*i+2])
        col_n = np.array([3*i, 3*i+1, 3*i+2, 3*i, 3*i+1, 3*i+2, 3*i, 3*i+1, 3*i+2])
        D_mn = D_mn + csr_matrix((curdata_nn.flatten(),(row_n,col_n)),shape=(3*nat,3*nat))

    return D_mn
### 

class LennardJonesCut():
    """
    Functional form for a 12-6 Lennard-Jones potential with a hard cutoff.
    Energy is shifted to zero at cutoff.
    """

    def __init__(self, epsilon, sigma, cutoff):
        self.epsilon = epsilon
        self.sigma = sigma 
        self.cutoff = cutoff
        self.offset = (sigma/cutoff)**12 -(sigma/cutoff)**6

    def __call__(self, r):
        """
        Return function value (potential energy).
        """
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((r6 - 1) * r6 - self.offset)

    def get_cutoff(self):
        return self.cutoff

    def first_derivative(self, r):
        r = (self.sigma / r)
        r6 = r**6
        return -24 * self.epsilon / self.sigma * (2 * r6 - 1) * r6 * r

    def second_derivative(self, r):
        r2 = (self.sigma / r)**2
        r6 = r2**3
        return 24 * self.epsilon/self.sigma**2 * (26 * r6 - 7) * r6 * r2

    def derivative(self, n=1):
        if n == 1:
            return self.first_derivative
        elif n == 2:
            return self.second_derivative
        else:
            raise ValueError("Don't know how to compute {}-th derivative.".format(n))

###

class LennardJonesQuadratic():
    """
    Functional form for a 12-6 Lennard-Jones potential with a soft cutoff.
    Energy, its first and second derivative are shifted to zero at cutoff.
    """

    def __init__(self, epsilon, sigma, cutoff):
        self.epsilon = epsilon
        self.sigma = sigma 
        self.cutoff = cutoff 
        self.offset_energy = (sigma/cutoff)**12 -(sigma/cutoff)**6
        self.offset_force = 6/cutoff * (-2 * (sigma/cutoff)**12 + (sigma/cutoff)**6)
        self.offset_dforce = (1/cutoff**2) * (156 * (sigma/cutoff)**12 - 42 * (sigma/cutoff)**6)
    
    def get_cutoff(self):
        return self.cutoff

    def __call__(self, r):
        """
        Return function value (potential energy).
        """
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((r6 - 1) * r6 - self.offset_energy - (r - self.cutoff) * self.offset_force - ((r - self.cutoff)**2 /2) * self.offset_dforce)

    def first_derivative(self, r):
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((6/r) * (-2 * r6 + 1 ) * r6 - self.offset_force - (r - self.cutoff) * self.offset_dforce)

    def second_derivative(self, r):
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((1/r**2) * (156 * r6 - 42) * r6 - self.offset_dforce)

    def derivative(self, n=1):
        if n == 1:
            return self.first_derivative
        elif n == 2:
            return self.second_derivative
        else:
            raise ValueError("Don't know how to compute {}-th derivative.".format(n))


###


class PairPotential(Calculator):
    implemented_properties = ['energy', 'stress', 'forces']
    default_parameters = {}
    name = 'PairPotential'

    def __init__(self, f, cutoff=None):
        Calculator.__init__(self)
        self.f = f

        self.dict = {x: obj.get_cutoff() for x,obj in f.items()}
        self.df = {x: obj.derivative(1) for x,obj in f.items()}
        self.df2 = {x: obj.derivative(2) for x,obj in f.items()}

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        nat = len(self.atoms)
        atnums = self.atoms.numbers
        atnums_in_system = set(atnums)

        i_n, j_n, dr_nc, abs_dr_n = neighbour_list('ijDd', self.atoms, self.dict)

        e_n = np.zeros_like(abs_dr_n)
        de_n = np.zeros_like(abs_dr_n)
        for params, pair in enumerate(self.dict):
            print("params, pair:", params, pair)
            if pair[0] == pair[1]:
                mask1 = atnums[i_n] == pair[0]
                mask2 = atnums[j_n] == pair[0]
                mask = np.logical_and(mask1, mask2)

                e_n[mask] = self.f[pair](abs_dr_n[mask])
                de_n[mask] = self.df[pair](abs_dr_n[mask])

            if pair[0] != pair[1]:
                mask1 = np.logical_and(atnums[i_n] == pair[0], atnums[j_n] == pair[1])
                mask2 = np.logical_and(atnums[i_n] == pair[1], atnums[j_n] == pair[0])
                mask = np.logical_or(mask1, mask2)

                e_n[mask] = self.f[pair](abs_dr_n[mask])
                de_n[mask] = self.df[pair](abs_dr_n[mask]) 

        epot = 0.5*np.sum(e_n)

        # Forces
        df_nc = -0.5*de_n.reshape(-1,1)*dr_nc/abs_dr_n.reshape(-1,1)

        # Sum for each atom
        fx_i = np.bincount(j_n, weights=df_nc[:,0], minlength=nat) - \
            np.bincount(i_n, weights=df_nc[:,0], minlength=nat)
        fy_i = np.bincount(j_n, weights=df_nc[:,1], minlength=nat) - \
            np.bincount(i_n, weights=df_nc[:,1], minlength=nat)
        fz_i = np.bincount(j_n, weights=df_nc[:,2], minlength=nat) - \
            np.bincount(i_n, weights=df_nc[:,2], minlength=nat)

        # Virial
        virial_v = -np.array([dr_nc[:,0]*df_nc[:,0],               # xx
                              dr_nc[:,1]*df_nc[:,1],               # yy
                              dr_nc[:,2]*df_nc[:,2],               # zz
                              dr_nc[:,1]*df_nc[:,2],               # yz
                              dr_nc[:,0]*df_nc[:,2],               # xz
                              dr_nc[:,0]*df_nc[:,1]]).sum(axis=1)  # xy


        self.results = {'energy': epot,
                        'stress': virial_v/self.atoms.get_volume(),
                        'forces': np.transpose([fx_i, fy_i, fz_i])}
