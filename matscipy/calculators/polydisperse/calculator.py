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

from __future__ import division

import os

import sys

import time

import numpy as np

import ase

from ase.calculators.calculator import Calculator

from matscipy.neighbours import neighbour_list, first_neighbours

try:
    from scipy.special import factorial2
    from scipy.sparse import bsr_matrix
except ImportError:
    warnings.warn('Warning: no scipy')

###


class InversePowerLawPotential():
    """
    Functional form for a smoothed inverse-power-law potential (IPL)
    with an repulsive exponent of 10.

    Parameters
    ----------
    epsilon : float
        Energy scale
    cutoff : float
        Cutoff for the pair-interaction
    minSize : float
        Minimal size of a particle, lower bound of distribtuion
    maxSize : float
        Maximal size of a particle, upper bound of distribtuion
    na : float
        Non-additivity paramter for pairwise sizes
    q : int
        Smooth the potential up to the q-th derivative.
        For q=0 the potential is smoothed, for q=1 the potential
        and its first derivative are zero at the cutoff,...

    Reference:
    ----------
        E. Lerner, Journal of Non-Crystalline Solids, 522, 119570.
    """

    def __init__(self, epsilon, cutoff, na, q, minSize, maxSize):
        self.epsilon = epsilon
        self.cutoff = cutoff
        self.minSize = minSize
        self.maxSize = maxSize
        self.na = na
        self.q = q
        self.coeffs = []
        for index in range(0, q+1):
            first_expr = np.power(-1, index+1) / (factorial2(
                2*q - 2*index, exact=True) * factorial2(2*index, exact=True))
            second_expr = factorial2(10+2*q, exact=True) / (factorial2(
                8, exact=True) * (10+2*index))
            third_expr = np.power(cutoff, -(10+2*index))
            self.coeffs.append(first_expr * second_expr * third_expr)

    def __call__(self, r, ijsize):
        """
        Return function value (potential energy)
        """
        ipl = self.epsilon * np.power(ijsize, 10) / np.power(r, 10)
        for l in range(0, self.q+1):
            ipl += self.epsilon * self.coeffs[l] * np.power(r/ijsize, 2*l)

        return ipl

    def mix_sizes(self, isize, jsize):
        """
        Nonadditive interaction rule for the cross size of particles i and j.
        """
        return 0.5 * (isize+jsize) * (1 - self.na * np.absolute(isize-jsize))

    def get_cutoff(self):
        """
        Return the cutoff.
        """
        return self.cutoff

    def get_coeffs(self):
        """
        Return the smoothing coefficients of the potential.
        """
        return self.coeffs

    def get_maxSize(self):
        """
        Return the maximal size of a particle (=Upper boundary of distribution)
        """
        return self.maxSize

    def get_minSize(self):
        """
        Return the minimal size of a particle (=Lower boundary of distribution)
        """
        return self.minSize

    def first_derivative(self, r, ijsize):
        """
        Return first derivative
        """
        dipl = -10 * self.epsilon * np.power(ijsize, 10) / np.power(r, 11)
        for l in range(0, self.q+1):
            dipl += 2*self.epsilon*l * \
                self.coeffs[l] * np.power(r, 2*l-1) / np.power(ijsize, 2*l)

        return dipl

    def second_derivative(self, r, ijsize):
        """
        Return second derivative
        """
        ddipl = 110 * self.epsilon * np.power(ijsize, 10) / np.power(r, 12)
        for l in range(0, self.q+1):
            ddipl += self.epsilon * \
                (4*np.power(l, 2)-2*l) * \
                self.coeffs[l] * np.power(r, 2*l-2) / np.power(ijsize, 2*l)

        return ddipl

    def derivative(self, n=1):
        if n == 1:
            return self.first_derivative
        elif n == 2:
            return self.second_derivative
        else:
            raise ValueError(
                "Don't know how to compute {}-th derivative.".format(n))

###


class Polydisperse(Calculator):
    implemented_properties = ["energy", "stress", "forces"]
    default_parameters = {}
    name = "Polydisperse"

    def __init__(self, f, cutoff=None):
        Calculator.__init__(self)
        self.f = f

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        f = self.f
        nat = len(self.atoms)
        if atoms.has("size"):
            size = self.atoms.get_array("size")
        else:
            raise AttributeError(
                "Attribute error: Unable to load atom sizes from atoms object!")

        i_n, j_n, dr_nc, abs_dr_n = neighbour_list(
            "ijDd", self.atoms, f.get_maxSize()*f.get_cutoff())
        ijsize = f.mix_sizes(size[i_n], size[j_n])

        # Mask neighbour list to consider only true neighbors
        mask = abs_dr_n <= f.get_cutoff() * ijsize
        i_n = i_n[mask]
        j_n = j_n[mask]
        dr_nc = dr_nc[mask]
        abs_dr_n = abs_dr_n[mask]
        ijsize = ijsize[mask]
        e_n = f(abs_dr_n, ijsize)
        de_n = f.first_derivative(abs_dr_n, ijsize)

        # Energy
        epot = 0.5*np.sum(e_n)

        # Forces
        df_nc = 0.5*de_n.reshape(-1, 1)*dr_nc/abs_dr_n.reshape(-1, 1)

        # Sum for each atom
        fx_i = np.bincount(i_n, weights=df_nc[:, 0], minlength=nat) - \
            np.bincount(j_n, weights=df_nc[:, 0], minlength=nat)
        fy_i = np.bincount(i_n, weights=df_nc[:, 1], minlength=nat) - \
            np.bincount(j_n, weights=df_nc[:, 1], minlength=nat)
        fz_i = np.bincount(i_n, weights=df_nc[:, 2], minlength=nat) - \
            np.bincount(j_n, weights=df_nc[:, 2], minlength=nat)

        # Virial
        virial_v = -np.array([dr_nc[:, 0]*df_nc[:, 0],               # xx
                              dr_nc[:, 1]*df_nc[:, 1],               # yy
                              dr_nc[:, 2]*df_nc[:, 2],               # zz
                              dr_nc[:, 1]*df_nc[:, 2],               # yz
                              dr_nc[:, 0]*df_nc[:, 2],               # xz
                              dr_nc[:, 0]*df_nc[:, 1]]).sum(axis=1)  # xy

        self.results = {'energy': epot,
                        'stress': virial_v/self.atoms.get_volume(),
                        'forces': np.transpose([fx_i, fy_i, fz_i])}

    ###

    def hessian_matrix(self, atoms, divide_by_masses=False):
        """
        Calculate the Hessian matrix for a polydisperse systems where atoms interact via a pair potential.
        For an atomic configuration with N atoms in d dimensions the hessian matrix is a symmetric, hermitian matrix
        with a shape of (d*N,d*N). The matrix is due to the cutoff function a sparse matrix, which consists of dense blocks of shape (d,d), which
        are the mixed second derivatives. The result of the derivation for a pair potential can be found in:
        L. Pastewka et. al. "Seamless elastic boundaries for atomistic calculations", Phys. Ev. B 86, 075459 (2012).

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        divide_by_masses: bool
            Divide the block "l,m" by the corresponding atomic masses "sqrt(m_l, m_m)" to obtain dynamical matrix.

        Restrictions
        ----------
        This method is currently only implemented for three dimensional systems

        """

        if self.atoms is None:
            self.atoms = atoms

        f = self.f
        nat = len(self.atoms)
        if atoms.has("size"):
            size = self.atoms.get_array("size")
        else:
            raise AttributeError(
                "Attribute error: Unable to load atom sizes from atoms object! Probably missing size array.")

        i_n, j_n, dr_nc, abs_dr_n = neighbour_list(
            "ijDd", self.atoms, f.get_maxSize()*f.get_cutoff())
        ijsize = f.mix_sizes(size[i_n], size[j_n])

        # Mask neighbour list to consider only true neighbors
        mask = abs_dr_n <= f.get_cutoff()*ijsize
        i_n = i_n[mask]
        j_n = j_n[mask]
        dr_nc = dr_nc[mask]
        abs_dr_n = abs_dr_n[mask]
        ijsize = ijsize[mask]
        first_i = first_neighbours(nat, i_n)

        if divide_by_masses:
            mass_nat = self.atoms.get_masses()
            geom_mean_mass_n = np.sqrt(mass_nat[i_n]*mass_nat[j_n])

        # Hessian 
        de_n = f.first_derivative(abs_dr_n, ijsize)
        dde_n = f.second_derivative(abs_dr_n, ijsize)
        e_nc = (dr_nc.T/abs_dr_n).T
        H_ncc = -(dde_n * (e_nc.reshape(-1, 3, 1)
                           * e_nc.reshape(-1, 1, 3)).T).T
        H_ncc += -(de_n/abs_dr_n * (np.eye(3, dtype=e_nc.dtype)
                                   - (e_nc.reshape(-1, 3, 1) * e_nc.reshape(-1, 1, 3))).T).T

        if divide_by_masses:
            H = bsr_matrix(((H_ncc.T/geom_mean_mass_n).T,
                            j_n, first_i), shape=(3*nat, 3*nat))

        else:
            H = bsr_matrix((H_ncc, j_n, first_i), shape=(3*nat, 3*nat))

        Hdiag_icc = np.empty((nat, 3, 3))
        for x in range(3):
            for y in range(3):
                Hdiag_icc[:, x, y] = - \
                    np.bincount(i_n, weights=H_ncc[:, x, y])

        if divide_by_masses:
            H += bsr_matrix(((Hdiag_icc.T/mass_nat).T, np.arange(nat),
                    np.arange(nat+1)), shape=(3*nat, 3*nat))         

        else:
            H += bsr_matrix((Hdiag_icc, np.arange(nat),
                     np.arange(nat+1)), shape=(3*nat, 3*nat))

        return H
