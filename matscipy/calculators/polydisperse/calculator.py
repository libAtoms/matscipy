#
# Copyright 2020-2021 Jan Griesser (U. Freiburg)
#           2020 griesserj@fp-10-126-132-144.eduroam-fp.privat
#           2020 Arnaud Allera (U. Lyon 1)
#           2014 Lars Pastewka (U. Freiburg)
#           2014 James Kermode (Warwick U.)
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

import numpy as np

from scipy.special import factorial2
from scipy.sparse import bsr_matrix

import ase

from ...neighbours import neighbour_list, first_neighbours
from ..calculator import MatscipyCalculator
from ...numpy_tricks import mabincount

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

class Polydisperse(MatscipyCalculator):
    implemented_properties = [
        "energy",
        "free_energy",
        "stress",
        "forces",
        "hessian",
        "dynamical_matrix",
        "nonaffine_forces",
        "birch_coefficients",
        "nonaffine_elastic_contribution",
        "stress_elastic_contribution",
        "born_constants",
        'elastic_constants',
    ]

    default_parameters = {}
    name = "Polydisperse"

    def __init__(self, f, cutoff=None):
        MatscipyCalculator.__init__(self)
        self.f = f
        self.reset()

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)

        f = self.f
        nb_atoms = len(self.atoms)
        if atoms.has("size"):
            size = self.atoms.get_array("size")
        else:
            raise AttributeError(
                "Attribute error: Unable to load atom sizes from atoms object!")

        i_p, j_p, r_pc, r_p = neighbour_list("ijDd", self.atoms, f.get_maxSize()*f.get_cutoff())
        ijsize = f.mix_sizes(size[i_p], size[j_p])

        # Mask neighbour list to consider only true neighbors
        mask = r_p <= f.get_cutoff() * ijsize
        i_p = i_p[mask]
        j_p = j_p[mask]
        r_pc = r_pc[mask]
        r_p = r_p[mask]
        ijsize = ijsize[mask]
        e_p = f(r_p, ijsize)
        de_p = f.first_derivative(r_p, ijsize)

        # Energy
        epot = 0.5*np.sum(e_p)

        # Forces
        df_pc = -0.5*de_p.reshape(-1, 1)*r_pc/r_p.reshape(-1, 1)

        f_nc = mabincount(j_p, df_pc, nb_atoms) - mabincount(i_p, df_pc, nb_atoms)

        # Virial
        virial_v = -np.array([r_pc[:, 0] * df_pc[:, 0],               # xx
                              r_pc[:, 1] * df_pc[:, 1],               # yy
                              r_pc[:, 2] * df_pc[:, 2],               # zz
                              r_pc[:, 1] * df_pc[:, 2],               # yz
                              r_pc[:, 0] * df_pc[:, 2],               # xz
                              r_pc[:, 0] * df_pc[:, 1]]).sum(axis=1)  # xy

        self.results.update(
            {
                'energy': epot,
                'free_energy': epot,
                'stress': virial_v / self.atoms.get_volume(),
                'forces': f_nc,
            }
        )

    ###

    def get_hessian(self, atoms, format='sparse', divide_by_masses=False):
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

        format: "sparse" or "neighbour-list"
            Output format of the hessian matrix.

        divide_by_masses: bool
            Divide the block "l,m" by the corresponding atomic masses "sqrt(m_l, m_m)" to obtain dynamical matrix.

        Restrictions
        ----------
        This method is currently only implemented for three dimensional systems

        """

        if self.atoms is None:
            self.atoms = atoms

        f = self.f
        nb_atoms = len(self.atoms)
        if atoms.has("size"):
            size = atoms.get_array("size")
        else:
            raise AttributeError(
                "Attribute error: Unable to load atom sizes from atoms object! Probably missing size array.")

        i_p, j_p, r_pc, r_p = neighbour_list("ijDd", self.atoms, f.get_maxSize()*f.get_cutoff())
        ijsize = f.mix_sizes(size[i_p], size[j_p])

        # Mask neighbour list to consider only true neighbors
        mask = r_p <= f.get_cutoff()*ijsize
        i_p = i_p[mask]
        j_p = j_p[mask]
        r_pc = r_pc[mask]
        r_p = r_p[mask]
        ijsize = ijsize[mask]
        first_i = first_neighbours(nb_atoms, i_p)

        if divide_by_masses:
            mass_n = atoms.get_masses()
            geom_mean_mass_p = np.sqrt(mass_n[i_p]*mass_n[j_p])

        # Hessian 
        de_p = f.first_derivative(r_p, ijsize)
        dde_p = f.second_derivative(r_p, ijsize)
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

        # Neighbour list format
        elif format == "neighbour-list":
            return H_pcc, i_p, j_p, r_pc, r_p
