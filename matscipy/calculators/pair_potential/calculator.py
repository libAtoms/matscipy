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

import ase

from ...neighbours import neighbour_list, first_neighbours
from ..calculator import MatscipyCalculator
from ...numpy_tricks import mabincount


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
        self.offset = (sigma/cutoff)**12 - (sigma/cutoff)**6

    def __call__(self, r):
        """
        Return function value (potential energy).
        """
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((r6-1) * r6 - self.offset)

    def get_cutoff(self):
        return self.cutoff

    def first_derivative(self, r):
        r = (self.sigma / r)
        r6 = r**6
        return -24 * self.epsilon / self.sigma * (2*r6-1) * r6 * r

    def second_derivative(self, r):
        r2 = (self.sigma / r)**2
        r6 = r2**3
        return 24 * self.epsilon/self.sigma**2 * (26*r6-7) * r6 * r2

    def derivative(self, n=1):
        if n == 1:
            return self.first_derivative
        elif n == 2:
            return self.second_derivative
        else:
            raise ValueError(
                "Don't know how to compute {}-th derivative.".format(n))

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
        self.offset_energy = (sigma/cutoff)**12 - (sigma/cutoff)**6
        self.offset_force = 6/cutoff * \
            (-2*(sigma/cutoff)**12+(sigma/cutoff)**6)
        self.offset_dforce = (1/cutoff**2) * \
            (156*(sigma/cutoff)**12-42*(sigma/cutoff)**6)

    def get_cutoff(self):
        return self.cutoff

    def __call__(self, r):
        """
        Return function value (potential energy).
        """
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((r6-1)*r6-self.offset_energy - (r-self.cutoff) * self.offset_force - ((r - self.cutoff)**2/2) * self.offset_dforce)

    def first_derivative(self, r):
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((6/r) * (-2*r6+1) * r6 - self.offset_force - (r-self.cutoff) * self.offset_dforce)

    def second_derivative(self, r):
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((1/r**2) * (156*r6-42) * r6 - self.offset_dforce)

    def derivative(self, n=1):
        if n == 1:
            return self.first_derivative
        elif n == 2:
            return self.second_derivative
        else:
            raise ValueError(
                "Don't know how to compute {}-th derivative.".format(n))

###


class LennardJonesLinear():
    """
    Function form of a 12-6 Lennard-Jones potential with a soft cutoff
    The energy and the force are shifted at the cutoff.
    """

    def __init__(self, epsilon, sigma, cutoff):
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.offset_energy = (sigma/cutoff)**12 - (sigma/cutoff)**6
        self.offset_force = 6/cutoff * \
            (-2*(sigma/cutoff)**12+(sigma/cutoff)**6)

    def get_cutoff(self):
        return self.cutoff

    def __call__(self, r):
        """
        Return function value (potential energy).
        """
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((r6-1) * r6 - self.offset_energy - (r-self.cutoff) * self.offset_force)

    def first_derivative(self, r):
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((6/r) * (-2*r6+1) * r6 - self.offset_force)

    def second_derivative(self, r):
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((1/r**2) * (156*r6-42) * r6)

    def derivative(self, n=1):
        if n == 1:
            return self.first_derivative
        elif n == 2:
            return self.second_derivative
        else:
            raise ValueError(
                "Don't know how to compute {}-th derivative.".format(n))


###

class FeneLJCut():
    """
    Finite extensible nonlinear elastic(FENE) potential for a bead-spring polymer model.
    For the Lennard-Jones interaction a LJ-cut potential is used. Due to choice of the cutoff (rc=2^(1/6) sigma)
    it ensures a continous potential and force at the cutoff.
    """

    def __init__(self, K, R0, epsilon, sigma):
        self.K = K
        self.R0 = R0
        self.epsilon = epsilon
        self.sigma = sigma

    def __call__(self, r):
        """
        Return function value (potential energy).
        """
        r6 = (self.sigma/r)**6
        bond = -0.5 * self.K * self.R0**2 * np.log(1-(r/self.R0)**2)
        lj = 4 * self.epsilon * (r6-1) * r6 + self.epsilon
        return bond + lj

    def first_derivative(self, r):
        r6 = (self.sigma/r)**6
        bond = self.K * r / (1-(r/self.R0)**2)
        lj = -24 * self.epsilon * (2*r6/r-1/r) * r6
        return bond + lj

    def second_derivative(self, r):
        r6 = (self.sigma/r)**6
        invLength = 1 / (1-(r/self.R0)**2)
        bond = K * invLength + 2 * K * r**2 * invLength**2 / self.R0**2
        lj = 4 * self.epsilon * ((1/r**2) * (156*r6-42) * r6)
        return bond + lj

    def derivative(self, n=1):
        if n == 1:
            return self.first_derivative
        elif n == 2:
            return self.second_derivative
        else:
            raise ValueError(
                "Don't know how to compute {}-th derivative.".format(n))


###

class LennardJones84():
    """
    Function form of a 8-4 Lennard-Jones potential, used to model the structure of a CuZr.
    Kobayashi, Shinji et. al. "Computer simulation of atomic structure of Cu57Zr43 amorphous alloy."
    Journal of the Physical Society of Japan 48.4 (1980): 1147-1152.
    """

    def __init__(self, C1, C2, C3, C4, cutoff):
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.cutoff = cutoff

    def get_cutoff():
        return self.cutoff

    def __call__(self, r):
        """
        Return function value (potential energy).
        """
        r4 = (1 / r)**4
        return (self.C2*r4-self.C1) * r4 + self.C3 * r + self.C4

    def first_derivative(self, r):
        r4 = (1 / r)**4
        return (-8 * self.C2*r4/r+4*self.C1/r) * r4 + self.C3

    def second_derivative(self, r):
        r4 = (1 / r)**4
        return (72 * self.C2 * r4 / r**2 - 20 * self.C1 / r**2) * r4

    def derivative(self, n=1):
        if n == 1:
            return self.first_derivative
        elif n == 2:
            return self.second_derivative
        else:
            raise ValueError(
                "Don't know how to compute {}-th derivative.".format(n))

###


class PairPotential(MatscipyCalculator):
    implemented_properties = ['energy', 'free_energy', 'stress', 'forces', 'hessian']
    default_parameters = {}
    name = 'PairPotential'

    def __init__(self, f, cutoff=None):
        MatscipyCalculator.__init__(self)
        self.f = f

        self.dict = {x: obj.get_cutoff() for x, obj in f.items()}
        self.df = {x: obj.derivative(1) for x, obj in f.items()}
        self.df2 = {x: obj.derivative(2) for x, obj in f.items()}

    def calculate(self, atoms, properties, system_changes):
        MatscipyCalculator.calculate(self, atoms, properties, system_changes)

        nb_atoms = len(self.atoms)
        atnums = self.atoms.numbers
        atnums_in_system = set(atnums)

        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', self.atoms, self.dict)

        e_p = np.zeros_like(r_p)
        de_p = np.zeros_like(r_p)
        for params, pair in enumerate(self.dict):
            if pair[0] == pair[1]:
                mask1 = atnums[i_p] == pair[0]
                mask2 = atnums[j_p] == pair[0]
                mask = np.logical_and(mask1, mask2)

                e_p[mask] = self.f[pair](r_p[mask])
                de_p[mask] = self.df[pair](r_p[mask])

            if pair[0] != pair[1]:
                mask1 = np.logical_and(
                    atnums[i_p] == pair[0], atnums[j_p] == pair[1])
                mask2 = np.logical_and(
                    atnums[i_p] == pair[1], atnums[j_p] == pair[0])
                mask = np.logical_or(mask1, mask2)

                e_p[mask] = self.f[pair](r_p[mask])
                de_p[mask] = self.df[pair](r_p[mask])

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
        dict = self.dict
        df = self.df
        df2 = self.df2

        nb_atoms = len(atoms)
        atnums = atoms.numbers

        i_p, j_p,  r_p, r_pc = neighbour_list('ijdD', atoms, dict)
        first_i = first_neighbours(nb_atoms, i_p)

        e_p = np.zeros_like(r_p)
        de_p = np.zeros_like(r_p)
        dde_p = np.zeros_like(r_p)
        for params, pair in enumerate(dict):
            if pair[0] == pair[1]:
                mask1 = atnums[i_p] == pair[0]
                mask2 = atnums[j_p] == pair[0]
                mask = np.logical_and(mask1, mask2)

                e_p[mask] = f[pair](r_p[mask])
                de_p[mask] = df[pair](r_p[mask])
                dde_p[mask] = df2[pair](r_p[mask])

            if pair[0] != pair[1]:
                mask1 = np.logical_and(
                    atnums[i_p] == pair[0], atnums[j_p] == pair[1])
                mask2 = np.logical_and(
                    atnums[i_p] == pair[1], atnums[j_p] == pair[0])
                mask = np.logical_or(mask1, mask2)

                e_p[mask] = f[pair](r_p[mask])
                de_p[mask] = df[pair](r_p[mask])
                dde_p[mask] = df2[pair](r_p[mask])
        
        n_pc = (r_pc.T/r_p).T
        H_pcc = -(dde_p * (n_pc.reshape(-1, 3, 1)
                           * n_pc.reshape(-1, 1, 3)).T).T
        H_pcc += -(de_p/r_p * (np.eye(3, dtype=n_pc.dtype)
                                    - (n_pc.reshape(-1, 3, 1) * n_pc.reshape(-1, 1, 3))).T).T

        # Sparse BSR-matrix
        if format == "sparse":
            if divide_by_masses:
                masses_n = self.atoms.get_masses()
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
                masses_p = (self.atoms.get_masses()).repeat(3)
                H /= np.sqrt(masses_p.reshape(-1,1)*masses_p.reshape(1,-1))
                return H

            else:
                return H

        # Neighbour list format
        elif format == "neighbour-list":
            return H_pcc, i_p, j_p, r_pc, r_p
