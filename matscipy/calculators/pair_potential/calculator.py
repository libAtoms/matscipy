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

from abc import ABC, abstractmethod

import numpy as np

from scipy.sparse import bsr_matrix

from ...neighbours import neighbour_list, first_neighbours
from ..calculator import MatscipyCalculator
from ...numpy_tricks import mabincount


class CuttoffInteraction(ABC):
    """Pair interaction potential with cutoff."""

    def __init__(self, cutoff):
        """Initialize with cutoff."""
        self._cutoff = cutoff

    @property
    def cutoff(self):
        """Physical cutoff distance for pair interaction."""
        return self._cutoff

    def get_cutoff(self):
        """Get cutoff. Deprecated."""
        return self.cutoff

    @abstractmethod
    def __call__(self, r, qi, qj):
        """Compute interaction energy."""

    @abstractmethod
    def first_derivative(self, r, qi, qj):
        """Compute derivative w/r to distance."""

    @abstractmethod
    def second_derivative(self, r, qi, qj):
        """Compute second derivative w/r to distance."""

    def derivative(self, n=1):
        """Return specified derivative."""
        if n == 1:
            return self.first_derivative
        elif n == 2:
            return self.second_derivative
        else:
            raise ValueError(
                "Don't know how to compute {}-th derivative.".format(n))


class LennardJonesCut(CuttoffInteraction):
    """
    Functional form for a 12-6 Lennard-Jones potential with a hard cutoff.
    Energy is shifted to zero at cutoff.
    """

    def __init__(self, epsilon, sigma, cutoff):
        super().__init__(cutoff)
        self.epsilon = epsilon
        self.sigma = sigma
        self.offset = (sigma/cutoff)**12 - (sigma/cutoff)**6

    def __call__(self, r, *args):
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((r6-1) * r6 - self.offset)

    def first_derivative(self, r, *args):
        r = (self.sigma / r)
        r6 = r**6
        return -24 * self.epsilon / self.sigma * (2*r6-1) * r6 * r

    def second_derivative(self, r, *args):
        r2 = (self.sigma / r)**2
        r6 = r2**3
        return 24 * self.epsilon/self.sigma**2 * (26*r6-7) * r6 * r2

###


class LennardJonesQuadratic(CuttoffInteraction):
    """
    Functional form for a 12-6 Lennard-Jones potential with a soft cutoff.
    Energy, its first and second derivative are shifted to zero at cutoff.
    """

    def __init__(self, epsilon, sigma, cutoff):
        super().__init__(cutoff)
        self.epsilon = epsilon
        self.sigma = sigma
        self.offset_energy = (sigma/cutoff)**12 - (sigma/cutoff)**6
        self.offset_force = 6/cutoff * \
            (-2*(sigma/cutoff)**12+(sigma/cutoff)**6)
        self.offset_dforce = (1/cutoff**2) * \
            (156*(sigma/cutoff)**12-42*(sigma/cutoff)**6)

    def __call__(self, r, *args):
        """
        Return function value (potential energy).
        """
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((r6-1)*r6-self.offset_energy - (r-self.cutoff) * self.offset_force - ((r - self.cutoff)**2/2) * self.offset_dforce)

    def first_derivative(self, r, *args):
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((6/r) * (-2*r6+1) * r6 - self.offset_force - (r-self.cutoff) * self.offset_dforce)

    def second_derivative(self, r, *args):
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((1/r**2) * (156*r6-42) * r6 - self.offset_dforce)

###


class LennardJonesLinear(CuttoffInteraction):
    """
    Function form of a 12-6 Lennard-Jones potential with a soft cutoff
    The energy and the force are shifted at the cutoff.
    """

    def __init__(self, epsilon, sigma, cutoff):
        super().__init__(cutoff)
        self.epsilon = epsilon
        self.sigma = sigma
        self.offset_energy = (sigma/cutoff)**12 - (sigma/cutoff)**6
        self.offset_force = 6/cutoff * \
            (-2*(sigma/cutoff)**12+(sigma/cutoff)**6)

    def __call__(self, r, *args):
        """
        Return function value (potential energy).
        """
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((r6-1) * r6 - self.offset_energy - (r-self.cutoff) * self.offset_force)

    def first_derivative(self, r, *args):
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((6/r) * (-2*r6+1) * r6 - self.offset_force)

    def second_derivative(self, r, *args):
        r6 = (self.sigma / r)**6
        return 4 * self.epsilon * ((1/r**2) * (156*r6-42) * r6)


###

class FeneLJCut(LennardJonesCut):
    """
    Finite extensible nonlinear elastic(FENE) potential for a bead-spring polymer model.
    For the Lennard-Jones interaction a LJ-cut potential is used. Due to choice of the cutoff (rc=2^(1/6) sigma)
    it ensures a continous potential and force at the cutoff.
    """

    def __init__(self, K, R0, epsilon, sigma):
        super().__init__(2**(1/6) * sigma)
        self.K = K
        self.R0 = R0
        self.epsilon = epsilon
        self.sigma = sigma

    def __call__(self, r, *args):
        return (-0.5 * self.K * self.R0**2 * np.log(1-(r/self.R0)**2)
                + super().__call__(r))

    def first_derivative(self, r, *args):
        return (self.K * r / (1-(r/self.R0)**2)
                + super().first_derivative(r))

    def second_derivative(self, r, *args):
        invLength = 1 / (1-(r/self.R0)**2)
        return (self.K * invLength
                + 2 * self.K * r**2 * invLength**2 / self.R0**2
                + super().second_derivative(r))


###

class LennardJones84(CuttoffInteraction):
    """
    Function form of a 8-4 Lennard-Jones potential, used to model the structure of a CuZr.
    Kobayashi, Shinji et. al. "Computer simulation of atomic structure of Cu57Zr43 amorphous alloy."
    Journal of the Physical Society of Japan 48.4 (1980): 1147-1152.
    """

    def __init__(self, C1, C2, C3, C4, cutoff):
        super().__init__(cutoff)
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4

    def __call__(self, r, *args):
        r4 = (1 / r)**4
        return (self.C2*r4-self.C1) * r4 + self.C3 * r + self.C4

    def first_derivative(self, r, *args):
        r4 = (1 / r)**4
        return (-8 * self.C2*r4/r+4*self.C1/r) * r4 + self.C3

    def second_derivative(self, r, *args):
        r4 = (1 / r)**4
        return (72 * self.C2 * r4 / r**2 - 20 * self.C1 / r**2) * r4

###
_c = np.s_[..., np.newaxis]


class PairPotential(MatscipyCalculator):
    implemented_properties = ['energy', 'free_energy',
                              'stress', 'forces', 'hessian']
    default_parameters = {}
    name = 'PairPotential'

    def __init__(self, f, cutoff=None):
        MatscipyCalculator.__init__(self)
        self.f = f

        self.dict = {x: obj.cutoff for x, obj in f.items()}
        self.df = {x: obj.derivative(1) for x, obj in f.items()}
        self.df2 = {x: obj.derivative(2) for x, obj in f.items()}

    def _mask_pairs(self, i_p, j_p):
        """Iterate over pair masks."""
        numi_p, numj_p = self.atoms.numbers[i_p], self.atoms.numbers[j_p]

        for pair in self.dict:
            mask = (numi_p == pair[0]) & (numj_p == pair[1])

            if pair[0] != pair[1]:
                mask |= (numi_p == pair[1]) & (numj_p == pair[0])

            yield mask, pair


    def calculate(self, atoms, properties, system_changes):
        MatscipyCalculator.calculate(self, atoms, properties, system_changes)

        nb_atoms = len(self.atoms)
        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', self.atoms, self.dict)

        e_p = np.zeros_like(r_p)
        de_p = np.zeros_like(r_p)

        for mask, pair in self._mask_pairs(i_p, j_p):
            e_p[mask] = self.f[pair](r_p[mask])
            de_p[mask] = self.df[pair](r_p[mask])

        epot = 0.5 * np.sum(e_p)

        # Forces
        df_pc = -0.5 * de_p[_c] * r_pc / r_p[_c]

        f_nc = mabincount(j_p, df_pc, nb_atoms) \
            - mabincount(i_p, df_pc, nb_atoms)

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

        for mask, pair in self._mask_pairs(i_p, j_p):
            e_p[mask] = f[pair](r_p[mask])
            de_p[mask] = df[pair](r_p[mask])
            dde_p[mask] = df2[pair](r_p[mask])

        n_pc = r_pc / r_p[_c]
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
