#
# Copyright 2021 Lars Pastewka (U. Freiburg)
#           2021 Jan Griesser (U. Freiburg)
#           2020-2021 Jonas Oldenstaedt (U. Freiburg)
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
Generic Many body potential.
"""

#
# Coding convention
# * All numpy arrays are suffixed with the array dimensions
# * The suffix stands for a certain type of dimension:
#   - n: Atomic index, i.e. array dimension of length nb_atoms
#   - p: Pair index, i.e. array dimension of length nb_pairs
#   - t: Triplet index, i.e. array dimension of length nb_triplets
#   - q: Index in n-uplet, i.e. pair or triplet (length # elements in n-uplet)
#   - c: Cartesian index, array dimension of length 3
#   - a: Cartesian index for the first dimension of the deformation gradient,
#        array dimension of length 3
#   - b: Cartesian index for the second dimension of the deformation gradient,
#        array dimension of length 3
#

from itertools import product, starmap
from abc import ABC, abstractmethod
import numpy as np

from scipy.sparse.linalg import cg
from scipy.sparse import bsr_matrix

from ase.calculators.calculator import Calculator
from ase.geometry import find_mic

from ...calculators.calculator import MatscipyCalculator
from ...elasticity import Voigt_6_to_full_3x3_stress
from ...neighbours import (
    find_indices_of_reversed_pairs,
    first_neighbours,
    Neighbourhood,
    CutoffNeighbourhood
)
from ...numpy_tricks import mabincount


def _o(x, y, z=None):
    """Outer product."""
    if z is None:
        return np.einsum('...i,...j', x, y)
    return np.einsum('...i,...j,...k', x, y, z)


# broadcast slices
_c, _cc = np.s_[..., np.newaxis], np.s_[..., np.newaxis, np.newaxis]


class Manybody(MatscipyCalculator):
    implemented_properties = [
        'free_energy',
        'energy',
        'stress',
        'forces',
        'hessian',
        'nonaffine_forces',
        'birch_coefficients',
        'dynamical_matrix',
        'elastic_constants',
    ]

    default_parameters = {}
    name = 'Manybody'

    def __init__(self, atom_type, pair_type,
                 F, G,
                 d1F, d2F,
                 d11F, d22F, d12F,
                 d1G, d11G, d2G, d22G, d12G,
                 cutoff, neighbourhood: Neighbourhood = None):
        super().__init__()
        self.atom_type = atom_type
        self.pair_type = pair_type
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

        if neighbourhood is None:
            self.neighbourhood = CutoffNeighbourhood(atom_types=atom_type,
                                                     pair_types=pair_type)
        else:
            self.neighbourhood = neighbourhood

        if not isinstance(self.neighbourhood, Neighbourhood):
            raise TypeError(
                f"{self.neighbourhood} is not of type Neighbourhood")

    def get_cutoff(self, atoms):
        """Return a valid cutoff."""
        if np.isscalar(self.cutoff):
            return self.cutoff

        # Get largest cutoff for all possible pair types
        atom_types = map(self.atom_type, np.unique(atoms.numbers))
        pair_types = starmap(self.pair_type, product(atom_types, repeat=2))
        return max(self.cutoff[p] for p in pair_types)

    def calculate(self, atoms, properties, system_changes):
        """Calculate system properties."""
        super().calculate(atoms, properties, system_changes)

        # Setting up cutoffs for neighbourhood
        self.neighbourhood.cutoff = self.get_cutoff(atoms)  # no-op on mols

        # get internal atom types from atomic numbers
        t_n = self.neighbourhood.atom_type(atoms.numbers)

        # construct neighbor list
        i_p, j_p, r_p, r_pc = self.neighbourhood.get_pairs(atoms, "ijdD")

        nb_atoms = len(self.atoms)
        nb_pairs = len(i_p)

        # normal vectors
        n_pc = r_pc / r_p[_c]

        # construct triplet list
        ij_t, ik_t, r_tq, r_tqc = self.neighbourhood.get_triplets(atoms,
                                                                  "ijdD")

        # construct lists with atom and pair types
        ti_p = t_n[i_p]
        tij_p = self.neighbourhood.pair_type(ti_p, t_n[j_p])
        ti_t = t_n[i_p[ij_t]]
        tij_t = self.neighbourhood.pair_type(ti_t, t_n[j_p[ij_t]])
        tik_t = self.neighbourhood.pair_type(ti_t, t_n[j_p[ik_t]])

        # potential-dependent functions
        G_t = self.G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d1G_tc = self.d1G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d2G_tc = self.d2G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)

        xi_p = np.bincount(ij_t, weights=G_t, minlength=nb_pairs)

        F_p = self.F(r_p, xi_p, ti_p, tij_p)
        d1F_p = self.d1F(r_p, xi_p, ti_p, tij_p)
        d2F_p = self.d2F(r_p, xi_p, ti_p, tij_p)
        d2F_d2G_t = d2F_p[ij_t][_c] * d2G_tc

        # calculate energy
        epot = 0.5 * np.sum(F_p)

        # calculate forces (per pair)
        f_pc = (d1F_p[_c] * n_pc
                + d2F_p[_c] * mabincount(ij_t, d1G_tc, nb_pairs)
                + mabincount(ik_t, d2F_d2G_t, nb_pairs))

        # collect atomic forces
        f_nc = 0.5 * (mabincount(i_p, f_pc, nb_atoms)
                      - mabincount(j_p, f_pc, nb_atoms))

        # Virial
        virial_v = np.concatenate([
            # diagonal components (xx, yy, zz)
            np.einsum('pi,pi->i', r_pc, f_pc),

            # off-diagonal (yz, xz, xy)
            np.einsum('pi,pi->i', r_pc[:, (1, 0, 0)], f_pc[:, (2, 2, 1)])
        ])

        virial_v *= 0.5

        self.results.update({'free_energy': epot,
                             'energy': epot,
                             'stress': virial_v / atoms.get_volume(),
                             'forces': f_nc})

    def get_hessian(self, atoms, format='sparse', divide_by_masses=False):
        """Calculate the Hessian matrix for a generic many-body potential.

        For an atomic configuration with N atoms in d dimensions the hessian
        matrix is a symmetric, hermitian matrix with a shape of (d*N,d*N). The
        matrix is in general a sparse matrix, which consists of dense blocks of
        shape (d,d), which are the mixed second derivatives.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic configuration in a local or global minima.

        format : "sparse" or "neighbour-list"
            Output format of the hessian matrix.

        divide_by_masses : bool
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

        # get internal atom types from atomic numbers
        t_n = self.neighbourhood.atom_type(atoms.numbers)
        cutoff = self.get_cutoff(atoms)
        self.neighbourhood.cutoff = 2 * cutoff

        # construct neighbor list
        i_p, j_p, r_p, r_pc = self.neighbourhood.get_pairs(atoms, "ijdD")

        mask_p = r_p > cutoff

        nb_atoms = len(self.atoms)
        nb_pairs = len(i_p)

        # reverse pairs
        tr_p = find_indices_of_reversed_pairs(i_p, j_p, r_p)

        # normal vectors
        n_pc = r_pc / r_p[_c]

        # construct triplet list (need jk_t here, neighbor must be to 2*cutoff)
        self.neighbourhood.cutoff = cutoff
        ij_t, ik_t, jk_t, r_tq, r_tqc = \
            self.neighbourhood.get_triplets(atoms, "ijkdD",
                                            neighbours=[i_p, j_p, r_p, r_pc])
        first_n = first_neighbours(nb_atoms, i_p)
        first_p = first_neighbours(len(i_p), ij_t)
        nb_triplets = len(ij_t)

        # construct lists with atom and pair types
        ti_p = t_n[i_p]
        tij_p = self.neighbourhood.pair_type(ti_p, t_n[j_p])
        ti_t = t_n[i_p[ij_t]]
        tij_t = self.neighbourhood.pair_type(ti_t, t_n[j_p[ij_t]])
        tik_t = self.neighbourhood.pair_type(ti_t, t_n[j_p[ik_t]])

        # potential-dependent functions
        G_t = self.G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d1G_tc = self.d1G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d2G_tc = self.d2G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d11G_tcc = self.d11G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d12G_tcc = self.d12G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d22G_tcc = self.d22G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)

        xi_p = np.bincount(ij_t, weights=G_t, minlength=nb_pairs)

        d1F_p = self.d1F(r_p, xi_p, ti_p, tij_p)
        d1F_p[mask_p] = 0.0  # explicitly exclude everything with r > cutoff
        d2F_p = self.d2F(r_p, xi_p, ti_p, tij_p)
        d2F_p[mask_p] = 0.0
        d11F_p = self.d11F(r_p, xi_p, ti_p, tij_p)
        d11F_p[mask_p] = 0.0
        d12F_p = self.d12F(r_p, xi_p, ti_p, tij_p)
        d12F_p[mask_p] = 0.0
        d22F_p = self.d22F(r_p, xi_p, ti_p, tij_p)
        d22F_p[mask_p] = 0.0

        # normal vectors for triplets
        n_tqc = r_tqc / r_tq[_c]

        # Hessian term #4
        nn_pcc = _o(n_pc, n_pc)
        H_pcc = -(d1F_p[_cc] * (np.eye(3) - nn_pcc) / r_p[_cc])

        # Hessian term #1
        H_pcc -= d11F_p[_cc] * nn_pcc

        # Hessian term #2
        H_temp3_t = d12F_p[ij_t][_cc] * _o(d2G_tc, n_tqc[:, 0])
        H_temp4_t = d12F_p[ij_t][_cc] * _o(d1G_tc, n_tqc[:, 0])

        # Hessian term #5
        H_temp2_t = d2F_p[ij_t][_cc] * d22G_tcc
        H_temp_t = d2F_p[ij_t][_cc] * d11G_tcc
        H_temp1_t = d2F_p[ij_t][_cc] * d12G_tcc

        # Hessian term #3

        # Terms involving D_1 * D_1
        d1G_pc = mabincount(ij_t, d1G_tc, nb_pairs)
        H_pcc -= d22F_p[_cc] * _o(d1G_pc, d1G_pc)

        # Terms involving D_2 * D_2
        d2G_pc = mabincount(ij_t, d2G_tc, nb_pairs)
        Q1 = _o((d22F_p[_c] * d2G_pc)[ij_t], d2G_tc)

        # Terms involving D_1 * D_2
        Q2 = _o((d22F_p[_c] * d1G_pc)[ij_t], d2G_tc)

        H_pcc -= d22F_p[_cc] * _o(d2G_pc, d1G_pc)

        H_pcc += \
            - mabincount(ij_t, weights=H_temp_t, minlength=nb_pairs) \
            + mabincount(jk_t, weights=H_temp1_t, minlength=nb_pairs) \
            - mabincount(tr_p[ij_t], weights=H_temp1_t, minlength=nb_pairs) \
            - mabincount(ik_t, weights=H_temp1_t, minlength=nb_pairs) \
            - mabincount(ik_t, weights=H_temp2_t, minlength=nb_pairs) \
            + mabincount(tr_p[jk_t], weights=H_temp3_t, minlength=nb_pairs) \
            - mabincount(ij_t, weights=H_temp3_t, minlength=nb_pairs) \
            - mabincount(tr_p[ik_t], weights=H_temp3_t, minlength=nb_pairs) \
            - mabincount(ij_t, weights=H_temp4_t, minlength=nb_pairs) \
            - mabincount(tr_p[ij_t], weights=H_temp4_t, minlength=nb_pairs) \
            - mabincount(ik_t, weights=Q1, minlength=nb_pairs) \
            + mabincount(jk_t, weights=Q2, minlength=nb_pairs) \
            - mabincount(ik_t, weights=Q2, minlength=nb_pairs)

        for il_im in range(nb_triplets):
            il = ij_t[il_im]
            im = ik_t[il_im]
            lm = jk_t[il_im]
            ti = ti_t[il_im]
            tij = tij_t[il_im]
            tim = tik_t[il_im]
            til = tij_t[il_im]
            for t in range(first_p[il], first_p[il + 1]):
                ij = ik_t[t]
                if ij != il and ij != im:
                    r_p_ij = r_pc[ij][np.newaxis, :]
                    r_p_il = r_pc[il][np.newaxis, :]
                    r_p_im = r_pc[im][np.newaxis, :]
                    H_pcc[lm, :, :] += np.squeeze(np.transpose(
                        0.5 * d22F_p[ij]
                        * (_o(self.d2G(r_p_ij, r_p_il, ti, tij, til),
                              self.d2G(r_p_ij, r_p_im, ti, tij, tim))).T))

        # Add the conjugate terms (symmetrize Hessian)
        H_pcc += H_pcc.transpose(0, 2, 1)[tr_p]

        if format == "sparse":
            # Construct full diagonal terms from off-diagonal terms
            H_acc = np.zeros([nb_atoms, 3, 3])
            for x in range(3):
                for y in range(3):
                    H_acc[:, x, y] = -np.bincount(i_p, weights=H_pcc[:, x, y],
                                                  minlength=nb_atoms)

            if divide_by_masses:
                mass_nat = atoms.get_masses()
                geom_mean_mass_n = np.sqrt(mass_nat[i_p] * mass_nat[j_p])
                return \
                    bsr_matrix(((H_pcc.T / (2 * geom_mean_mass_n)).T, j_p, first_n),
                               shape=(3 * nb_atoms, 3 * nb_atoms)) \
                    + bsr_matrix(((H_acc.T / (2 * mass_nat)).T, np.arange(nb_atoms),
                                  np.arange(nb_atoms + 1)),
                                 shape=(3 * nb_atoms, 3 * nb_atoms))
            else:
                return \
                    bsr_matrix((H_pcc / 2, j_p, first_n), shape=(3 * nb_atoms, 3 * nb_atoms)) \
                    + bsr_matrix((H_acc / 2, np.arange(nb_atoms), np.arange(nb_atoms + 1)),
                                 shape=(3 * nb_atoms, 3 * nb_atoms))

        # Neighbour list format
        elif format == "neighbour-list":
            return H_pcc / 2, i_p, j_p, r_pc, r_p

    def get_second_derivative(self, atoms, drda_pc, drdb_pc,
                              i_p=None, j_p=None, r_p=None, r_pc=None):
        """Calculate the second derivative of the energy with respect to
        arbitrary variables a and b.

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

        # get internal atom types from atomic numbers
        t_n = self.neighbourhood.atom_type(atoms.numbers)
        cutoff = self.get_cutoff(atoms)
        self.neighbourhood.cutoff = cutoff

        if i_p is None or j_p is None or r_p is None or r_pc is None:
            # We need to construct the neighbor list ourselves
            i_p, j_p, r_p, r_pc = self.neighbourhood.get_pairs(atoms, "ijdD")

        nb_pairs = len(i_p)

        # normal vectors
        n_pc = r_pc / r_p[_c]

        # derivative of the lengths of distance vectors
        drda_p = np.einsum('...i,...i', n_pc, drda_pc)
        drdb_p = np.einsum('...i,...i', n_pc, drdb_pc)

        # construct triplet list (no jk_t here, hence 1x cutoff suffices)
        ij_t, ik_t, r_tq, r_tqc = \
            self.neighbourhood.get_triplets(atoms, "ijdD",
                                            neighbours=[i_p, j_p, r_p, r_pc])

        # construct lists with atom and pair types
        ti_p = t_n[i_p]
        tij_p = self.neighbourhood.pair_type(ti_p, t_n[j_p])
        ti_t = t_n[i_p[ij_t]]
        tij_t = self.neighbourhood.pair_type(ti_t, t_n[j_p[ij_t]])
        tik_t = self.neighbourhood.pair_type(ti_t, t_n[j_p[ik_t]])

        # potential-dependent functions
        G_t = self.G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d1G_tc = self.d1G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d2G_tc = self.d2G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d11G_tcc = self.d11G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d12G_tcc = self.d12G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d22G_tcc = self.d22G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)

        xi_p = np.bincount(ij_t, weights=G_t, minlength=nb_pairs)

        d1F_p = self.d1F(r_p, xi_p, ti_p, tij_p)
        d2F_p = self.d2F(r_p, xi_p, ti_p, tij_p)
        d11F_p = self.d11F(r_p, xi_p, ti_p, tij_p)
        d12F_p = self.d12F(r_p, xi_p, ti_p, tij_p)
        d22F_p = self.d22F(r_p, xi_p, ti_p, tij_p)

        # Term 1
        T1 = np.einsum('i,i,i', d11F_p, drda_p, drdb_p)

        # Term 2
        T2_path = np.einsum_path('p,pc,pc,p', d12F_p[ij_t], d2G_tc,
                                 drda_pc[ik_t], drdb_p[ij_t])[0]
        T2 = np.einsum('p,pc,pc,p', d12F_p[ij_t], d2G_tc,
                       drda_pc[ik_t], drdb_p[ij_t], optimize=T2_path)
        T2 += np.einsum('p,pc,pc,p', d12F_p[ij_t], d2G_tc,
                        drdb_pc[ik_t], drda_p[ij_t], optimize=T2_path)
        T2 += np.einsum('p,pc,pc,p', d12F_p[ij_t], d1G_tc,
                        drda_pc[ij_t], drdb_p[ij_t], optimize=T2_path)
        T2 += np.einsum('p,pc,pc,p', d12F_p[ij_t], d1G_tc,
                        drdb_pc[ij_t], drda_p[ij_t], optimize=T2_path)

        # Term 3
        contract = lambda x, y: np.einsum('...j,...j', x, y)
        dxida_t = \
            contract(d1G_tc, drda_pc[ij_t]) + contract(d2G_tc, drda_pc[ik_t])
        dxidb_t = \
            contract(d1G_tc, drdb_pc[ij_t]) + contract(d2G_tc, drdb_pc[ik_t])
        T3 = np.einsum(
            'i,i,i',
            d22F_p.flatten(),
            np.bincount(ij_t, weights=dxida_t, minlength=nb_pairs).flatten(),
            np.bincount(ij_t, weights=dxidb_t, minlength=nb_pairs).flatten())

        # Term 4
        Q_pcc = (np.eye(3) - _o(n_pc, n_pc)) / r_p[_cc]

        T4 = np.einsum('p,pij,pi,pj', d1F_p, Q_pcc, drda_pc, drdb_pc)

        # Term 5
        T5_t = np.einsum('tij,ti,tj->t', d11G_tcc, drdb_pc[ij_t], drda_pc[ij_t])
        T5_t += np.einsum('tij,tj,ti->t', d12G_tcc, drdb_pc[ik_t], drda_pc[ij_t])
        T5_t += np.einsum('tij,ti,tj->t', d12G_tcc, drdb_pc[ij_t], drda_pc[ik_t])
        T5_t += np.einsum('tij,ti,tj->t', d22G_tcc, drdb_pc[ik_t], drda_pc[ik_t])
        T5 = contract(d2F_p.flatten(),
                      np.bincount(ij_t, weights=T5_t, minlength=nb_pairs)).sum()

        return T1 + T2 + T3 + T4 + T5

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

        self.neighbourhood.cutoff = 2 * self.get_cutoff(atoms)
        i_p, j_p, r_p, r_pc = self.neighbourhood.get_pairs(atoms, "ijdD")

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
                        H_ab[3 * m + cm, 3 * l + cl] = \
                            self.get_second_derivative(atoms, drda_pc, drdb_pc,
                                                       i_p=i_p, j_p=j_p,
                                                       r_p=r_p, r_pc=r_pc)

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

        self.neighbourhood.cutoff = 2 * self.get_cutoff(atoms)
        i_p, j_p, r_p, r_pc = self.neighbourhood.get_pairs(atoms, "ijdD")

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
                        naF_ncab[m, cm, alpha, beta] = \
                            self.get_second_derivative(atoms, drda_pc, drdb_pc, i_p=i_p, j_p=j_p, r_p=r_p, r_pc=r_pc)
        return naF_ncab / 2

    def get_born_elastic_constants(self, atoms):
        """
        Compute the Born elastic constants.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """

        if self.atoms is None:
            self.atoms = atoms

        self.neighbourhood.cutoff = 2 * self.get_cutoff(atoms)
        i_p, j_p, r_p, r_pc = self.neighbourhood.get_pairs(atoms, "ijdD")

        nb_pairs = len(i_p)

        C_abab = np.zeros((3, 3, 3, 3))

        for alpha in range(3):
            for beta in range(3):
                drda_pc = np.zeros((nb_pairs, 3))
                drda_pc[:, alpha] = r_pc[:, beta] / 2
                drda_pc[:, beta] += r_pc[:, alpha] / 2
                for nu in range(3):
                    for mu in range(3):
                        drdb_pc = np.zeros((nb_pairs, 3))
                        drdb_pc[:, nu] = r_pc[:, mu] / 2
                        drdb_pc[:, mu] += r_pc[:, nu] / 2
                        C_abab[alpha, beta, nu, mu] = \
                            self.get_second_derivative(atoms, drda_pc, drdb_pc, i_p=i_p, j_p=j_p, r_p=r_p, r_pc=r_pc)

        C_abab /= (2 * atoms.get_volume())

        return C_abab

    def get_nonaffine_forces(self, atoms):
        if self.atoms is None:
            self.atoms = atoms

        # get internal atom types from atomic numbers
        t_n = self.neighbourhood.atom_type(atoms.numbers)
        self.neighbourhood.cutoff = self.get_cutoff(atoms)

        # construct neighbor list
        i_p, j_p, r_p, r_pc = self.neighbourhood.get_pairs(atoms, 'ijdD')

        nb_atoms = len(self.atoms)
        nb_pairs = len(i_p)

        # normal vectors
        n_pc = r_pc / r_p[_c]
        dn_pcc = (np.eye(3) - _o(n_pc, n_pc)) / r_p[_cc]

        # construct triplet list (no jk_t here, neighbor to cutoff suffices)
        ij_t, ik_t, r_tq, r_tqc = self.neighbourhood.get_triplets(atoms, "ijdD")

        # construct lists with atom and pair types
        ti_p = t_n[i_p]
        tij_p = self.neighbourhood.pair_type(ti_p, t_n[j_p])
        ti_t = t_n[i_p[ij_t]]
        tij_t = self.neighbourhood.pair_type(ti_t, t_n[j_p[ij_t]])
        tik_t = self.neighbourhood.pair_type(ti_t, t_n[j_p[ik_t]])

        # potential-dependent functions
        G_t = self.G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d1G_tc = self.d1G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d2G_tc = self.d2G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d11G_tcc = self.d11G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d12G_tcc = self.d12G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)
        d22G_tcc = self.d22G(r_tqc[:, 0], r_tqc[:, 1], ti_t, tij_t, tik_t)

        xi_p = np.bincount(ij_t, weights=G_t, minlength=nb_pairs)

        d1F_p = self.d1F(r_p, xi_p, ti_p, tij_p)
        d2F_p = self.d2F(r_p, xi_p, ti_p, tij_p)
        d11F_p = self.d11F(r_p, xi_p, ti_p, tij_p)
        d12F_p = self.d12F(r_p, xi_p, ti_p, tij_p)
        d22F_p = self.d22F(r_p, xi_p, ti_p, tij_p)

        # Derivative of xi with respect to the deformation gradient
        dxidF_pab = mabincount(ij_t, _o(d1G_tc, r_pc[ij_t])
                               + _o(d2G_tc, r_pc[ik_t]), minlength=nb_pairs)

        # Term 1
        naF1_ncab = d11F_p.reshape(-1, 1, 1, 1) * _o(n_pc, n_pc, r_pc)

        # Term 2
        naF21_tcab = (d12F_p[ij_t] * (_o(n_pc[ij_t], d1G_tc, r_pc[ij_t])
                                      + _o(n_pc[ij_t], d2G_tc, r_pc[ik_t])
                                      + _o(d1G_tc, n_pc[ij_t], r_pc[ij_t])
                                      + _o(d2G_tc, n_pc[ij_t], r_pc[ij_t])).T).T

        naF22_tcab = -(d12F_p[ij_t] *
                       (_o(n_pc[ij_t], d1G_tc, r_pc[ij_t])
                        + _o(n_pc[ij_t], d2G_tc, r_pc[ik_t])
                        + _o(d1G_tc, n_pc[ij_t], r_pc[ij_t])).T).T

        naF23_tcab = -(d12F_p[ij_t] * (_o(d2G_tc, n_pc[ij_t], r_pc[ij_t])).T).T

        # Term 3
        naF31_tcab = (d22F_p[ij_t].reshape(-1, 1, 1, 1)
                      * d1G_tc.reshape(-1, 3, 1, 1)
                      * dxidF_pab[ij_t].reshape(-1, 1, 3, 3))
        naF32_tcab = (d22F_p[ij_t].reshape(-1, 1, 1, 1)
                      * d2G_tc.reshape(-1, 3, 1, 1)
                      * dxidF_pab[ij_t].reshape(-1, 1, 3, 3))

        # Term 4
        naF4_ncab = (d1F_p * (dn_pcc.reshape(-1, 3, 3, 1)
                              * r_pc.reshape(-1, 1, 1, 3)).T).T

        # Term 5
        naF51_tcab = (d2F_p[ij_t] * (
            d11G_tcc.reshape(-1, 3, 3, 1) * r_pc[ij_t].reshape(-1, 1, 1, 3)
            + d12G_tcc.reshape(-1, 3, 3, 1) * r_pc[ik_t].reshape(-1, 1, 1, 3)
            + d22G_tcc.reshape(-1, 3, 3, 1) * r_pc[ik_t].reshape(-1, 1, 1, 3)
            + (d12G_tcc.reshape(-1, 3, 3, 1)).swapaxes(1, 2)
            * r_pc[ij_t].reshape(-1, 1, 1, 3)).T).T

        naF52_tcab = -(d2F_p[ij_t] * (
            d11G_tcc.reshape(-1, 3, 3, 1) * r_pc[ij_t].reshape(-1, 1, 1, 3)
            + d12G_tcc.reshape(-1, 3, 3, 1) * r_pc[ik_t].reshape(-1, 1, 1, 3)).T).T

        naF53_tcab = -(d2F_p[ij_t] * (
            d12G_tcc.reshape(-1, 3, 3, 1).swapaxes(1, 2) * r_pc[ij_t].reshape(-1, 1, 1, 3)
            + d22G_tcc.reshape(-1, 3, 3, 1) * r_pc[ik_t].reshape(-1, 1, 1, 3)).T).T

        naforces_icab = \
            mabincount(i_p, naF1_ncab, minlength=nb_atoms) \
            - mabincount(j_p, naF1_ncab, minlength=nb_atoms) \
            + mabincount(i_p[ij_t], naF21_tcab, minlength=nb_atoms) \
            + mabincount(j_p[ij_t], naF22_tcab, minlength=nb_atoms) \
            + mabincount(j_p[ik_t], naF23_tcab, minlength=nb_atoms) \
            + mabincount(i_p[ij_t], naF31_tcab, minlength=nb_atoms) \
            - mabincount(j_p[ij_t], naF31_tcab, minlength=nb_atoms) \
            + mabincount(i_p[ij_t], naF32_tcab, minlength=nb_atoms) \
            - mabincount(j_p[ik_t], naF32_tcab, minlength=nb_atoms) \
            + mabincount(i_p, naF4_ncab, minlength=nb_atoms) \
            - mabincount(j_p, naF4_ncab, minlength=nb_atoms) \
            + mabincount(i_p[ij_t], naF51_tcab, minlength=nb_atoms) \
            + mabincount(j_p[ij_t], naF52_tcab, minlength=nb_atoms) \
            + mabincount(j_p[ik_t], naF53_tcab, minlength=nb_atoms)

        return naforces_icab / 2


class NiceManybody(Manybody):
    """Manybody calculator with nicer API."""

    class F(ABC):
        """Pair potential."""

        @abstractmethod
        def __call__(self, r, xi, atom_type, pair_type):
            """Compute energy."""
        @abstractmethod
        def gradient(self, r, xi, atom_type, pair_type):
            """Compute gradient."""
        @abstractmethod
        def hessian(self, r, xi, atom_type, pair_type):
            """Compute hessian."""

    class G(ABC):
        """Triplet potential."""

        @abstractmethod
        def __call__(self, r_ij, r_ik, atom_type, ij_type, ik_type):
            """Compute energy."""
        @abstractmethod
        def gradient(self, r_ij, r_ik, atom_type, ij_type, ik_type):
            """Compute gradient."""
        @abstractmethod
        def hessian(self, r_ij, r_ik, atom_type, ij_type, ik_type):
            """Compute hessian."""

        @staticmethod
        def _distance_triplet(rij, rik, cell, pbc):
            D, d = find_mic(
                np.concatenate([rij, rik, rik-rij]),
                cell, pbc)
            return D.reshape(3, -1, 3), d.reshape(3, -1)

    def __init__(self, F, G, neighbourhood):
        """Init with pair & triplet potential + neighbourhood."""
        d1F, d2F = self._split_call(F.gradient, 2)
        d11F, d22F, d12F = self._split_call(F.hessian, 3)
        d1G, d2G = self._split_call(G.gradient, 2)
        d11G, d22G, d12G = self._split_call(G.hessian, 3)

        super().__init__(
            neighbourhood.atom_type,
            neighbourhood.pair_type,
            F=F.__call__, G=G.__call__,
            d1F=d1F, d2F=d2F,
            d11F=d11F, d22F=d22F, d12F=d12F,
            d1G=d1G, d11G=d11G, d2G=d2G, d22G=d22G, d12G=d12G,
            cutoff=neighbourhood.cutoff,
            neighbourhood=neighbourhood)

    @staticmethod
    def _split_call(func, n):
        def wrap(func, i, *args):
            return np.asanyarray(func(*args))[i]

        def make_lambda(i):
            return lambda *args: wrap(func, i, *args)
        return (make_lambda(i) for i in range(n))
