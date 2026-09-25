#
# Copyright 2020-2021 Lars Pastewka (U. Freiburg)
#           2015 m.a.aldegunde-rodriguez@warwick.ac.uk
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

# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
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

import unittest

import numpy as np
import pytest

from matscipy.elasticity import elastic_moduli, poisson_ratio


###

def test_rotation():
    Cs = [
        np.array([
            [165.7, 63.9, 63.9, 0, 0, 0],
            [63.9, 165.7, 63.9, 0, 0, 0],
            [63.9, 63.9, 165.7, 0, 0, 0],
            [0, 0, 0, 79.6, 0, 0],
            [0, 0, 0, 0, 79.6, 0],
            [0, 0, 0, 0, 0, 79.6],
        ])
    ]
    l = [np.array([1, 0, 0]), np.array([1, 1, 0])]
    EM = [
        {'E': np.array([130, 130, 130]),
         'nu': np.array([0.28, 0.28, 0.28]),
         'G': np.array([79.6, 79.6, 79.6])},
        {'E': np.array([169, 169, 130]),
         'nu': np.array([0.36, 0.28, 0.064]),
         'G': np.array([79.6, 79.6, 50.9])}
    ]
    for C in Cs:
        for i, directions in enumerate(l):
            directions = directions / np.linalg.norm(directions)

            E, nu, Gm, B, K = elastic_moduli(C, l=directions)

            nu_v = np.array([nu[1, 2], nu[2, 0], nu[0, 1]])
            G = np.array([Gm[1, 2], Gm[2, 0], Gm[0, 1]])

            np.testing.assert_array_almost_equal(E, EM[i]['E'], decimal=1)
            np.testing.assert_array_almost_equal(nu_v, EM[i]['nu'], decimal=2)
            np.testing.assert_array_almost_equal(G, EM[i]['G'], decimal=9)


def test_monoclinic():
    C = np.array([
            [5,2,2,0,1,0],
            [2,5,2,0,1,0],
            [2,2,5,0,1,0],
            [0,0,0,2,0,1],
            [1,1,1,0,2,0],
            [0,0,0,1,0,2]
        ])
    l = [np.array([1, 0, 1]), np.array([1, 0, -1])]
    EM = np.array([5.4545,3.1579])

    for i, directions in enumerate(l):
        directions = directions / np.linalg.norm(directions)
        E, nu, Gm, B, K = elastic_moduli(C, l=directions)
        np.testing.assert_almost_equal(E[0], EM[i], decimal=1)


###

if __name__ == '__main__':
    unittest.main()


def _poisson_ratio_from_compliance(C, l, m):
    """Exact -eps_mm/eps_ll for uniaxial stress along l, from the full compliance tensor"""
    S = np.linalg.inv(C)
    voigt = [[0, 5, 4], [5, 1, 3], [4, 3, 2]]
    S_ijkl = np.zeros((3, 3, 3, 3))
    for i, j, k, n in np.ndindex(3, 3, 3, 3):
        a, b = voigt[i][j], voigt[k][n]
        S_ijkl[i, j, k, n] = S[a, b] * (1 if a < 3 else 0.5) * (1 if b < 3 else 0.5)
    l = np.asarray(l, float) / np.linalg.norm(l)
    m = np.asarray(m, float) / np.linalg.norm(m)
    eps = np.einsum('ijkl,kl->ij', S_ijkl, np.outer(l, l))
    return -(m @ eps @ m) / (l @ eps @ l)


def test_poisson_ratio():
    # strongly anisotropic cubic crystal (bcc Fe-like, GPa): nu depends on m (issue #13)
    C11, C12, C44 = 243., 145., 116.
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    np.fill_diagonal(C[:3, :3], C11)
    C[3, 3] = C[4, 4] = C[5, 5] = C44
    for l, m in [([1, 0, 0], [0, 1, 0]), ([0, 1, 1], [1, 0, 0]),
                 ([0, 1, 1], [0, 1, -1]), ([1, 1, 1], [1, -1, 0]),
                 ([1, 1, 2], [1, -1, 0])]:
        np.testing.assert_allclose(poisson_ratio(C, np.array(l), np.array(m)),
                                   _poisson_ratio_from_compliance(C, l, m), rtol=1e-10)
    with pytest.raises(ValueError):
        poisson_ratio(C, np.array([0, 1, 1]), np.array([0, 1, 0]))
