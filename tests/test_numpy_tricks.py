#
# Copyright 2014-2015, 2017, 2021 Lars Pastewka (U. Freiburg)
#           2020 Wolfram G. Nöhring (U. Freiburg)
#           2015 Adrien Gola (KIT)
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

from matscipy.numpy_tricks import mabincount


def test_mabincount():
    w = np.array([[0.5, 1, 1],
                  [2, 2, 2]])

    x = np.array([1, 2])
    r = mabincount(x, w, 4, axis=0)
    assert r.shape == (4, 3)
    np.testing.assert_allclose(r, [[0, 0, 0], [0.5, 1, 1], [2, 2, 2], [0, 0, 0]])

    x = np.array([1, 2, 2])
    r = mabincount(x, w.T, 4, axis=0)
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r, [[0, 0], [0.5, 2], [2, 4], [0, 0]])
