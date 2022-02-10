#
# Copyright 2022 Lucas Frérot (U. Freiburg)
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
import numpy.testing as nt

from ase import Atoms
from matscipy.calculators.manybody.calculator import NiceManybody
from matscipy.molecules import Molecules
from matscipy.neighbours import MolecularNeighbourhood


class HarmonicBond(NiceManybody.F):
    def __init__(self, r0, k):
        self.r0 = r0
        self.k = k

    def __call__(self, r, *args):
        return 0.5 * self.k * (r - self.r0)**2

    def gradient(self, r, *args):
        return [self.k * (r - self.r0), np.zeros_like(r)]

    def hessian(self, r, *args):
        return [np.ones_like(r), np.zeros_like(r), np.zeros_like(r)]


class ZeroAngle(NiceManybody.G):
    def __call__(self, *args):
        return np.zeros_like(args[0])

    def gradient(self, *args):
        return np.zeros([2] + list(args[0].shape))

    def hessian(self, *args):
        return np.zeros([3] + list(args[0].shape))


class HarmonicAngle(NiceManybody.G):
    def __init__(self, a0, k):
        self.a0 = a0
        self.k = k

    def __call__(self, r_ij_c, r_ik_c, *args):
        r_ij, r_ik, r_jk = map(lambda x: np.linalg.norm(x, axis=-1),
                               [r_ij_c, r_ik_c, r_ik_c - r_ij_c])
        a = np.arccos(-(r_ij**2 + r_jk**2 - r_ik**2) / (2 * r_ij * r_jk))
        return 0.5 * self.k * (a - self.a0)**2

    def gradient(self, r_ij, r_ik, *args):
        pass


def test_harmonic_bond():
    r0 = 0.5
    atoms = Atoms("CO2",
                  positions=[[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
                  cell=[5, 5, 5])
    molecules = Molecules(bonds_connectivity=[[0, 1], [1, 2]])
    neigh = MolecularNeighbourhood(molecules)
    pot = HarmonicBond(0.5, 1)
    calc = NiceManybody(pot, ZeroAngle(), neigh)
    atoms.calc = calc

    epot = atoms.get_potential_energy()
    epot_ref = 2 * (0.5 * r0**2)
    nt.assert_allclose(epot, epot_ref, rtol=1e-15)
