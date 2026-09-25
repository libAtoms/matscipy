#
# Copyright 2026 James Kermode (U. Warwick)
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

"""Per-atom decomposition of the (public) Manybody calculator.

The calculator now exposes per-atom site energies (``energies``) and per-atom
stresses (``stresses``). They are defined so that they sum exactly to the
global energy and global virial respectively; this test verifies that
consistency for the Stillinger-Weber form on bulk Si, including a sheared cell
that breaks the cubic symmetry (so the off-diagonal virial components are
exercised).
"""

import numpy as np
import numpy.testing as nt

from ase.lattice.cubic import Diamond

from matscipy.calculators.manybody import Manybody
from matscipy.calculators.manybody.explicit_forms import StillingerWeber
from matscipy.calculators.manybody.explicit_forms.stillinger_weber import (
    Stillinger_Weber_PRB_31_5262_Si,
)


def _si(cell_transform=None):
    atoms = Diamond("Si", latticeconstant=5.43, size=(2, 2, 2))
    if cell_transform is not None:
        atoms.set_cell(atoms.cell @ cell_transform, scale_atoms=True)
    atoms.calc = Manybody(**StillingerWeber(Stillinger_Weber_PRB_31_5262_Si))
    return atoms


def test_site_energies_sum_to_total():
    for T in (None, np.array([[1.0, 0.02, 0.0],
                              [0.0, 0.99, 0.0],
                              [0.0, 0.0, 1.01]])):
        atoms = _si(T)
        E = atoms.get_potential_energy()
        e_n = atoms.get_potential_energies()
        assert e_n.shape == (len(atoms),)
        nt.assert_allclose(e_n.sum(), E, rtol=0, atol=1e-10)


def test_per_atom_virial_sums_to_global():
    # shear so all six virial components are non-trivial
    T = np.array([[1.0, 0.02, 0.0],
                  [0.0, 0.99, 0.0],
                  [0.0, 0.0, 1.01]])
    atoms = _si(T)
    atoms.get_potential_energy()
    W_nv = atoms.calc.results["stresses"]            # (N, 6), per-atom stress
    assert W_nv.shape == (len(atoms), 6)
    global_v = atoms.get_stress()                    # (6,) global stress
    nt.assert_allclose(W_nv.sum(axis=0), global_v, rtol=0, atol=1e-10)
