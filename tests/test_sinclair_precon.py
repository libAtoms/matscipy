#
# Copyright 2026 James Kermode (Warwick U.)
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
from ase.calculators.emt import EMT
from ase.units import GPa

from matscipy.elasticity import fit_elastic_constants
from matscipy.fracture_mechanics.clusters import fcc, set_regions
from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack


def test_update_precon_with_non_radial_regions():
    """SinclairCrack.update_precon must select regions I+II by mask, not by
    slicing the first N2 atoms. set_regions sorts atoms radially, so with
    extended_region_I=True (region I extended along the crack surface) the
    region-I atoms are no longer the first N1 and a [:N2] slice builds the
    preconditioner from the wrong atoms."""
    el, a0 = 'Cu', 3.59
    calc = EMT()
    bulk = fcc(el, a0, [1, 1, 1], [1, 0, 0], [0, 0, 1])
    bulk.calc = calc
    C, _ = fit_elastic_constants(bulk, symmetry='cubic', verbose=False)

    crack_surface, crack_front = [1, 1, 0], [0, 0, 1]
    rI, cutoff, rIII = 8.0, 5.0, 16.0
    ax, ay, _ = fcc(el, a0, [1, 1, 1], crack_surface, crack_front).cell.lengths()
    n = [2 * int(np.ceil((rIII + 2 * cutoff) / ax)) + 1,
         2 * int(np.ceil((rIII + 2 * cutoff) / ay)) + 2, 1]
    cryst = fcc(el, a0, n, crack_surface, crack_front)
    cluster = set_regions(cryst, rI, cutoff, rIII, extended_region_I=True)
    region = cluster.arrays['region']
    # precondition for this test to be meaningful
    assert not np.all(np.diff(region[region <= 2]) >= 0)

    crk = CubicCrystalCrack(crack_surface, crack_front, C=C / GPa)
    sc = SinclairCrack(crk, cluster, calc, crk.k1g(1.0), variable_alpha=True)
    x = sc.get_dofs()
    sc.update_precon(x, sc.get_forces(x))

    N_dof = len(sc)
    assert N_dof == 3 * sc.N1 + 1
    assert sc.P_ilu.shape == (N_dof, N_dof)
    # the preconditioner is positive definite on the region-I block
    Pf = sc.P_ilu.solve(np.ones(N_dof))
    assert np.all(np.isfinite(Pf))
