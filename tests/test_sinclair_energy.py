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


def test_rescale_k_refreshes_energy_reference():
    """After rescale_k, get_potential_energy (and its alpha derivative) must
    match a SinclairCrack constructed at the new K (issue #321): the far-field
    term E2 is referenced to the u = 0 state, which depends on K."""
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
    cluster = set_regions(fcc(el, a0, n, crack_surface, crack_front), rI, cutoff, rIII)

    crk = CubicCrystalCrack(crack_surface, crack_front, C=C / GPa)
    k1g = crk.k1g(1.0)

    rescaled = SinclairCrack(crk, cluster.copy(), calc, 0.8 * k1g, alpha=0.0)
    rescaled.rescale_k(1.2 * k1g)
    fresh = SinclairCrack(crk, cluster.copy(), calc, 1.2 * k1g, alpha=0.0)

    u = 0.01 * np.random.default_rng(0).standard_normal(fresh.u.shape)
    h = 1e-3
    energies = {}
    for name, sc in [('rescaled', rescaled), ('fresh', fresh)]:
        sc.u[:] = u
        E = []
        for alpha in (-h, 0.0, h):
            sc.alpha = alpha
            sc.update_atoms()
            E.append(sc.get_potential_energy())
        energies[name] = np.array(E)

    np.testing.assert_allclose(energies['rescaled'], energies['fresh'], rtol=0, atol=1e-8)
    dE_rescaled = (energies['rescaled'][2] - energies['rescaled'][0]) / (2 * h)
    dE_fresh = (energies['fresh'][2] - energies['fresh'][0]) / (2 * h)
    np.testing.assert_allclose(dE_rescaled, dE_fresh, rtol=0, atol=1e-5)
