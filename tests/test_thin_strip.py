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
"""The modulus that relates strain to energy release rate in a thin strip."""

import numpy as np
import pytest
from ase.lattice.cubic import Diamond
from ase.optimize import LBFGS
from ase.units import GPa

import matscipy.fracture_mechanics.crack as crack
from matscipy.calculators.manybody import Manybody
from matscipy.calculators.manybody.explicit_forms import StillingerWeber
from matscipy.calculators.manybody.explicit_forms.stillinger_weber import (
    Stillinger_Weber_PRB_31_5262_Si,
)
from matscipy.elasticity import fit_elastic_constants

SI_110 = dict(crack_direction=[1, -1, 0], cleavage_plane=[1, 1, 0], crack_front=[0, 0, 1])


def cubic(C11, C12, C44):
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    C[np.arange(3), np.arange(3)] = C11
    C[np.arange(3, 6), np.arange(3, 6)] = C44
    return C


def test_isotropic_modulus_is_lambda_plus_two_mu():
    lam, mu = 60.0, 40.0
    C = cubic(lam + 2 * mu, lam, mu)
    for dirs in [SI_110, dict(crack_direction=[1, 0, 0], cleavage_plane=[0, 1, 0],
                              crack_front=[0, 0, 1])]:
        assert crack.thin_strip_modulus(C, **dirs) == pytest.approx(lam + 2 * mu, rel=1e-12)


def test_cubic_110_pull_is_rotated_stiffness():
    C11, C12, C44 = 151.4, 76.4, 56.4
    M = crack.thin_strip_modulus(cubic(C11, C12, C44), **SI_110)
    assert M == pytest.approx((C11 + C12) / 2 + C44, rel=1e-12)


def test_G_strain_roundtrip_and_formula():
    M, h, eps = 1.1, 150.0, 0.03
    G = crack.thin_strip_G(eps, M, h)
    assert G == pytest.approx(0.5 * M * eps**2 * h, rel=1e-14)
    assert crack.thin_strip_strain(G, M, h) == pytest.approx(eps, rel=1e-14)


def test_modulus_matches_strain_energy_of_uniaxially_strained_si():
    """Far ahead of a thin-strip crack only eps_yy is nonzero: the energy density is M eps^2 / 2."""
    calc = Manybody(**StillingerWeber(Stillinger_Weber_PRB_31_5262_Si))
    a0 = 5.431
    unit = Diamond("Si", latticeconstant=a0, pbc=True)
    unit.calc = calc
    C, _ = fit_elastic_constants(unit, symmetry="cubic", optimizer=LBFGS, fmax=1e-6, verbose=False)
    M = crack.thin_strip_modulus(C, **SI_110)
    cell = Diamond("Si", directions=[SI_110["crack_direction"], SI_110["cleavage_plane"],
                                     SI_110["crack_front"]],
                   size=(2, 2, 2), latticeconstant=a0, pbc=True)
    cell.calc = calc
    e0, V, eps = cell.get_potential_energy(), cell.get_volume(), 1e-3

    def energy(e):
        strained = cell.copy()
        strained.calc = calc
        strained.set_cell(cell.cell @ np.diag([1.0, 1.0 + e, 1.0]), scale_atoms=True)
        LBFGS(strained, logfile=None).run(fmax=1e-6)
        return strained.get_potential_energy()

    # central difference: any residual stress at a0 cancels
    measured = (energy(eps) + energy(-eps) - 2 * e0) / V / eps**2
    assert measured == pytest.approx(M, rel=5e-3)
    # the plane-strain form E / (1 - nu_yx^2) misses it by more than 20%
    from matscipy.elasticity import poisson_ratio, youngs_modulus

    E = youngs_modulus(C, SI_110["cleavage_plane"])
    nu = poisson_ratio(C, SI_110["cleavage_plane"], SI_110["crack_direction"])
    assert E / (1 - nu**2) < 0.8 * measured


def test_energy_release_rate_uses_stored_thin_strip_modulus():
    a = Diamond("Si", latticeconstant=5.431, size=(2, 2, 2), pbc=True)
    h = a.positions[:, 1].max() - a.positions[:, 1].min()
    a.positions[:, 1] *= 1.02
    a.info.update(OrigHeight=h, ThinStripModulus=1.1, YoungsModulus=0.9, PoissonRatio_yx=0.2)
    G = crack.get_energy_release_rate(a)
    assert G == pytest.approx(crack.thin_strip_G(a.info["strain"], 1.1, h), rel=1e-12)


def test_energy_release_rate_without_modulus_warns():
    a = Diamond("Si", latticeconstant=5.431, size=(2, 2, 2), pbc=True)
    h = a.positions[:, 1].max() - a.positions[:, 1].min()
    a.info.update(OrigHeight=h, YoungsModulus=0.9, PoissonRatio_yx=0.2)
    with pytest.warns(UserWarning, match="ThinStripModulus"):
        crack.get_energy_release_rate(a)
