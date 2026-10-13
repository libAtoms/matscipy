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
import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from ase.units import GPa, J, m

from matscipy.fracture_mechanics.crack import CubicCrystalCrack

DATA = Path(__file__).parent / "data" / "clamped_crack"


def _crk(args):
    return CubicCrystalCrack(args["surface"], args["front"], C11=args["C11"] * GPa,
                             C12=args["C12"] * GPa, C44=args["C44"] * GPa)


def _build(system, **kw):
    from matscipy.fracture_mechanics.clusters import clamped_cluster
    args = json.loads((DATA / f"{system}.json").read_text())
    crk = _crk(args)
    KG = crk.k1g(args["surface_energy"] * J / m**2)
    a = clamped_cluster(crk, KG, tuple(args["n"]), tuple(args["skin"]), a0=args["a0"],
                        crack_surface=args["surface"], crack_front=args["front"],
                        tip_offset=args["tip_offset"], stagger_caps=args["stagger_caps"], **kw)
    return a, crk, KG, args


@pytest.mark.parametrize("system", ["110_110", "110_001", "111_110"])
def test_clamped_cluster_reproduces_the_crackcell_builds(system):
    ref = read(DATA / f"{system}.xyz")
    a, _, _, _ = _build(system)
    assert a.get_chemical_symbols() == ref.get_chemical_symbols()
    assert np.allclose(a.positions, ref.positions, atol=1e-8)
    assert np.allclose(a.cell, ref.cell, atol=1e-8)
    assert (a.arrays["groups"] == ref.arrays["groups"]).all()
    assert (a.arrays["h_host"] == ref.arrays["h_host"]).all()
    assert (a.info["bond1"], a.info["bond2"]) == (ref.info["bond1"], ref.info["bond2"])
    assert a.info["tip_x"] == pytest.approx(ref.info["tip_x"], abs=1e-8)
    assert a.info["tip_y"] == pytest.approx(ref.info["tip_y"], abs=1e-8)


def test_tip_bond_midpoint_is_at_the_cell_centre_in_y():
    a, _, _, _ = _build("110_110")
    assert a.info["tip_y"] == pytest.approx(a.cell[1, 1] / 2, abs=1e-8)


def test_caps_take_the_host_specific_length():
    a, _, _, _ = _build("110_001", cap_lengths={"Si": 1.50})
    h = np.flatnonzero(a.arrays["h_host"] >= 0)
    d = np.linalg.norm(a.positions[h] - a.positions[a.arrays["h_host"][h]], axis=1)
    assert np.allclose(d, 1.50, atol=1e-6)


def test_cluster_like_keeps_the_topology_on_another_lattice():
    from matscipy.fracture_mechanics.clusters import cluster_like
    a, crk, KG, args = _build("111_110")
    crk2 = CubicCrystalCrack(args["surface"], args["front"], C11=140 * GPa, C12=60 * GPa,
                             C44=70 * GPa)
    KG2 = crk2.k1g(1.3 * J / m**2)
    b = cluster_like(a, crk2, KG2, 5.5)
    s = 5.5 / args["a0"]
    assert b.get_chemical_symbols() == a.get_chemical_symbols()
    assert (b.arrays["groups"] == a.arrays["groups"]).all()
    assert (b.arrays["h_host"] == a.arrays["h_host"]).all()
    assert (b.info["bond1"], b.info["bond2"]) == (a.info["bond1"], a.info["bond2"])
    assert np.allclose(b.arrays["reference"], s * a.arrays["reference"])
    assert np.allclose(b.cell, s * np.asarray(a.cell))
    h = np.flatnonzero(a.arrays["h_host"] >= 0)
    host = a.arrays["h_host"][h]
    assert np.allclose(b.positions[h] - b.positions[host], s * (a.positions[h] - a.positions[host]))
    # unloaded (K = 0) it sits on its own scaled reference (heavy atoms)
    c = cluster_like(a, crk2, 0.0, 5.5)
    heavy = a.arrays["h_host"] < 0
    assert np.allclose(c.positions[heavy], c.arrays["reference"][heavy])


from ase.calculators.calculator import Calculator, all_changes  # noqa: E402


class _NoPerAtom(Calculator):
    """Wraps a calculator and refuses per-atom energies, like most MLIPs."""
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        b = atoms.copy()
        b.calc = self.inner
        self.results = {"energy": b.get_potential_energy(), "free_energy": b.get_potential_energy(),
                        "forces": b.get_forces()}

    def get_potential_energies(self, atoms=None):
        raise NotImplementedError("no per-atom energies")


def _sw():
    from matscipy.calculators.manybody import Manybody
    from matscipy.calculators.manybody.explicit_forms.stillinger_weber import (
        Stillinger_Weber_PRB_31_5262_Si, StillingerWeber)
    return Manybody(**StillingerWeber(Stillinger_Weber_PRB_31_5262_Si))


def _sw_clamped(n=(4, 4, 1)):
    """Unpassivated SW (110)[1-10] cluster: the crackcell reference trace's settings."""
    from matscipy.fracture_mechanics.clusters import clamped_cluster
    crk = CubicCrystalCrack([1, -1, 0], [1, 1, 0], C11=151.42363758176955 * GPa,
                            C12=76.42352 * GPa, C44=56.44842 * GPa)
    KG = crk.k1g(1.6656936 * J / m**2)
    a = clamped_cluster(crk, 0.6 * KG, n, (0.5, 1.0), a0=5.430949878916217,
                        crack_surface=[1, -1, 0], crack_front=[1, 1, 0], passivate=False)
    return a, crk, KG


def _clamped_sinclair(a, crk, calc, K):
    from matscipy.fracture_mechanics.crack import SinclairCrack, regions_from_groups
    cryst = a.copy()
    cryst.positions = a.arrays["reference"]
    regions_from_groups(cryst)
    alpha = a.info["tip_x"] - cryst.cell[0, 0] / 2
    return SinclairCrack(crk, cryst, calc, K, alpha=alpha, vacuum=0.0, variable_alpha=False,
                         variable_k=True, energy="total")


def test_regions_from_groups_maps_free_and_clamped():
    from matscipy.fracture_mechanics.crack import regions_from_groups
    a, _, _ = _sw_clamped()
    regions_from_groups(a)
    assert ((a.arrays["region"] == 1) == (a.arrays["groups"] != 0)).all()
    assert ((a.arrays["region"] == 3) == (a.arrays["groups"] == 0)).all()


def test_u_cle_honours_its_k_argument():
    a, crk, KG = _sw_clamped()
    sc = _clamped_sinclair(a, crk, _NoPerAtom(_sw()), 0.6 * KG)
    assert np.allclose(sc.u_cle(kI=2 * KG), 2 * sc.u_cle(kI=KG))
    assert np.allclose(sc.u_cle(kI=0.6 * KG), sc.u_cle())


def test_total_energy_mode_works_without_per_atom_energies():
    a, crk, KG = _sw_clamped()
    sc = _clamped_sinclair(a, crk, _NoPerAtom(_sw()), 0.6 * KG)
    sc.set_dofs(sc.get_dofs())
    b = sc.atoms.copy()
    b.calc = _sw()
    assert sc.get_potential_energy() == pytest.approx(b.get_potential_energy())


def test_clamped_mode_reproduces_the_build_at_u_zero():
    a, crk, KG = _sw_clamped()
    sc = _clamped_sinclair(a, crk, _NoPerAtom(_sw()), 0.6 * KG)
    assert np.allclose(sc.atoms.positions, a.positions, atol=1e-10)


def _problem(n=(4, 4, 1), k=0.6):
    from matscipy.fracture_mechanics.continuation import ClampedProblem
    a, crk, KG = _sw_clamped(n)
    sc = _clamped_sinclair(a, crk, _NoPerAtom(_sw()), k * KG)
    return ClampedProblem(sc, KG, atoms=a)


def test_residual_is_minus_the_energy_gradient():
    p = _problem()
    rng = np.random.default_rng(0)
    x = p.x_from(p.atoms.positions, 0.6)
    x[:-1] += 0.01 * rng.normal(size=p.nf)
    F = p.residual(x)
    v = np.zeros_like(x)
    v[:-1] = rng.normal(size=p.nf)
    v /= np.linalg.norm(v)
    h = 1e-4
    dE = (p.evaluate(x + h * v)[0] - p.evaluate(x - h * v)[0]) / (2 * h)
    assert -F @ v[:-1] == pytest.approx(dE, rel=1e-5, abs=1e-6)


def test_surrogate_jacobian_matches_finite_differences_for_its_own_potential():
    from matscipy.fracture_mechanics.continuation import HessianSurrogate
    p = _problem()
    x = p.x_from(p.atoms.positions, 0.6)
    J = HessianSurrogate(p, _sw()).jac(x)
    rng = np.random.default_rng(1)
    v = rng.normal(size=len(x))
    v /= np.linalg.norm(v)
    h = 1e-5
    fd = (p.residual(x + h * v) - p.residual(x - h * v)) / (2 * h)
    assert np.allclose(J @ v, fd, rtol=1e-4, atol=1e-5)


def test_surrogate_with_h_caps_has_springs_and_stays_finite():
    from matscipy.fracture_mechanics.continuation import ClampedProblem, HessianSurrogate
    a, crk, KG, _ = _build("110_001")
    sc = _clamped_sinclair(a, crk, _NoPerAtom(_sw()), KG)
    p = ClampedProblem(sc, KG, atoms=a)
    J = HessianSurrogate(p, _sw()).jac(p.x_from(p.atoms.positions, 1.0))
    assert J.shape == (p.nf, p.nf + 1) and np.isfinite(J).all()
    h = np.flatnonzero(a.arrays["h_host"] >= 0)[0]
    row = 3 * int(np.flatnonzero(p.free == h)[0])
    assert np.abs(J[row:row + 3, row:row + 3]).max() > 1.0  # the X-H spring block


def test_broyden_corrector_solves_a_bordered_linear_toy():
    from matscipy.fracture_mechanics.continuation import BroydenCorrector
    rng = np.random.default_rng(2)
    A = rng.normal(size=(4, 5))
    F = lambda y: -A @ y  # noqa: E731
    t = rng.normal(size=5)
    t /= np.linalg.norm(t)
    yp = rng.normal(size=5)
    c = BroydenCorrector(ftol=1e-10, max_iter=20, step_limit=1e9)
    c.B = -A + 0.1 * rng.normal(size=(4, 5))
    r = c.correct(F, yp, F(yp), yp, t, lambda y1, y0: float(np.abs(y1 - y0).max()))
    assert r.ok and np.abs(F(r.y)).max() < 1e-10 and abs(t @ (r.y - yp)) < 1e-6
