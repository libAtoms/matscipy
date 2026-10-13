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
