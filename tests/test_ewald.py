#
# Copyright 2018-2021 Jan Griesser (U. Freiburg)
#           2014, 2020-2021 Lars Pastewka (U. Freiburg)
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

import pytest
import numpy as np

from ase.optimize import FIRE
from ase.units import GPa
from ase.lattice.hexagonal import HexagonalFactory
from ase.lattice.cubic import SimpleCubicFactory
from ase.calculators.mixing import SumCalculator

from matscipy.calculators.ewald import Ewald
from matscipy.calculators.pair_potential.calculator import (
    PairPotential,
    BeestKramerSanten,
)
from matscipy.numerical import (
    numerical_hessian,
    numerical_nonaffine_forces,
    numerical_forces,
    numerical_stress,
)
from matscipy.elasticity import (
    fit_elastic_constants,
    full_3x3x3x3_to_Voigt_6x6,
    nonaffine_elastic_contribution,
)


###


class alpha_quartz(HexagonalFactory):
    """
    Factory to create an alpha quartz crystal structure
    """

    xtal_name = "alpha_quartz"
    bravais_basis = [
        [0, 0.4763, 0.6667],
        [0.4763, 0, 0.3333],
        [0.5237, 0.5237, 0],
        [0.1588, 0.7439, 0.4612],
        [0.2561, 0.4149, 0.7945],
        [0.4149, 0.2561, 0.2055],
        [0.5851, 0.8412, 0.1279],
        [0.7439, 0.1588, 0.5388],
        [0.8412, 0.5851, 0.8721],
    ]
    element_basis = (0, 0, 0, 1, 1, 1, 1, 1, 1)


class beta_cristobalite(SimpleCubicFactory):
    """
    Factory to create a beta cristobalite crystal structure
    """

    xtal_name = "beta_cristobalite"
    bravais_basis = [
        [0.98184000, 0.51816000, 0.48184000],
        [0.51816000, 0.48184000, 0.98184000],
        [0.48184000, 0.98184000, 0.51816000],
        [0.01816000, 0.01816000, 0.01816000],
        [0.72415800, 0.77584200, 0.22415800],
        [0.77584200, 0.22415800, 0.72415800],
        [0.22415800, 0.72415800, 0.77584200],
        [0.27584200, 0.27584200, 0.27584200],
        [0.86042900, 0.34389100, 0.55435200],
        [0.36042900, 0.15610900, 0.44564800],
        [0.13957100, 0.84389100, 0.94564800],
        [0.15610900, 0.44564800, 0.36042900],
        [0.94564800, 0.13957100, 0.84389100],
        [0.44564800, 0.36042900, 0.15610900],
        [0.34389100, 0.55435200, 0.86042900],
        [0.55435200, 0.86042900, 0.34389100],
        [0.14694300, 0.14694300, 0.14694300],
        [0.35305700, 0.85305700, 0.64694300],
        [0.64694300, 0.35305700, 0.85305700],
        [0.85305700, 0.64694300, 0.35305700],
        [0.63957100, 0.65610900, 0.05435200],
        [0.05435200, 0.63957100, 0.65610900],
        [0.65610900, 0.05435200, 0.63957100],
        [0.84389100, 0.94564800, 0.13957100],
    ]
    element_basis = tuple([0] * 8 + [1] * 16)


calc_alpha_quartz = {
    (14, 14): BeestKramerSanten(0, 1, 0, 2.9),
    (14, 8): BeestKramerSanten(18003.7572, 4.87318, 133.5381, 2.9),
    (8, 8): BeestKramerSanten(1388.7730, 2.76000, 175.0000, 2.9),
}

ewald_alpha_params = dict(
    accuracy=1e-6,
    cutoff=2.9,
    kspace={
        "alpha": 1.28,
        "nbk_c": np.array([8, 11, 9]),
        "cutoff": 10.43,
    },
)

calc_beta_cristobalite = calc_alpha_quartz.copy()
for k in calc_beta_cristobalite:
    calc_beta_cristobalite[k].cutoff = 3.8

ewald_beta_params = dict(
    accuracy=1e-6,
    cutoff=3.8,
    kspace={
        "alpha": 0.96,
        "nbk_c": np.array([9, 9, 9]),
        "cutoff": 7,
    },
)


@pytest.fixture(scope="module", params=[4.7, 4.9, 5.1])
def alpha_quartz_ewald(request):
    structure = alpha_quartz()
    a0 = request.param
    atoms = structure(
        ["Si", "O"],
        size=[1, 1, 1],
        latticeconstant={"a": a0, "b": a0, "c": 5.4},
    )

    charges = np.zeros(len(atoms))
    charges[atoms.symbols == "Si"] = +2.4
    charges[atoms.symbols == "O"] = -1.2
    atoms.set_array("charge", charges, dtype=float)

    b = Ewald()
    b.set(**ewald_alpha_params)
    atoms.calc = b
    return atoms


@pytest.fixture(scope="module")
def alpha_quartz_bks(alpha_quartz_ewald):
    atoms = alpha_quartz_ewald
    atoms.calc = SumCalculator([atoms.calc, PairPotential(calc_alpha_quartz)])

    FIRE(atoms, logfile=None).run(fmax=1e-3)
    return atoms


@pytest.fixture(scope="module")
def beta_cristobalite_ewald(request):
    a0 = request.param
    structure = beta_cristobalite()
    atoms = structure(["Si", "O"], size=[1, 1, 1], latticeconstant=a0)
    charges = np.zeros(len(atoms))
    charges[atoms.symbols == "Si"] = +2.4
    charges[atoms.symbols == "O"] = -1.2
    atoms.set_array("charge", charges, dtype=float)

    b = Ewald()
    b.set(**ewald_beta_params)
    atoms.calc = b
    return atoms


@pytest.fixture(scope="module")
def beta_cristobalite_bks(request):
    a0 = request.param
    structure = beta_cristobalite()
    atoms = structure(["Si", "O"], size=[1, 1, 1], latticeconstant=a0)
    charges = np.zeros(len(atoms))
    charges[atoms.symbols == "Si"] = +2.4
    charges[atoms.symbols == "O"] = -1.2
    atoms.set_array("charge", charges, dtype=float)

    b = Ewald()
    b.set(**ewald_beta_params)
    atoms.calc = b
    atoms.calc = SumCalculator(
        [atoms.calc, PairPotential(calc_beta_cristobalite)]
    )

    FIRE(atoms, logfile=None).run(fmax=1e-3)
    return atoms


# Test for alpha quartz
def test_stress_alpha_quartz(alpha_quartz_ewald):
    """
    Test the computation of stress
    """
    atoms = alpha_quartz_ewald
    s = atoms.get_stress()
    sn = numerical_stress(atoms, d=0.001)

    print("Stress ana: \n", s)
    print("Stress num: \n", sn)

    np.testing.assert_allclose(s, sn, atol=1e-3)


def test_forces_alpha_quartz(alpha_quartz_ewald):
    """
    Test the computation of forces
    """
    atoms = alpha_quartz_ewald
    f = atoms.get_forces()
    fn = numerical_forces(atoms, d=0.0001)

    print("f_ana: \n", f[:5, :])
    print("f_num: \n", fn[:5, :])

    np.testing.assert_allclose(f, fn, atol=1e-3)


def test_hessian_alpha_quartz(alpha_quartz_bks):
    """
    Test the computation of the Hessian matrix
    """
    atoms = alpha_quartz_bks
    H_num = numerical_hessian(atoms, dx=1e-5, indices=None)
    H_ana = atoms.calc.get_property("hessian", atoms)

    print("H_num: \n", H_num.todense()[:6, :6])
    print("H_ana: \n", H_ana[:6, :6])

    np.testing.assert_allclose(H_num.todense(), H_ana, atol=1e-3)


def test_non_affine_forces_alpha_quartz(alpha_quartz_bks):
    """
    Test the computation of non-affine forces
    """
    atoms = alpha_quartz_bks

    naForces_ana = atoms.calc.get_property("nonaffine_forces")
    naForces_num = numerical_nonaffine_forces(atoms, d=1e-5)

    print("Num: \n", naForces_num[:1])
    print("Ana: \n", naForces_ana[:1])

    np.testing.assert_allclose(naForces_num, naForces_ana, atol=1e-2)


def test_birch_coefficients_alpha_quartz(alpha_quartz_bks):
    """
    Test the computation of the affine elastic constants + stresses
    """
    atoms = alpha_quartz_bks
    C_ana = full_3x3x3x3_to_Voigt_6x6(
        atoms.calc.get_property("birch_coefficients"), check_symmetry=False
    )

    C_num, Cerr = fit_elastic_constants(
        atoms,
        symmetry="triclinic",
        N_steps=11,
        delta=1e-5,
        optimizer=None,
        verbose=False,
    )

    print("Stress: \n", atoms.get_stress() / GPa)
    print("C_num: \n", C_num / GPa)
    print("C_ana: \n", C_ana / GPa)

    np.testing.assert_allclose(C_num, C_ana, atol=1e-1)


def test_full_elastic_alpha_quartz(alpha_quartz_bks):
    """
    Test the computation of the affine elastic constants + stresses
    + non-affine elastic constants
    """
    atoms = alpha_quartz_bks
    C_num, Cerr = fit_elastic_constants(
        atoms,
        symmetry="triclinic",
        N_steps=11,
        delta=1e-4,
        optimizer=FIRE,
        fmax=1e-4,
        logfile=None,
        verbose=False,
    )
    C_ana = full_3x3x3x3_to_Voigt_6x6(
        atoms.calc.get_property("birch_coefficients"), check_symmetry=False
    )
    C_na = full_3x3x3x3_to_Voigt_6x6(
        nonaffine_elastic_contribution(atoms)
    )

    print("stress: \n", atoms.get_stress())
    print("C_num: \n", C_num)
    print("C_ana: \n", C_ana + C_na)

    np.testing.assert_allclose(C_num, C_ana + C_na, atol=1e-1)


###


# Beta crisotbalite
@pytest.mark.parametrize(
    "beta_cristobalite_ewald", [6, 7, 8, 9], indirect=True
)
def test_stress_beta_cristobalite(beta_cristobalite_ewald):
    """
    Test the computation of stress
    """
    atoms = beta_cristobalite_ewald
    s = atoms.get_stress()
    sn = numerical_stress(atoms, d=0.0001)

    print("Stress ana: \n", s)
    print("Stress num: \n", sn)

    np.testing.assert_allclose(s, sn, atol=1e-3)


@pytest.mark.parametrize(
    "beta_cristobalite_ewald", [6, 7, 8, 9], indirect=True
)
def test_forces_beta_cristobalite(beta_cristobalite_ewald):
    """
    Test the computation of forces
    """
    atoms = beta_cristobalite_ewald
    f = atoms.get_forces()
    fn = numerical_forces(atoms, d=1e-5)

    print("forces ana: \n", f[:5, :])
    print("forces num: \n", fn[:5, :])

    np.testing.assert_allclose(f, fn, atol=1e-3)


@pytest.mark.parametrize("beta_cristobalite_bks", [6, 7, 8, 9], indirect=True)
def test_hessian_beta_cristobalite(beta_cristobalite_bks):
    """
    Test the computation of the Hessian matrix
    """
    atoms = beta_cristobalite_bks
    H_num = numerical_hessian(atoms, dx=1e-5, indices=None)
    H_ana = atoms.calc.get_property("hessian")

    np.testing.assert_allclose(H_num.todense(), H_ana, atol=1e-3)


@pytest.mark.parametrize("beta_cristobalite_bks", [6], indirect=True)
def test_non_affine_forces_beta_cristobalite(beta_cristobalite_bks):
    """
    Test the computation of non-affine forces
    """
    atoms = beta_cristobalite_bks
    naForces_ana = atoms.calc.get_property("nonaffine_forces")
    naForces_num = numerical_nonaffine_forces(atoms, d=1e-5)

    print("Num: \n", naForces_num[:1])
    print("Ana: \n", naForces_ana[:1])

    np.testing.assert_allclose(naForces_num, naForces_ana, atol=1e-2)


@pytest.mark.parametrize("beta_cristobalite_bks", [6], indirect=True)
def test_birch_coefficients_beta_cristobalite(beta_cristobalite_bks):
    """
    Test the computation of the affine elastic constants + stresses
    """
    atoms = beta_cristobalite_bks
    C_num, Cerr = fit_elastic_constants(
        atoms,
        symmetry="triclinic",
        N_steps=11,
        delta=1e-5,
        optimizer=None,
        verbose=False,
    )
    C_ana = full_3x3x3x3_to_Voigt_6x6(
        atoms.calc.get_property("birch_coefficients"), check_symmetry=False
    )

    print("C_num: \n", C_num)
    print("C_ana: \n", C_ana)

    np.testing.assert_allclose(C_num, C_ana, atol=1e-1)


@pytest.mark.parametrize("beta_cristobalite_bks", [6], indirect=True)
def test_non_affine_elastic_beta_cristobalite(beta_cristobalite_bks):
    """
    Test the computation of the affine elastic constants + stresses
    + non-affine elastic constants
    """
    atoms = beta_cristobalite_bks
    C_num, Cerr = fit_elastic_constants(
        atoms,
        symmetry="triclinic",
        N_steps=11,
        delta=1e-3,
        optimizer=FIRE,
        fmax=1e-3,
        logfile=None,
        verbose=False,
    )
    C_ana = full_3x3x3x3_to_Voigt_6x6(
        atoms.calc.get_property("birch_coefficients"), check_symmetry=False
    )
    C_na = full_3x3x3x3_to_Voigt_6x6(
        nonaffine_elastic_contribution(atoms)
    )

    print("stress: \n", atoms.get_stress())
    print("C_num: \n", C_num)
    print("C_ana: \n", C_ana + C_na)

    np.testing.assert_allclose(C_num, C_ana + C_na, atol=1e-1)
