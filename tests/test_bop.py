#! /usr/bin/env python

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

import numpy as np

import pytest

import ase
from ase import Atoms
from ase.optimize import FIRE
from ase.lattice.compounds import B3
from ase.lattice.cubic import Diamond

import matscipy.calculators.bop_sw.explicit_forms.stillinger_weber as sw
import matscipy.calculators.bop_sw.explicit_forms.kumagai as kum
import matscipy.calculators.bop_sw.explicit_forms.tersoff3 as t3
from matscipy.calculators.bop_sw import AbellTersoffBrennerStillingerWeber
from matscipy.calculators.bop_sw.explicit_forms import KumagaiTersoff, AbellTersoffBrenner, StillingerWeber
from matscipy.hessian_finite_differences import fd_hessian
from matscipy.elasticity import fit_elastic_constants, full_3x3x3x3_to_Voigt_6x6
from matscipy.calculators.calculator import MatscipyCalculator


@pytest.mark.parametrize('a0', [5.2, 5.3, 5.4, 5.5])
def test_stress(a0):
    atoms = Diamond('Si', size=[1, 1, 1], latticeconstant=a0)
    kumagai_potential = kum.kumagai
    calculator = AbellTersoffBrennerStillingerWeber(**KumagaiTersoff(kumagai_potential))
    atoms.set_calculator(calculator)
    s = atoms.get_stress()
    sn = calculator.calculate_numerical_stress(atoms, d=0.0001)
    np.testing.assert_allclose(s, sn, atol=1e-6)


def test_hessian_divide_by_masses():
    # Test the computation of dynamical matrix
    atoms = ase.io.read('aSi.cfg')
    masses_n = np.random.randint(1, 10, size=len(atoms))
    atoms.set_masses(masses=masses_n)
    kumagai_potential = kum.kumagai
    calc = AbellTersoffBrennerStillingerWeber(**KumagaiTersoff(kumagai_potential))
    D_ana = calc.get_hessian(atoms, divide_by_masses=True).todense()
    H_ana = calc.get_hessian(atoms).todense()
    masses_nc = masses_n.repeat(3)
    H_ana /= np.sqrt(masses_nc.reshape(-1, 1) * masses_nc.reshape(1, -1))
    np.testing.assert_allclose(D_ana, H_ana, atol=1e-4)


@pytest.mark.parametrize('d', np.arange(1.0, 2.3, 0.15))
def test_kumagai_small1(d):
    # Test forces and hessian matrix for Kumagai
    small = Atoms([14] * 4, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d)], cell=(100, 100, 100))
    small.center(vacuum=10.0)
    compute_forces_and_hessian(small, KumagaiTersoff(kum.kumagai))


@pytest.mark.parametrize('d', np.arange(1.0, 2.3, 0.15))
def test_kumagai_small2(d):
    # Test forces and hessian matrix for Kumagai
    small2 = Atoms([14] * 5, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d), (0, d, d)], cell=(100, 100, 100))
    small2.center(vacuum=10.0)
    compute_forces_and_hessian(small2, KumagaiTersoff(kum.kumagai))


@pytest.mark.parametrize('a0', [5.2, 5.3, 5.4, 5.5])
def test_kumagai_crystal(a0):
    # Test forces, hessian, non-affine forces and elastic constants for a Si crystal
    Si_crystal = Diamond('Si', size=[1, 1, 1], latticeconstant=a0)
    compute_forces_and_hessian(Si_crystal, KumagaiTersoff(kum.kumagai))
    compute_elastic_constants(Si_crystal, KumagaiTersoff(kum.kumagai))


def test_kumagai_amorphous():
    # Tests for amorphous Si
    aSi = ase.io.read('aSi_N8.xyz')
    aSi.set_calculator(AbellTersoffBrennerStillingerWeber(**KumagaiTersoff(kum.kumagai)))
    # Non-zero forces and Hessian
    compute_forces_and_hessian(aSi, KumagaiTersoff(kum.kumagai))
    # Test forces, hessian, non-affine forces and elastic constants for amorphous Si
    FIRE(aSi).run(fmax=1e-5, steps=1e3)
    compute_forces_and_hessian(aSi, KumagaiTersoff(kum.kumagai))
    compute_elastic_constants(aSi, KumagaiTersoff(kum.kumagai))


@pytest.mark.parametrize('d', np.arange(1.0, 2.3, 0.15))
def test_tersoff_small1(d):
    # Test forces and hessian matrix for Tersoff3
    small = Atoms([14] * 4, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d)], cell=(100, 100, 100))
    small.center(vacuum=10.0)
    compute_forces_and_hessian(small, AbellTersoffBrenner(t3.Tersoff_PRB_39_5566_Si_C))


@pytest.mark.parametrize('d', np.arange(1.0, 2.3, 0.15))
def test_tersoff_small2(d):
    # Test forces and hessian matrix for Tersoff3
    small2 = Atoms([14] * 5, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d), (0, d, d)], cell=(100, 100, 100))
    small2.center(vacuum=10.0)
    compute_forces_and_hessian(small2, AbellTersoffBrenner(t3.Tersoff_PRB_39_5566_Si_C))


@pytest.mark.parametrize('a0', [5.2, 5.3, 5.4, 5.5])
def test_tersoff_crystal(a0):
    # Test forces, hessian, non-affine forces and elastic constants for a Si crystal
    Si_crystal = Diamond('Si', size=[1, 1, 1], latticeconstant=a0)
    compute_forces_and_hessian(Si_crystal, AbellTersoffBrenner(t3.Tersoff_PRB_39_5566_Si_C))
    compute_elastic_constants(Si_crystal, AbellTersoffBrenner(t3.Tersoff_PRB_39_5566_Si_C))


@pytest.mark.parametrize('a0', [4.2, 4.3, 4.4, 4.5])
def test_tersoff_multicomponent_crystal(a0):
    # Test forces, hessian, non-affine forces and elastic constants for a Si-C crystal
    Si_crystal = B3(['Si', 'C'], size=[1, 1, 1], latticeconstant=a0)
    compute_forces_and_hessian(Si_crystal, AbellTersoffBrenner(t3.Tersoff_PRB_39_5566_Si_C))
    compute_elastic_constants(Si_crystal, AbellTersoffBrenner(t3.Tersoff_PRB_39_5566_Si_C))


def test_tersoff_amorphous():
    # Tests for amorphous Si
    aSi = ase.io.read('aSi_N8.xyz')
    aSi.set_calculator(AbellTersoffBrennerStillingerWeber(**AbellTersoffBrenner(t3.Tersoff_PRB_39_5566_Si_C)))
    # Non-zero forces and Hessian
    compute_forces_and_hessian(aSi, AbellTersoffBrenner(t3.Tersoff_PRB_39_5566_Si_C))
    # Test forces, hessian, non-affine forces and elastic constants for amorphous Si
    FIRE(aSi).run(fmax=1e-5, steps=1e3)
    compute_forces_and_hessian(aSi, AbellTersoffBrenner(t3.Tersoff_PRB_39_5566_Si_C))
    compute_elastic_constants(aSi, AbellTersoffBrenner(t3.Tersoff_PRB_39_5566_Si_C))


@pytest.mark.parametrize('d', np.arange(1.0, 1.8, 0.15))
def test_stillinger_weber_small1(d):
    # Test forces and hessian matrix for Stillinger-Weber
    small = Atoms([14] * 4, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d)], cell=(100, 100, 100))
    small.center(vacuum=10.0)
    compute_forces_and_hessian(small, StillingerWeber(sw.original_SW))


@pytest.mark.parametrize('d', np.arange(1.0, 1.8, 0.15))
def test_stillinger_weber_small2(d):
    # Test forces and hessian matrix for Stillinger-Weber
    small2 = Atoms([14] * 5, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d), (0, d, d)], cell=(100, 100, 100))
    small2.center(vacuum=10.0)
    compute_forces_and_hessian(small2, StillingerWeber(sw.original_SW))


@pytest.mark.parametrize('a0', [5.2, 5.3, 5.4, 5.5])
def test_stillinger_weber_crystal(a0):
    # Test forces, hessian, non-affine forces and elastic constants for a Si crystal
    Si_crystal = Diamond('Si', size=[1, 1, 1], latticeconstant=a0)
    compute_forces_and_hessian(Si_crystal, StillingerWeber(sw.original_SW))
    compute_elastic_constants(Si_crystal, StillingerWeber(sw.original_SW))


def test_stillinger_weber_amorphous():
    # Tests for amorphous Si
    aSi = ase.io.read('aSi_N8.xyz')
    aSi.set_calculator(AbellTersoffBrennerStillingerWeber(**StillingerWeber(sw.original_SW)))
    # Non-zero forces and Hessian
    compute_forces_and_hessian(aSi, StillingerWeber(sw.original_SW))
    # Test forces, hessian, non-affine forces and elastic constants for amorphous Si
    FIRE(aSi).run(fmax=1e-5, steps=1e3)
    compute_forces_and_hessian(aSi, StillingerWeber(sw.original_SW))
    compute_elastic_constants(aSi, StillingerWeber(sw.original_SW))


def test_generic_potential_form():
    test_cutoff = 2.4
    d = 2.0  # Si2 bondlength
    small = Atoms([14] * 4, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d)], cell=(100, 100, 100))
    small.center(vacuum=10.0)
    small2 = Atoms([14] * 5, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d), (0, d, d)], cell=(100, 100, 100))
    small2.center(vacuum=10.0)
    compute_forces_and_hessian(small, term1())
    compute_forces_and_hessian(small, term4())
    compute_forces_and_hessian(small, d11_term5())
    compute_forces_and_hessian(small, d22_term5())

    compute_forces_and_hessian(small2, term1())
    compute_forces_and_hessian(small2, term4())
    compute_forces_and_hessian(small2, d11_term5())
    compute_forces_and_hessian(small2, d22_term5())


# 0 - Tests Hessian term #4 (with all other terms turned off)
def term4(test_cutoff):
    return {
        'atom_type': lambda n: np.zeros_like(n),
        'pair_type': lambda i, j: np.zeros_like(i),
        'F': lambda x, y, p: x,
        'G': lambda x, y, i, ij: np.ones_like(x[:, 0]),
        'd1F': lambda x, y, p: np.ones_like(x),
        'd11F': lambda x, y, p: np.zeros_like(x),
        'd2F': lambda x, y, p: np.zeros_like(y),
        'd22F': lambda x, y, p: np.zeros_like(y),
        'd12F': lambda x, y, p: np.zeros_like(y),
        'd1G': lambda x, y, i, ij: np.zeros_like(y),
        'd2G': lambda x, y, i, ij: np.zeros_like(y),
        'd11G': lambda x, y, i, ij: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        # if beta <= 1 else beta*(beta-1)*x.**(beta-2) * y[:, 2]**gamma,
        'd12G': lambda x, y, i, ij: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'd22G': lambda x, y, i, ij: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'cutoff': test_cutoff}


# 1 - Tests Hessian term #1 (and #4, with all other terms turned off)
def term1(test_cutoff):
    return {
        'atom_type': lambda n: np.zeros_like(n),
        'pair_type': lambda i, j: np.zeros_like(i),
        'F': lambda x, y, p: x ** 2,
        'G': lambda x, y, i, ij: np.ones_like(x[:, 0]),
        'd1F': lambda x, y, p: 2 * x,
        'd11F': lambda x, y, p: 2 * np.ones_like(x),
        'd2F': lambda x, y, p: np.zeros_like(y),
        'd22F': lambda x, y, p: np.zeros_like(y),
        'd12F': lambda x, y, p: np.zeros_like(y),
        'd1G': lambda x, y, i, ij: np.zeros_like(x),
        'd2G': lambda x, y, i, ij: np.zeros_like(y),
        'd11G': lambda x, y, i, ij: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'd12G': lambda x, y, i, ij: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'd22G': lambda x, y, i, ij: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'cutoff': test_cutoff}


# 2 - Tests D_11 parts of Hessian term #5
def d11_term5(test_cutoff):
    return {
        'atom_type': lambda n: np.zeros_like(n),
        'pair_type': lambda i, j: np.zeros_like(i),
        'F': lambda x, y, p: y,
        'G': lambda x, y, i, ij: np.sum(x ** 2, axis=1),
        'd1F': lambda x, y, p: np.zeros_like(x),
        'd11F': lambda x, y, p: np.zeros_like(x),
        'd2F': lambda x, y, p: np.ones_like(x),
        'd22F': lambda x, y, p: np.zeros_like(x),
        'd12F': lambda x, y, p: np.zeros_like(x),
        'd1G': lambda x, y, i, ij: 2 * x,
        'd2G': lambda x, y, i, ij: np.zeros_like(y),
        'd11G': lambda x, y, i, ij: np.array([2 * np.eye(3)] * x.shape[0]),
        # np.ones_like(x).reshape(-1,3,1)*np.ones_like(y).reshape(-1,1,3), #if beta <= 1 else beta*(beta-1)*x.**(beta-2) * y[:, 2]**gamma,
        'd12G': lambda x, y, i, ij: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'd22G': lambda x, y, i, ij: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'cutoff': test_cutoff}


# 3 - Tests D_22 parts of Hessian term #5
def d22_term5(test_cutoff):
    return {
        'atom_type': lambda n: np.zeros_like(n),
        'pair_type': lambda i, j: np.zeros_like(i),
        'F': lambda x, y, p: y,
        'G': lambda x, y, i, ij: np.sum(y ** 2, axis=1),
        'd1F': lambda x, y, p: np.zeros_like(x),
        'd11F': lambda x, y, p: np.zeros_like(x),
        'd2F': lambda x, y, p: np.ones_like(x),
        'd22F': lambda x, y, p: np.zeros_like(x),
        'd12F': lambda x, y, p: np.zeros_like(x),
        'd2G': lambda x, y, i, ij: 2 * y,
        'd1G': lambda x, y, i, ij: np.zeros_like(x),
        'd22G': lambda x, y, i, ij: np.array([2 * np.eye(3)] * x.shape[0]),
        # np.ones_like(x).reshape(-1,3,1)*np.ones_like(y).reshape(-1,1,3), #if beta <= 1 else beta*(beta-1)*x.**(beta-2) * y[:, 2]**gamma,
        'd12G': lambda x, y, i, ij: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'd11G': lambda x, y, i, ij: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'cutoff': test_cutoff}


def compute_forces_and_hessian(a, par):
    # function to test the bop AbellTersoffBrenner class on
    # a potential given by the form defined in par

    # Parameters
    # ----------
    # a : ase atoms object
    #    passes an atomic configuration as an ase atoms object
    # par : bop explicit form
    #   defines the explicit form of the bond order potential

    calculator = AbellTersoffBrennerStillingerWeber(**par)
    a.set_calculator(calculator)

    # Forces
    ana_forces = a.get_forces()
    num_forces = calculator.calculate_numerical_forces(a, d=1e-5)
    # print('num\n', num_forces)
    # print('ana\n', ana_forces)
    np.testing.assert_allclose(ana_forces, num_forces, atol=1e-3)

    # Hessian
    ana_hessian = calculator.get_hessian(a).todense()
    num_hessian = fd_hessian(a, dx=1e-5, indices=None).todense()
    # print('ana\n', ana_hessian)
    # print('num\n', num_hessian)
    # print('ana - num\n', (np.abs(ana_hessian - num_hessian) > 1e-6).astype(int))
    np.testing.assert_allclose(ana_hessian, ana_hessian.T, atol=1e-6)
    np.testing.assert_allclose(ana_hessian, num_hessian, atol=1e-4)

    ana2_hessian = calculator.get_hessian_from_second_derivative(a)
    np.testing.assert_allclose(ana2_hessian, ana2_hessian.T, atol=1e-6)
    np.testing.assert_allclose(ana_hessian, ana2_hessian, atol=1e-5)


def compute_elastic_constants(a, par):
    # function to test the bop AbellTersoffBrenner class on
    # a potential given by the form defined in par

    # Parameters
    # ----------
    # a : ase atoms object
    #    passes an atomic configuration as an ase atoms object
    # par : bop explicit form
    #   defines the explicit form of the bond order potential

    calculator = AbellTersoffBrennerStillingerWeber(**par)
    a.set_calculator(calculator)

    # Non-affine forces
    num_naF = MatscipyCalculator().get_numerical_non_affine_forces(a, d=1e-5)
    ana_naF1 = calculator.get_non_affine_forces_from_second_derivative(a)
    ana_naF2 = calculator.get_non_affine_forces(a)
    # print("num_naF[0]: \n", num_naF[0])
    # print("ana_naF1[0]: \n", ana_naF1[0])
    # print("ana_naF2[0]: \n", ana_naF2[0])
    np.testing.assert_allclose(ana_naF1, num_naF, atol=0.01)
    np.testing.assert_allclose(ana_naF1, ana_naF2, atol=1e-4)

    # Birch elastic constants
    C_num, Cerr = fit_elastic_constants(a, symmetry="triclinic", N_steps=7, delta=1e-4, optimizer=None,
                                        verbose=False)
    B_ana = calculator.get_birch_coefficients(a)
    # print("C (fit_elastic_constants): \n", C_num[0, 0], C_num[0, 1], C_num[3, 3])
    # print("B_ana: \n", full_3x3x3x3_to_Voigt_6x6(B_ana)[0, 0], full_3x3x3x3_to_Voigt_6x6(B_ana)[0, 1], full_3x3x3x3_to_Voigt_6x6(B_ana)[3, 3])
    np.testing.assert_allclose(C_num, full_3x3x3x3_to_Voigt_6x6(B_ana), atol=0.1)

    # Non-affine elastic constants
    C_num, Cerr = fit_elastic_constants(a, symmetry="triclinic", N_steps=7, delta=1e-4, optimizer=FIRE, fmax=1e-5,
                                        verbose=False)
    B_ana = calculator.get_birch_coefficients(a)
    C_na = calculator.get_non_affine_contribution_to_elastic_constants(a, tol=1e-5)
    # print("C (fit_elastic_constants): \n", C_num[0, 0], C_num[0, 1], C_num[3, 3])
    # print("B_ana + C_na: \n", full_3x3x3x3_to_Voigt_6x6(B_ana+C_na)[0, 0], full_3x3x3x3_to_Voigt_6x6(B_ana+C_na)[0, 1], full_3x3x3x3_to_Voigt_6x6(B_ana+C_na)[3, 3])
    np.testing.assert_allclose(C_num, full_3x3x3x3_to_Voigt_6x6(B_ana + C_na), atol=0.1)
