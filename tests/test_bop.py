#
# Copyright 2020-2021 Lars Pastewka (U. Freiburg)
#           2021 Jan Griesser (U. Freiburg)
#           2020-2021 Jonas Oldenstaedt (U. Freiburg)
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

import pytest

import ase
import ase.constraints
from ase import Atoms
from ase.optimize import FIRE
from ase.lattice.compounds import B3
from ase.lattice.cubic import Diamond

import matscipy.calculators.manybody.explicit_forms.stillinger_weber \
    as stillinger_weber
import matscipy.calculators.manybody.explicit_forms.kumagai as kumagai
import matscipy.calculators.manybody.explicit_forms.tersoff_brenner \
    as tersoff_brenner
from matscipy.calculators.manybody import Manybody
from matscipy.calculators.manybody.explicit_forms import (
    Kumagai,
    TersoffBrenner,
    StillingerWeber,
)
from matscipy.numerical import (
    numerical_hessian,
    numerical_forces,
    numerical_stress,
    numerical_nonaffine_forces,
)
from matscipy.elasticity import (
    fit_elastic_constants,
    measure_triclinic_elastic_constants,
)


@pytest.mark.parametrize('a0', [5.2, 5.3, 5.4, 5.5])
@pytest.mark.parametrize('par', [
    Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si),
    TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
    StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si)
])
def test_stress(a0, par):
    atoms = Diamond('Si', size=[1, 1, 1], latticeconstant=a0)
    calculator = Manybody(**par)
    atoms.calc = calculator
    s = atoms.get_stress()
    sn = numerical_stress(atoms, d=0.0001)
    np.testing.assert_allclose(s, sn, atol=1e-6)


def test_hessian_divide_by_masses():
    # Test the computation of dynamical matrix
    atoms = ase.io.read('aSi.cfg')
    masses_n = np.random.randint(1, 10, size=len(atoms))
    atoms.set_masses(masses=masses_n)
    kumagai_potential = kumagai.Kumagai_Comp_Mat_Sci_39_Si
    calc = Manybody(**Kumagai(kumagai_potential))
    D_ana = calc.get_hessian(atoms, divide_by_masses=True).todense()
    H_ana = calc.get_hessian(atoms).todense()
    masses_nc = masses_n.repeat(3)
    H_ana /= np.sqrt(masses_nc.reshape(-1, 1) * masses_nc.reshape(1, -1))
    np.testing.assert_allclose(D_ana, H_ana, atol=1e-4)


@pytest.mark.parametrize('d', np.arange(1.0, 2.3, 0.15))
@pytest.mark.parametrize('par', [
    Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si),
    TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
    StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si)
])
def test_small1(d, par):
    # Test forces and hessian matrix for Kumagai
    small = Atoms(
        [14] * 4, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d)],
        cell=(100, 100, 100))
    small.center(vacuum=10.0)
    compute_forces_and_hessian(small, par)


@pytest.mark.parametrize('d', np.arange(1.0, 2.3, 0.15))
@pytest.mark.parametrize('par', [
    Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si),
    TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
    StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si)
])
def test_small2(d, par):
    # Test forces and hessian matrix for Kumagai
    small2 = Atoms(
        [14] * 5, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d), (0, d, d)],
        cell=(100, 100, 100))
    small2.center(vacuum=10.0)
    compute_forces_and_hessian(small2, par)


@pytest.mark.parametrize('a0', [5.2, 5.3, 5.4, 5.5])
@pytest.mark.parametrize('par', [
    Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si),
    TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
    StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si)
])
def test_crystal_forces_and_hessian(a0, par):
    # Test forces, hessian, non-affine forces and elastic constants for a Si crystal
    Si_crystal = Diamond('Si', size=[1, 1, 1], latticeconstant=a0)
    compute_forces_and_hessian(Si_crystal, par)


@pytest.mark.parametrize('a0', [5.0, 5.2, 5.3, 5.4, 5.5])
@pytest.mark.parametrize('par', [
    Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si),
    TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
    StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si)
])
def test_crystal_elastic_constants(a0, par):
    # Test forces, hessian, non-affine forces and elastic constants for a Si crystal
    Si_crystal = Diamond('Si', size=[1, 1, 1], latticeconstant=a0)
    compute_elastic_constants(Si_crystal, par)


@pytest.mark.parametrize('par', [
    Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si),
    TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
    StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si)
])
def test_amorphous(par):
    # Tests for amorphous Si
    aSi = ase.io.read('aSi_N8.xyz')
    aSi.calc = Manybody(**par)
    # Non-zero forces and Hessian
    compute_forces_and_hessian(aSi, par)
    # Test forces, hessian, non-affine forces and elastic constants for a stress-free amorphous Si configuration
    FIRE(
        ase.constraints.UnitCellFilter(
            aSi, mask=[1, 1, 1, 1, 1, 1], hydrostatic_strain=False),
        logfile=None).run(fmax=1e-5)
    compute_forces_and_hessian(aSi, par)
    compute_elastic_constants(aSi, par)


@pytest.mark.parametrize('a0', [4.3, 4.4, 4.5])
@pytest.mark.parametrize('par', [
    TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
    TersoffBrenner(tersoff_brenner.Erhart_PRB_71_035211_SiC)
])
def test_tersoff_multicomponent_crystal_forces_and_hessian(a0, par):
    # Test forces, hessian, non-affine forces and elastic constants for a Si-C crystal
    Si_crystal = B3(['Si', 'C'], size=[1, 1, 1], latticeconstant=a0)
    compute_forces_and_hessian(Si_crystal, par)


@pytest.mark.parametrize('a0', [4.3, 4.4, 4.5])
@pytest.mark.parametrize('par', [
    TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
    TersoffBrenner(tersoff_brenner.Erhart_PRB_71_035211_SiC)
])
def test_tersoff_multicomponent_crystal_elastic_constants(a0, par):
    # Test forces, hessian, non-affine forces and elastic constants for a Si-C crystal
    Si_crystal = B3(['Si', 'C'], size=[1, 1, 1], latticeconstant=a0)
    compute_elastic_constants(Si_crystal, par)


# 0 - Tests Hessian term #4 (with all other terms turned off)
def term4(test_cutoff):
    return {
        'atom_type':
        lambda n: np.zeros_like(n),
        'pair_type':
        lambda i, j: np.zeros_like(i),
        'F':
        lambda x, y, i, p: x,
        'G':
        lambda x, y, i, ij, ik: np.ones_like(x[:, 0]),
        'd1F':
        lambda x, y, i, p: np.ones_like(x),
        'd11F':
        lambda x, y, i, p: np.zeros_like(x),
        'd2F':
        lambda x, y, i, p: np.zeros_like(y),
        'd22F':
        lambda x, y, i, p: np.zeros_like(y),
        'd12F':
        lambda x, y, i, p: np.zeros_like(y),
        'd1G':
        lambda x, y, i, ij, ik: np.zeros_like(y),
        'd2G':
        lambda x, y, i, ij, ik: np.zeros_like(y),
        'd11G':
        lambda x, y, i, ij, ik: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        # if beta <= 1 else beta*(beta-1)*x.**(beta-2) * y[:, 2]**gamma,
        'd12G':
        lambda x, y, i, ij, ik: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'd22G':
        lambda x, y, i, ij, ik: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'cutoff':
        test_cutoff
    }


# 1 - Tests Hessian term #1 (and #4, with all other terms turned off)
def term1(test_cutoff):
    return {
        'atom_type':
        lambda n: np.zeros_like(n),
        'pair_type':
        lambda i, j: np.zeros_like(i),
        'F':
        lambda x, y, i, p: x**2,
        'G':
        lambda x, y, i, ij, ik: np.ones_like(x[:, 0]),
        'd1F':
        lambda x, y, i, p: 2 * x,
        'd11F':
        lambda x, y, i, p: 2 * np.ones_like(x),
        'd2F':
        lambda x, y, i, p: np.zeros_like(y),
        'd22F':
        lambda x, y, i, p: np.zeros_like(y),
        'd12F':
        lambda x, y, i, p: np.zeros_like(y),
        'd1G':
        lambda x, y, i, ij, ik: np.zeros_like(x),
        'd2G':
        lambda x, y, i, ij, ik: np.zeros_like(y),
        'd11G':
        lambda x, y, i, ij, ik: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'd12G':
        lambda x, y, i, ij, ik: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'd22G':
        lambda x, y, i, ij, ik: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'cutoff':
        test_cutoff
    }


# 2 - Tests D_11 parts of Hessian term #5
def d11_term5(test_cutoff):
    return {
        'atom_type':
        lambda n: np.zeros_like(n),
        'pair_type':
        lambda i, j: np.zeros_like(i),
        'F':
        lambda x, y, i, p: y,
        'G':
        lambda x, y, i, ij, ik: np.sum(x**2, axis=1),
        'd1F':
        lambda x, y, i, p: np.zeros_like(x),
        'd11F':
        lambda x, y, i, p: np.zeros_like(x),
        'd2F':
        lambda x, y, i, p: np.ones_like(x),
        'd22F':
        lambda x, y, i, p: np.zeros_like(x),
        'd12F':
        lambda x, y, i, p: np.zeros_like(x),
        'd1G':
        lambda x, y, i, ij, ik: 2 * x,
        'd2G':
        lambda x, y, i, ij, ik: np.zeros_like(y),
        'd11G':
        lambda x, y, i, ij, ik: np.array([2 * np.eye(3)] * x.shape[0]),
        # np.ones_like(x).reshape(-1,3,1)*np.ones_like(y).reshape(-1,1,3), #if beta <= 1 else beta*(beta-1)*x.**(beta-2) * y[:, 2]**gamma,
        'd12G':
        lambda x, y, i, ij, ik: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'd22G':
        lambda x, y, i, ij, ik: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'cutoff':
        test_cutoff
    }


# 3 - Tests D_22 parts of Hessian term #5
def d22_term5(test_cutoff):
    return {
        'atom_type':
        lambda n: np.zeros_like(n),
        'pair_type':
        lambda i, j: np.zeros_like(i),
        'F':
        lambda x, y, i, p: y,
        'G':
        lambda x, y, i, ij, ik: np.sum(y**2, axis=1),
        'd1F':
        lambda x, y, i, p: np.zeros_like(x),
        'd11F':
        lambda x, y, i, p: np.zeros_like(x),
        'd2F':
        lambda x, y, i, p: np.ones_like(x),
        'd22F':
        lambda x, y, i, p: np.zeros_like(x),
        'd12F':
        lambda x, y, i, p: np.zeros_like(x),
        'd2G':
        lambda x, y, i, ij, ik: 2 * y,
        'd1G':
        lambda x, y, i, ij, ik: np.zeros_like(x),
        'd22G':
        lambda x, y, i, ij, ik: np.array([2 * np.eye(3)] * x.shape[0]),
        # np.ones_like(x).reshape(-1,3,1)*np.ones_like(y).reshape(-1,1,3), #if beta <= 1 else beta*(beta-1)*x.**(beta-2) * y[:, 2]**gamma,
        'd12G':
        lambda x, y, i, ij, ik: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'd11G':
        lambda x, y, i, ij, ik: 0 * x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3),
        'cutoff':
        test_cutoff
    }


@pytest.mark.parametrize('term', [term1, term4, d11_term5, d22_term5])
@pytest.mark.parametrize('test_cutoff', [3.0])
def test_generic_potential_form1(test_cutoff, term):
    d = 2.0  # Si2 bondlength
    small = Atoms(
        [14] * 4, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d)],
        cell=(100, 100, 100))
    small.center(vacuum=10.0)
    compute_forces_and_hessian(small, term(test_cutoff))


@pytest.mark.parametrize('term', [term1, term4, d11_term5, d22_term5])
@pytest.mark.parametrize('test_cutoff', [3.5])
def test_generic_potential_form2(test_cutoff, term):
    d = 2.0  # Si2 bondlength
    small2 = Atoms(
        [14] * 5, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d), (0, d, d)],
        cell=(100, 100, 100))
    small2.center(vacuum=10.0)
    compute_forces_and_hessian(small2, term(test_cutoff))


def compute_forces_and_hessian(a, par):
    # function to test the bop AbellTersoffBrenner class on
    # a potential given by the form defined in par

    # Parameters
    # ----------
    # a : ase atoms object
    #    passes an atomic configuration as an ase atoms object
    # par : bop explicit form
    #   defines the explicit form of the bond order potential
    calculator = Manybody(**par)
    a.calc = calculator

    # Forces
    ana_forces = a.get_forces()
    num_forces = numerical_forces(a, d=1e-5)
    # print('num\n', num_forces)
    # print('ana\n', ana_forces)
    np.testing.assert_allclose(ana_forces, num_forces, atol=1e-3)

    # Hessian
    ana_hessian = np.array(calculator.get_hessian(a).todense())
    num_hessian = np.array(
        numerical_hessian(a, dx=1e-5, indices=None).todense())
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

    calculator = Manybody(**par)
    a.calc = calculator

    # Non-affine forces
    num_naF = numerical_nonaffine_forces(a, d=1e-5)
    ana_naF1 = calculator.get_non_affine_forces_from_second_derivative(a)
    ana_naF2 = calculator.get_property('nonaffine_forces', a)
    # print("num_naF[0]: \n", num_naF[0])
    # print("ana_naF1[0]: \n", ana_naF1[0])
    # print("ana_naF2[0]: \n", ana_naF2[0])
    np.testing.assert_allclose(ana_naF1, num_naF, atol=0.01)
    np.testing.assert_allclose(ana_naF1, ana_naF2, atol=1e-4)

    # Birch elastic constants
    C_ana = a.calc.get_property('birch_coefficients', a)
    C_num = measure_triclinic_elastic_constants(a, delta=1e-3)
    np.testing.assert_allclose(C_ana, C_num, atol=.3)

    # Non-affine elastic constants
    C_ana = a.calc.get_property('elastic_constants', a)
    C_num = measure_triclinic_elastic_constants(
        a, optimizer=FIRE, fmax=1e-6, delta=1e-3,
    )
    np.testing.assert_allclose(C_ana, C_num, atol=.3)
