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

import matscipy.calculators.manybody.explicit_forms.stillinger_weber as stillinger_weber
import matscipy.calculators.manybody.explicit_forms.kumagai as kumagai
import matscipy.calculators.manybody.explicit_forms.tersoff_brenner as tersoff_brenner
from matscipy.calculators.manybody import Manybody
from matscipy.calculators.manybody.explicit_forms import Kumagai, TersoffBrenner, StillingerWeber
from matscipy.hessian_finite_differences import fd_hessian
from matscipy.elasticity import fit_elastic_constants, full_3x3x3x3_to_Voigt_6x6, Voigt_6x6_to_full_3x3x3x3, birch_coefficients
from matscipy.calculators.calculator import MatscipyCalculator

### These parameter sets disable specific terms

# Tests term #1 (with all other terms turned off)
def term1(test_cutoff):
    return {
        'atom_type': lambda n: np.zeros_like(n),
        'pair_type': lambda i, j: np.zeros_like(i),
        'phi': lambda R, r, xi, i, p: R,
        'd1phi': lambda R, r, xi, i, p: np.ones_like(R),
        'd2phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'd11phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'd22phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'd12phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.ones_like(Rij),
        'd1theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd2theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd3theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd11theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd22theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd33theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd12theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd13theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd23theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'cutoff': test_cutoff}


# Tests term #2 (and #1, with all other terms turned off)
def term2(test_cutoff):
    return {
        'atom_type': lambda n: np.zeros_like(n),
        'pair_type': lambda i, j: np.zeros_like(i),
        'phi': lambda R, r, xi, i, p: R * R,
        'd1phi': lambda R, r, xi, i, p: 2 * R,
        'd2phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'd11phi': lambda R, r, xi, i, p: 2 * np.ones_like(R),
        'd22phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'd12phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.ones_like(Rij),
        'd1theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd2theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd3theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd11theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd22theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd33theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd12theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd13theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'd23theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: np.zeros_like(Rij),
        'cutoff': test_cutoff}


# Tests term #3 (with all other terms turned off)
def term3(test_cutoff):
    return {
        'atom_type': lambda n: np.zeros_like(n),
        'pair_type': lambda i, j: np.zeros_like(i),
        'phi': lambda R, r, xi, i, p: xi,
        'd1phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'd2phi': lambda R, r, xi, i, p: np.ones_like(R),
        'd11phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'd22phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'd12phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: Rij * Rij * Rik * Rik * Rjk * Rjk,
        'd1theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rij * Rik * Rik * Rjk * Rjk,
        'd2theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rij * Rij * Rik * Rjk * Rjk,
        'd3theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rij * Rij * Rik * Rik * Rjk,
        'd11theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rik * Rik * Rjk * Rjk,
        'd22theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rij * Rij * Rjk * Rjk,
        'd33theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rij * Rij * Rik * Rik,
        'd12theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 4 * Rij * Rik * Rjk * Rjk,
        'd13theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 4 * Rij * Rik * Rik * Rjk,
        'd23theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 4 * Rij * Rij * Rik * Rjk,
        'cutoff': test_cutoff}


# Tests term #5 (and #3, with all other terms turned off)
def term5(test_cutoff):
    return {
        'atom_type': lambda n: np.zeros_like(n),
        'pair_type': lambda i, j: np.zeros_like(i),
        'phi': lambda R, r, xi, i, p: xi * xi,
        'd1phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'd2phi': lambda R, r, xi, i, p: 2 * xi,
        'd11phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'd22phi': lambda R, r, xi, i, p: 2 * np.ones_like(R),
        'd12phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: Rij * Rij * Rik * Rik * Rjk * Rjk,
        'd1theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rij * Rik * Rik * Rjk * Rjk,
        'd2theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rij * Rij * Rik * Rjk * Rjk,
        'd3theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rij * Rij * Rik * Rik * Rjk,
        'd11theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rik * Rik * Rjk * Rjk,
        'd22theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rij * Rij * Rjk * Rjk,
        'd33theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rij * Rij * Rik * Rik,
        'd12theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 4 * Rij * Rik * Rjk * Rjk,
        'd13theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 4 * Rij * Rik * Rik * Rjk,
        'd23theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 4 * Rij * Rij * Rik * Rjk,
        'cutoff': test_cutoff}


# Tests term #4 (and #1 and #3, with terms #2 and #5 turned off)
def term4(test_cutoff):
    return {
        'atom_type': lambda n: np.zeros_like(n),
        'pair_type': lambda i, j: np.zeros_like(i),
        'phi': lambda R, r, xi, i, p: R * xi,
        'd1phi': lambda R, r, xi, i, p: xi,
        'd2phi': lambda R, r, xi, i, p: R,
        'd11phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'd22phi': lambda R, r, xi, i, p: np.zeros_like(R),
        'd12phi': lambda R, r, xi, i, p: np.ones_like(R),
        'theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: Rij * Rij * Rik * Rik * Rjk * Rjk,
        'd1theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rij * Rik * Rik * Rjk * Rjk,
        'd2theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rij * Rij * Rik * Rjk * Rjk,
        'd3theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rij * Rij * Rik * Rik * Rjk,
        'd11theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rik * Rik * Rjk * Rjk,
        'd22theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rij * Rij * Rjk * Rjk,
        'd33theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 2 * Rij * Rij * Rik * Rik,
        'd12theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 4 * Rij * Rik * Rjk * Rjk,
        'd13theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 4 * Rij * Rik * Rik * Rjk,
        'd23theta': lambda Rij, rij, Rik, rik, Rjk, rjk, i, ij, ik: 4 * Rij * Rij * Rik * Rjk,
        'cutoff': test_cutoff}

###

def assert_numerical_first_derivatives(err_msg, f, *args):
    eps = 1e-6
    nb_args = len(args)
    a0 = 3 * np.random.random(nb_args) + 0.001
    for i in range(nb_args):
        a = a0.copy()
        a[i] += eps
        fp = f(*a)
        a[i] -= 2 * eps
        fm = f(*a)
        np.testing.assert_allclose(args[i](*a0), (fp - fm) / (2 * eps),
                                   err_msg=f'when varying argument {i} of function {err_msg}',
                                   rtol=1e-6)

@pytest.mark.parametrize('par', [
    term1(3.0),
    term2(3.0),
    term3(3.0),
    term5(3.0),
    term4(3.0),
    Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si)])
#                                 TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
#                                 StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si)])
def test_derivatives_phi(par):
    phi = par['phi']
    d1phi = par['d1phi']
    d2phi = par['d2phi']
    d11phi = par['d11phi']
    d12phi = par['d12phi']
    d22phi = par['d22phi']
    assert_numerical_first_derivatives(
        'phi',
        lambda R, xi: phi(R, np.sqrt(R), xi, 0, 0),
        lambda R, xi: d1phi(R, np.sqrt(R), xi, 0, 0),
        lambda R, xi: d2phi(R, np.sqrt(R), xi, 0, 0)
    )
    assert_numerical_first_derivatives(
        'd1phi',
        lambda R, xi: d1phi(R, np.sqrt(R), xi, 0, 0),
        lambda R, xi: d11phi(R, np.sqrt(R), xi, 0, 0),
        lambda R, xi: d12phi(R, np.sqrt(R), xi, 0, 0)
    )
    assert_numerical_first_derivatives(
        'd2phi',
        lambda R, xi: d2phi(R, np.sqrt(R), xi, 0, 0),
        lambda R, xi: d12phi(R, np.sqrt(R), xi, 0, 0),
        lambda R, xi: d22phi(R, np.sqrt(R), xi, 0, 0)
    )


@pytest.mark.parametrize('par', [
    term1(3.0),
    term2(3.0),
    term3(3.0),
    term5(3.0),
    term4(3.0),
    Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si)])
#                                 TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
#                                 StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si)])
def test_derivatives_theta(par):
    theta = par['theta']
    d1theta = par['d1theta']
    d2theta = par['d2theta']
    d3theta = par['d3theta']
    d11theta = par['d11theta']
    d12theta = par['d12theta']
    d13theta = par['d13theta']
    d22theta = par['d22theta']
    d23theta = par['d23theta']
    d33theta = par['d33theta']
    assert_numerical_first_derivatives(
        'theta',
        lambda Rij, Rik, Rjk: theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
        lambda Rij, Rik, Rjk: d1theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
        lambda Rij, Rik, Rjk: d2theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
        lambda Rij, Rik, Rjk: d3theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
    )
    assert_numerical_first_derivatives(
        'd1theta',
        lambda Rij, Rik, Rjk: d1theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
        lambda Rij, Rik, Rjk: d11theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
        lambda Rij, Rik, Rjk: d12theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
        lambda Rij, Rik, Rjk: d13theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
    )
    assert_numerical_first_derivatives(
        'd2theta',
        lambda Rij, Rik, Rjk: d2theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
        lambda Rij, Rik, Rjk: d12theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
        lambda Rij, Rik, Rjk: d22theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
        lambda Rij, Rik, Rjk: d23theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
    )
    assert_numerical_first_derivatives(
        'd3theta',
        lambda Rij, Rik, Rjk: d3theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
        lambda Rij, Rik, Rjk: d13theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
        lambda Rij, Rik, Rjk: d23theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
        lambda Rij, Rik, Rjk: d33theta(Rij, np.sqrt(Rij), Rik, np.sqrt(Rik), Rjk, np.sqrt(Rjk), 0, 0, 0),
    )

@pytest.mark.parametrize('term', [term1, term2, term3, term5, term4])
@pytest.mark.parametrize('test_cutoff', [3.0])
def test_generic_potential_form1(test_cutoff, term):
    d = 2.0  # Si2 bondlength
    small = Atoms([14] * 4, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d)], cell=(2 * d, 2 * d, 2 * d))
    #small.center(vacuum=10.0)
    compute_elastic_constants(small, term(test_cutoff))
    #compute_forces_and_hessian(small, term(test_cutoff))


@pytest.mark.parametrize('term', [term1, term2, term3, term5, term4])
@pytest.mark.parametrize('test_cutoff', [3.5])
def test_generic_potential_form2(test_cutoff, term):
    d = 2.0  # Si2 bondlength
    small2 = Atoms([14] * 5, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d), (0, d, d)], cell=(2 * d, 2 * d, 2 * d))
    #small2.center(vacuum=10.0)
    compute_elastic_constants(small2, term(test_cutoff))
    #compute_forces_and_hessian(small2, term(test_cutoff))


@pytest.mark.parametrize('a0', [5.2, 5.3, 5.4, 5.5])
@pytest.mark.parametrize('par', [Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si)])
#                                 TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
#                                 StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si)])
def test_stress(a0, par):
    atoms = Diamond('Si', size=[1, 1, 1], latticeconstant=a0)
    calculator = Manybody(**par)
    atoms.calc = calculator
    s = atoms.get_stress()
    sn = calculator.calculate_numerical_stress(atoms, d=0.0001)
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
@pytest.mark.parametrize('par', [Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si)])
#                                 TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
#                                 StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si)])
def test_small1(d, par):
    # Test forces and hessian matrix for Kumagai
    small = Atoms([14] * 4, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d)], cell=(100, 100, 100))
    small.center(vacuum=10.0)
    compute_forces_and_hessian(small, par)


@pytest.mark.parametrize('d', np.arange(1.0, 2.3, 0.15))
@pytest.mark.parametrize('par', [Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si),
                                 TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
                                 StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si)])
def test_small2(d, par):
    # Test forces and hessian matrix for Kumagai
    small2 = Atoms([14] * 5, [(d, 0, d / 2), (0, 0, 0), (d, 0, 0), (0, 0, d), (0, d, d)], cell=(100, 100, 100))
    small2.center(vacuum=10.0)
    compute_forces_and_hessian(small2, par)


@pytest.mark.parametrize('a0', [5.2, 5.3, 5.4, 5.5])
@pytest.mark.parametrize('par', [Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si)]) #,
#                                 TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
#                                 StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si)])
def test_crystal_forces_and_hessian(a0, par):
    # Test forces, hessian, non-affine forces and elastic constants for a Si crystal
    Si_crystal = Diamond('Si', size=[1, 1, 1], latticeconstant=a0)
    compute_forces_and_hessian(Si_crystal, par)


@pytest.mark.parametrize('a0', [5.0, 5.2, 5.3, 5.4, 5.5])
@pytest.mark.parametrize('par', [Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si)])
#                                 TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
#                                 StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si)])
def test_crystal_elastic_constants(a0, par):
    # Test forces, hessian, non-affine forces and elastic constants for a Si crystal
    Si_crystal = Diamond('Si', size=[1, 1, 1], latticeconstant=a0)
    compute_elastic_constants(Si_crystal, par)


@pytest.mark.parametrize('par', [Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si),
                                 TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
                                 StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si)])
def test_amorphous(par):
    # Tests for amorphous Si
    aSi = ase.io.read('aSi_N8.xyz')
    aSi.calc = Manybody(**par)
    # Non-zero forces and Hessian
    compute_forces_and_hessian(aSi, par)
    # Test forces, hessian, non-affine forces and elastic constants for a stress-free amorphous Si configuration
    FIRE(ase.constraints.UnitCellFilter(aSi, mask=[1,1,1,1,1,1], hydrostatic_strain=False), logfile=None).run(fmax=1e-5)    
    compute_forces_and_hessian(aSi, par)
    compute_elastic_constants(aSi, par)


@pytest.mark.parametrize('a0', [4.3, 4.4, 4.5])
@pytest.mark.parametrize('par', [TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
                                 TersoffBrenner(tersoff_brenner.Erhart_PRB_71_035211_SiC)])
def test_tersoff_multicomponent_crystal_forces_and_hessian(a0, par):
    # Test forces, hessian, non-affine forces and elastic constants for a Si-C crystal
    Si_crystal = B3(['Si', 'C'], size=[1, 1, 1], latticeconstant=a0)
    compute_forces_and_hessian(Si_crystal, par)


@pytest.mark.parametrize('a0', [4.3, 4.4, 4.5])
@pytest.mark.parametrize('par', [TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
                                 TersoffBrenner(tersoff_brenner.Erhart_PRB_71_035211_SiC)])
def test_tersoff_multicomponent_crystal_elastic_constants(a0, par):
    # Test forces, hessian, non-affine forces and elastic constants for a Si-C crystal
    Si_crystal = B3(['Si', 'C'], size=[1, 1, 1], latticeconstant=a0)
    compute_elastic_constants(Si_crystal, par)


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
    num_forces = calculator.calculate_numerical_forces(a, d=1e-5)
    # print('num\n', num_forces)
    # print('ana\n', ana_forces)
    np.testing.assert_allclose(ana_forces, num_forces, atol=1e-3)

    # Hessian
    ana_hessian = np.array(calculator.get_hessian(a).todense())
    num_hessian = np.array(fd_hessian(a, dx=1e-5, indices=None).todense())
    # print('ana\n', ana_hessian)
    # print('num\n', num_hessian)
    # print('ana - num\n', (np.abs(ana_hessian - num_hessian) > 1e-6).astype(int))
    np.testing.assert_allclose(ana_hessian, ana_hessian.T, atol=1e-6, err_msg='when checking for symmetric Hessian')
    np.testing.assert_allclose(ana_hessian, num_hessian, atol=1e-4, err_msg='when comparing to numeric Hessian')

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
#    num_naF = MatscipyCalculator().get_numerical_non_affine_forces(a, d=1e-5)
#    ana_naF1 = calculator.get_non_affine_forces_from_second_derivative(a)
#    ana_naF2 = calculator.get_non_affine_forces(a)
    # print("num_naF[0]: \n", num_naF[0])
    # print("ana_naF1[0]: \n", ana_naF1[0])
    # print("ana_naF2[0]: \n", ana_naF2[0])
#    np.testing.assert_allclose(ana_naF1, num_naF, atol=0.01)
#    np.testing.assert_allclose(ana_naF1, ana_naF2, atol=1e-4)

    # Birch elastic constants
#    C_num, Cerr = fit_elastic_constants(a, symmetry="triclinic", N_steps=7, delta=1e-4, optimizer=None,
#                                        verbose=False)
    C_num = birch_coefficients(a)
    B_ana = calculator.get_birch_coefficients(a)

    # print("C (fit_elastic_constants): \n", C_num)
    # print("B_ana: \n", full_3x3x3x3_to_Voigt_6x6(B_ana))
    #np.testing.assert_allclose(C_num, full_3x3x3x3_to_Voigt_6x6(B_ana), atol=0.1)
    np.testing.assert_allclose(C_num, B_ana, rtol=1e-4, atol=1e-6)

    return

    # Non-affine elastic constants
    C_num, Cerr = fit_elastic_constants(a, symmetry="triclinic", N_steps=7, delta=1e-4, optimizer=FIRE, fmax=1e-5,
                                        verbose=False)
    C_na = calculator.get_non_affine_contribution_to_elastic_constants(a, tol=1e-5)
    # print("C (fit_elastic_constants): \n", C_num[0, 0], C_num[0, 1], C_num[3, 3])
    # print("B_ana + C_na: \n", full_3x3x3x3_to_Voigt_6x6(B_ana+C_na)[0, 0], full_3x3x3x3_to_Voigt_6x6(B_ana+C_na)[0, 1], full_3x3x3x3_to_Voigt_6x6(B_ana+C_na)[3, 3])
    np.testing.assert_allclose(C_num, full_3x3x3x3_to_Voigt_6x6(B_ana + C_na), atol=0.1)
