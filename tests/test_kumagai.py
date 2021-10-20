#
# Copyright 2021 Lars Pastewka (U. Freiburg)
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

"""Finite differences tests for Kumagai paramterization"""

import inspect

import numpy as np

from matscipy.calculators.manybody.explicit_forms.kumagai import Kumagai, Kumagai_Comp_Mat_Sci_39_Si


def assert_numerical_first_derivatives(err_msg, f, *args):
    eps = 1e-6
    nb_args = len(args)
    a0 = np.random.random(nb_args) - 0.5
    for i in range(nb_args):
        a = a0.copy()
        a[i] += eps
        fp = f(*a)
        a[i] -= 2 * eps
        fm = f(*a)
        np.testing.assert_allclose(args[i](*a0), (fp - fm) / (2 * eps),
                                   err_msg=f'when varying argument {i} of function {err_msg}',
                                   rtol=1e-6)


def test_derivatives_phi():
    k = Kumagai(Kumagai_Comp_Mat_Sci_39_Si)
    phi = k['phi']
    d1phi = k['d1phi']
    d2phi = k['d2phi']
    d11phi = k['d11phi']
    d12phi = k['d12phi']
    d22phi = k['d22phi']
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


def test_derivatives_theta():
    k = Kumagai(Kumagai_Comp_Mat_Sci_39_Si)
    theta = k['theta']
    d1theta = k['d1theta']
    d2theta = k['d2theta']
    d3theta = k['d3theta']
    d11theta = k['d11theta']
    d12theta = k['d12theta']
    d13theta = k['d13theta']
    d22theta = k['d22theta']
    d23theta = k['d23theta']
    d33theta = k['d33theta']
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
