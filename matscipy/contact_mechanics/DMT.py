#
# Copyright 2014, 2021 Lars Pastewka (U. Freiburg)
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

import math

import numpy as np

import matscipy.contact_mechanics.Hertz as Hertz

###

def radius_and_pressure(N, R, Es, w):
    """
    Given normal load, sphere radius and contact modulus compute contact radius
    and peak pressure.

    Parameters
    ----------
    N : float
        Normal force.
    R : float
        Sphere radius.
    Es : float
        Contact modulus: Es = E/(1-nu**2) with Young's modulus E and Poisson
        number nu.
    w : float
        Work of adhesion.
    """

    return Hertz.radius_and_pressure(N+2*math.pi*w*R, R, Es)


def surface_stress(r, a, nu):
    """
    Given distance from the center of the sphere, contact radius and Poisson
    number contact, compute the stress at the surface.

    Parameters
    ----------
    r : array_like
        Array of distance (from the center of the sphere).
    a : float
        Contact radius.
    nu : float
        Poisson number.

    Returns
    -------
    pz : array
        Contact pressure.
    sr : array
        Radial stress.
    stheta : array
        Azimuthal stress.
    """

    return Hertz.surface_stress(r, a, nu)
    
    
def surface_displacements(r, a):
    return Hertz.surface_displacements(r, a)