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

import math

import numpy as np

###

def radius_and_pressure(N, R, Es):
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
    """

    a = R*(3./4*( N/(Es*R**2) ))**(1./3)
    p0 = 3*N/(2*math.pi*a*a)
    
    return a, p0


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

    mask0 = np.abs(r) < 1e-6
    maski = np.logical_and(r < a, np.logical_not(mask0))
    masko = np.logical_and(np.logical_not(maski), np.logical_not(mask0))
    r_0 = r[mask0]
    r_i = r[maski]
    r_o = r[masko]
    
    # Initialize
    pz = np.zeros_like(r)
    pr = np.zeros_like(r)
    ptheta = np.zeros_like(r)
    
    # Solution at r=0
    if mask0.sum() > 0:
        pz[mask0] = np.ones_like(r_0)
        pr[mask0] = np.ones_like(r_0)
        ptheta[mask0] = np.ones_like(r_0)

    # Solution inside the contact radius
    if maski.sum() > 0:
        r_a_sq = (r_i/a)**2
        pz[maski] = np.sqrt(1-r_a_sq)
        pr[maski] = (1.-2.*nu)/(3.*r_a_sq)*(1.-(1.-r_a_sq)**(3./2))-np.sqrt(1.-r_a_sq)
        ptheta[maski] = -(1.-2.*nu)/(3.*r_a_sq)*(1.-(1.-r_a_sq)**(3./2))-2*nu*np.sqrt(1.-r_a_sq)
    
    # Solution outside of the contact radius
    if mask0.sum() > 0:
        r_a_sq = (r_o/a)**2
        po = (1.-2.*nu)/(3.*r_a_sq)
        pr[masko] = po
        ptheta[masko] = -po
    
    return pz, pr, ptheta
    
    
def surface_displacements(r, a):
    maski = r < a
    masko = np.logical_not(maski)
    r_i = r[maski]
    r_o = r[masko]
    
    # Initialize
    uz = np.zeros_like(r)
    
    # Solution inside the contact circle
    if maski.sum() > 0:
        uz[maski] = -math.pi*(2*a**2-r_i**2)/(4*a)

    # Solution outside the contact circle
    if masko.sum() > 0:
        uz[masko] = (-(2*a**2-r_o**2)*np.arcsin(a/r_o) - 
            a*r_o*np.sqrt(1-(a/r_o)**2))/(2*a)

    return uz