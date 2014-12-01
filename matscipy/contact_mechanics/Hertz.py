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

    Return
    ------
    a : float
        Contact radius.
    p0 : float
        Maximum pressure inside the contacting area (right under the apex).
    """

    a = R*(3./4*( N/(Es*R**2) ))**(1./3)
    p0 = 3*N/(2*math.pi*a*a)
    
    return a, p0


def surface_stress(r, nu=0.5):
    """
    Given distance from the center of the sphere, contact radius and Poisson
    number contact, compute the stress at the surface.

    Parameters
    ----------
    r : array_like
        Array of distance (from the center of the sphere in units of contact
        radius a).
    nu : float
        Poisson number.

    Returns
    -------
    pz : array
        Contact pressure (in units of maximum pressure p0).
    sr : array
        Radial stress (in units of maximum pressure p0).
    stheta : array
        Azimuthal stress (in units of maximum pressure p0).
    """

    mask0 = np.abs(r) < 1e-6
    maski = np.logical_and(r < 1.0, np.logical_not(mask0))
    masko = np.logical_and(np.logical_not(maski), np.logical_not(mask0))
    r_0 = r[mask0]
    r_i = r[maski]
    r_o = r[masko]
    
    # Initialize
    pz = np.zeros_like(r)
    sr = np.zeros_like(r)
    stheta = np.zeros_like(r)
    
    # Solution at r=0
    if mask0.sum() > 0:
        pz[mask0] = np.ones_like(r_0)
        sr[mask0] = -(1.+2*nu)/2.*np.ones_like(r_0)
        stheta[mask0] = -(1.+2*nu)/2.*np.ones_like(r_0)

    # Solution inside the contact radius
    if maski.sum() > 0:
        r_a_sq = r_i**2
        pz[maski] = np.sqrt(1-r_a_sq)
        sr[maski] = (1.-2.*nu)/(3.*r_a_sq)*(1.-(1.-r_a_sq)**(3./2))- \
            np.sqrt(1.-r_a_sq)
        stheta[maski] = -(1.-2.*nu)/(3.*r_a_sq)*(1.-(1.-r_a_sq)**(3./2))- \
            2*nu*np.sqrt(1.-r_a_sq)
    
    # Solution outside of the contact radius
    if masko.sum() > 0:
        r_a_sq = r_o**2
        po = (1.-2.*nu)/(3.*r_a_sq)
        sr[masko] = po
        stheta[masko] = -po
    
    return pz, sr, stheta


def centerline_stress(z, nu=0.5):
    """
    Given distance from the center of the sphere, contact radius and Poisson
    number contact, compute the stress at the surface.

    Parameters
    ----------
    z : array_like
        Array of depths (from the center of the sphere in units of contact
        radius a).
    nu : float
        Poisson number.

    Returns
    -------
    srr : array
        Radial stress (in units of maximum pressure p0).
    szz : array
        Contact pressure (in units of maximum pressure p0).
    """

    srr = -(1.+nu)*(1.-z*np.arctan(1./z)) + 1./(2.*(1.+z**2))
    szz = -1./(1.+z**2)

    return srr, szz

    
def surface_displacements(r):
    """
    Return the displacements at the surface due to an indenting sphere.
    See: K.L. Johnson, Contact Mechanics, p. 61

    Parameters
    ----------
    r : array_like
        Radial position normalized by contact radius a.

    Returns
    -------
    uz : array
        Normal displacements at the surface of the contact (in units of
        p0/Es * a where p0 is maximum pressure, Es contact modulus and a 
        contact radius).
    """

    maski = r < 1.0
    masko = np.logical_not(maski)
    r_i = r[maski]
    r_o = r[masko]
    
    # Initialize
    uz = np.zeros_like(r)
    
    # Solution inside the contact circle
    if maski.sum() > 0:
        uz[maski] = -math.pi*(2.-r_i**2)/4.

    # Solution outside the contact circle
    if masko.sum() > 0:
        uz[masko] = (-(2.-r_o**2)*np.arcsin(1./r_o) - 
            r_o*np.sqrt(1.-(1./r_o)**2))/2.

    return uz


def stress(r, z, nu=0.5):
    """
    Return components of the stress tensor in the bulk of the Hertz solid.
    This is the solution given by: M.T. Huber, Ann. Phys. 319, 153 (1904)

    Note that the stress tensor at any point in the solid has the form below.
    Zero off-diagonal components are zero by rotational symmetry. stt is the
    circumferential component, srr the radial component and szz the normal
    component of the stress tensor.

            / stt  0   0  \ 
        s = |  0  srr srz |
            \  0  srz szz /

    Parameters
    ----------
    r : array_like
        Radial position (in units of the contact radius a).
    z : array_like
        Depth (in units of the contact radius a).
    nu : float
        Poisson number.

    Returns
    -------
    stt : array
        Circumferential component of the stress tensor (in units of maximum
        pressure p0).
    srr : array
        Radial component of the stress tensor (in units of maximum pressure p0).
    szz : array
        Normal component of the stress tensor (in units of maximum pressure p0).
    srz : array
        Shear component of the stress tensor (in units of maximum pressure p0).
    """

    p = r**2+z**2-1
    u = p/2 + np.sqrt(p**2/4+z**2)
    sqrtu = np.sqrt(u)

    # Variable substitution: r->r/sqrt(1+u), z->z/sqrt(u)
    r = r/np.sqrt(1+u)
    z = np.sqrt(1-r**2) # equiv. to z=u/sqrt(u), but defined for z=0

    # Precompute arctan
    sqrtu_arctan_inv_sqrtu = sqrtu*np.arctan(1./sqrtu)

    # The next two expressions give numerical problems at the tip center and the
    # contact edge, regularize with the asymptotic value.

    # Note: r**2/(1-z**3)->3/2 for r->0
    one_minus_z3_div_r2 = np.where(r > 0., (1.-z**3)/r**2, 3./2.*np.ones_like(r))

    # Note: z**2/(u+z**2)->1 for u+z**2->0
    u_plus_z2 = u+z**2
    z2_div_u_plus_z2 = np.where(u_plus_z2 > 0., z**2/u_plus_z2, np.ones_like(z))

    # Compute stresses
    stt = (1.-2.*nu)/3. * 1./(1.+u) * one_minus_z3_div_r2 + \
        z*(2.*nu + (1.-nu)*u/(1.+u) - (1.+nu)*sqrtu_arctan_inv_sqrtu)
    szz = z*z2_div_u_plus_z2
    srr = -( (1.-2.*nu)/3. * 1./(1+u) * one_minus_z3_div_r2 + \
         z*z2_div_u_plus_z2 + \
         z*((1.-nu)*u/(1.+u) + (1.+nu)*sqrtu_arctan_inv_sqrtu - 2.) )
    srz = r*z2_div_u_plus_z2 * sqrtu/np.sqrt(1.+u)

    return -stt, -srr, -szz, -srz