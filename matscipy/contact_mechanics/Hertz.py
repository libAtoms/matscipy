#
# Copyright 2014-2015, 2021 Lars Pastewka (U. Freiburg)
#           2015 Till Junge (EPFL)
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

    Returns
    -------
    a : float
        Contact radius.
    p0 : float
        Maximum pressure inside the contacting area (right under the apex).
    """

    a = R*(3./4*( N/(Es*R**2) ))**(1./3)
    p0 = 3*N/(2*math.pi*a*a)
    
    return a, p0


def surface_stress(r, poisson=0.5):
    """
    Given distance from the center of the sphere, contact radius and Poisson
    number contact, compute the stress at the surface.

    Parameters
    ----------
    r : array_like
        Array of distance (from the center of the sphere in units of contact
        radius a).
    poisson : float
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
        sr[mask0] = -(1.+2*poisson)/2.*np.ones_like(r_0)
        stheta[mask0] = -(1.+2*poisson)/2.*np.ones_like(r_0)

    # Solution inside the contact radius
    if maski.sum() > 0:
        r_a_sq = r_i**2
        pz[maski] = np.sqrt(1-r_a_sq)
        sr[maski] = (1.-2.*poisson)/(3.*r_a_sq)*(1.-(1.-r_a_sq)**(3./2))- \
            np.sqrt(1.-r_a_sq)
        stheta[maski] = -(1.-2.*poisson)/(3.*r_a_sq)*(1.-(1.-r_a_sq)**(3./2))- \
            2*poisson*np.sqrt(1.-r_a_sq)
    
    # Solution outside of the contact radius
    if masko.sum() > 0:
        r_a_sq = r_o**2
        po = (1.-2.*poisson)/(3.*r_a_sq)
        sr[masko] = po
        stheta[masko] = -po
    
    return pz, sr, stheta


def centerline_stress(z, poisson=0.5):
    """
    Given distance from the center of the sphere, contact radius and Poisson
    number contact, compute the stress at the surface.

    Parameters
    ----------
    z : array_like
        Array of depths (from the center of the sphere in units of contact
        radius a).
    poisson : float
        Poisson number.

    Returns
    -------
    srr : array
        Radial stress (in units of maximum pressure p0).
    szz : array
        Contact pressure (in units of maximum pressure p0).
    """

    srr = -(1.+poisson)*(1.-z*np.arctan(1./z)) + 1./(2.*(1.+z**2))
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


def stress(r, z, poisson=0.5):
    """
    Return components of the stress tensor in the interior of the Hertz solid.
    This is the solution given by: M.T. Huber, Ann. Phys. 319, 153 (1904)

    Note that the stress tensor at any point in the solid has the form below in
    a cylindrical coordinate system centered at the tip apex. Some off-diagonal
    components are zero by rotational symmetry. stt is the circumferential
    component, srr the radial component and szz the normal component of the
    stress tensor.

            / stt  0   0  \ 
        s = |  0  srr srz |
            \  0  srz szz /

    Parameters
    ----------
    r : array_like
        Radial position (in units of the contact radius a).
    z : array_like
        Depth (in units of the contact radius a).
    poisson: float
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

    # Note: The equation for u is
    #     r**2/(1+u) + z**2/u = 1
    # With r' = r/sqrt(1+u) and z' = z/sqrt(u) this expression becomes
    #     r'**2 + z'**2 = 1, hence z' = sqrt(1-r'**2)
    # Variables r and z below are substituted by r->r' and z->z'.
    r = r/np.sqrt(1+u)
    z = np.sqrt(1-r**2) # equiv. to z=u/sqrt(u), but defined for z=0

    # Precompute arctan
    sqrtu_arctan_inv_sqrtu = sqrtu*np.arctan(1./sqrtu)

    # The next two expressions give numerical problems at the tip center and the
    # contact edge, regularize with the asymptotic value.

    # Note: (1-z**3)/r**2->3/2 for r->0
    one_minus_z3_div_r2 = 3./2.*np.ones_like(r)
    mask = r > 0
    one_minus_z3_div_r2[mask] = (1.-z[mask]**3)/r[mask]**2

    # Note: z**2/(u+z**2)->1 for u+z**2->0
    u_plus_z2 = u+z**2
    z2_div_u_plus_z2 = np.where(u_plus_z2 > 0., z**2/u_plus_z2, np.ones_like(z))

    # Compute stresses
    # Note: the factor 1/(1+u) stems from the substitution r->r' and z->z' above
    stt = (1.-2.*poisson)/3. * 1./(1.+u) * one_minus_z3_div_r2 + \
        z*(2.*poisson+ (1.-poisson)*u/(1.+u) - (1.+poisson)*sqrtu_arctan_inv_sqrtu)
    szz = z*z2_div_u_plus_z2
    srr = -( (1.-2.*poisson)/3. * 1./(1+u) * one_minus_z3_div_r2 + \
         z*z2_div_u_plus_z2 + \
         z*((1.-poisson)*u/(1.+u) + (1.+poisson)*sqrtu_arctan_inv_sqrtu - 2.) )
    srz = r*z2_div_u_plus_z2 * sqrtu/np.sqrt(1.+u)

    return -stt, -srr, -szz, -srz


def stress_Cartesian(x, y, z, poisson=0.5):
    """
    Return components of the stress tensor in the interior of solid due to
    normal Hertz loading.
    This is the solution given by:
    G.M. Hamilton, Proc. Instn. Mech. Engrs. 197C, 53-59 (1983)

    Parameters
    ----------
    x, y : array_like
        In-plane positions (in units of the contact radius a).
    z : array_like
        Depth (in units of the contact radius a).
    poisson : float
        Poisson number.

    Returns
    -------
    sxx, syy, szz, syz, sxz, sxy : array
        Individual components of the Cartesian stress tensor.
    """

    def stress_offcenter(x, y, z, r_sq, poisson=0.5):
        A = r_sq + z**2 - 1
        S = np.sqrt(A**2 + 4*z**2)

        M = np.sqrt((S+A)/2)
        N = np.sqrt((S-A)/2)
        phi = np.arctan2(1, M)

        G = M**2 - N**2 + z*M - N
        H = 2*M*N + M + z*N

        sxx = (1+poisson)*z*phi+1/r_sq*((y**2-x**2)/r_sq*((1-poisson)*N*z**2-(1-2*poisson)/3*(N*S+2*A*N+1)-poisson*M*z)-N*(x**2+2*poisson*y**2)-M*x**2*z/S)
        syy = (1+poisson)*z*phi+1/r_sq*((x**2-y**2)/r_sq*((1-poisson)*N*z**2-(1-2*poisson)/3*(N*S+2*A*N+1)-poisson*M*z)-N*(y**2+2*poisson*x**2)-M*y**2*z/S)
        szz = -N+z*M/S
        sxy = x*y*(1-2*poisson)/r_sq**2*(-N*r_sq+2/3*N*(S+2*A)-z*(z*N+M)+2/3)+x*y*z/r_sq**2*(-M*r_sq/S-z*N+M)
        syz = -z*(y*N/S-y*z*H/(G**2+H**2))
        sxz = -z*(x*N/S-x*z*H/(G**2+H**2))

        return sxx, syy, szz, syz, sxz, sxy

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    sxx = np.zeros_like(x)
    syy = np.zeros_like(x)
    szz = np.zeros_like(x)
    syz = np.zeros_like(x)
    sxz = np.zeros_like(x)
    sxy = np.zeros_like(x)

    r_sq = x**2 + y**2

    mask = r_sq > 0
    xx, yy, zz, yz, xz, xy = stress_offcenter(x[mask], y[mask], z[mask],
                                              r_sq[mask], poisson=poisson)
    sxx[mask] = xx
    syy[mask] = yy
    szz[mask] = zz
    syz[mask] = yz
    sxz[mask] = xz
    sxy[mask] = xy

    mask = np.logical_not(mask)
    if mask.sum() > 0:
        z = z[mask]
        sxx[mask] = ((1+poisson)*(z*np.arctan2(1, z)-1)+1/(2*(1+z**2)))
        syy[mask] = sxx[mask]
        szz[mask] = -1/(1+z**2)

    return sxx, syy, szz, syz, sxz, sxy


def stress_for_tangential_loading(x, y, z, poisson=0.5):
    """
    Return components of the stress tensor in the interior of solid due to
    tangential (Hertz) loading.
    This is the solution given by:
    G.M. Hamilton, Proc. Instn. Mech. Engrs. 197C, 53-59 (1983)

    Parameters
    ----------
    x, y : array_like
        In-plane positions (in units of the contact radius a).
    z : array_like
        Depth (in units of the contact radius a).
    poisson : float
        Poisson number.

    Returns
    -------
    sxx, syy, szz, syz, sxz, sxy : array
        Individual components of the Cartesian stress tensor.
    """

    def stress_offcenter(x, y, z, r_sq, poisson=0.5):
        A = r_sq + z**2 - 1
        S = np.sqrt(A**2 + 4*z**2)

        M = np.sqrt((S+A)/2)
        N = np.sqrt((S-A)/2)
        phi = np.arctan2(1, M)

        G = M**2 - N**2 + z*M - N
        H = 2*M*N + M + z*N

        sxx = -x*(poisson/4+1)*phi+x*M/r_sq**2*((3/2-2*x**2/r_sq)*(S*poisson-2*A*poisson+z**2)+x**2*z**2/S+7*poisson*r_sq/4-2*poisson*x**2+r_sq)+x*z*N/r_sq**2*((3/2-2*x**2/r_sq)*(-S/6*(1-2*poisson)-A/3*(1-2*poisson)-1/2*(z**2+3))+x**2/S-poisson*r_sq/4-7*r_sq/4)+4*x*z/(3*r_sq**2)*(3/2-2*x**2/r_sq)*(1-2*poisson)
        syy = -3*poisson*x*phi/4+x*M/r_sq**2*((1/2-2*y**2/r_sq)*(poisson*(S-2*A+r_sq)+z**2)+y**2*z**2/S+3/4*poisson*r_sq)+z*x*N/r_sq**2*((1/2-2*y**2/r_sq)*(-S/6*(1-2*poisson)-A/3*(1-2*poisson)-z**2/2-3/2)+y**2/S-3/4*poisson*r_sq-r_sq/4)+4/3*z*x/r_sq**2*(1/2-2*y**2/r_sq)*(1-2*poisson)
        szz = z*x*N/(2*r_sq)*(1-(r_sq+z**2+1)/S)
        sxy = y/2*(poisson/2-1)*phi+y*M/r_sq**2*(x**2*z**2/S+poisson*((S-2*A)*(1/2-2*x**2/r_sq)-2*x**2+r_sq/4)+r_sq/2+z**2*(1/2-2*x**2/r_sq))+y*z*N/r_sq**2*((1/2-2*x**2/r_sq)*((2*poisson-1)*(S/6+A/3)-z**2/2-3/2-r_sq/2)+r_sq*poisson/4+x**2/S-y**2/2-3*x**2/2)+4*y*z/(3*r_sq**2)*(1/2-2*x**2/r_sq)*(1-2*poisson)
        syz = x*y*z/(2*r_sq**2)*(M*(1/2+1/S*(z**2/2-3/2-r_sq/2))+z*N/2*(-3+1/S*(5+z**2+r_sq)))
        sxz = 3*z*phi/2+z*M/r_sq*(1+x**2/r_sq-x**2/S)+N/r_sq*(-3/4*(S+2*A)+z**2-3/4-1/4*r_sq+z**2/2*(1/2-2*x**2/r_sq))

        return sxx, syy, szz, syz, sxz, sxy

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    sxx = np.zeros_like(x)
    syy = np.zeros_like(x)
    szz = np.zeros_like(x)
    syz = np.zeros_like(x)
    sxz = np.zeros_like(x)
    sxy = np.zeros_like(x)

    r_sq = x**2 + y**2

    mask = r_sq > 0
    xx, yy, zz, yz, xz, xy = stress_offcenter(x[mask], y[mask], z[mask],
                                              r_sq[mask], poisson=poisson)
    sxx[mask] = xx
    syy[mask] = yy
    szz[mask] = zz
    syz[mask] = yz
    sxz[mask] = xz
    sxy[mask] = xy

    mask = np.logical_not(mask)
    if mask.sum() > 0:
        z = z[mask]
        sxz[mask] += -1+3/2*z*np.arctan2(1, z)-z**2/(2*(1+z**2))

    return sxx, syy, szz, syz, sxz, sxy

