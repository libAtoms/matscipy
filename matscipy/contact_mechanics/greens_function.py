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

from math import pi

import numpy as np

###

def gf_displacement_nonperiodic(x, y, a=0.5, b=0.5):
    """
    Real-space representation of Green's function for the normal displacements
    of a non-periodic linear elastic half-space with contact modulus 2 and
    Poisson number 1/2 in response to a uniform pressure applied to a
    rectangular region.
    See: K.L. Johnson, Contact Mechanics, p. 54

    Parameters
    ----------
    x : array_like
        x-coordinates.
    y : array_like
        y-coordinates.
    a, b : float
        Size of rectangle. Rectangle has corners at -a,-b and a,b, i.e.
        rectangle has extend of 2*a x 2*b. Solution converges to
        Boussinesq-Cerrutti form for a->0 and b->0.

    Returns
    -------
    gf : array
        Green's function for displacements at coordinates given by x,y as a
        function of pressure applied to the rectangular region.
    """

    return ( (x+a)*np.log( ( (y+b)+np.sqrt((y+b)*(y+b)+(x+a)*(x+a)) )/
                           ( (y-b)+np.sqrt((y-b)*(y-b)+(x+a)*(x+a)) ) )+
             (y+b)*np.log( ( (x+a)+np.sqrt((y+b)*(y+b)+(x+a)*(x+a)) ) /
                           ( (x-a)+np.sqrt((y+b)*(y+b)+(x-a)*(x-a)) ) )+
             (x-a)*np.log( ( (y-b)+np.sqrt((y-b)*(y-b)+(x-a)*(x-a)) ) /
                           ( (y+b)+np.sqrt((y+b)*(y+b)+(x-a)*(x-a)) ) )+
             (y-b)*np.log( ( (x-a)+np.sqrt((y-b)*(y-b)+(x-a)*(x-a)) ) /
                           ( (x+a)+np.sqrt((y-b)*(y-b)+(x+a)*(x+a)) ) ) )/(2*pi);


def gf_subsurface_stress_nonperiodic(x, y, z, nu=0.5):
    """
    Real-space representation of Green's function for the stress in the bulk of
    a non-periodic linear elastic half-space in response to a concentrated
    normal force. This is the Boussinesq-Cerrutti solution.
    See: K.L. Johnson, Contact Mechanics, p. 51

    Parameters
    ----------
    x : array_like
        x-coordinates.
    y : array_like
        y-coordinates.
    z : array_like
        z_coordinates.
    nu : float
        Poisson number.

    Returns
    -------
    sxx : array
        Green's function xx-component of the stress tensor at coordinates given
        by x,y,z as a function of pressure applied at point 0,0.
    syy : array
        yy-component of the stress tensor.
    szz : array
        zz-component of the stress tensor.
    syz : array
        yz-component of the stress tensor.
    sxz : array
        xz-component of the stress tensor.
    sxy : array
        xy-component of the stress tensor.
    """

    rho = np.sqrt(x**2 + y**2 + z**2)
    r_sq = x**2 + y**2

    sxx = ( (1-2*nu)/r_sq * ((1 - z/rho) * (x**2 - y**2)/r_sq + z*y**2/rho**3) - 
            3*z*x**2/rho**5 )/(2*pi)
    syy = ( (1-2*nu)/r_sq * ((1 - z/rho) * (y**2 - x**2)/r_sq + z*x**2/rho**3) - 
            3*z*y**2/rho**5 )/(2*pi)
    szz = -3*z**3/(2*pi*rho**5)

    sxy = ( (1-2*nu)/r_sq * ((1 - z/rho) * x*y/r_sq - x*y*z/rho**3) -
            3*x*y*z/rho**5 )/(2*pi)
    sxz = -3*x*z**2/(2*pi*rho**5)
    syz = -3*y*z**2/(2*pi*rho**5)

    return sxx, syy, szz, syz, sxz, sxy


def reciprocal_grid(nx, ny=None, gf=gf_displacement_nonperiodic,
                    coordinates=False):
    """
    Return the reciprocal space representation of a Green's function on an FFT
    grid.
    Note: If the Green's function is for the non-periodic (free-bounday)
    problem, then only a section of 1/2 nx by 1/2 ny of the grid can have
    non-zero pressure. The other region is a padding region. See R.W. Hockney,
    Methods in Computational Physics, Vol. 9, pp. 135-211 (1970) for an
    application of this method to the electrostatic problem.

    Parameters
    ----------
    nx : int
        Number of grid points in x-direction.
    ny : int
        Number of grid points in y-direction.
    gf : function
        Function returning the real-space Green's function.
    coordinates : bool
        If True, return grid coordinates.

    Returns
    -------
    G : array
        Reciprocal space representation of the stiffness on an nx,ny grid.
    x : array
        x-coordinates of grid.
    y : array
        y-coordinates of grid.
    """

    if ny is None:
        nx, ny = nx

    x = np.arange(nx)
    x = np.where(x <= nx/2, x, x-nx)
    x.shape = (-1,1)
    y = np.arange(ny)
    y = np.where(y <= ny/2, y, y-ny)
    y.shape = (1,-1)

    G = gf(x, y)
    if isinstance(G, tuple) or isinstance(G, list):
        r = [ np.fft.fft2(np.real(_G)) for _G in G ]
    else:
        r = np.fft.fft2(np.real(G))

    if coordinates:
        return r, x, y
    else:
        return r


def reciprocal_stiffness_periodic(nx, ny=None, phi0=None, size=None):
    """
    Return reciprocal space stiffness coefficients (i.e. inverse of the Green's
    function) for a periodic system with contact modulus 2 and Poisson number
    1/2.

    Parameters
    ----------
    nx : int
        Number of grid points in x-direction.
    ny : int
        Number of grid points in y-direction.
    phi0 : float
        Stiffness at q=0 (gamma point).
    size : tuple
        System size. Assumed to be nx,ny if parameter is omitted.

    Returns
    -------
    phi : array
        Reciprocal space representation of the stiffness on an nx,ny grid.
    """

    if ny is None:
        nx, ny = nx

    qx = np.arange(nx, dtype=np.float64)
    qx = np.where(qx <= nx/2, 2*pi*qx/sx, 2*pi*(nx-qx)/sx)
    qy = np.arange(ny, dtype=np.float64)
    qy = np.where(qy <= ny/2, 2*pi*qy/sy, 2*pi*(ny-qy)/sy)
    phi  = np.sqrt( (qx*qx).reshape(-1, 1) + (qy*qy).reshape(1, -1) )
    if phi0 is None:
        phi[0, 0] = (phi[1, 0].real + phi[0, 1].real)/2
    else:
        phi[0, 0] = phi0

    return phi
