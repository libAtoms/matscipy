#
# Copyright 2014-2015, 2021 Lars Pastewka (U. Freiburg)
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

"""
Contact mechanics module.

This module contains functions creating real and reciprocal space Green's
function. Continuous real space Green's function can be converted to their
discrete reciprocal space representation by `real_to_reciprocal_space`.
Typically, all *nonperiodic* kernel return real space Green's functions,
all *periodic* kernel reciprocal space ones.
"""

from math import isnan, pi, sqrt

import numpy as np

###

def square_pressure__nonperiodic(x, y, a=0.5, b=0.5):
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


def point_traction__nonperiodic(quantities, x, y, z, G=1.0, poisson=0.5):
    """
    Real-space representation of Green's function for the displacement and
    stress in the bulk of a non-periodic linear elastic half-space in response
    to a concentrated surface force. This is the Boussinesq-Cerrutti solution.
    See: K.L. Johnson, Contact Mechanics, p. 51 and p. 69
    Sign convention is as in Johnson!

    Parameters
    ----------
    quantities : str
        Each character in this string defines a return quantity. They are
        returned in a tuple of the same order. Possible quantities are
            'x' : Displacement for a concentrated surface traction in
                  x-direction.
            'X' : Stress for a concentrated surface traction in x-direction.
            'Y' : Stress for a concentrated surface traction in y-direction.
            'Z' : Stress for a concentrated surface pressure
                  (i.e. "traction" in z-direction).
    x : array_like
        x-coordinates.
    y : array_like
        y-coordinates.
    z : array_like
        z-coordinates. Into the solid is positive.
    G : float
        Shear modulus.
    poisson : float
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

    r_sq = x**2 + y**2
    r_sq = np.array(r_sq, dtype=float)
    rho = np.sqrt(r_sq + z**2)

    r_sq[r_sq <= 0.0] = 1e-9
    rho[rho <= 0.0] = 1e-9

    retvals = []
    for q in quantities:
        if q == 'x':
            ux = (1/rho+x**2/rho**3+(1-2*poisson)*(1/(rho+z)-x**2/(rho*(rho+z)**2)))/(4*pi*G)
            uy = (x*y/rho**3-(1-2*poisson)*x*y/(rho*(rho+z)**2))/(4*pi*G)
            uz = (x*z/rho**3+(1-2*poisson)*x/(rho*(rho+z)))/(4*pi*G)
            retvals += [np.array([ux, uy, uz])]
        elif q == 'y':
            raise NotImplementedError()
        elif q == 'z':
            ux = (x*z/rho**3-(1-2*poisson)*x/(rho*(rho+z)))/(4*pi*G)
            uy = (y*z/rho**3-(1-2*poisson)*y/(rho*(rho+z)))/(4*pi*G)
            uz = (z**2/rho**3+2*(1-poisson)/rho)/(4*pi*G)
            retvals += [np.array([ux, uy, uz])]
        elif q == 'X':
            sxx = ( -3*x**3/rho**5 + (1-2*poisson)*(x/rho**3 - 3*x/(rho*(rho+z)**2) + x**3/(rho**3*(rho+z)**2) + 2*x**3/(rho**2*(rho+z)**3)) )/(2*pi)
            syy = ( -3*x*y**2/rho**5 + (1-2*poisson)*(x/rho**3 - x/(rho*(rho+z)**2) + x*y**2/(rho**3*(rho+z)**2) + 2*x*y**2/(rho**2*(rho+z)**3)) )/(2*pi)
            szz = ( -3*x*z**2/rho**5 )/(2*pi)
            sxy = ( -3*x**2*y/rho**5 + (1-2*poisson)*(-y/(rho*(rho+z)**2) + x**2*y/(rho**3*(rho+z)**2) + 2*x**2*y/(rho**2*(rho+z)**3)) )/(2*pi)
            sxz = ( -3*x**2*z/rho**5 )/(2*pi)
            syz = ( -3*x*y*z/rho**5 )/(2*pi)
            retvals += [np.array([sxx, syy, szz, syz, sxz, sxy])]
        elif q == 'Y':
            raise NotImplementedError()
        elif q == 'Z':
            sxx = ( (1-2*poisson)/r_sq * ((1 - z/rho) * (x**2 - y**2)/r_sq + z*y**2/rho**3) - 3*z*x**2/rho**5 )/(2*pi)
            syy = ( (1-2*poisson)/r_sq * ((1 - z/rho) * (y**2 - x**2)/r_sq + z*x**2/rho**3) - 3*z*y**2/rho**5 )/(2*pi)
            szz = -3*z**3/(2*pi*rho**5)
            # Note: Johnson is lacking a factor of 2 here!!!
            sxy = ( (1-2*poisson)/r_sq * (2*(1 - z/rho) * x*y/r_sq - x*y*z/rho**3) - 3*x*y*z/rho**5 )/(2*pi)
            sxz = -3*x*z**2/(2*pi*rho**5)
            syz = -3*y*z**2/(2*pi*rho**5)
            retvals += [np.array([sxx, syy, szz, syz, sxz, sxy])]
        else:
            raise ValueError("Unknown quantity '{0}' requested.".format(q))

    if len(quantities) == 1:
        return retvals[0]
    else:
        return retvals


def real_to_reciprocal_space(nx, ny=None, gf=square_pressure__nonperiodic,
                             coordinates=False):
    """
    Return the reciprocal space representation of a real-space Green's function
    on an FFT grid.
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
    x = np.where(x <= nx//2, x, x-nx)
    x.shape = (-1,1)
    y = np.arange(ny)
    y = np.where(y <= ny//2, y, y-ny)
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


def point_displacement__periodic(nx, ny=None, phi0=None, size=None):
    """
    Return reciprocal space stiffness coefficients (i.e. inverse of the Green's
    function) for a periodic system with contact modulus 2 and Poisson number
    1/2. This gives force as a function of displacement.

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
    qx = np.where(qx <= nx//2, 2*pi*qx/sx, 2*pi*(nx-qx)/sx)
    qy = np.arange(ny, dtype=np.float64)
    qy = np.where(qy <= ny//2, 2*pi*qy/sy, 2*pi*(ny-qy)/sy)
    phi  = np.sqrt( (qx*qx).reshape(-1, 1) + (qy*qy).reshape(1, -1) )
    if phi0 is None:
        phi[0, 0] = (phi[1, 0].real + phi[0, 1].real)/2
    else:
        phi[0, 0] = phi0

    return phi


def min_ccg(h_r, gf_q, u_r=None, pentol=1e-6, maxiter=100000,
            logger=None):
    """
    Use a constrained conjugate gradient optimization to find the equilibrium
    configuration deflection of an elastic manifold. The conjugate gradient
    iteration is reset using the steepest descent direction whenever the contact
    area changes.
    Method is described in I.A. Polonsky, L.M. Keer, Wear 231, 206 (1999)

    Parameters
    ----------
    h_r : array_like
        Height profile of the rigid counterbody.
    gf_q : array_like
        Green's function (in reciprocal space).
    u_r : array
        Array used for initial displacements. A new array is created if omitted.
    pentol : float
        Maximum penetration of contacting regions required for convergence.
    maxiter : float
        Maximum number of iterations.

    Returns
    -------
    u : array
        2d-array of displacements.
    p : array
        2d-array of pressure.
    """

    # Note: Suffix _r deontes real-space _q reciprocal space 2d-arrays

    if logger is not None:
        logger.pr('maxiter = {0}'.format(maxiter))
        logger.pr('pentol = {0}'.format(pentol))

    if u_r is None:
        u_r = np.zeros_like(h_r)

    u_r[:, :] = np.where(u_r > h_r, h_r, u_r)

    # Compute forces
    p_r = -np.fft.ifft2(np.fft.fft2(u_r)/gf_q).real

    # iteration
    delta = 0
    delta_str = 'reset'
    G_old = 1.0
    for it in range(1, maxiter+1):
        # Reset contact area
        c_r = p_r > 0.0

        # Compute total contact area (area with repulsive force)
        A = np.sum(c_r)

        # Compute G = sum(g*g) (over contact area only)
        g_r = h_r-u_r
        G = np.sum(c_r*g_r*g_r)

        # t = (g + delta*(G/G_old)*t) inside contact area and 0 outside
        if delta > 0:
            t_r = c_r*(g_r + delta*(G/G_old)*t_r)
        else:
            t_r = c_r*g_r

        # Compute elastic displacement that belong to t_r
        # (Note: r_r is negative of Polonsky, Kerr's r)
        r_r = -np.fft.ifft2(gf_q*np.fft.fft2(t_r)).real

        # Note: Sign reversed from Polonsky, Keer because this r_r is negative
        # of theirs.
        tau = 0.0
        if A > 0:
            # tau = -sum(g*t)/sum(r*t) where sum is only over contact region
            x = -np.sum(c_r*r_r*t_r)
            if x > 0.0:
                tau = np.sum(c_r*g_r*t_r)/x
            else:
                G = 0.0

        # Save forces for later
        fold_r = p_r.copy()

        p_r -= tau*c_r*t_r

        # Find area with negative forces and negative gap
        # (i.e. penetration of the two surfaces)
        nc_r = np.logical_and(p_r <= 0.0, g_r < 0.0)

        # Set all negative forces to zero
        p_r *= p_r > 0.0

        if np.sum(nc_r) > 0:
            # nc_r contains area that just jumped into contact. Update their
            # forces.
            p_r -= tau*nc_r*g_r
        
            delta = 0
            delta_str = 'sd'
        else:
            delta = 1
            delta_str = 'cg'

        # Compute new displacements from updated forces
        u_r = -np.fft.ifft2(gf_q*np.fft.fft2(p_r)).real
       
        # Store G for next step
        G_old = G
        
        # Compute root-mean square penetration, max penetration and max force
        # difference between the steps
        if A > 0:
            rms_pen = sqrt(G/A)
        else:
            rms_pen = sqrt(G)
        max_pen = max(0.0, np.max(c_r*(u_r-h_r)))

        # Elastic energy would be
        # e_el = -0.5*np.sum(p_r*u_r)

        if rms_pen < pentol and max_pen < pentol:
            if logger is not None:
                logger.st(['status', 'it', 'A', 'tau', 'rms_pen', 'max_pen'],
                          ['CONVERGED', it, A, tau, rms_pen, max_pen],
                          force_print=True)
            return u_r, p_r

        if logger is not None:
            logger.st(['status', 'it', 'A', 'tau', 'rms_pen', 'max_pen'],
                      [delta_str, it, A, tau, rms_pen, max_pen])

        if isnan(G) or isnan(rms_pen):
            raise RuntimeError('nan encountered.')

    raise RuntimeError('Maximum number of iterations ({0}) exceeded.' \
                           .format(maxiter))

