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

from math import isnan, pi, sqrt

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
    sxx = np.where(r_sq > 0.0, sxx, np.zeros_like(sxx))
    syy = ( (1-2*nu)/r_sq * ((1 - z/rho) * (y**2 - x**2)/r_sq + z*x**2/rho**3) - 
            3*z*y**2/rho**5 )/(2*pi)
    syy = np.where(r_sq > 0.0, syy, np.zeros_like(syy))
    szz = -3*z**3/(2*pi*rho**5)

    sxy = ( (1-2*nu)/r_sq * ((1 - z/rho) * x*y/r_sq - x*y*z/rho**3) -
            3*x*y*z/rho**5 )/(2*pi)
    sxy = np.where(r_sq > 0.0, sxy, np.zeros_like(sxy))
    sxz = -3*x*z**2/(2*pi*rho**5)
    syz = -3*y*z**2/(2*pi*rho**5)

    return -sxx, -syy, -szz, -syz, -sxz, -sxy


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


def min_ccg(h_xy, gf, u_xy=None, pentol=1e-6, maxiter=100000,
            logger=None):
    """
    Use a constrained conjugate gradient optimization to find the equilibrium
    configuration deflection of an elastic manifold. The conjugate gradient
    iteration is reset using the steepest descent direction whenever the contact
    area changes.
    Method is described in I.A. Polonsky, L.M. Keer, Wear 231, 206 (1999)

    Parameters
    ----------
    h_xy : array_like
        Height profile of the rigid counterbody.
    gf : array_like
        Green's function (in reciprocal space).
    u_xy : array
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

    if logger is not None:
        logger.pr('maxiter = {0}'.format(maxiter))
        logger.pr('pentol = {0}'.format(pentol))

    if u_xy is None:
        u_xy = np.zeros_like(h_xy)

    u_xy[:, :] = np.where(u_xy > h_xy, h_xy, u_xy)

    # Compute forces
    p_xy = -np.fft.ifft2(np.fft.fft2(u_xy)/gf).real

    # iteration
    delta = 0
    delta_str = 'reset'
    G_old = 1.0
    for it in range(1, maxiter+1):
        # Reset contact area
        c_xy = p_xy > 0.0

        # Compute total contact area (area with repulsive force)
        A = np.sum(c_xy)

        # Compute G = sum(g*g) (over contact area only)
        g_xy = h_xy-u_xy
        G = np.sum(c_xy*g_xy*g_xy)

        # t = (g + delta*(G/G_old)*t) inside contact area and 0 outside
        if delta > 0:
            t_xy = c_xy*(g_xy + delta*(G/G_old)*t_xy)
        else:
            t_xy = c_xy*g_xy

        # Compute elastic displacement that belong to t_xy
        # (Note: r_xy is negative of Polonsky, Kerr's r)
        r_xy = -np.fft.ifft2(gf*np.fft.fft2(t_xy)).real

        # Note: Sign reversed from Polonsky, Keer because this r_xy is negative
        # of theirs.
        tau = 0.0
        if A > 0:
            # tau = -sum(g*t)/sum(r*t) where sum is only over contact region
            x = -np.sum(c_xy*r_xy*t_xy)
            if x > 0.0:
                tau = np.sum(c_xy*g_xy*t_xy)/x
            else:
                G = 0.0

        # Save forces for later
        fold_xy = p_xy.copy()

        p_xy -= tau*c_xy*t_xy

        # Find area with negative forces and negative gap
        # (i.e. penetration of the two surfaces)
        nc_xy = np.logical_and(p_xy <= 0.0, g_xy < 0.0)

        # Set all negative forces to zero
        p_xy *= p_xy > 0.0

        if np.sum(nc_xy) > 0:
            # nc_xy contains area that just jumped into contact. Update their
            # forces.
            p_xy -= tau*nc_xy*g_xy
        
            delta = 0
            delta_str = 'sd'
        else:
            delta = 1
            delta_str = 'cg'

        # Compute new displacements from updated forces
        u_xy = -np.fft.ifft2(gf*np.fft.fft2(p_xy)).real
       
        # Store G for next step
        G_old = G
        
        # Compute root-mean square penetration, max penetration and max force
        # difference between the steps
        if A > 0:
            rms_pen = sqrt(G/A)
        else:
            rms_pen = sqrt(G)
        max_pen = max(0.0, np.max(c_xy*(u_xy-h_xy)))

        # Elastic energy would be
        # e_el = -0.5*np.sum(p_xy*u_xy)

        if rms_pen < pentol and max_pen < pentol:
            if logger is not None:
                logger.st(['status', 'it', 'A', 'tau', 'rms_pen', 'max_pen'],
                          ['CONVERGED', it, A, tau, rms_pen, max_pen],
                          force_print=True)
            return u_xy, p_xy

        if logger is not None:
            logger.st(['status', 'it', 'A', 'tau', 'rms_pen', 'max_pen'],
                      [delta_str, it, A, tau, rms_pen, max_pen])

        if isnan(G) or isnan(rms_pen):
            raise RuntimeError('nan encountered.')

    raise RuntimeError('Maximum number of iterations ({0}) exceeded.' \
                           .format(maxiter))

