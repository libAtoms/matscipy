# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2020) Johannes Hoermann, University of Freiburg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ======================================================================
"""
Assumes steric particles by enforcing minimum distances on coordinates within
discrete distribtution.

Copyright 2020 IMTEK Simulation
University of Freiburg

Authors:

  Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>
"""
import logging, os, sys
import os.path

import numpy as np

# import scipy.constants as sc
#from scipy import integrate, optimize
import scipy.optimize

logger = logging.getLogger(__name__)

def min_dist(x):
    """Finds minimum distance ||xi-xj|| within coordinate distribution

    Parameters
    ----------
    x: (N,dim) ndarray
      coordinates

    Returns
    -------
    float: minimum distance
    """
    # sum_i sum_{j!=i} max(0,d^2-||xi-xj||^2)^2
    n = x.shape[0]
    for i in np.arange(n):
        for j in np.arange(i):
            dx  = x[i,:] - x[j,:]
            dxsq = np.square(dx)
            dxnormsq = np.sum( dxsq )
            if i == 1 and j == 0:
                mindsq = dxnormsq
            elif dxnormsq < mindsq:
                mindsq = dxnormsq
    mind = np.sqrt(mindsq)
    logger.debug("Minimum distance: {:.4g}.".format(mind))
    return mind

def target_function(x, r=1.0, constraints=None):
    """Target function. Penalizes dense packing for coordinates ||xi-xj||<ri+rj.

    Parameters
    ----------
    x: (N,dim) ndarray
        particle coordinates
    r: float or (N,) ndarray(sample_size), optional (default=1.0)
        steric radii of particles

    Returns
    -------
    float: target function value
    """
    assert x.ndim == 2, "2d array expected for x"
    # sum_i sum_{j!=i} max(0,(r_i+r_j)"^2-||xi-xj||^2)^2
    f = 0
    n = x.shape[0]
    xi  = x

    ri = r
    if not isinstance(r, np.ndarray) or r.shape != (n,):
        ri = ri*np.ones(n)
    assert  ri.shape == (n,)

    zeros = np.zeros(n)
    for i in np.arange(n):
        rj = np.roll(ri,i,axis=0)
        xj = np.roll(xi,i,axis=0)
        d  = ri + rj
        dsq = np.square(d)
        dx = xi - xj
        dxsq = np.square(dx)
        dxnormsq = np.sum( dxsq, axis=1 )
        sqdiff = dsq - dxnormsq
        penalty = np.maximum(zeros,sqdiff)
        penaltysq = np.square(penalty)
        # half for double-counting
        f += 0.5*np.sum(penaltysq)


    if constraints:
        logger.debug(
            "Unconstrained penalty: {:.4g}.".format(f))
        f += constraints(x)

    logger.debug(
            "Total penalty:         {:.4g}.".format(f))
    return f


def box_constraint(x, box=np.array([[0.,0.,0],[1.0,1.0,1.0]]), r=0.):
    """Constraint function confining coordinates within box.

    Parameters
    ----------
    x: (N,dim) ndarray
      coordinates
    box: (2,dim) ndarray, optional (default: [[0.,0.,0.],[1.,1.,1.]])
      box corner coordinates
    r: float or np.(N,) ndarray, optional (default=0)
      steric radii of particles

    Returns
    -------
    float: penalty, positive
    """
    zeros = np.zeros(x.shape)

    # positive if coordinates out of box
    ldist = box[0,:] - r - x
    rdist = x - box[1,:] + r

    lpenalty = np.maximum(zeros,ldist)
    rpenalty = np.maximum(zeros,rdist)

    lpenaltysq = np.square(lpenalty)
    rpenaltysq = np.square(rpenalty)

    g = np.sum(lpenaltysq) + np.sum(rpenaltysq)
    logger.debug("Constraint penalty: {:.4g}.".format(f))
    return g

def make_steric(x, box=Noned, r=None,
    options={'gtol':1e-3,'maxiter':10,'disp':True,'eps':1e-3}):
    """Enforces steric constraints on coordinate distribution within box.

    Parameters
    ----------
    x : (N,dim) ndarray
        particle coordinates
    box: (2,dim) ndarray, optional (default: None)
        box corner coordinates
    r : float or (N,) ndarray, optional (default=None)
        steric radius of particles. Can be specified particle-wise.
    options : dict, optional
        forwarded to scipy BFGS minimzer
        https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html
        (default: {'gtol':1e-3,'maxiter':10,'disp':True,'eps':1e-3} )

    Returns
    -------
    float : (N,dim) ndarray
        modified particle coordinates, meeting steric constraints
    """

    assert isinstance(x, np.ndarry), "x must be np.ndarray"
    assert x.ndim == 2, "x must be 2d array"

    n   = x.shape[0]
    dim = x.shape[1]

    if r is None:
        r = 0.0
        logger.info("No steric radii explicitly specified, using none (zero).")

    r = np.array(r)

    assert r.ndim == 1, "only isotropic steric radii r, no spatial dimensions"
    if r.shape[0] == 1:
        r = r*np.ones(n)
    assert r.shape[0] == n, "either one steric radius for all paricles or one each"

    if box is None:
        box = np.array(x.min(axis=0),x.max(axis=0))
        logger.info(
            """No bounding box explicitly specified, using extreme coordinates
            ({}) of coordinate set as default.""".format(box))

    assert isinstance(box, np.ndarray), "box must be np.ndarray"
    assert x.ndim == 2, "box must be 2d array"
    assert box.shape[0] == 2, "box must have two rows for outer corners"
    assert box.shape[1] == dim, "spatial dimensions of x and box must agree"

    V = np.product(box)[1,:]-box[0,:])
    L = np.power(V, (1./dim))
    logger.info(
        """Normalizing coordinates by reference length
        L = V^(1/dim) = ({:.2g})^(1/{:d}) = {:.2g}.""".format(V,dim,L) )

    # normalizing to unit volume necessary,
    # as target function apparently not dimension-insensitive
    BOX = box / L
    X0  = x / L
    R   = r / L

    logger.info("Normalized bounding box: {}.".format(BOX))

    # flatten coordinates for scipy optimizer
    x0 = X.reshape(np.product(X.shape))

    # define constraint and target wrapper for scipy optimizer
    g = lambda x: box_constraint(x, box=BOX, r=R )
    f = lambda x: target_function(x.reshape((n,dim)),r=R,constraints=g)

    callback_count = 0
    # https://stackoverflow.com/questions/16739065/how-to-display-progress-of-scipy-optimize-function
    def minimizer_callback(xk, res, *_):
        """Callback function that can be used by optimizers of scipy.optimize.
        The third argument "*_" makes sure that it still works when the
        optimizer calls the callback function with more than one argument. Pass
        to optimizer without arguments or parentheses."""
        # s1 = ""
        # xk = np.atleast_1d(xk)

        if callback_count == 0:
            logger.debug(
                "{:>10s} {:>10s} {:>10s}".format(
                    "#callback","objective","min. dist.") )

        X1 = res.x.reshape((n,dim))
        mind = min_dist(X1)    

        logger.debug(
            "{:10d} {:10.5e} {:10.5e}".format(callback_count, res.fun, mind))

        callback_count += 1


    res = scipy.optimize.minimize(f,X0,method='BFGS',
               options={'gtol':1e-3,'maxiter':10,'disp':True,'eps':1e-3})
