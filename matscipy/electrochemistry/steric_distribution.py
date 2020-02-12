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

Examples:

    Performance of target functions:

    >>> from matscipy.electrochemistry.steric_distribution import scipy_distance_based_target_function
    >>> from matscipy.electrochemistry.steric_distribution import numpy_only_target_function
    >>> from matscipy.electrochemistry.steric_distribution import brute_force_target_function
    >>> import itertools
    >>> import pandas as pd
    >>> import timeit
    >>>
    >>> funcs = [
    >>>         brute_force_target_function,
    >>>         numpy_only_target_function,
    >>>         scipy_distance_based_target_function ]
    >>> func_names = ['brute','numpy','scipy']
    >>>
    >>> stats = []
    >>> K = np.exp(np.log(10)*np.arange(-3,3))
    >>> for k in K:
    >>>     lambdas = [ (lambda x0=x0,k=k,f=f: f(x0*k)) for f in funcs ]
    >>>     vals    = [ f() for f in lambdas ]
    >>>     times   = [ timeit.timeit(f,number=1) for f in lambdas ]
    >>>     diffs = pdist(np.atleast_2d(vals).T,metric='euclidean')
    >>>     stats.append((k,*vals,*diffs,*times))
    >>>
    >>> func_name_tuples = list(itertools.combinations(func_names,2))
    >>> diff_names = [ 'd_{:s}_{:s}'.format(f1,f2) for (f1,f2) in func_name_tuples ]
    >>> perf_names = [ 't_{:s}'.format(f) for f in func_names ]
    >>> fields =  ['k',*func_names,*diff_names,*perf_names]
    >>> dtypes = [ (field, '>f4') for field in fields ]
    >>> labeled_stats = np.array(stats,dtype=dtypes)
    >>> stats_df = pd.DataFrame(labeled_stats)
    >>> print(stats_df.to_string(float_format='%8.6g'))
             k       brute       numpy       scipy  d_brute_numpy  d_brute_scipy  d_numpy_scipy  t_brute  t_numpy   t_scipy
    0    0.001  3.1984e+07  3.1984e+07  3.1984e+07    5.58794e-08    6.70552e-08    1.11759e-08 0.212432 0.168858 0.0734278
    1     0.01 3.19829e+07 3.19829e+07 3.19829e+07    9.31323e-08    7.82311e-08    1.49012e-08 0.212263  0.16846 0.0791856
    2      0.1 3.18763e+07 3.18763e+07 3.18763e+07    7.45058e-09    1.86265e-08    1.11759e-08 0.201706 0.164867 0.0711544
    3        1 2.27418e+07 2.27418e+07 2.27418e+07    3.72529e-08    4.84288e-08    1.11759e-08  0.20762 0.166005 0.0724238
    4       10      199751      199751      199751    1.16415e-10    2.91038e-11    8.73115e-11 0.202635 0.161932 0.0772684
    5      100     252.548     252.548     252.548    3.28555e-11              0    3.28555e-11 0.202512 0.161217 0.0726705

"""
import logging, os, sys
import os.path
import time

import numpy as np

import scipy.optimize
import scipy.spatial.distance

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
    # TODO: vectorize loop
    t0 = time.perf_counter()
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
    t1 = time.perf_counter()-t0
    logger.debug("Found minimum distance {:10.5e} within {:10.5e} s.".format(
        mind,t1))
    return mind

def brute_force_closest_pair(x):
    """Finds coordinate pair with minimum distance squared ||xi-xj||^2

    Parameters
    ----------
    x: (N,dim) ndarray
      coordinates

    Returns
    -------
    float, (ndarray, ndarray): minimum distance squared and coodinates pair
    """
    t0 = time.perf_counter()

    n = x.shape[0]
    imin = 0
    jmin = 1
    if n < 2:
        return (None,None), float('inf')

    dx = x[0,:] - x[1,:]
    dxsq   = np.square(dx)
    mindsq = np.sum( dxsq )

    for i in np.arange(n):
        for j in np.arange(i+1,n):
            dx = x[i,:] - x[j,:]
            dxsq = np.square(dx)
            dxnormsq = np.sum( dxsq )
            if dxnormsq < mindsq:
                imin = i
                jmin = j
                mindsq = dxnormsq

    t1 = time.perf_counter()-t0
    logger.debug("""Found minimum distance squared {:10.5e} for pair
        ({:d},{:d}) with coodinates {} and {} within {:10.5e} s.""".format(
        mindsq,imin,jmin,x[imin,:],x[jmin,:],t1))
    return mindsq, (x[imin,:], x[jmin,:])

def recursive_closest_pair(x,y):
    """Finds coordinate pair with minimum distance squared ||xi-xj||^2
    with one point from x and the other point from y

    Parameters
    ----------
    x: (N,dim) ndarray
        coordinates

    Returns
    -------
    float, (ndarray, ndarray): minimum distance squared and coodinate pair
    """
    t0 = time.perf_counter()

    n = x.shape[0]
    if n < 4:
        return brute_force_closest_pair(x)

    mid = n // 2
    xl = x[:mid]
    xr = x[mid:]

    xdivider = x[mid,0]
    yl = []
    yr = []

    m = y.shape[0]
    for j in np.arange(m):
        if y[j,0] <= xdivider:
            yl.append(y[j,:])
        else:
            yr.append(y[j,:])

    yl = np.array(yl)
    yr = np.array(yr)
    mindsql,(pil,pjl) = recursive_closest_pair(xl,yl)
    mindsqr,(pir,pjr) = recursive_closest_pair(xr,yr)

    mindsq,(pim,pjm) = (mindsql,(pil,pjl)) if mindsql < mindsqr else (mindsqr,(pir,pjr))

    # TODO: this latter part only valid for 2d problems,
    # see https://sites.cs.ucsb.edu/~suri/cs235/ClosestPair.pdf
    # some 3d implementation at
    # https://github.com/eyny/closest-pair-3d/blob/master/src/ballmanager.cpp
    close_y = np.array(
        [y[j,:] for j in np.arange(m) if (np.square(y[j,0]-xdivider) < mindsq)])

    close_n = close_y.shape[0]
    if close_n > 1:
        for i in np.arange(close_n-1):
            for j in np.arange(i+1,min(i+8,close_n)):
                dx = close_y[i,:] - close_y[j,:]
                dxsq = np.square(dx)
                dxnormsq = np.sum( dxsq )
                if dxnormsq < mindsq:
                    pim = close_y[i,:]
                    pjm = close_y[j,:]
                    mindsq = dxnormsq

    return mindsq, (pim, pjm)

def planar_closest_pair(x):
    """Finds coordinate pair with minimum distance ||xi-xj||

    ATTENTION: this implementation tackles the planar problem!

    Parameters
    ----------
    x: (N,dim) ndarray
      coordinates

    Returns
    -------
    float, (ndarray, ndarray): minimum distance squared and coodinates pair
    """
    assert isinstance(x, np.ndarray), "np.ndarray expected for x"
    assert x.ndim == 2, "x is expected to be 2d array"

    t0 = time.perf_counter()

    I = np.argsort(x[:,0])
    J = np.argsort(x[:,-1])
    X = x[I,:]
    Y = x[J,:]
    mindsq, (pim, pjm) = recursive_closest_pair(X,Y)

    # mind = np.sqrt(mindsq)
    t1 = time.perf_counter()-t0
    logger.debug("""Found minimum distance squared {:10.5e} for pair with
        coodinates {} and {} within {:10.5e} s.""".format(mindsq,pim,pjm,t1))
    return mindsq, (pim, pjm)

def scipy_distance_based_closest_pair(x):
    """Finds coordinate pair with minimum distance ||xi-xj||

    Parameters
    ----------
    x: (N,dim) ndarray
      coordinates

    Returns
    -------
    float, (ndarray, ndarray): minimum distance squared and coodinates pair

    Examples
    --------
    Handling condensed distance matrix indices:

        >>> c = np.array([1, 2, 3, 4, 5, 6])
        >>> print(c)
        [1 2 3 4 5 6]

        >>> d = scipy.spatial.distance.squareform(c)
        >>> print(d)
        [[0 1 2 3]
         [1 0 4 5]
         [2 4 0 6]
         [3 5 6 0]]

        >>> I = np.tril_indices(d.shape[0], -1)
        >>> print(I)
        (array([1, 2, 2, 3, 3, 3]), array([0, 0, 1, 0, 1, 2]))
        
        >>> print(d[I])
        [1 2 4 3 5 6]
    """
    t0 = time.perf_counter()

    n = x.shape[0]

    dxnormsq  = scipy.spatial.distance.pdist(x, metric='sqeuclidean')

    ij = np.argmin(dxnormsq)


    t1 = time.perf_counter()-t0
    logger.debug("""Found minimum distance squared {:10.5e} for pair
        ({:d},{:d}) with coodinates {} and {} within {:10.5e} s.""".format(
        mindsq,imin,jmin,x[imin,:],x[jmin,:],t1))
    return mindsq, (x[imin,:], x[jmin,:])

def brute_force_target_function(x, r=1.0, constraints=None):
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
    for i in np.arange(1,n):
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
            "Unconstrained penalty: {:10.5e}.".format(f))
        f += constraints(x)

    logger.debug(
            "Total penalty:         {:10.5e}.".format(f))
    return f

def scipy_distance_based_target_function(x, r=1.0, constraints=None):
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
    n = x.shape[0]

    if not isinstance(r, np.ndarray) or r.shape != (n,):
        r = r*np.ones(n)
    assert  r.shape == (n,)


    # r(Nx1) kron ones(1xN) = Ri(NxN)
    Ri = np.kron(r, np.ones((n,1)))
    Rj = Ri.T
    Dij = Ri + Rj
    dij = scipy.spatial.distance.squareform(Dij,force='tovector',checks=False)

    zeros = np.zeros(dij.shape)

    dsq = np.square(dij)

    dxnormsq  = scipy.spatial.distance.pdist(x, metric='sqeuclidean')
    # computes the squared Euclidean distance ||u-v||_2^2 between vectors
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    sqdiff    = dsq - dxnormsq

    penalty   = np.maximum(zeros,sqdiff)
    penaltysq = np.square(penalty)
    # no double-counting here
    f = np.sum(penaltysq)

    if constraints:
        logger.debug(
            "Unconstrained penalty: {:10.5e}.".format(f))
        f += constraints(x)

    logger.debug(
            "Total penalty:         {:10.5e}.".format(f))
    return f

# https://www.researchgate.net/publication/266617010_NumPy_SciPy_Recipes_for_Data_Science_Squared_Euclidean_Distance_Matrices
def numpy_only_target_function(x, r=1.0, constraints=None):
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
    n = x.shape[0]

    if not isinstance(r, np.ndarray) or r.shape != (n,):
        r = r*np.ones(n)
    assert  r.shape == (n,)

    zeros = np.zeros((n,n))

    # r(Nx1) kron  ones(1xN) = Ri(NxN)
    Ri = np.kron(r, np.ones((n,1)))
    Rj = Ri.T
    Dij = Ri + Rj
    dsq = np.square(Dij)
    np.fill_diagonal(dsq,0.)

    G = np.dot(x,x.T)
    H = np.tile(np.diag(G), (n,1))
    dxnormsq = H + H.T - 2*G
    sqdiff    = dsq - dxnormsq

    penalty   = np.maximum(zeros,sqdiff)
    penaltysq = np.square(penalty)
    # half for double-counting
    f = 0.5*np.sum(penaltysq)

    if constraints:
        logger.debug(
            "Unconstrained penalty: {:10.5e}.".format(f))
        f += constraints(x)

    logger.debug(
            "Total penalty:         {:10.5e}.".format(f))
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
    r = np.atleast_2d(r).T

    # positive if coordinates out of box
    ldist = box[0,:] - (x+r)
    rdist = (x+r) - box[1,:]

    lpenalty = np.maximum(zeros,ldist)
    rpenalty = np.maximum(zeros,rdist)

    lpenaltysq = np.square(lpenalty)
    rpenaltysq = np.square(rpenalty)

    g = np.sum(lpenaltysq) + np.sum(rpenaltysq)
    logger.debug("Constraint penalty: {:.4g}.".format(g))
    return g

def make_steric(x, box=None, r=None,
    options={'gtol':1e-3,'maxiter':10,'disp':True,'eps':0.1},
    target_function=scipy_distance_based_target_function):
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
        (default: {'gtol':1e-3,'maxiter':10,'disp':True,'eps':1e-3})
    target_function: func, optional
        one of the target functions within this submodule, or function
        of same signature (default: scipy_distance_based_target_function)

    Returns
    -------
    float : (N,dim) ndarray
        modified particle coordinates, meeting steric constraints
    """

    assert isinstance(x, np.ndarray), "x must be np.ndarray"
    assert x.ndim == 2, "x must be 2d array"

    n   = x.shape[0]
    dim = x.shape[1]

    if r is None:
        r = 0.0
        logger.info("No steric radii explicitly specified, using none (zero).")

    r = np.atleast_1d(r)

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

    V = np.product(box[1,:]-box[0,:])
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
    x0 = X0.reshape(np.product(X0.shape))

    # define constraint and target wrapper for scipy optimizer
    g = lambda x: box_constraint(x, box=BOX, r=R )
    f = lambda x: target_function(x.reshape((n,dim)),r=R,constraints=g)

    callback_count = 0

    t0 = time.perf_counter()
    tk = t0

    def minimizer_callback(xk, *_):
        """Callback function that can be used by optimizers of scipy.optimize.
        The second argument "*_" makes sure that it still works when the
        optimizer calls the callback function with more than one argument. See
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        """
        nonlocal callback_count, tk
        if callback_count == 0:
            logger.info(
                "{:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
                    "#callback","objective","min. dist.", "timing, step", "timing, tot.") )

        fk = f(xk)
        Xk = xk.reshape((n,dim))
        mind = min_dist(Xk)

        t1 = time.perf_counter()
        dt = t1 - tk
        dT = t1 - t0
        tk = t1

        logger.info(
            "{:12d} {:12.5e} {:12.5e} {:12.5e} {:12.5e}".format(
                callback_count, fk, mind, dt, dT))

        callback_count += 1
        return

    res = scipy.optimize.minimize(f,x0,method='BFGS',
        callback=minimizer_callback, options=options)

    if not res.success:
        logger.warn(res.message)

    X1 = res.x
    f1 = f(X1)
    X1 = X1.reshape((n,dim))
    minD = min_dist(X1)
    x1 = X1*L
    mind = min_dist(x1)

    logger.info(
        """Final distribution has residual penalty {:10.5e} with minimum
        distance {:10.5e} or {:10.5e} (normalized by L = {:.2g}""".format(
            f1, minD, mind, L ))

    dT = time.perf_counter() - t0
    logger.info("Ellapsed time: {:10.5} s.".format(dT))

    return x1, res
