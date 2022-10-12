#
# Copyright 2020 Johannes Hoermann (U. Freiburg)
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
"""Enforces minimum distances on coordinates within discrete distribtution.

Copyright 2020 IMTEK Simulation
University of Freiburg

Authors:

    Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>

Examples
-------
Benchmark different scipy optimizers for the steric correction problem:

    >>> # measures of box
    >>> xsize = ysize = 5e-9 # nm, SI units
    >>> zsize = 10e-9    # nm, SI units
    >>>
    >>> # get continuum distribution, z direction
    >>> x = np.linspace(0, zsize, 2000)
    >>> c = [0.1,0.1]
    >>> z = [1,-1]
    >>> u = 0.05
    >>>
    >>> phi = potential(x, c, z, u)
    >>> C   = concentration(x, c, z, u)
    >>> rho = charge_density(x, c, z, u)
    >>>
    >>> # create distribution functions
    >>> distributions = [interpolate.interp1d(x,c) for c in C]
    >>>
    >>> # sample discrete coordinate set
    >>> box = np.array([xsize, ysize, zsize])m
    >>> sample_size = 100
    >>>
    >>> samples = [ continuous2discrete(
    >>>     distribution=d, box=box, count=sample_size) for d in distributions ]
    >>>
    >>> # apply penalty for steric overlap
    >>> x = np.vstack(samples)
    >>>
    >>> box = np.array([[0.,0.,0],box]) # needs lower corner
    >>>
    >>> n = x.shape[0]
    >>> dim = x.shape[1]
    >>>
    >>> # benchmakr methods
    >>> mindsq, (p1,p2) = scipy_distance_based_closest_pair(x)
    >>> pmin = np.min(x,axis=0)
    >>> pmax = np.max(x,axis=0)
    >>> mind = np.sqrt(mindsq)
    >>> logger.info("Minimum pair-wise distance in sample: {}".format(mind))
    >>> logger.info("First sample point in pair:    ({:8.4e},{:8.4e},{:8.4e})".format(*p1))
    >>> logger.info("Second sample point in pair    ({:8.4e},{:8.4e},{:8.4e})".format(*p2))
    >>> logger.info("Box lower boundary:            ({:8.4e},{:8.4e},{:8.4e})".format(*box[0]))
    >>> logger.info("Minimum coordinates in sample: ({:8.4e},{:8.4e},{:8.4e})".format(*pmin))
    >>> logger.info("Maximum coordinates in sample: ({:8.4e},{:8.4e},{:8.4e})".format(*pmax))
    >>> logger.info("Box upper boundary:            ({:8.4e},{:8.4e},{:8.4e})".format(*box[1]))
    >>>
    >>> # stats: method, x, res, dt, mind, p1, p2 , pmin, pmax
    >>> stats = [('initial',x,None,0,mind,p1,p2,pmin,pmax)]
    >>>
    >>> r = 4e-10 # 4 Angstrom steric radius
    >>> logger.info("Steric radius: {:8.4e}".format(r))
    >>>
    >>> methods = [
    >>>     'Powell',
    >>>     'CG',
    >>>     'BFGS',
    >>>     'L-BFGS-B'
    >>> ]
    >>>
    >>> for m in methods:
    >>>     try:
    >>>         logger.info("### {} ###".format(m))
    >>>         t0 = time.perf_counter()
    >>>         x1, res = apply_steric_correction(x,box=box,r=r,method=m)
    >>>         t1 = time.perf_counter()
    >>>         dt = t1 - t0
    >>>         logger.info("{} s runtime".format(dt))
    >>>
    >>>         mindsq, (p1,p2) = scipy_distance_based_closest_pair(x1)
    >>>         mind = np.sqrt(mindsq)
    >>>         pmin = np.min(x1,axis=0)
    >>>         pmax = np.max(x1,axis=0)
    >>>
    >>>         stats.append([m,x1,res,dt,mind,p1,p2,pmin,pmax])
    >>>
    >>>         logger.info("Minimum pair-wise distance in final configuration: {:8.4e}".format(mind))
    >>>         logger.info("First sample point in pair:    ({:8.4e},{:8.4e},{:8.4e})".format(*p1))
    >>>         logger.info("Second sample point in pair    ({:8.4e},{:8.4e},{:8.4e})".format(*p2))
    >>>         logger.info("Box lower boundary:            ({:8.4e},{:8.4e},{:8.4e})".format(*box[0]))
    >>>         logger.info("Minimum coordinates in sample: ({:8.4e},{:8.4e},{:8.4e})".format(*pmin))
    >>>         logger.info("Maximum coordinates in sample: ({:8.4e},{:8.4e},{:8.4e})".format(*pmax))
    >>>         logger.info("Box upper boundary:            ({:8.4e},{:8.4e},{:8.4e})".format(*box[1]))
    >>>     except:
    >>>         logger.warn("{} failed.".format(m))
    >>>         continue
    >>>
    >>> stats_df = pd.DataFrame( [ {
    >>>     'method':  s[0],
    >>>     'runtime': s[3],
    >>>     'mind':    s[4],
    >>>     **{'p1{:d}'.format(i): c for i,c in enumerate(s[5]) },
    >>>     **{'p2{:d}'.format(i): c for i,c in enumerate(s[6]) },
    >>>     **{'pmin{:d}'.format(i): c for i,c in enumerate(s[7]) },
    >>>     **{'pmax{:d}'.format(i): c for i,c in enumerate(s[8]) }
    >>> } for s in stats] )
    >>>
    >>> print(stats_df.to_string(float_format='%8.6g'))
    method  runtime        mind         p10         p11         p12         p20         p21         p22       pmin0       pmin1       pmin2       pmax0       pmax1       pmax2
    0   initial        0 1.15674e-10 2.02188e-09 4.87564e-10 5.21835e-09 2.03505e-09 3.72691e-10 5.22171e-09 1.17135e-12 1.49124e-10 6.34126e-12 4.98407e-09 4.99037e-09 9.86069e-09
    1    Powell  75.2704 8.02318e-10 4.23954e-09 3.36242e-09 8.80092e-09 4.31183e-09 2.56345e-09 8.81278e-09 4.01789e-10  4.0081e-10  4.2045e-10 4.59284e-09 4.54413e-09  9.5924e-09
    2        CG  27.0756  7.9992e-10 3.39218e-09 4.00079e-09 8.27255e-09 3.86337e-09 4.27807e-09 7.68863e-09 4.00018e-10 4.00146e-10 4.00565e-10 4.59941e-09 4.59989e-09 9.59931e-09
    3      BFGS  19.0255 7.99527e-10 1.82802e-09 3.54397e-09 9.69736e-10 2.41411e-09   3.936e-09 1.34664e-09 4.00514e-10 4.01874e-10  4.0002e-10 4.59695e-09 4.59998e-09 9.58155e-09
    4  L-BFGS-B  11.7869 7.99675e-10 4.34395e-09 3.94096e-09 1.28996e-09 4.44064e-09 3.15999e-09 1.14778e-09 4.12146e-10 4.01506e-10 4.03583e-10     4.6e-09 4.59898e-09  9.5982e-09
"""
import logging
import time

import _matscipy
import numpy as np

import scipy.optimize
import scipy.spatial.distance


# https://stackoverflow.com/questions/21377020/python-how-to-do-lazy-debug-logging
class DeferredMessage(object):
    """Lazy evaluation for log messages."""

    def __init__(self, msg, func, *args, **kwargs):
        self.msg = msg
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.msg.format(self.func(*self.args, **self.kwargs))


def brute_force_closest_pair(x):
    """Find coordinate pair with minimum distance squared ||xi-xj||^2.

    Parameters
    ----------
    x: (N,dim) ndarray
      coordinates

    Returns
    -------
    float, (ndarray, ndarray): minimum distance squared and coodinates pair

    Examples
    --------
    Compare the performance of closest pair algorithms:

        >>> from matscipy.electrochemistry.steric_distribution import scipy_distance_based_target_function
        >>> from matscipy.electrochemistry.steric_distribution import numpy_only_target_function
        >>> from matscipy.electrochemistry.steric_distribution import brute_force_target_function
        >>> import itertools
        >>> import pandas as pd
        >>> import scipy.spatial.distance
        >>> import timeit
        >>>
        >>> funcs = [
        >>>         brute_force_closest_pair,
        >>>         scipy_distance_based_closest_pair,
        >>>         planar_closest_pair ]
        >>> func_names = ['brute','scipy','planar']
        >>> stats = []
        >>> N = 1000
        >>> dim = 3
        >>> for k in range(5):
        >>>     x = np.random.rand(N,dim)
        >>>     lambdas = [ (lambda x=x,f=f: f(x)) for f in funcs ]
        >>>     rets    = [ f() for f in lambdas ]
        >>>     vals    = [ v[0] for v in rets ]
        >>>     coords  = [ c for v in rets for p in v[1] for c in p ]
        >>>     times   = [ timeit.timeit(f,number=1) for f in lambdas ]
        >>>     diffs   = scipy.spatial.distance.pdist(
        >>>         np.atleast_2d(vals).T,metric='euclidean')
        >>>     stats.append((*vals,*diffs,*times,*coords))
        >>>
        >>> func_name_tuples = list(itertools.combinations(func_names,2))
        >>> diff_names =  [ 'd_{:s}_{:s}'.format(f1,f2) for (f1,f2) in func_name_tuples ]
        >>> perf_names =  [ 't_{:s}'.format(f) for f in func_names ]
        >>> coord_names = [
        >>>     'p{:d}{:s}_{:s}'.format(i,a,f) for f in func_names for i in (1,2) for a in ('x','y','z') ]
        >>> float_fields = [*func_names,*diff_names,*perf_names,*coord_names]
        >>> dtypes = [ (field, 'f4') for field in float_fields ]
        >>> labeled_stats = np.array(stats,dtype=dtypes)
        >>> stats_df = pd.DataFrame(labeled_stats)
        >>> print(stats_df.T.to_string(float_format='%8.6g'))
                                 0           1           2           3           4
        brute          2.24089e-05 5.61002e-05 8.51047e-05 3.48424e-05 5.37235e-05
        scipy          2.24089e-05 5.61002e-05 8.51047e-05 3.48424e-05 5.37235e-05
        planar         2.24089e-05 5.61002e-05 8.51047e-05 3.48424e-05 5.37235e-05
        d_brute_scipy            0           0           0           0           0
        d_brute_planar           0           0           0           0           0
        d_scipy_planar           0           0           0           0           0
        t_brute            4.02697     3.85543      4.1414     3.90338     3.86993
        t_scipy         0.00708364  0.00698962  0.00762594  0.00703242  0.00703579
        t_planar           0.38302     0.39462    0.434342    0.407233    0.420773
        p1x_brute         0.132014    0.331441    0.553405    0.534633    0.977582
        p1y_brute         0.599688    0.186959     0.90897    0.575864    0.636278
        p1z_brute          0.49631    0.993856    0.246418    0.853567    0.411793
        p2x_brute         0.134631    0.333526     0.55322    0.534493    0.977561
        p2y_brute         0.603598    0.179771    0.915063    0.576894    0.629313
        p2z_brute         0.496833    0.994145    0.239493    0.859377    0.409509
        p1x_scipy         0.132014    0.331441    0.553405    0.534633    0.977582
        p1y_scipy         0.599688    0.186959     0.90897    0.575864    0.636278
        p1z_scipy          0.49631    0.993856    0.246418    0.853567    0.411793
        p2x_scipy         0.134631    0.333526     0.55322    0.534493    0.977561
        p2y_scipy         0.603598    0.179771    0.915063    0.576894    0.629313
        p2z_scipy         0.496833    0.994145    0.239493    0.859377    0.409509
        p1x_planar        0.132014    0.331441     0.55322    0.534633    0.977561
        p1y_planar        0.599688    0.186959    0.915063    0.575864    0.629313
        p1z_planar         0.49631    0.993856    0.239493    0.853567    0.409509
        p2x_planar        0.134631    0.333526    0.553405    0.534493    0.977582
        p2y_planar        0.603598    0.179771     0.90897    0.576894    0.636278
        p2z_planar        0.496833    0.994145    0.246418    0.859377    0.411793
    """
    logger = logging.getLogger(__name__)
    t0 = time.perf_counter()

    n = x.shape[0]
    imin = 0
    jmin = 1
    if n < 2:
        return (None, None), float('inf')

    dx = x[0,:] - x[1,:]
    dxsq = np.square(dx)
    mindsq = np.sum(dxsq)

    for i in np.arange(n):
        for j in np.arange(i+1, n):
            dx = x[i,:] - x[j,:]
            dxsq = np.square(dx)
            dxnormsq = np.sum(dxsq)
            if dxnormsq < mindsq:
                imin = i
                jmin = j
                mindsq = dxnormsq

    t1 = time.perf_counter()-t0
    logger.debug("""Found minimum distance squared {:10.5e} for pair
        ({:d},{:d}) with coodinates {} and {} within {:10.5e} s.""".format(
        mindsq, imin, jmin, x[imin,:], x[jmin,:], t1))
    return mindsq, (x[imin,:], x[jmin,:])


def recursive_closest_pair(x, y):
    """Find coordinate pair with minimum distance squared ||xi-xj||^2.

    Find coordinate pair with minimum distance squared ||xi-xj||^2
    with one point from x and the other point from y

    Parameters
    ----------
    x: (N,dim) ndarray
        coordinates

    Returns
    -------
    float, (ndarray, ndarray): minimum distance squared and coodinate pair
    """
    # t0 = time.perf_counter()

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
    mindsql, (pil, pjl) = recursive_closest_pair(xl, yl)
    mindsqr, (pir, pjr) = recursive_closest_pair(xr, yr)

    mindsq, (pim, pjm) = (mindsql, (pil, pjl)) if mindsql < mindsqr else (
        mindsqr, (pir, pjr))

    # TODO: this latter part only valid for 2d problems,
    # see https://sites.cs.ucsb.edu/~suri/cs235/ClosestPair.pdf
    # some 3d implementation at
    # https://github.com/eyny/closest-pair-3d/blob/master/src/ballmanager.cpp
    close_y = np.array(
        [y[j,:] for j in np.arange(m) if (np.square(y[j,0]-xdivider) < mindsq)])

    close_n = close_y.shape[0]
    if close_n > 1:
        for i in np.arange(close_n-1):
            for j in np.arange(i+1, min(i+8, close_n)):
                dx = close_y[i,:] - close_y[j,:]
                dxsq = np.square(dx)
                dxnormsq = np.sum(dxsq)
                if dxnormsq < mindsq:
                    pim = close_y[i,:]
                    pjm = close_y[j,:]
                    mindsq = dxnormsq

    return mindsq, (pim, pjm)


def planar_closest_pair(x):
    """Find coordinate pair with minimum distance ||xi-xj||.

    ATTENTION: this implementation tackles the planar problem!

    Parameters
    ----------
    x: (N,dim) ndarray
      coordinates

    Returns
    -------
    float, (ndarray, ndarray): minimum distance squared and coodinates pair
    """
    logger = logging.getLogger(__name__)
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
    """Find coordinate pair with minimum distance ||xi-xj||.

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

        >>> I,J = np.tril_indices(d.shape[0],-1)
        >>> print(I,J)
        [1 2 2 3 3 3] [0 0 1 0 1 2]

        >>> I,J = np.triu_indices(d.shape[0],1)
        >>> print(I,J)
        [0 0 0 1 1 2] [1 2 3 2 3 3]

        >>> print(d[I])
        [1 2 4 3 5 6]
    """
    logger = logging.getLogger(__name__)
    t0 = time.perf_counter()

    n = x.shape[0]

    dxnormsq = scipy.spatial.distance.pdist(x, metric='sqeuclidean')

    ij = np.argmin(dxnormsq)
    mindsq = dxnormsq[ij]

    # I,J = np.tril_indices(n,-1)
    I,J = np.triu_indices(n,1)
    imin,jmin = (I[ij],J[ij])

    t1 = time.perf_counter()-t0
    logger.debug("""Found minimum distance squared {:10.5e} for pair
        ({:d},{:d}) with coodinates {} and {} within {:10.5e} s.""".format(
            mindsq,imin,jmin,x[imin,:],x[jmin,:],t1))
    return mindsq, (x[imin,:], x[jmin,:])


def brute_force_target_function(x, r=1.0, constraints=None):
    """Target function. Penalize dense packing for coordinates ||xi-xj||<ri+rj.

    Parameters
    ----------
    x: (N,dim) ndarray
        particle coordinates
    r: float or (N,) ndarray, optional (default=1.0)
        steric radii of particles

    Returns
    -------
    float: target function value

    Examples
    --------
    Compare performance of target functions:


        >>> from matscipy.electrochemistry.steric_distribution import scipy_distance_based_target_function
        >>> from matscipy.electrochemistry.steric_distribution import numpy_only_target_function
        >>> from matscipy.electrochemistry.steric_distribution import brute_force_target_function
        >>> import itertools
        >>> import pandas as pd
        >>> import scipy.spatial.distance
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
        >>>     diffs = scipy.spatial.distance.pdist(np.atleast_2d(vals).T,metric='euclidean')
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
    logger = logging.getLogger(__name__)
    assert x.ndim == 2, "2d array expected for x"
    # sum_i sum_{j!=i} max(0,(r_i+r_j)"^2-||xi-xj||^2)^2
    f = 0
    n = x.shape[0]
    xi = x

    ri = r
    if not isinstance(r, np.ndarray) or r.shape != (n,):
        ri = ri*np.ones(n)
    assert  ri.shape == (n,)

    zeros = np.zeros(n)
    for i in np.arange(1,n):
        rj = np.roll(ri, i, axis=0)
        xj = np.roll(xi, i, axis=0)
        d = ri + rj
        dsq = np.square(d)
        dx = xi - xj
        dxsq = np.square(dx)
        dxnormsq = np.sum(dxsq, axis=1)
        sqdiff = dsq - dxnormsq
        penalty = np.maximum(zeros, sqdiff)
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


# TODO: code explicit Jacobian
def scipy_distance_based_target_function(x, r=1.0, constraints=None):
    """Target function. Penalize dense packing for coordinates ||xi-xj||<ri+rj.

    Parameters
    ----------
    x: (N,dim) ndarray
        particle coordinates
    r: float or (N,) ndarray, optional (default=1.0)
        steric radii of particles

    Returns
    -------
    float: target function value
    """
    logger = logging.getLogger(__name__)
    assert x.ndim == 2, "2d array expected for x"
    # sum_i sum_{j!=i} max(0,(r_i+r_j)"^2-||xi-xj||^2)^2
    n = x.shape[0]

    if not isinstance(r, np.ndarray) or r.shape != (n,):
        r = r*np.ones(n)
    assert r.shape == (n,)

    # r(Nx1) kron ones(1xN) = Ri(NxN)
    Ri = np.kron(r, np.ones((n, 1)))
    Rj = Ri.T
    Dij = Ri + Rj
    dij = scipy.spatial.distance.squareform(
        Dij, force='tovector', checks=False)

    zeros = np.zeros(dij.shape)

    dsq = np.square(dij)

    dxnormsq = scipy.spatial.distance.pdist(x, metric='sqeuclidean')
    # computes the squared Euclidean distance ||u-v||_2^2 between vectors
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    sqdiff = dsq - dxnormsq

    penalty = np.maximum(zeros, sqdiff)
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
    """Target function. Penalize dense packing for coordinates ||xi-xj||<ri+rj.

    Parameters
    ----------
    x: (N,dim) ndarray
        particle coordinates
    r: float or (N,) ndarray, optional (default=1.0)
        steric radii of particles

    Returns
    -------
    float: target function value
    """
    logger = logging.getLogger(__name__)
    assert x.ndim == 2, "2d array expected for x"
    # sum_i sum_{j!=i} max(0,(r_i+r_j)"^2-||xi-xj||^2)^2
    n = x.shape[0]

    if not isinstance(r, np.ndarray) or r.shape != (n,):
        r = r*np.ones(n)
    assert r.shape == (n,)

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
    sqdiff = dsq - dxnormsq

    penalty = np.maximum(zeros, sqdiff)
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


def neigh_list_based_target_function(x, r=1.0, constraints=None, Dij=None):
    """Target function. Penalize dense packing for coordinates ||xi-xj||<ri+rj.

    Parameters
    ----------
    x: (N,dim) ndarray
        particle coordinates
    r: float or (N,) ndarray, optional (default=1.0)
        steric radii of particles
    constraints: callable, returns (float, (N,dim) ndarray)
        constraint function value and gradien
    Dij: (N, N) ndarray, optional (default=None)
        pairwise minimum allowed distance matrix, overrides r

    Returns
    -------
    (float, (N,dim) ndarray): target function value and gradient

        f: float, target (or penalty) function, value evaluates to
              sum_i sum_{j!=i} max(0,(r_i+r_j)^2-||xi-xj||^2)^2
            without double-counting pairs, where r_i and r_j are
            the steric radii of coordinate points i and j
        df: (N,dim) ndarray of float, the gradient, evaluates to
              4*si sj ((r_i'+r_j)^2-||xi-xj||^2)^2)*(xik'-xjk')*(kdi'j-kdi'i)
            for entry (i',k'), where i subscribes the coordinate point and k
            the spatial dimension. kd is Kronecker delta, si is sum over i
    """
    logger = logging.getLogger(__name__)
    #
    # function:
    #
    # f(x) = sum_i sum_{j!=i} max(0,(r_i+r_j)^2-||xi-xj||^2)^2
    #
    # gradient:
    #
    # dfdxi'k'=4*si sj ((r_i'+r_j)^2-||xi-xj||^2)^2)*(xik'-xjk')*(kdi'j-kdi'i)
    #
    # for all pairs i'j within ri'+rj cutoff
    # where k is the spatial direction x, y or z.

    assert x.ndim == 2, "2d array expected for x"
    n = x.shape[0]

    if not Dij:
        if not isinstance(r, np.ndarray) or r.shape != (n,):
            r = r*np.ones(n)
        assert r.shape == (n,)

        # compute minimum allowed pairwise distances Dij (NxN matrix)
        # r(Nx1) kron ones(1xN) = Ri(NxN)
        Ri = np.kron(r, np.ones((n, 1)))
        Rj = Ri.T
        Dij = Ri + Rj

    # TODO: allow for periodic boundaries, for now use shrink wrapped box
    box = np.array([x.min(axis=0), x.max(axis=0)])
    cell_origin = box[0,:]
    cell = np.diag(box[1,:] - box[0,:])

    # get all pairs within their allowed minimum distance
    # parameters are
    # _matscipy.neighbour_list(quantities, cell_origin, cell,
    #                          np.linalg.inv(cell.T), pbc, positions,
    #                          cutoff, numbers)
    # If thhe parameter 'cutoff' is a per-atom value, then it must be a
    # diamater, not a radius (as wrongly stated within the function's
    # docstring)
    i, j, dxijnorm, dxijvec = _matscipy.neighbour_list(
        'ijdD', cell_origin, cell, np.linalg.inv(cell.T), [0,0,0],
        x, 2.0*Ri, np.ones(len(x),dtype=np.int32))
    # i, j are coordinate point indices, dxijnorm is pairwise distance,
    #   dxvijvec is distance vector

    # nl contains redundnancies, i.e. ij AND ji
    # pairs = list(zip(i,j))
    logger.debug("Number of pairs within minimum allowed distance: {:d}"
                 .format(len(i)))

    # get minimum allowed pairwise distance for all pairs within this distance
    dij = Dij[i,j]

    # (r_i'+r_j)^2
    dsq = np.square(dij)

    # ||xi-xj||^2
    dxnormsq = np.square(dxijnorm)

    # (r_i+r_j)^2-||xi-xj||^2
    sqdiff = dsq - dxnormsq

    # ((r_i+r_j)^2-||xi-xj||^2)^2
    penaltysq = np.square(sqdiff)

    # correct for double counting due to redundancies
    f = 0.5*np.sum(penaltysq)

    # 4*si sj ((r_i'+r_j)^2-||xi-xj||^2)^2)*(xik'-xjk')*(kdi'j-kdi'i)
    # (N x 1) column vector (a1 ... aN) * (N x dim) matrix ((d11,))
    # Other than the formula above implicates, _matscipy.neighbour_list
    # returns the distance vector dxijvec = xj-xi (pointing from i to j),
    # thus positive sign in gradient below:
    gradij = 4*np.atleast_2d(sqdiff).T*dxijvec  # let's hope for proper broadcasting
    grad = np.zeros(x.shape)
    grad[i] += gradij
    # Since neighbour list always includes ij and ji, we only have to treat i,
    # not j. The following line is obsolete (otherwise we would introduce
    # double-counting again):
    # grad[j] -= gradij

    if constraints:
        logger.debug(
            "Unconstrained penalty:       {:10.5e}.".format(f))
        logger.debug(DeferredMessage(
            "Unconstrained gradient norm: {:10.5e}.",
            np.linalg.norm, grad))
        g, g_grad = constraints(x)
        f += g
        grad += g_grad

    grad_1d = grad.reshape(np.product(grad.shape))

    logger.debug(
        "Total penalty:               {:10.5e}.".format(f))
    logger.debug(DeferredMessage(
        "Gradient norm:               {:10.5e}.",
        np.linalg.norm, grad_1d))

    return f, grad_1d


def box_constraint(x, box=np.array([[0., 0., 0], [1.0, 1.0, 1.0]]), r=0.):
    """Constraint function. Confine coordinates within box.

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
    logger = logging.getLogger(__name__)
    zeros = np.zeros(x.shape)
    r = np.atleast_2d(r).T

    # positive if coordinates out of box
    ldist = box[0, :] - x + r
    rdist = x + r - box[1, :]

    lpenalty = np.maximum(zeros, ldist)
    rpenalty = np.maximum(zeros, rdist)

    lpenaltysq = np.square(lpenalty)
    rpenaltysq = np.square(rpenalty)

    g = np.sum(lpenaltysq) + np.sum(rpenaltysq)
    logger.debug("Constraint penalty: {:.4g}.".format(g))
    return g


def box_constraint_with_gradient(
        x, box=np.array([[0., 0., 0], [1.0, 1.0, 1.0]]), r=0.):
    """Constraint function. Confine coordinates within box.

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
    (float, (N,dim) ndarray of float): penalty g(x) and its gradient dg(x)

        g(x), positive:
                lik = box[0,k] - xik + ri  > 0 for xik out of lower boundaries
                rik = xik + ri - box[1,k]  > 0 for xik out of upper boundaries
            where box[0] and box[1] are lower and upper corner of bounding box
            and k marks spatial dimension.
                g(x) = sum_i sum_k ( max(0,lik) + max(0,rik) )^2

        dg(x): gradient, entries accordingly evaluates to
                dgdxik(x) = 2*( -max(0,lik) + max(0,rik) )
    """
    logger = logging.getLogger(__name__)
    zeros = np.zeros(x.shape)
    r = np.atleast_2d(r).T

    # positive if coordinates out of box
    ldist = box[0, :] - x + r
    rdist = x + r - box[1, :]

    lpenalty = np.maximum(zeros, ldist)
    rpenalty = np.maximum(zeros, rdist)

    lpenaltysq = np.square(lpenalty)
    rpenaltysq = np.square(rpenalty)

    g = np.sum(lpenaltysq) + np.sum(rpenaltysq)
    logger.debug("Constraint penalty: {:.4g}.".format(g))

    grad = -2*lpenalty + 2*rpenalty
    logger.debug(DeferredMessage(
        "Norm of constraint penalty gradient: {:.4g}.",
        np.linalg.norm, grad))

    return g, grad


def apply_steric_correction(
        x, box=None, r=None,
        method='L-BFGS-B',
        options={'gtol':1.e-8,'maxiter':100,'disp':True,'eps':1.0e-8},
        target_function=neigh_list_based_target_function,
        returns_gradient=True,
        closest_pair_function=scipy_distance_based_closest_pair):
    """Enforce steric constraints on coordinate distribution within box.

    Parameters
    ----------
    x : (N,dim) ndarray
        Particle coordinates.
    box: (2,dim) ndarray, optional (default: None)
        Box corner coordinates.
    r : float or (N,) ndarray, optional (default=None)
        Steric radius of particles. Can be specified particle-wise.
    options : dict, optional
        Forwarded to scipy minimzer.
        https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html
        (default: {'gtol':1.e-5,'maxiter':10,'disp':True,'eps':1.e-8})
    target_function: func, optional
        One of the target functions within this submodule, or function
        of same signature. (default: neigh_list_based_target_function)
    returns_gradient: bool, optional (default: Trze)
        If True, then 'target_function' is expected to return a tuple (f, df)
        with f the actual target function value and df its (N,dim) gradient.
        This flag must be set for 'neigh_list_based_target_function'.
    closest_pair_function: func, optional
        One of the closest pair functions within this submodule, or function
        of same signature. (default: scipy_distance_based_closest_pair)

    Returns
    -------
    float : (N,dim) ndarray
        Modified particle coordinates, meeting steric constraints.
    """
    logger = logging.getLogger(__name__)
    assert isinstance(x, np.ndarray), "x must be np.ndarray"
    assert x.ndim == 2, "x must be 2d array"

    n = x.shape[0]
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
        box = np.array(x.min(axis=0), x.max(axis=0))
        logger.info("No bounding box explicitly specified, using extreme")
        logger.info("coordinates ({}) of coordinate set as default.".format(
            box))

    assert isinstance(box, np.ndarray), "box must be np.ndarray"
    assert x.ndim == 2, "box must be 2d array"
    assert box.shape[0] == 2, "box must have two rows for outer corners"
    assert box.shape[1] == dim, "spatial dimensions of x and box must agree"

    V = np.product(box[1, :]-box[0, :])
    L = np.power(V, (1./dim))
    logger.info("Normalizing coordinates by reference length")
    logger.info("    L = V^(1/dim) = ({:.2g})^(1/{:d}) = {:.2g}.".format(
        V, dim, L))

    # normalizing to unit volume necessary,
    # as target function apparently not dimension-insensitive
    BOX = box / L
    X0 = x / L
    R = r / L

    logger.info("Normalized bounding box: ")
    logger.info("    {}.".format(BOX[0]))
    logger.info("    {}.".format(BOX[1]))

    # flatten coordinates for scipy optimizer
    x0 = X0.reshape(np.product(X0.shape))


    # define constraint and target wrapper for scipy optimizer
    if returns_gradient:

        def g(x):
            return box_constraint_with_gradient(x, box=BOX, r=R)

        def f(x):
            f, grad = target_function(x.reshape((n, dim)), r=R, constraints=g)
            return f, grad.reshape(np.product(grad.shape))

        gval, ggrad = g(X0)
        fval, fgrad = f(x0)
        logger.info("Initial constraint penalty:       {:10.5e}.".format(gval))
        logger.info("Initial total penalty:            {:10.5e}.".format(fval))
        logger.info("Initial constraint gradient norm: {:10.5e}.".format(
            np.linalg.norm(ggrad)))
        logger.info("Initial total gradient norm:      {:10.5e}.".format(
            np.linalg.norm(fgrad)))

    else:

        def g(x):
            return box_constraint(x, box=BOX, r=R)

        def f(x):
            return target_function(x.reshape((n, dim)), r=R, constraints=g)

        logger.info("Initial constraint penalty: {:10.5e}.".format(g(X0)))
        logger.info("Initial total penalty:      {:10.5e}.".format(f(x0)))

    # log initial minimum distance between pairs
    minDsq, (P1, P2) = closest_pair_function(X0)  # dimensionless
    mindsq, (p1, p2) = closest_pair_function(x)  # dimensional

    minD = np.sqrt(minDsq)
    mind = np.sqrt(mindsq)

    # logger.info("Final distribution has residual penalty {:10.5e}.".format(f1))
    logger.info("Min. dist. {:10.5e} between points".format(mind))
    logger.info("    {} and".format(p1))
    logger.info("    {}.".format(p2))
    logger.info("Min. dist. {:10.5e} between dimensionless points".format(minD))
    logger.info("    {} and".format(P1))
    logger.info("    {}.".format(P2))
    logger.info("    normalized by L = {:.2g}.".format(L))




    def minimizer_callback(xk, *_):
        """Callback function to be used by optimizers of scipy.optimize.

        The second argument "*_" makes sure that it still works when the
        optimizer calls the callback function with more than one argument. See
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        """
        nonlocal closest_pair_function, callback_count, tk, returns_gradient
        if callback_count == 0 and returns_gradient:
            logger.info(
                "{:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
                    "#callback", "objective", "gradient", "min. dist.",
                    "timing, step", "timing, tot."))
        elif callback_count == 0:
            logger.info(
                "{:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
                    "#callback", "objective", "min. dist.",
                    "timing, step", "timing, tot."))

        if returns_gradient:
            fk, gradk = f(xk)
            normgradk = np.linalg.norm(gradk)
        else:
            fk = f(xk)

        Xk = xk.reshape((n, dim))
        mindsq, _ = closest_pair_function(Xk)
        mind = np.sqrt(mindsq)

        t1 = time.perf_counter()
        dt = t1 - tk
        dT = t1 - t0
        tk = t1

        if returns_gradient:
            logger.info(
                "{:12d} {:12.5e} {:12.5e} {:12.5e} {:12.5e} {:12.5e}".format(
                    callback_count, fk, normgradk, mind, dt, dT))
        else:
            logger.info(
                "{:12d} {:12.5e} {:12.5e} {:12.5e} {:12.5e}".format(
                    callback_count, fk, mind, dt, dT))

        callback_count += 1

    callback_count = 0
    t0 = time.perf_counter()
    tk = t0  # previosu callback timer value
    # call once for initial configuration
    if logger.isEnabledFor(logging.INFO):
        callback = minimizer_callback
        callback(x0)
    else:
        callback = None

    # neat lecture on scipy optimizers
    # http://scipy-lectures.org/advanced/mathematical_optimization/
    res = scipy.optimize.minimize(f, x0, method=method, jac=returns_gradient,
                                  callback=callback, options=options)

    if not res.success:
        logger.warn(res.message)

    x1 = res.x  # dimensionless, flat
    X1 = x1.reshape((n, dim))  # dimensionless, 2d

    if returns_gradient:
        gval, ggrad = g(X1)
        fval, fgrad = f(x1)
        logger.info("Final constraint penalty:       {:10.5e}.".format(gval))
        logger.info("Final total penalty:            {:10.5e}.".format(fval))
        logger.info("Final constraint gradient norm: {:10.5e}.".format(
            np.linalg.norm(ggrad)))
        logger.info("Final total gradient norm:      {:10.5e}.".format(
            np.linalg.norm(fgrad)))

    else:
        logger.info("Final constraint penalty: {:10.5e}.".format(g(X1)))
        logger.info("Final total penalty:      {:10.5e}.".format(f(x1)))

    x1 = X1*L  # dimensional
    minDsq, (P1, P2) = closest_pair_function(X1)  # dimensionless
    mindsq, (p1, p2) = closest_pair_function(x1)  # dimensional

    minD = np.sqrt(minDsq)
    mind = np.sqrt(mindsq)

    # logger.info("Final distribution has residual penalty {:10.5e}.".format(f1))
    logger.info("Min. dist. {:10.5e} between points".format(mind))
    logger.info("    {} and".format(p1))
    logger.info("    {}.".format(p2))
    logger.info("Min. dist. {:10.5e} between dimensionless points".format(minD))
    logger.info("    {} and".format(P1))
    logger.info("    {}.".format(P2))
    logger.info("    normalized by L = {:.2g}.".format(L))

    dT = time.perf_counter() - t0
    logger.info("Ellapsed time: {:10.5} s.".format(dT))

    return x1, res
