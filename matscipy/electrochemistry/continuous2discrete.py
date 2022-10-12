#
# Copyright 2019-2020 Johannes Hoermann (U. Freiburg)
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
Generates atomic structure following a given distribution.

Copyright 2019, 2020 IMTEK Simulation
University of Freiburg

Authors:

  Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>
  Lukas Elflein <elfleinl@cs.uni-freiburg.de>
"""
import logging, os, sys
import os.path
from six.moves import builtins
from collections.abc import Iterable

import numpy as np
# import matplotlib.pyplot as plt

import scipy.constants as sc
from scipy import integrate, optimize

logger = logging.getLogger(__name__)

def exponential(x, rate=0.1):
    """Exponential distribution."""
    return rate * np.exp(-1 * rate * x)


def uniform(x, *args, **kwargs):
    """Uniform distribution."""
    return np.ones(np.array(x).shape) / 2

def pdf_to_cdf(pdf):
    """Transform partial distribution to cumulative distribution function

    >>> pdf_to_cdf(pdf=np.array([0.5, 0.3, 0.2]))
    array([ 0.5,  0.8,  1. ])
    """
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]

    return cdf

def get_nearest_pos(array, value):
    """Find the value of an array clostest to the second argument.

    Example:
    >>> get_nearest_pos(array=[0, 0.25, 0.5, 0.75, 1.0], value=0.55)
    2
    """
    array = np.asarray(array)
    pos = np.abs(array - value).argmin()
    return pos


def get_histogram(struc, box, n_bins=100):
    """Slice the list of atomic positions, aggregate positions into histogram."""
    # Extract x/y/z positions only
    # x, y, z = struc[:, 0], struc[:, 1], struc[:, 2]

    histograms = []
    for dimension in range(struc.shape[1]):
        bins = np.linspace(0, box[dimension], n_bins)
        hist, bins = np.histogram(struc[:, dimension], bins=bins, density=True)
        # Normalize the histogram for all values to sum to 1
        hist /= sum(hist)

        histograms += [(hist, bins)]
    return histograms

def quartile_function(distribution, p, support=None):
    """Inverts a distribution x->p, and returns the x-value belonging to the provided p.

    Assumption: The distribution to be inverted must have a strictly increasing CDF!
    Also see 'https://en.wikipedia.org/wiki/Quantile_function'.

    Parameters
    ----------
    distribution: a function x -> p; x should be approximatable by a compact support
    p: an output of the distribution function, probablities in (0,1) are preferrable
    """
    if support is None:
        # Define the x-values to evaluate the function on
        support = np.arange(0,1,0.01)

    # Calculate the histogram of the distribution
    hist = distribution(support)

    # Sum the distribution to get the cumulatative distribution
    cdf =  pdf_to_cdf(hist)

    # If the p is not in the image of the support, get the nearest hit instead
    nearest_pos = get_nearest_pos(cdf, p)

    # Get the x-value belonging to the probablity value provided in the input
    x = support[nearest_pos]
    return x

def inversion_sampler(distribution, support):
    """Wrapper for quartile_function."""
    # z is distributed according to the given distribution
    # To approximate this, we insert an atom with probablity dis(z) at place z.
    # This we do by inverting the distribution, and sampling uniformely from distri^-1:
    p = np.random.uniform()
    sample = quartile_function(distribution, p, support=support)

    return sample


def rejection_sampler(distribution, support=(0.0,1.0), max_tries=10000, scale_M=1.1):
    """Sample distribution by drawing from support and keeping according to distribution.

    Draw a random sample from our support, and keep it if another random number is
    smaller than our target distribution at the support location.

        Algorithm: https://en.wikipedia.org/wiki/Rejection_sampling

    Parameters
    ----------
    distribution: callable(x)
        target distribut10on
    support: list or 2-tuple
        either discrete list of locations in space where our distribution is
        defined, or 2-tuple defining conitnuous support interval
    max_tries: how often the sampler should attempt to draw before giving up.
       If the distribution is very sparse, increase this parameter to still get results.
    scale_M: float, optional
        scales bound M for likelihood ratio

    Returns
    -------
    sample: a location which is conistent (in expectation) with being drawn from the distribution.
    """

    # rejection sampling (https://en.wikipedia.org/wiki/Rejection_sampling):
    # Generates sampling values from a target distribution X with arbitrary
    # probability density function f(x) by using a proposal distribution Y
    # with probability density g(x).
    # Concept: Generates a sample value from X by instead sampling from Y and
    # accepting this sample with probability f(x) / ( M g(x) ), repeating the
    # draws from Y until a value is accepted. M here is a constant, finite bound
    # on the likelihood ratio f(x)/g(x), satisfying 1 < M < infty over the
    # support of X; in other words, M must satisfy f(x) <= Mg(x) for all values
    # of x. The support of Y must include the support of X.

    # Here, f(x) = distribution(x), g(x) is uniform density on [0,1)
    # X are f-distributed positions from support, Y are uniformly distributed
    # values from [0,1)]
    logger.debug("Rejection sampler on distribution f(x) ({}) with".format(
        distribution))

    # coninuous support case
    if isinstance(support,tuple) and len(support) == 2:
        a = support[0]
        b = support[1]
        logger.debug("continuous support X (interval [{},{}]".format(a,b))
        # uniform probability density g(x) on support is
        g = 1 / ( b - a )
        # find maximum value fmax on distribution at x0
        xatol = (b - a)*1e-6 # optimization absolute tolerance
        x0 = optimize.minimize_scalar( lambda x: -distribution(x),
            bounds=(a,b), method='bounded', options={'xatol':xatol}).x
        fmax = distribution(x0)
        M = scale_M*fmax / g
        logger.debug("Uniform probability density g(x) = {:g} and".format(g))
        logger.debug("maximum probability density f(x0) = {:g} at x0 = {:g}".format(fmax, x0))
        logger.debug("require M >= scale_M*g(x)/max(f(x)), i.e. M = {:g}.".format(M))

        for i in range(max_tries):
            # draw a sample from a uniformly distributed support
            sample = np.random.random() * (b-a) + a

            # Generate random float in the half-open interval [0.0, 1.0) and .
            # keep sample with probablity of distribution
            if np.random.random() < distribution(sample) / (M*g):
                return sample

    else: # discrete support case
        logger.debug("discrete support X ({:d} points in interval [{},{}]".format(
            len(support), np.min(support), np.max(support)))
        # uniform probability density g(x) on support is
        g = 1.0 / len(support) # for discrete support
        # maximum probability on distributiom f(x) is
        fmax = np.max(distribution(support))
        # thus M must be at least
        M = scale_M * fmax / g
        logger.debug("Uniform probability g(x) = {:g} and".format(g))
        logger.debug("maximum probability max(f(x)) = {:g} require".format(fmax))
        logger.debug("M >= scale_M*g(x)/max(f(x)), i.e. M = {:g}.".format(M))

        for i in range(max_tries):
            # draw a sample from support
            sample = np.random.choice(support)

            # Generate random float in the half-open interval [0.0, 1.0) and .
            # keep sample with probablity of distribution
            if np.random.random() < distribution(sample) / (M*g):
                return sample

    raise RuntimeError('Maximum of attempts max_tries {} exceeded!'.format(max_tries))
    

def generate_structure(
    distribution, box=np.array([50, 50, 100]),
    count=100, n_gridpoints=np.nan):
    """Generate 'atoms' from continuous distributuion(s).

    Coordinates are distributed according to given distributions.

    Per default, X and Y coordinates are drawn uniformely.

    Parameters
    ----------
    distribution: func(x) or list of func(x)
      With one function, uniform sampling appplies along x and y axes,
      while applying 'distribution' along z axis. With a list of functions,
      apllies the respective distribution function along x, y and z direction.
    box: np.ndarray(3), optional (default: np.array([50, 50, 100]) )
      dimensions of volume to be filled with samples
    count: int, optional (default: 100)
      number of samples to draw
    n_gridpoints: int or (int,int,int), optional (default: np.nan)
      If spcefified, samples are not placed arbitrarily, but on an evenly spaced
      grid of this many grid points along each axis. Specify np.nan for
      continuous sampling, i.e. (10,np.nan,20)

    Returns
    -------
    np.ndarray((sample_size,3)): sample coordinates
    """
    global logger

    if callable(distribution):
        logger.info("Using uniform distribution along x and y direction.")
        logger.info("Using distribution {} along z direction.".format(
            distribution))
        distribution = [ uniform, uniform, distribution]

    for d in distribution:
        assert callable(d), "distribution {} must be callable".format(d)

    assert np.array(box).shape == (3,), "wrong specification of 3d box dimensions"

    #if isinstance(n_gridpoints,int) or n_gridpoints == np.nan:
    if not isinstance(n_gridpoints, Iterable):
        n_gridpoints = 3*[n_gridpoints] # to list

    n_gridpoints = np.array(n_gridpoints,dtype=float)
    logger.info("Using {} grid as sampling support.".format(
        n_gridpoints))

    assert n_gridpoints.shape == (3,), "n_gridpoints must be int or list of int"

    # We define which positions in space the atoms can be placed
    support = []
    normalized_distribution = []
    for k, d in enumerate(distribution):
        # Using the box parameter, we construct a grid inside the box
        # This results in a 100x100x100 grid:
        if np.isnan( n_gridpoints[k] ): # continuous support
            support.append((0,box[k])) # interval
            # Normalization constant:
            Z, _ = integrate.quad(d, support[-1][0], support[-1][1])
        else : # discrete supoport
            support.append(np.linspace(0, box[k], n_gridpoints[k]))
            Z = np.sum(d(support[-1])) # Normalization constant

        logger.info("Normalizing 'distribution' {} by {}.".format(d,Z))
        normalized_distribution.append(
            lambda x,k=k,Z=Z: distribution[k](x) / Z )

    # For every atom, draw random x, y and z coordinates
    positions = np.array( [ [
        rejection_sampler(d,s) for d,s in zip(normalized_distribution,support) ]
            for i in range(int(count)) ])

    logger.info("Drew {} samples from distributions.".format(positions.shape))
    return positions
