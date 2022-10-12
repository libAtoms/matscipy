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
"""
Electrochemistry module utility functions

Copyright 2019, 2020 IMTEK Simulation
University of Freiburg

Authors:

  Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>
  Lukas Elflein <elfleinl@cs.uni-freiburg.de>
"""

import matplotlib.pyplot as plt

def get_centers(bins):
    """Return the center of the provided bins.

    Example:
    >>> get_centers(bins=np.array([0.0, 1.0, 2.0]))
    array([ 0.5,  1.5])
    """
    bins = bins.astype(float)
    return (bins[:-1] + bins[1:]) / 2


def plot_dist(histogram, name, reference_distribution=None):
    """Plot histogram with an optional reference distribution."""
    hist, bins = histogram
    width = 1 * (bins[1] - bins[0])
    centers = get_centers(bins)

    fi, ax = plt.subplots()
    ax.bar( centers, hist, align='center', width=width, label='Empirical distribution',
            edgecolor="none")

    if reference_distribution is not None:
        ref = reference_distribution(centers)
        ref /= sum(ref)
        ax.plot(centers, ref, color='red', label='Target distribution')

    plt.title(name)
    plt.legend()
    plt.xlabel('Distance ' + name)
    plt.savefig(name + '.png')
