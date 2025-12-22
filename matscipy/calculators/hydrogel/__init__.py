#
# Copyright 2025 Antoine Sanner (ETH Zürich)
#           2025 Lars Pastewka (University of Freiburg)
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

"""Hydrogel calculator for coarse-grained hydrogel simulations.

This module provides an ASE calculator for coarse-grained hydrogel simulations
combining:
1. Flory-Huggins mixing free energy (density-dependent repulsion)
2. Langevin chain conformational free energy (bond stretching)

The model represents crosslinkers as particles connected by polymer chains.
"""

from .calculator import Hydrogel
from .potentials import FloryHuggins, LangevinChain, LucyWeightFunction


__all__ = [
    'Hydrogel',
    'FloryHuggins',
    'LangevinChain',
    'LucyWeightFunction',
]
