# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014-2018) James Kermode, King's College London
#                       Lars Pastewka, University of Freiburg
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

from __future__ import division

import os

import sys

import time

import numpy as np

from scipy.special import factorial2

import ase

from ase.calculators.calculator import Calculator

from matscipy.neighbours import neighbour_list, first_neighbours

###

class IPL():
	"""
    Functional form for a inverse-power-law potential with an exponent of 10.
    Energy, its first, second and third derivative are shifted to zero at cutoff.
    Reference:
    E. Lerner, Journal of Non-Crystalline Solids, 522, 119570.
	"""
    
    def __init__(self, epsilon, cutoff):
        self.epsilon = epsilon
        self.cutoff = cutoff

    def __call__(self, r):
        """
        Return function value (potential energy)
        """

    def smoothing_coefficients(self, q, xc):
    	"""
        Return coefficients which smooth the potential up to q-th derivative
        """

    def get_cutoff(self):
        return self.cutoff

    def first_derivative(self, r):

    def second_derivative(self, r):

    def derivative(self, n=1):
        if n == 1:
            return self.first_derivative
        elif n == 2:
            return self.second_derivative
        else:
            raise ValueError(
                "Don't know how to compute {}-th derivative.".format(n))     	

###