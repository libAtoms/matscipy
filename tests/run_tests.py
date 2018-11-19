#! /usr/bin/env python

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

import unittest

from analysis import *
from angle_distribution import *
from cubic_crystal_crack import *
from eam_io import *
from elastic_moduli import *
from fit_elastic_constants import *
from full_to_Voigt import *
from greens_function import *
from Hertz import *
from hydrogenate import *
from idealbrittlesolid import *
from invariants import *
from neighbours import *
from ring_statistics import *
from spatial_correlation_function import *
from test_io import *
from crack_tests import *
from mcfm_test import *

try:
    from scipy.interpolate import InterpolatedUnivariateSpline
except:
    print('No scipy.interpolate.InterpolatedUnivariateSpline, skipping '
          'EAM test.')
else:
    from eam_calculator import *
    from rotation_of_elastic_constants import *


try:
    from scipy.optimize import minimize
except:
    print('No scipy.optimize.minimise, skipping '
          'dislocation test.')
else:
    from test_dislocation import *

###

unittest.main()

