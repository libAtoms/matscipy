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

import sys
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


if sys.version_info.major > 2:
    from test_c2d import *
    from test_poisson_nernst_planck_solver import *
else:
    print('Electrochemistry module requires python 3, related tests '
          '(test_c2d, test_poisson_nernst_planck_solver) skipped.')

try:
    from scipy.interpolate import InterpolatedUnivariateSpline
except:
    print('No scipy.interpolate.InterpolatedUnivariateSpline, skipping '
          'EAM test.')
else:
    print('EAM tests (eam_calculate, rotation_of_elastic_constants) are '
          'broken with added scipy 1.2.3 and otherwise current matscipy 0.3.0 '
          'Travis CI configuration (ase 3.13.0, numpy 1.12.1), hence skipped.')
    # from eam_calculator import *
    # from rotation_of_elastic_constants import *


# tests requiring these imports are skipped individually with unittest.skipIf()
# try:
#     from scipy.optimize import minimize
#     import matplotlib.pyplot
#     import atomman
# except:
#     print('One of these missing: scipy.optimize.minimize, matplotlib.pyplot, '
#           ' atomman. Skipping dislocation test.')
# else:
from test_dislocation import *
from test_pressurecoupling import *

###

unittest.main()
