#
# Copyright 2014-2015, 2017, 2021 Lars Pastewka (U. Freiburg)
#           2018-2021 Jan Griesser (U. Freiburg)
#           2020 Jonas Oldenstaedt (U. Freiburg)
#           2015 Adrien Gola (KIT)
#           2014 James Kermode (Warwick U.)
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

from .eam import EAM
from .pair_potential import PairPotential
from .supercell_calculator import SupercellCalculator
from .polydisperse import Polydisperse
from .manybody import Manybody
from .ewald import Ewald

try:
    import scipy.sparse as sp
except ImportError:
    warnings.warn('Warning: no scipy')

