#
# Copyright 2016 Lars Pastewka (U. Freiburg)
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
from ase.units import fs

from matscipy.calculators.manybody import Manybody
from matscipy.calculators.manybody.explicit_forms import TersoffBrenner
from matscipy.calculators.manybody.explicit_forms.tersoff_brenner import Tersoff_PRB_39_5566_Si_C

# Quick and robust calculator to relax initial positions
quick_calc = Manybody(**TersoffBrenner(Tersoff_PRB_39_5566_Si_C))
# Calculator for actual quench
calc = quick_calc

stoichiometry = 'Si128C128'
densities = [3.21]

# These times are too low for production runs and are intended for demonstration only
dtdump = 10 * fs
teq = 1e3 * fs
tqu = 1e3 * fs


