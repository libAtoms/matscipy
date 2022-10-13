#
# Copyright 2014-2015, 2017, 2021 Lars Pastewka (U. Freiburg)
#           2020 Wolfram G. Nöhring (U. Freiburg)
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
"""Implements the Embedded Atom Method"""

from .calculator import EAM
from .io import mix_eam, read_eam, write_eam
from .average_atom import average_potential
