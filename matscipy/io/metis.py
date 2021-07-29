#
# Copyright 2014-2016, 2021 Lars Pastewka (U. Freiburg)
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

def save_metis(fn, a, i, j):
    """
    Save neighbour list as the METIS graph file format.

    See here: http://glaros.dtc.umn.edu/gkhome/views/metis

    Parameters
    ----------
    fn : str
        File name.
    a : Atoms
        Atoms object.
    i, j : array_like
        Neighbour list.
    """
    f = open(fn, 'w')

    # Output number of vertices and number of edges
    print('{} {}'.format(len(a), len(i)//2), file=f)

    s = ''
    lasti = i[0]
    for _i, _j in zip(i, j):
        if _i != lasti:
            print(s.strip(), file=f)
            s = ''
        s += '{} '.format(_j+1)
        lasti = _i
    print(s.strip(), file=f)
    f.close()
