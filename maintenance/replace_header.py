#
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


import sys

file_lines = open(sys.argv[1], 'r').readlines()
header_lines = sys.stdin.readlines()

while file_lines[0].startswith('#'):
    file_lines = file_lines[1:]

file_lines.insert(0, '#\n')
for header_line in header_lines[::-1]:
    file_lines.insert(0, '# {}'.format(header_line).strip() + '\n')
file_lines.insert(0, '#\n')

open(sys.argv[1], 'w').writelines(file_lines)
