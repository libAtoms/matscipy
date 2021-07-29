#
# Copyright 2015 James Kermode (Warwick U.)
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
import logging
#logging.root.setLevel(logging.DEBUG)

import os

from matscipy.socketcalc import QUIPClient, SocketCalculator
from matscipy.fracture_mechanics.clusters import diamond

from quippy import Potential

quip_client = QUIPClient(client_id=0,
                         exe=os.path.join(os.environ['QUIP_ROOT'],
                                          'build.'+os.environ['QUIP_ARCH'],
                                          'socktest'),
                         param_files=['params.xml'],
                         env=os.environ)

sock_calc = SocketCalculator(quip_client)

el = 'Si'
a0              = 5.43 
n               = [ 6, 4, 1 ]
crack_surface   = [ 1, 1, 1 ]
crack_front     = [ 1, -1, 0 ]

cryst = diamond(el, a0, n, crack_surface, crack_front)
cryst.pbc = [True, True, True] # QUIP doesn't handle open BCs
cryst.rattle(0.01)

cryst.set_calculator(sock_calc)
sock_e = cryst.get_potential_energy()
sock_f = cryst.get_forces()
sock_s = cryst.get_stress()
sock_calc.shutdown()

# compare to results from quippy
quippy_calc = Potential('IP SW', param_filename='params.xml')
cryst.set_calculator(quippy_calc)
quippy_e = cryst.get_potential_energy()
quippy_f = cryst.get_forces()
quippy_s = cryst.get_stress()

print 'energy difference', sock_e - quippy_e
print 'force difference', abs((sock_f - quippy_f).max())
print 'stress difference', abs((sock_s - quippy_s).max())



