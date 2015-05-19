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



