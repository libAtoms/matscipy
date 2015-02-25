import logging
#logging.root.setLevel(logging.DEBUG)

import os

from matscipy.socketcalc import VaspClient, SocketCalculator
from matscipy.fracture_mechanics.clusters import diamond

from quippy import Potential

quip_client = VaspClient(client_id=0,
                         exe=os.path.join(os.environ['QUIP_ROOT'],
                                          'build.'+os.environ['QUIP_ARCH'],
                                          'socktest'),
                         gamma=True)

sock_calc = SocketCalculator(quip_client)

el = 'Si'
a0              = 5.43 
n               = [ 1, 1, 1 ]
crack_surface   = [ 1, 1, 1 ]
crack_front     = [ 1, -1, 0 ]

cryst = diamond(el, a0, n, crack_surface, crack_front)
cryst.pbc = [True, True, True]
cryst.rattle(0.01)

cryst.set_calculator(sock_calc)
sock_e = cryst.get_potential_energy()
sock_f = cryst.get_forces()
sock_s = cryst.get_stress()
sock_calc.shutdown()



