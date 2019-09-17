#!/usr/bin/env python

import logging
logging.root.setLevel(logging.INFO)

import sys
import os
from distutils import spawn

from ase.lattice import bulk
from ase.io import read

from matscipy.socketcalc import CastepClient, SocketCalculator

if 'CASTEP_COMMAND' in os.environ:
      castep = os.environ['CASTEP_COMMAND']
else:
      castep = spawn.find_executable('castep.serial')

atoms = bulk('Si')

castep_client = CastepClient(client_id=0,
                             exe=castep,
                             devel_code="""PP=T
pp: NL=T SW=T :endpp
""")

try:
      sock_calc = SocketCalculator(castep_client)

      atoms.set_calculator(sock_calc)
      sock_e = atoms.get_potential_energy()
      sock_f = atoms.get_forces()
      sock_s = atoms.get_stress()

      print('energy', sock_e)
      print('forces', sock_f)
      print('stress', sock_s)

      atoms.cell *= 2.0 # trigger a restart by changing cell

      sock_e = atoms.get_potential_energy()
      sock_f = atoms.get_forces()
      sock_s = atoms.get_stress()

      print('energy', sock_e)
      print('forces', sock_f)
      print('stress', sock_s)

      atoms.rattle() # small change in position, no restart

      sock_e = atoms.get_potential_energy()
      sock_f = atoms.get_forces()
      sock_s = atoms.get_stress()

      print('energy', sock_e)
      print('forces', sock_f)
      print('stress', sock_s)
finally:
      sock_calc.shutdown()
