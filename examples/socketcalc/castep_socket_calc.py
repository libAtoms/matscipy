#!/usr/bin/env python

import logging
logging.root.setLevel(logging.INFO)

import os
from distutils import spawn

from ase import Atoms
from matscipy.socketcalc import CastepClient, SocketCalculator

if 'CASTEP' in os.environ:
      castep = os.environ['CASTEP']
else:
      castep = spawn.find_executable('castep.serial')

a = 5.404
bulk = Atoms(symbols='Si8',
             positions=[(0, 0, 0),
                        (0, 0.5, 0.5),
                        (0.5, 0, 0.5),
                        (0.5, 0.5, 0),
                        (0.25, 0.25, 0.25),
                        (0.25, 0.75, 0.75),
                        (0.75, 0.25, 0.75),
                        (0.75, 0.75, 0.25)],
             pbc=True)
bulk.set_cell((a, a, a), scale_atoms=True)

castep_client = CastepClient(client_id=0,
                             exe=castep)

try:
      sock_calc = SocketCalculator(castep_client)

      bulk.set_calculator(sock_calc)
      sock_e = bulk.get_potential_energy()
      sock_f = bulk.get_forces()
      sock_s = bulk.get_stress()

      print 'energy', sock_e
      print 'forces', sock_f
      print 'stress', sock_s

      bulk.rattle(0.01)

      sock_e = bulk.get_potential_energy()
      sock_f = bulk.get_forces()
      sock_s = bulk.get_stress()

      print 'energy', sock_e
      print 'forces', sock_f
      print 'stress', sock_s
finally:
      sock_calc.shutdown()
