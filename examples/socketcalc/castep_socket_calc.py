#!/usr/bin/env python

import logging
logging.root.setLevel(logging.INFO)

import sys
import os
from distutils import spawn

from ase.lattice import bulk
from ase.io import read

import ase.calculators.castep

from matscipy.socketcalc import CastepClient, SocketCalculator

from matscipy.socketcalc_old import CastepClient as CastepClient_old, SocketCalculator as SocketCalculator_old

if 'CASTEP_COMMAND' in os.environ:
      castep = os.environ['CASTEP_COMMAND']
else:
      castep = spawn.find_executable('castep.serial')

atoms = bulk('Si', cubic = 'True')

castep_client = CastepClient(client_id=0,
                             exe=castep,
                             devel_code="""PP=T
pp: NL=T SW=T :endpp
""")

old_castep_client = CastepClient_old(client_id=0,
                                     exe=castep,
                                     devel_code="""PP=T
pp: NL=T SW=T :endpp
""")



non_sock_calc = ase.calculators.castep.Castep()


#PROBLEM: I DO NOT KNOW HOW TO INSERT DEVEL CODE INTO CASTEP
#non_sock_calc.params.devel_code='PP=T pp: NL=T SW=T :endpp'
#non_sock_calc.params.PP=T
#non_sock_calc.params.SW=T


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




      atoms[0].number = 6
      atoms[1].number = 6
      atoms[4].number = 6
      atoms[5].number = 6

      atoms2 = atoms.copy()
      atoms.set_calculator(non_sock_calc)

      atoms3 = atoms.copy()
      atoms3.set_calculator(old_castep_client)

      socketcalc_energies = atoms.get_potential_energy()
      socketcalc_forces = atoms.get_forces()

      atoms2.set_calculator(non_sock_calc)

      nonesocketcalc_energies = atoms2.get_potential_energy()
      nonesocketcalc_forces = atoms2.get_forces()

      old_socketcalc_energies = atoms3.get_potential_energy()
      old_socketcalc_forces = atoms3.get_forces()

      print(socketcalc_energies - nonesocketcalc_energies)
      print(socketcacl_forces - nonesocketcalc_forces)


      print(old_socketcalc_energies - nonesocketcalc_energies)
      print(old_socketcalc_forces - nonesocketcalc_forces)

finally:
      sock_calc.shutdown()
