import logging
logging.root.setLevel(logging.DEBUG)

import os
from distutils import spawn

from ase import Atoms
from matscipy.socketcalc import VaspClient, SocketCalculator

# look for mpirun and vasp on $PATH
mpirun = spawn.find_executable('mpirun')
vasp = spawn.find_executable('vasp')

a = 5.404
bulk = Atoms(symbols='Si8',
             positions=[(0, 0, 0.1 / a),
                        (0, 0.5, 0.5),
                        (0.5, 0, 0.5),
                        (0.5, 0.5, 0),
                        (0.25, 0.25, 0.25),
                        (0.25, 0.75, 0.75),
                        (0.75, 0.25, 0.75),
                        (0.75, 0.75, 0.25)],
             pbc=True)
bulk.set_cell((a, a, a), scale_atoms=True)

vasp_client = VaspClient(client_id=0,
                         exe=vasp,
                         mpirun=mpirun,
                         parmode='mpi',
                         xc='LDA',
                         kpts=[4, 4, 4])

sock_calc = SocketCalculator(vasp_client)

bulk.set_calculator(sock_calc)
sock_e = bulk.get_potential_energy()
sock_f = bulk.get_forces()
sock_s = bulk.get_stress()
sock_calc.shutdown()



