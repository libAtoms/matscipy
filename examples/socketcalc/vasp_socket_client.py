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
logging.root.setLevel(logging.DEBUG)

import os
from distutils import spawn

from ase import Atoms
from matscipy.socketcalc import VaspClient, SocketCalculator

# look for mpirun and vasp on $PATH
mpirun = spawn.find_executable('mpirun')
vasp = spawn.find_executable('vasp')
#vasp = '/home/eng/essswb/vasp5/vasp.5.3.new/vasp'

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
                         npj=1,
                         ppn=8,
                         exe=vasp,
                         mpirun=mpirun,
                         parmode='mpi',
                         xc='LDA',
                         lreal=False, ibrion=13, nsw=1000000,
                         algo='VeryFast', npar=8, 
                         lplane=False, lwave=False, lcharg=False, nsim=1,
                         voskown=1, ismear=0, sigma=0.01, iwavpr=11, isym=0, nelm=150)

sock_calc = SocketCalculator(vasp_client)

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

sock_calc.shutdown()
