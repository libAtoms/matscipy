#
# Copyright 2020 James Kermode (Warwick U.)
#           2017, 2020 Lars Pastewka (U. Freiburg)
#           2016 Punit Patel (Warwick U.)
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
import unittest

import numpy as np
import ase.io
import ase.units as units
from ase.build import bulk
from ase.constraints import FixAtoms, UnitCellFilter
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE

import matscipy.fracture_mechanics.crack as crack
from matscipy.elasticity import fit_elastic_constants
from matscipy.fracture_mechanics.crack import ConstantStrainRate
from matscipy.fracture_mechanics.clusters import diamond, set_groups
from matscipy.neighbours import neighbour_list

try:
    import atomistica
    from atomistica import TersoffScr, Tersoff_PRB_39_5566_Si_C__Scr
    from atomistica import Tersoff, Tersoff_PRB_39_5566_Si_C
    have_atomistica = True
except ImportError:
    have_atomistica = False

if have_atomistica:

    class TestConstantStrain(unittest.TestCase):

        def test_apply_strain(self):
            calc = TersoffScr(**Tersoff_PRB_39_5566_Si_C__Scr)
            timestep = 1.0*units.fs

            atoms = ase.io.read('cryst_rot_mod.xyz')
            atoms.set_calculator(calc)

            # constraints
            top = atoms.positions[:, 1].max()
            bottom = atoms.positions[:, 1].min()
            fixed_mask = ((abs(atoms.positions[:, 1] - top) < 1.0) |
                          (abs(atoms.positions[:, 1] - bottom) < 1.0))
            fix_atoms = FixAtoms(mask=fixed_mask)

            # strain
            orig_height = (atoms.positions[:, 1].max() - atoms.positions[:, 1].min())
            delta_strain = timestep*1e-5*(1/units.fs)
            rigid_constraints = False
            strain_atoms = ConstantStrainRate(orig_height, delta_strain)
            atoms.set_constraint(fix_atoms)

            # dynamics
            np.random.seed(0)
            simulation_temperature = 300*units.kB
            MaxwellBoltzmannDistribution(atoms, 2.0*simulation_temperature)
            dynamics = VelocityVerlet(atoms, timestep)

            def apply_strain(atoms, ConstantStrainRate, rigid_constraints):
                ConstantStrainRate.apply_strain(atoms, rigid_constraints)

            dynamics.attach(apply_strain, 1, atoms, strain_atoms, rigid_constraints)
            dynamics.run(100)

            # tests
            if rigid_constraints:
                answer = 0
                temp_answer = 238.2066417638124
            else:
                answer = 0.013228150080099255
                temp_answer = 236.76904696481486

            newpos = atoms.get_positions()
            current_height = newpos[:, 1].max() - newpos[:, 1].min()
            diff_height = (current_height - orig_height)
            self.assertAlmostEqual(diff_height, answer, places=3)

            temperature = (atoms.get_kinetic_energy()/(1.5*units.kB*len(atoms)))
            self.assertAlmostEqual(temperature, temp_answer, places=2)

        def test_embedding_size_convergence(self):
            calc = Tersoff(**Tersoff_PRB_39_5566_Si_C)

            el = 'C'
            a0 = 3.566
            surface_energy = 2.7326 * 10
            crack_surface = [1, 1, 1]
            crack_front = [1, -1, 0]
            skin_x, skin_y = 1, 1

            cryst = bulk(el, cubic=True)
            cryst.set_calculator(calc)
            FIRE(UnitCellFilter(cryst), logfile=None).run(fmax=1e-6)
            a0 = cryst.cell.diagonal().mean()
            bondlength = cryst.get_distance(0, 1)
            #print('a0 =', a0, ', bondlength =', bondlength)

            cryst = diamond(el, a0, [1,1,1], crack_surface, crack_front)
            cryst.set_pbc(True)
            cryst.set_calculator(calc)
            cryst.set_cell(cryst.cell.diagonal(), scale_atoms=True)

            C, C_err = fit_elastic_constants(cryst, verbose=False,
                                             symmetry='cubic',
                                             optimizer=FIRE,
                                             fmax=1e-6)
            #print('Measured elastic constants (in GPa):')
            #print(np.round(C*10/units.GPa)/10)

            bondlengths = []
            refcell = None
            reftip_x = None
            reftip_y = None
            #[41, 39, 1], 
            for i, n in enumerate([[21, 19, 1], [11, 9, 1], [6, 5, 1]]):
                #print(n)
                cryst = diamond(el, a0, n, crack_surface, crack_front)
                set_groups(cryst, n, skin_x, skin_y)
                cryst.set_pbc(True)
                cryst.set_calculator(calc)
                FIRE(UnitCellFilter(cryst), logfile=None).run(fmax=1e-6)
                cryst.set_cell(cryst.cell.diagonal(), scale_atoms=True)

                ase.io.write('cryst_{}.xyz'.format(i), cryst, format='extxyz')

                crk = crack.CubicCrystalCrack(crack_surface,
                                              crack_front,
                                              Crot=C/units.GPa)
                k1g = crk.k1g(surface_energy)

                tip_x = cryst.cell.diagonal()[0]/2
                tip_y = cryst.cell.diagonal()[1]/2

                a = cryst.copy()
                a.set_pbc([False, False, True])

                k1 = 1.0
                ux, uy = crk.displacements(cryst.positions[:,0], cryst.positions[:,1],
                                           tip_x, tip_y, k1*k1g)

                a.positions[:, 0] += ux
                a.positions[:, 1] += uy

                # Center notched configuration in simulation cell and ensure enough vacuum.
                oldr = a[0].position.copy()
                if refcell is None:
                    a.center(vacuum=10.0, axis=0)
                    a.center(vacuum=10.0, axis=1)
                    refcell = a.cell.copy()
                    tip_x += a[0].x - oldr[0]
                    tip_y += a[0].y - oldr[1]
                    reftip_x = tip_x
                    reftip_y = tip_y
                else:
                    a.set_cell(refcell)

                # Shift tip position so all systems are exactly centered at the same spot
                a.positions[:, 0] += reftip_x - tip_x
                a.positions[:, 1] += reftip_y - tip_y

                refpositions = a.positions.copy()

                # Move reference crystal by same amount
                cryst.set_cell(a.cell)
                cryst.set_pbc([False, False, True])
                cryst.translate(a[0].position - oldr)

                bond1, bond2 = crack.find_tip_coordination(a, bondlength=bondlength*1.2)

                # Groups mark the fixed region and the region use for fitting the crack tip.
                g = a.get_array('groups')
                gcryst = cryst.get_array('groups')

                ase.io.write('cryst_{}.xyz'.format(i), cryst)

                a.set_calculator(calc)
                a.set_constraint(FixAtoms(mask=g==0))
                FIRE(a, logfile=None).run(fmax=1e-6)

                dpos = np.sqrt(((a.positions[:, 0]-refpositions[:, 0])/ux)**2 + ((a.positions[:, 1]-refpositions[:, 1])/uy)**2)
                a.set_array('dpos', dpos)

                distance_from_tip = np.sqrt((a.positions[:, 0]-reftip_x)**2 + (a.positions[:, 1]-reftip_y)**2)

                ase.io.write('crack_{}.xyz'.format(i), a)

                # Compute average bond length per atom
                neighi, neighj, neighd = neighbour_list('ijd', a, cutoff=bondlength*1.2)
                coord = np.bincount(neighi)
                assert coord.max() == 4

                np.savetxt('dpos_{}.out'.format(i), np.transpose([distance_from_tip[coord==4], dpos[coord==4]]))

                # Compute distances from tipcenter
                neighdist = np.sqrt(((a.positions[neighi,0]+a.positions[neighj,0])/2-reftip_x)**2 +
                                    ((a.positions[neighi,1]+a.positions[neighj,1])/2-reftip_y)**2)

                np.savetxt('bl_{}.out'.format(i), np.transpose([neighdist, neighd]))

                bondlengths += [a.get_distance(bond1, bond2)]
            print(bondlengths, np.diff(bondlengths), bondlengths/bondlengths[-1]-1)
            assert np.all(np.diff(bondlengths) > 0)
            assert np.max(bondlengths/bondlengths[0]-1) < 0.01


if __name__ == '__main__':
    unittest.main()
