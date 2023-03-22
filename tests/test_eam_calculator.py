#! /usr/bin/env python
#
# Copyright 2014-2015, 2017, 2019-2021 Lars Pastewka (U. Freiburg)
#           2019-2020 Wolfram G. Nöhring (U. Freiburg)
#           2020 Johannes Hoermann (U. Freiburg)
#           2014 James Kermode (Warwick U.)
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

import gzip
import random
import unittest

import numpy as np
from numpy.linalg import norm


import ase.io as io
from ase.calculators.test import numeric_force
from ase.constraints import StrainFilter, UnitCellFilter
from ase.lattice.compounds import B1, B2, L1_0, L1_2
from ase.lattice.cubic import FaceCenteredCubic
from ase.lattice.hexagonal import HexagonalClosedPacked
from ase.optimize import FIRE
from ase.units import GPa

import matscipytest
from matscipy.elasticity import fit_elastic_constants, Voigt_6x6_to_cubic
from matscipy.neighbours import neighbour_list
from matscipy.numerical import numerical_stress
from matscipy.calculators.eam import EAM

###

class TestEAMCalculator(matscipytest.MatSciPyTestCase):

    disp = 1e-6
    tol = 2e-6

    def test_forces(self):
        for calc in [EAM('Au-Grochola-JCP05.eam.alloy')]:
            a = io.read('Au_923.xyz')
            a.center(vacuum=10.0)
            a.calc = calc
            f = a.get_forces()
            for i in range(9):
                atindex = i*100
                fn = [numeric_force(a, atindex, 0, self.disp),
                      numeric_force(a, atindex, 1, self.disp),
                      numeric_force(a, atindex, 2, self.disp)]
                self.assertArrayAlmostEqual(f[atindex], fn, tol=self.tol)

    def test_stress(self):
        a = FaceCenteredCubic('Au', size=[2,2,2])
        calc = EAM('Au-Grochola-JCP05.eam.alloy')
        a.calc = calc
        self.assertArrayAlmostEqual(a.get_stress(), numerical_stress(a), tol=self.tol)

        sx, sy, sz = a.cell.diagonal()
        a.set_cell([sx, 0.9*sy, 1.2*sz], scale_atoms=True)
        self.assertArrayAlmostEqual(a.get_stress(), numerical_stress(a), tol=self.tol)

        a.set_cell([[sx, 0.1*sx, 0], [0, 0.9*sy, 0], [0, -0.1*sy, 1.2*sz]], scale_atoms=True)
        self.assertArrayAlmostEqual(a.get_stress(), numerical_stress(a), tol=self.tol)

    def test_Grochola(self):
        a = FaceCenteredCubic('Au', size=[2,2,2])
        calc = EAM('Au-Grochola-JCP05.eam.alloy')
        a.calc = calc
        FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        a0 = a.cell.diagonal().mean()/2
        self.assertTrue(abs(a0-4.0701)<2e-5)
        self.assertTrue(abs(a.get_potential_energy()/len(a)+3.924)<0.0003)
        C, C_err = fit_elastic_constants(a, symmetry='cubic', verbose=False)
        C11, C12, C44 = Voigt_6x6_to_cubic(C)
        self.assertTrue(abs((C11-C12)/GPa-32.07)<0.7)
        self.assertTrue(abs(C44/GPa-45.94)<0.5)

    def test_direct_evaluation(self):
        a = FaceCenteredCubic('Au', size=[2,2,2])
        a.rattle(0.1)
        calc = EAM('Au-Grochola-JCP05.eam.alloy')
        a.calc = calc
        f = a.get_forces()

        calc2 = EAM('Au-Grochola-JCP05.eam.alloy')
        i_n, j_n, dr_nc, abs_dr_n = neighbour_list('ijDd', a, cutoff=calc2.cutoff)
        epot, virial, f2 = calc2.energy_virial_and_forces(a.numbers, i_n, j_n, dr_nc, abs_dr_n)
        self.assertArrayAlmostEqual(f, f2)

        a = FaceCenteredCubic('Cu', size=[2,2,2])
        calc = EAM('CuAg.eam.alloy')
        a.calc = calc
        FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        e_Cu = a.get_potential_energy()/len(a)

        a = FaceCenteredCubic('Ag', size=[2,2,2])
        a.calc = calc
        FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        e_Ag = a.get_potential_energy()/len(a)
        self.assertTrue(abs(e_Ag+2.85)<1e-6)

        a = L1_2(['Ag', 'Cu'], size=[2,2,2], latticeconstant=4.0)
        a.calc = calc
        FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        e = a.get_potential_energy()
        syms = np.array(a.get_chemical_symbols())
        self.assertTrue(abs((e-(syms=='Cu').sum()*e_Cu-
                               (syms=='Ag').sum()*e_Ag)/len(a)-0.096)<0.0005)

        a = B1(['Ag', 'Cu'], size=[2,2,2], latticeconstant=4.0)
        a.calc = calc
        FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        e = a.get_potential_energy()
        syms = np.array(a.get_chemical_symbols())
        self.assertTrue(abs((e-(syms=='Cu').sum()*e_Cu-
                               (syms=='Ag').sum()*e_Ag)/len(a)-0.516)<0.0005)

        a = B2(['Ag', 'Cu'], size=[2,2,2], latticeconstant=4.0)
        a.calc = calc
        FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        e = a.get_potential_energy()
        syms = np.array(a.get_chemical_symbols())
        self.assertTrue(abs((e-(syms=='Cu').sum()*e_Cu-
                               (syms=='Ag').sum()*e_Ag)/len(a)-0.177)<0.0003)

        a = L1_2(['Cu', 'Ag'], size=[2,2,2], latticeconstant=4.0)
        a.calc = calc
        FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        e = a.get_potential_energy()
        syms = np.array(a.get_chemical_symbols())
        self.assertTrue(abs((e-(syms=='Cu').sum()*e_Cu-
                                (syms=='Ag').sum()*e_Ag)/len(a)-0.083)<0.0005)

    def test_CuZr(self):
        # This is a test for the potential published in:
        # Mendelev, Sordelet, Kramer, J. Appl. Phys. 102, 043501 (2007)
        a = FaceCenteredCubic('Cu', size=[2,2,2])
        calc = EAM('CuZr_mm.eam.fs', kind='eam/fs')
        a.calc = calc
        FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        a_Cu = a.cell.diagonal().mean()/2
        #print('a_Cu (3.639) = ', a_Cu)
        self.assertAlmostEqual(a_Cu, 3.639, 3)

        a = HexagonalClosedPacked('Zr', size=[2,2,2])
        a.calc = calc
        FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=1e-3)
        a, b, c = a.cell/2
        print('a_Zr (3.220) = ', norm(a), norm(b))
        print('c_Zr (5.215) = ', norm(c))
        self.assertAlmostEqual(norm(a), 3.220, 3)
        self.assertAlmostEqual(norm(b), 3.220, 3)
        self.assertAlmostEqual(norm(c), 5.215, 3)

        # CuZr3
        a = L1_2(['Cu', 'Zr'], size=[2,2,2], latticeconstant=4.0)
        a.calc = calc
        FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        self.assertAlmostEqual(a.cell.diagonal().mean()/2, 4.324, 3)

        # Cu3Zr
        a = L1_2(['Zr', 'Cu'], size=[2,2,2], latticeconstant=4.0)
        a.calc = calc
        FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        self.assertAlmostEqual(a.cell.diagonal().mean()/2, 3.936, 3)

        # CuZr
        a = B2(['Zr', 'Cu'], size=[2,2,2], latticeconstant=3.3)
        a.calc = calc
        FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        self.assertAlmostEqual(a.cell.diagonal().mean()/2, 3.237, 3)

    def test_forces_CuZr_glass(self):
        """Calculate interatomic forces in CuZr glass

        Reference: tabulated forces from a calculation
        with Lammmps (git version patch_29Mar2019-2-g585403d65)

        The forces can be re-calculated using the following
        Lammps commands:
            units metal
            atom_style atomic
            boundary p p p
            read_data CuZr_glass_460_atoms.lammps.data.gz
            pair_style eam/alloy
            pair_coeff * * ZrCu.onecolumn.eam.alloy Zr Cu
            # The initial configuration is in equilibrium
            # and the remaining forces are small
            # Swap atom types to bring system out of
            # equilibrium and create nonzero forces
            group originally_Zr type 1
            group originally_Cu type 2
            set group originally_Zr type 2
            set group originally_Cu type 1
            run 0
            write_dump all custom &
                CuZr_glass_460_atoms_forces.lammps.dump.gz &
                id type x y z fx fy fz &
                modify sort id format float "%.14g"
        """
        format = "lammps-dump" if "lammps-dump" in io.formats.all_formats.keys() else "lammps-dump-text"
        atoms = io.read("CuZr_glass_460_atoms_forces.lammps.dump.gz", format=format)
        old_atomic_numbers = atoms.get_atomic_numbers()
        sel, = np.where(old_atomic_numbers == 1)
        new_atomic_numbers = np.zeros_like(old_atomic_numbers)
        new_atomic_numbers[sel] = 40 # Zr
        sel, = np.where(old_atomic_numbers == 2)
        new_atomic_numbers[sel] = 29 # Cu
        atoms.set_atomic_numbers(new_atomic_numbers)
        calculator = EAM('ZrCu.onecolumn.eam.alloy')
        atoms.calc = calculator
        atoms.pbc = [True, True, True]
        forces = atoms.get_forces()
        # Read tabulated forces and compare
        with gzip.open("CuZr_glass_460_atoms_forces.lammps.dump.gz") as file:
            for line in file:
                if line.startswith(b"ITEM: ATOMS "): # ignore header
                    break
            dump = np.loadtxt(file)
        forces_dump = dump[:, 5:8]
        self.assertArrayAlmostEqual(forces, forces_dump, tol=1e-3)

    def test_funcfl(self):
        """Test eam kind 'eam' (DYNAMO funcfl format)

            variable da   equal 0.02775
            variable amin equal 2.29888527117067752084
            variable amax equal 5.55*sqrt(2.0)
            variable i loop 201
                label loop_head
                clear
                variable lattice_parameter equal ${amin}+${da}*${i}
                units metal
                atom_style atomic
                boundary p p p
                lattice fcc ${lattice_parameter}
                region box block 0 5 0 5 0 5
                create_box 1 box
                pair_style eam
                pair_coeff * * Au_u3.eam
                create_atoms 1 box
                thermo 1
                run 0
                variable x equal pe
                variable potential_energy_fmt format x "%.14f"
                print "#a,E: ${lattice_parameter} ${potential_energy_fmt}"
                next i
                jump SELF loop_head
            # use
            # grep '^#a,E' log.lammps  | awk '{print $2,$3}' > aE.txt
            # to extract info from log file

        The reference data was calculated using Lammps
        (git commit a73f1d4f037f670cd4295ecc1a576399a31680d2).
        """
        da   = 0.02775
        amin = 2.29888527117067752084
        amax = 5.55 * np.sqrt(2.0)
        for i in range(1, 202):
            latticeconstant = amin + i * da
            atoms = FaceCenteredCubic(symbol='Au', size=[5,5,5], pbc=(1,1,1), latticeconstant=latticeconstant)
            calc = EAM('Au-Grochola-JCP05.eam.alloy')
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            print(energy)


###

if __name__ == '__main__':
    unittest.main()
