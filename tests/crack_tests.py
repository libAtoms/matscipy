import unittest

import numpy as np
import ase.io
import ase.units as units
from ase.constraints import FixAtoms
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from matscipy.fracture_mechanics.crack import (ConstantStrainRate)

try:
    import atomistica
    from atomistica import TersoffScr, Tersoff_PRB_39_5566_Si_C__Scr
    have_atomistica = True
except ImportError:
    have_atomistica = False

import sys

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
            if rigid_constraints == True:
                answer = 0
                temp_answer = 238.2066417638124
            else:
                answer = 0.013228150080099255
                temp_answer = 236.76904696481486

            newpos = atoms.get_positions()
            current_height = newpos[:, 1].max() - newpos[:, 1].min()
            diff_height = (current_height - orig_height)
            self.assertAlmostEqual(diff_height, answer)

            temperature = (atoms.get_kinetic_energy()/(1.5*units.kB*len(atoms)))
            self.assertAlmostEqual(temperature, temp_answer)

if __name__ == '__main__':
    unittest.main()
