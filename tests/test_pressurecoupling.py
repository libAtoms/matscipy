#! /usr/bin/env python

# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2020) Alexander Held,
#                  Thomas Reichenbach
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
# ======================================================================

from __future__ import (
    division,
    absolute_import,
    print_function,
    unicode_literals
)
import unittest
import matscipytest
from matscipy import pressurecoupling as pc
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.units import GPa, kB, fs, m, s
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import numpy as np
from io import StringIO


class TestSlidingP(matscipytest.MatSciPyTestCase):

    def test_usage(self):
        atoms = fcc111('Al', size=(4, 4, 9), orthogonal=True)
        atoms.set_pbc(True)
        atoms.center(axis=2, vacuum=10.0)
        z = atoms.positions[:, 2]
        top_mask = z > z[115] - 0.1
        bottom_mask = z < z[19] + 0.1
        calc = EMT()
        atoms.calc = calc
        damping = pc.AutoDamping(C11=500 * GPa, p_c=0.2)
        Pdir = 2
        vdir = 0
        P = 5 * GPa
        v = 100.0 * m / s
        dt = 1.0 * fs
        T = 400.0
        t_langevin = 75 * fs
        gamma_langevin = 1. / t_langevin
        slider = pc.SlideWithNormalPressureCuboidCell(
            top_mask,
            bottom_mask,
            Pdir,
            P,
            vdir,
            v,
            damping
        )
        atoms.set_constraint(slider)
        MaxwellBoltzmannDistribution(atoms, 2 * kB * T)
        atoms.arrays['momenta'][top_mask, :] = 0
        atoms.arrays['momenta'][bottom_mask, :] = 0
        handle = StringIO()
        beginning = handle.tell()
        temps = np.zeros((len(atoms), 3))
        temps[slider.middle_mask, slider.Tdir] = kB * T
        gammas = np.zeros((len(atoms), 3))
        gammas[slider.middle_mask, slider.Tdir] = gamma_langevin
        integrator = Langevin(atoms, dt, temps, gammas, fixcm=False)
        logger = pc.SlideLogger(handle, atoms, slider, integrator)
        logger.write_header()
        logger()
        images = []
        integrator.attach(logger)
        integrator.attach(lambda: images.append(atoms.copy()))
        integrator.run(50)
        handle.seek(beginning)
        pc.SlideLog(handle)
        handle.close()

if __name__ == '__main__':
    unittest.main()
