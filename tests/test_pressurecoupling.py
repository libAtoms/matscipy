#
# Copyright 2021 Lars Pastewka (U. Freiburg)
#           2020, 2022 Thomas Reichenbach (Fraunhofer IWM)
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
import matscipytest
from matscipy import pressurecoupling as pc
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.units import GPa, kB, fs, m, s
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import numpy as np
from io import StringIO
from ase.data import atomic_masses


class TestPressureCoupling(matscipytest.MatSciPyTestCase):

    def test_damping(self):
        """
        Test damping classes of pressure coupling module
        """
        atoms = fcc111('Al', size=(1, 2, 2), orthogonal=True)
        atoms.set_pbc(True)
        atoms.center(axis=2, vacuum=10.0)
        top_mask = np.array([False, False, True, True])
        bottom_mask = np.array([True, True, False, False])

        damping = pc.AutoDamping(C11=100 * GPa, p_c=0.2)
        slider = pc.SlideWithNormalPressureCuboidCell(
            top_mask, bottom_mask, 2, 0, 0, 1 * m / s, damping)
        M, gamma = damping.get_M_gamma(slider, atoms)
        A = atoms.get_cell()[0][0] * atoms.get_cell()[1][1]
        h = atoms[2].position[2] - atoms[0].position[2]
        lx = atoms.get_cell()[0][0]
        self.assertAlmostEqual(M,
                               1/(4*np.pi**2) * np.sqrt(1/0.2**2 - 1) * 100 *
                               GPa * A / h * lx**2 / 0.01**2 * (1000 * fs)**2,
                               places=5)
        self.assertAlmostEqual(gamma, 2 * np.sqrt(M * 100 * GPa * A / h),
                               places=5)

        damping = pc.FixedDamping(1, 1)
        slider = pc.SlideWithNormalPressureCuboidCell(
            top_mask, bottom_mask, 2, 0, 0, 1 * m / s, damping)
        M, gamma = damping.get_M_gamma(slider, atoms)
        self.assertAlmostEqual(M, 2 * atomic_masses[13], places=5)
        self.assertAlmostEqual(gamma, 1, places=5)

        damping = pc.FixedMassCriticalDamping(C11=100 * GPa, M_factor=2)
        slider = pc.SlideWithNormalPressureCuboidCell(
            top_mask, bottom_mask, 2, 0, 0, 1 * m / s, damping)
        M, gamma = damping.get_M_gamma(slider, atoms)
        self.assertAlmostEqual(M, 2 * 2 * atomic_masses[13], places=5)
        self.assertAlmostEqual(gamma, 2 * np.sqrt(M * 100 * GPa * A / h),
                               places=5)

    def test_slider(self):
        """
        Test SlideWithNormalPressureCuboidCell class
        """
        atoms = fcc111('Al', size=(1, 2, 2), orthogonal=True)
        atoms.set_pbc(True)
        atoms.center(axis=2, vacuum=10.0)
        top_mask = np.array([False, False, True, True])
        bottom_mask = np.array([True, True, False, False])
        calc = EMT()
        atoms.calc = calc
        damping = pc.AutoDamping(C11=100 * GPa, p_c=0.2)

        slider = pc.SlideWithNormalPressureCuboidCell(
            top_mask, bottom_mask, 2, 1 * GPa, 0, 1 * m / s, damping)
        self.assertEqual(slider.Tdir, 1)
        self.assertEqual(np.sum(slider.middle_mask), 0)
        self.assertAlmostEqual(slider.get_A(atoms),
                               atoms.get_cell()[0][0] * atoms.get_cell()[1][1],
                               places=5)

        forces = atoms.get_forces()
        Fz = forces[top_mask, 2].sum()
        atoms.set_velocities([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        vz = atoms.get_velocities()[top_mask, 2].sum() / np.sum(top_mask)
        M, gamma = damping.get_M_gamma(slider, atoms)
        scale = atoms.get_masses()[top_mask].sum() / M
        slider.adjust_forces(atoms, forces)
        self.assertEqual(np.sum(np.abs(forces[0])), 0)
        self.assertEqual(np.sum(np.abs(forces[1])), 0)
        self.assertEqual(np.sum(np.abs(forces[2][:2])), 0)
        self.assertEqual(np.sum(np.abs(forces[3][:2])), 0)
        self.assertEqual(forces[2][2], forces[3][2])
        self.assertAlmostEqual(forces[2][2],
                               (Fz - GPa * slider.get_A(atoms) - gamma * vz)
                               * scale / np.sum(top_mask),
                               places=5)

        momenta = atoms.get_momenta()
        mom_z = momenta[top_mask, 2].sum() / np.sum(top_mask)
        slider.adjust_momenta(atoms, momenta)
        self.assertEqual(np.sum(np.abs(momenta[0])), 0)
        self.assertEqual(np.sum(np.abs(momenta[1])), 0)
        self.assertEqual(momenta[2][1], 0)
        self.assertEqual(momenta[3][1], 0)
        self.assertAlmostEqual(momenta[2][0], m/s * atomic_masses[13],
                               places=7)
        self.assertAlmostEqual(momenta[3][0], m/s * atomic_masses[13],
                               places=7)
        self.assertAlmostEqual(momenta[2][2], mom_z, places=5)
        self.assertAlmostEqual(momenta[3][2], mom_z, places=5)

    def test_logger(self):
        """
        Test SlideLogger and SlideLog classes
        """
        atoms = fcc111('Al', size=(4, 4, 9), orthogonal=True)
        atoms.set_pbc(True)
        atoms.center(axis=2, vacuum=10.0)
        calc = EMT()
        atoms.calc = calc
        damping = pc.AutoDamping(C11=1 * GPa, p_c=0.2)
        z = atoms.positions[:, 2]
        top_mask = z > z[115] - 0.1
        bottom_mask = z < z[19] + 0.1
        Pdir = 2
        vdir = 0
        P = 1 * GPa
        v = 1.0 * m / s
        dt = 0.5 * fs
        T = 300.0
        t_langevin = 75 * fs
        gamma_langevin = 1. / t_langevin
        slider = pc.SlideWithNormalPressureCuboidCell(
            top_mask,
            bottom_mask,
            Pdir,
            P,
            vdir,
            v,
            damping)
        atoms.set_constraint(slider)
        temps = np.zeros((len(atoms), 3))
        temps[slider.middle_mask, slider.Tdir] = T
        gammas = np.zeros((len(atoms), 3))
        gammas[slider.middle_mask, slider.Tdir] = gamma_langevin
        integrator = Langevin(atoms, dt, temperature_K=temps,
                              friction=gammas, fixcm=False)

        forces_ini = atoms.get_forces(apply_constraint=False)
        forces_ini_const = atoms.get_forces(apply_constraint=True)
        h_ini = atoms[115].position[2] - atoms[19].position[2]
        A = atoms.get_cell()[0][0] * atoms.get_cell()[1][1]

        handle = StringIO()
        beginning = handle.tell()
        logger = pc.SlideLogger(handle, atoms, slider, integrator)
        logger.write_header()
        integrator.attach(logger)
        integrator.run(1)
        integrator.logfile.close()

        handle.seek(beginning)
        log = pc.SlideLog(handle)
        self.assertArrayAlmostEqual(log.step, np.array([0, 1]))
        self.assertArrayAlmostEqual(log.time, np.array([0, 0.5]))
        self.assertEqual(log.T_thermostat[0], 0)
        Tref = (atomic_masses[13] *
                (atoms.get_velocities()[slider.middle_mask, 1]**2).sum() /
                (np.sum(slider.middle_mask) * kB))
        self.assertAlmostEqual(log.T_thermostat[1], Tref, places=5)
        self.assertAlmostEqual(log.P_top[0],
                               forces_ini[slider.top_mask, 2].sum()
                               / A / GPa,
                               places=5)
        self.assertAlmostEqual(log.P_top[1],
                               atoms.get_forces(
                                   apply_constraint=False)[slider.top_mask,
                                                           2].sum()
                               / A / GPa,
                               places=5)

        self.assertAlmostEqual(log.P_bottom[0],
                               forces_ini[slider.bottom_mask, 2].sum()
                               / A / GPa,
                               places=5)

        self.assertAlmostEqual(log.P_bottom[1],
                               atoms.get_forces(
                                   apply_constraint=False)[slider.bottom_mask,
                                                           2].sum()
                               / A / GPa,
                               places=5)
        self.assertArrayAlmostEqual(log.h, [h_ini,
                                            atoms[115].position[2] -
                                            atoms[19].position[2]])
        self.assertArrayAlmostEqual(log.v,
                                    [0, atoms.get_velocities()[115][2] * fs])
        self.assertArrayAlmostEqual(log.a,
                                    [forces_ini_const[115][2]
                                     / atomic_masses[13] * fs**2,
                                     atoms.get_forces(
                                         apply_constraint=True)[115][2]
                                     / atomic_masses[13] * fs**2])
        self.assertAlmostEqual(log.tau_top[0],
                               forces_ini[top_mask, 0].sum() / A / GPa,
                               places=13)
        self.assertAlmostEqual(log.tau_top[1],
                               atoms.get_forces(
                                   apply_constraint=False)[top_mask, 0].sum()
                               / A / GPa,
                               places=13)
        self.assertAlmostEqual(log.tau_bottom[0],
                               forces_ini[bottom_mask, 0].sum() / A / GPa,
                               places=13)
        self.assertAlmostEqual(log.tau_bottom[1],
                               atoms.get_forces(apply_constraint=False)
                               [bottom_mask, 0].sum() / A / GPa,
                               places=13)
        handle.close()

    def test_usage(self):
        """
        Test if sliding simulation with pressure coupling module runs
        without errors
        """
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
        MaxwellBoltzmannDistribution(atoms, temperature_K=2 * T)
        atoms.arrays['momenta'][top_mask, :] = 0
        atoms.arrays['momenta'][bottom_mask, :] = 0
        handle = StringIO()
        beginning = handle.tell()
        temps = np.zeros((len(atoms), 3))
        temps[slider.middle_mask, slider.Tdir] = T
        gammas = np.zeros((len(atoms), 3))
        gammas[slider.middle_mask, slider.Tdir] = gamma_langevin
        integrator = Langevin(atoms, dt, temperature_K=temps,
                              friction=gammas, fixcm=False)
        logger = pc.SlideLogger(handle, atoms, slider, integrator)
        logger.write_header()
        integrator.attach(logger)
        integrator.run(10)
        integrator.logfile.close()
        handle.seek(beginning)
        pc.SlideLog(handle)
        handle.close()

if __name__ == '__main__':
    unittest.main()
