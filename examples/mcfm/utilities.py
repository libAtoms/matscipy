#
# Copyright 2021 Lars Pastewka (U. Freiburg)
#           2018 Jacek Golebiowski (Imperial College London)
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
"""Copy of ASE Morse calculator that supports per-atom potential energy.
Also extra utilities"""

import numpy as np
from math import exp, sqrt

import ase.io
import ase.units as units
from ase.calculators.calculator import Calculator
from ase.constraints import FixConstraintSingle


class MorsePotentialPerAtom(Calculator):
    """Morse potential.

    Default values chosen to be similar as Lennard-Jones.
    """

    implemented_properties = ['energy', 'forces', 'potential_energies']
    default_parameters = {'epsilon': 1.0,
                          'rho0': 6.0,
                          'r0': 1.0}
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges', 'magmoms']):

        Calculator.calculate(self, atoms, properties, system_changes)
        epsilon = self.parameters.epsilon
        rho0 = self.parameters.rho0
        r0 = self.parameters.r0

        positions = self.atoms.get_positions()
        energy = 0.0
        energies = np.zeros(len(self.atoms))
        forces = np.zeros((len(self.atoms), 3))

        preF = 2 * epsilon * rho0 / r0
        for i1, p1 in enumerate(positions):
            for i2, p2 in enumerate(positions[:i1]):
                diff = p2 - p1
                r = sqrt(np.dot(diff, diff))
                expf = exp(rho0 * (1.0 - r / r0))
                energy += epsilon * expf * (expf - 2)
                energies[i1] += epsilon * expf * (expf - 2)
                energies[i2] += epsilon * expf * (expf - 2)
                F = preF * expf * (expf - 1) * diff / r
                forces[i1] -= F
                forces[i2] += F
        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['potential_energies'] = energies


def trajectory_writer(atoms, outputFile, writeResults=False):
    ase.io.write(outputFile, atoms, format="extxyz", write_results=writeResults)
    outputFile.flush()


def mcfm_thermo_printstatus(frequency, dynamics, atoms, potential, logfile=None):
    """Thermodynamicsal data printout"""

    if dynamics.nsteps == frequency:
        header = "%5.5s %16.15s %16.15s %16.15s %16.15s %16.15s :thermo\n" % (
            "nsteps",
            "temperature",
            "potential_E",
            "kinetic_E",
            "total_E",
            "QM_atoms_no")

        if logfile is not None:
            logfile.write(header)
            logfile.flush()
        print(header)

    if "energy" in potential.classical_calculator.results:
        potential_energy = potential.classical_calculator.results["energy"]
    elif "potential_energy" in potential.classical_calculator.results:
        potential_energy = potential.classical_calculator.results["potential_energy"]
    else:
        potential_energy = potential.get_potential_energy(atoms)

    kinetic_energy = atoms.get_kinetic_energy()
    total_energy = potential_energy + kinetic_energy
    temp = kinetic_energy / (1.5 * units.kB * len(atoms))
    full_qm_atoms_no = len([item for sublist in potential.cluster_list for item in sublist])

    log = "%5.1f %16.5e %16.5e %16.5e %16.5e %5.0f :thermo" % (
        dynamics.nsteps,
        temp,
        potential_energy,
        kinetic_energy,
        total_energy,
        full_qm_atoms_no,
    )
    if logfile is not None:
        logfile.write(log + "\n")
        logfile.flush()

    print(log)


def mcfm_error_logger(frequency, dynamics, atoms, mcfm_pot, logfile=None):
    """This is the logger for mcfm potential to be used with a DS simulation"""
    # Evaluate errors
    mcfm_pot.evaluate_errors(atoms, heavy_only=True)

    # Print output
    if dynamics.nsteps == frequency:
        header = "%5.5s %16.10s %16.10s %16.10s %16.10s %16.10s %16.10s %16.10s :errLog\n" % (
            "nsteps",
            "MAX_abs_fe",
            "RMS_abs_fe",
            "MAX_rel_fe",
            "RMS_rel_fe",
            "cumulFError",
            "cumEcontribution",
            "n_QM_atoms")

        if logfile is not None:
            logfile.write(header)
            logfile.flush()

    log = "%5.1f %16.5e %16.5e %16.5e %16.5e %16.5e %16.5e %16.0f :errLog" % (
        dynamics.nsteps,
        mcfm_pot.errors["max absolute error"],
        mcfm_pot.errors["rms absolute error"],
        mcfm_pot.errors["max relative error"],
        mcfm_pot.errors["rms relative error"],
        np.mean(mcfm_pot.errors["Cumulative fError vector length"]),
        mcfm_pot.errors["Cumulative energy change"],
        mcfm_pot.errors["no of QM atoms"]
    )
    if logfile is not None:
        logfile.write(log + "\n")
        logfile.flush()

    if (len(mcfm_pot.cluster_list) > 0):
        print("\tAbs errors:= RMS: %0.5f, MAX: %0.5f\t Rel errors:= RMS: %0.5f, MAX: %0.5f :errLog" %
              (mcfm_pot.errors["rms absolute error"],
               mcfm_pot.errors["max absolute error"],
               mcfm_pot.errors["rms relative error"],
               mcfm_pot.errors["max relative error"]))


class LinearConstraint(FixConstraintSingle):
    """Constrain an atom to move along a given direction only."""

    def __init__(self, a, direction, velocity):
        self.a = a
        self.dir = direction / np.sqrt(np.dot(direction, direction))
        self.velo = velocity
        self.removed_dof = 0

    def adjust_positions(self, atoms, newpositions):
        step = self.dir * self.velo
        newpositions[self.a] = atoms.positions[self.a] + step

    def adjust_forces(self, atoms, forces):
        forces[self.a] = np.zeros(3)

    def __repr__(self):
        return 'LinearConstraint(%d, %s, %.2e)' % (self.a, self.dir.tolist(), self.velo)

    def todict(self):
        return {'name': 'LinearConstraint',
                'kwargs': {'a': self.a, 'direction': self.dir, "velocity": self.velo}}
