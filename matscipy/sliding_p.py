# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
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

"""

This python module provides classes to be used together with ASE in order to
perform pressure relaxation and/or sliding simulations under pressure.

A usage example is found in the example directory.

Some parts are based on L. Pastewka, S. Moser, and M. Moseler, Tribol. Lett. 39, 49 (2010)
as indicated again below.

Author: Alexander Held
"""

from __future__ import (
    division,
    absolute_import,
    print_function,
    unicode_literals
)
import numpy as np
from ase.units import kB, fs, GPa
import logging

logger = logging.getLogger(__name__)


class AutoDamping(object):
    """Automatic damping

    Following L. Pastewka, S. Moser, and M. Moseler,
    Tribol. Lett. 39, 49 (2010).
    """

    def __init__(self, C11, p_c=0.01):
        """Constructor.

        Arguments as described in L. Pastewka, S. Moser, and M. Moseler,
        Tribol. Lett. 39, 49 (2010).
        """
        self.C11 = float(C11)
        self.p_c = float(p_c)

    def get_M_gamma(self, slider, atoms):
        A = slider.get_A(atoms)
        l = atoms.cell[slider.vdir, slider.vdir]
        t_c = l / slider.v
        omega_c = 2. * np.pi / t_c
        h1 = atoms.positions[slider.top_mask, slider.Pdir].min()
        h2 = atoms.positions[slider.bottom_mask, slider.Pdir].max()
        h = h1 - h2
        k = self.C11 * A / h
        M = k * omega_c ** -2 * np.sqrt(self.p_c ** -2 - 1.)
        # gamma = np.sqrt(2. * M * k)  # <- expression from paper, but this is wrong
        gamma = 2. * np.sqrt(M * k)
        return M, gamma


class FixedDamping(object):
    """Damping with fixed damping constant and fixed mass."""

    def __init__(self, gamma, M_factor=1.0):
        self.gamma = float(gamma)
        self.M_factor = float(M_factor)

    def get_M_gamma(self, slider, atoms):
        M_top = atoms.get_masses()[slider.top_mask].sum()
        return M_top * self.M_factor, self.gamma


class FixedMassCriticalDamping(object):
    """Damping with fixed mass and critical damping constant.

    Useful for fast pressure equilibration with small lid mass.
    """

    def __init__(self, C11, M_factor=1.0):
        self.C11 = float(C11)
        self.M_factor = float(M_factor)

    def get_M_gamma(self, slider, atoms):
        M_top = atoms.get_masses()[slider.top_mask].sum()
        M = M_top * self.M_factor
        A = slider.get_A(atoms)
        h1 = atoms.positions[slider.top_mask, slider.Pdir].min()
        h2 = atoms.positions[slider.bottom_mask, slider.Pdir].max()
        h = h1 - h2
        k = self.C11 * A / h
        gamma = 2. * np.sqrt(M * k)
        return M, gamma


class SlideWithNormalPressureCuboidCell(object):
    """ASE constraint used for sliding with pressure coupling

    Only works with diagonal cuboid cells so far.
    Sliding only works along cell vectors so far.
    Following L. Pastewka, S. Moser, and M. Moseler,
    Tribol. Lett. 39, 49 (2010).
    """

    def __init__(self, top_mask, bottom_mask, Pdir, P, vdir, v, damping):
        """Constructor.

        top_mask -- boolean numpy array a with a[i] == True for i the index
                    of a constraint top atom (the atoms which slide with
                    constant speed)
        bottom_mask -- same as top_mask but for completely fixed bottom
                       atoms
        Pdir -- index of cell axis (0, 1, 2) along which normal pressure
                is applied
        P -- normal pressure in ASE units (e.g. 10.0 * ase.units.GPa)
        vdir -- index of cell axis (0, 1, 2) along which to slide
        v -- constant sliding speed in ASE units
             (e.g. 100.0 * ase.units.m / ase.units.s)
        damping -- a damping object (e.g. AutoDamping instance)
        """
        self.top_mask = top_mask
        self.Ntop = top_mask.sum()
        self.bottom_mask = bottom_mask
        self.Pdir = int(Pdir)
        self.P = float(P)
        self.vdir = int(vdir)
        self.v = float(v)
        self.damping = damping

    @property
    def Tdir(self):
        all_dirs = {0, 1, 2}
        all_dirs.remove(self.Pdir)
        all_dirs.remove(self.vdir)
        return all_dirs.pop()

    @property
    def middle_mask(self):
        return np.logical_not(np.logical_or(self.top_mask, self.bottom_mask))

    def adjust_positions(self, atoms, positions):
        pass

    def get_A(self, atoms):
        A = 1.0
        for c in (0, 1, 2):
            if c != self.Pdir:
                A *= atoms.cell[c, c]
        return A

    def adjust_forces(self, atoms, forces):
        A = self.get_A(atoms)
        M, gamma = self.damping.get_M_gamma(self, atoms)
        Ftop = forces[self.top_mask, self.Pdir].sum()
        vtop = atoms.get_velocities()[self.top_mask, self.Pdir].sum() / self.Ntop
        F = Ftop - self.P * A - gamma * vtop
        a = F / M
        forces[self.bottom_mask, :] = .0
        forces[self.top_mask, :] = .0
        forces[self.top_mask, self.Pdir] = atoms.get_masses()[self.top_mask] * a

    def adjust_momenta(self, atoms, momenta):
        top_masses = atoms.get_masses()[self.top_mask]
        vtop = (momenta[self.top_mask, self.Pdir] / top_masses).sum() / self.Ntop
        momenta[self.bottom_mask, :] = 0.0
        momenta[self.top_mask, :] = 0.0
        momenta[self.top_mask, self.vdir] = self.v * top_masses
        momenta[self.top_mask, self.Pdir] = vtop * top_masses

    def adjust_potential_energy(self, atoms):
        return 0.0


class SlideLogger(object):
    """Logger to be attached to an ASE integrator."""

    def __init__(self, handle, atoms, slider, integrator, step_offset=0):
        """Constructor.

        handle -- filehandle e.g. pointing to a file opened in w or a mode
        atoms -- the ASE atoms object
        slider -- slider object
                  (e.g. instance of SlideWithNormalPressureCuboidCell)
        integrator -- ASE integrator object,
                      e.g. ase.md.langevin.Langevin instance
        step_offset -- for restart jobs: last step already written to log file

        For new files (not restart jobs), the write_header method should
        be called once in order to write the header to the file. Also note
        that ASE does *NOT* write the time step 0 automatically.
        You can do so by calling the SlideLogger instance once
        before starting the integration, like so:

        ...
        log_handle = open(logfn, 'w', 1)  # line buffered
        logger = SlideLogger(log_handle, ...)
        logger.write_header()
        logger()
        integrator.attach(logger)
        integrator.run(steps_integrate)
        log_handle.close()  # or some try ... finally clause
        ...

        For a restart job, you can use the following recipe:

        ...
        with open(logfn, 'r') as log_handle:
            step_offset = SlideLog(log_handle).step[-1]
        log_handle = open(logfn, 'a', 1)  # line buffered append
        logger = SlideLogger(log_handle, ..., step_offset=step_offset)
        integrator.attach(logger)
        integrator.run(steps_integrate)
        log_handle.close()  # or some try ... finally clause
        ...
        """
        self.handle = handle
        self.atoms = atoms
        self.slider = slider
        self.integrator = integrator
        self.step_offset = step_offset

    def write_header(self):
        self.handle.write('# step | time / fs | T_thermostat / K | P_top / GPa | P_bottom / GPa | h / Ang | v / Ang * fs | a / Ang * fs ** 2 | tau_top / GPa | tau_bottom / GPa\n')

    def __call__(self):
        slider = self.slider
        atoms = self.atoms
        integrator = self.integrator
        p = atoms.get_momenta()[slider.middle_mask, slider.Tdir]
        v = atoms.get_velocities()[slider.middle_mask, slider.Tdir]
        dof = len(p)
        Ekin = 0.5 * (p * v).sum()
        T = 2. * Ekin / (dof * kB)
        step = integrator.nsteps + self.step_offset
        t = step * integrator.dt / fs
        A = atoms.cell[slider.vdir, slider.vdir] * atoms.cell[slider.Tdir, slider.Tdir]
        F = atoms.get_forces(apply_constraint=False)
        F_top = F[slider.top_mask, slider.Pdir].sum()
        F_bottom = F[slider.bottom_mask, slider.Pdir].sum()
        P_top = F_top / A / GPa
        P_bottom = F_bottom / A / GPa
        h1 = atoms.positions[slider.top_mask, slider.Pdir].min()
        h2 = atoms.positions[slider.bottom_mask, slider.Pdir].max()
        h = h1 - h2
        v = atoms.get_velocities()[slider.top_mask, slider.Pdir][0] * fs
        a = atoms.get_forces(md=True)[slider.top_mask, slider.Pdir][0] / atoms.get_masses()[slider.top_mask][0] * fs ** 2
        F_v_top = F[slider.top_mask, slider.vdir].sum()
        tau_top = F_v_top / A / GPa
        F_v_bottom = F[slider.bottom_mask, slider.vdir].sum()
        tau_bottom = F_v_bottom / A / GPa
        self.handle.write('%d   %r   %r   %r   %r   %r   %r   %r   %r   %r\n' % (step, float(t), float(T), float(P_top), float(P_bottom), float(h), float(v), float(a), float(tau_top), float(tau_bottom)))


class SlideLog(object):
    """Reader for logs written with SlideLogger instance.

    The data of the log files is found as attributes
    (numpy arrays with step as axis):
    step -- step indices 0, 1, 2, ...
    time -- simulation time in fs at step
    T_thermostat -- instantaneous temperature in K from thermostatet region
                    only from degrees of freedom along thermalized
                    direction
    P_top -- normal pressure on lid in GPa
    P_bottom -- normal pressure on base in GPa
    h -- separation of lid and base in Ang
    v -- normal speed of lid in Ang / fs
    a -- normal acceleration of lid in Ang / fs ** 2
    tau_top -- shear stress on lid in GPa
    tau_bottom -- shear stress on base in GPa
    rows -- all data in a 2d array with axis 0 step and axis 1
            the values in the order as above
    """

    def __init__(self, handle):
        """Constructor.

        handle -- handle or filename pointing to log file
        """
        self.rows = np.loadtxt(handle)
        self.step, self.time, self.T_thermostat, self.P_top, self.P_bottom, self.h, self.v, self.a, self.tau_top, self.tau_bottom = self.rows.T
        self.step = self.step.astype(int)
