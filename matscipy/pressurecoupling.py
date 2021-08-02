#
# Copyright 2021 Lars Pastewka (U. Freiburg)
#           2020 Thomas Reichenbach (Fraunhofer IWM)
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

"""

Classes to be used with ASE in order to perform pressure relaxation and/or sliding simulations under pressure.

A usage example is found in the example directory.

Some parts are based on L. Pastewka, S. Moser, and M. Moseler, Tribol. Lett. 39, 49 (2010)
as indicated again below.

"""

import logging
import numpy as np
from ase.units import kB, fs, GPa


logger = logging.getLogger(__name__)


class AutoDamping(object):
    """Automatic damping.

    Following L. Pastewka, S. Moser, and M. Moseler,
    Tribol. Lett. 39, 49 (2010).

    Parameters
    ----------
    C11 : float
        Elastic material constant.
    p_c : float
        Empirical cut-off parameter.
    """

    def __init__(self, C11, p_c=0.01):
        self.C11 = float(C11)
        self.p_c = float(p_c)

    def get_M_gamma(self, slider, atoms):
        """Calculate mass M and dissipation constant gamma.

        Parameters
        ----------
        slider : matscipy.pressurecoupling.SlideWithNormalPressureCuboidCell
            ASE constraint used for sliding with pressure coupling.
        atoms : ase.Atoms
            Atomic configuration.

        Returns
        -------
        M: float
            Mass parameter.
        gamma : float
            Dissipation constant parameter.
        """
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
    """Damping with fixed damping constant and fixed mass.

    Parameters
    ----------
    gamma : float
        Damping constant.
    M_factor : float
        Multiplicative factor to increase actual mass of upper rigid atoms.
    """

    def __init__(self, gamma, M_factor=1.0):
        self.gamma = float(gamma)
        self.M_factor = float(M_factor)

    def get_M_gamma(self, slider, atoms):
        """Calculate mass M and damping constant gamma.

        Parameters
        ----------
        slider : matscipy.pressurecoupling.SlideWithNormalPressureCuboidCell
            ASE constraint used for sliding with pressure coupling.
        atoms : ase.Atoms
            Atomic configuration.

        Returns
        -------
        M: float
            Mass parameter.
        gamma : float
            Damping parameter.
        """
        M_top = atoms.get_masses()[slider.top_mask].sum()
        return M_top * self.M_factor, self.gamma


class FixedMassCriticalDamping(object):
    """Damping with fixed mass and critical damping constant.

    Useful for fast pressure equilibration with small lid mass.

    Parameters
    ----------
    C11 : float
        Elastic material constant.
    M_factor : float
        Multiplicative factor to increase actual mass of upper rigid atoms.
    """

    def __init__(self, C11, M_factor=1.0):
        self.C11 = float(C11)
        self.M_factor = float(M_factor)

    def get_M_gamma(self, slider, atoms):
        """Calculate mass M and damping constant gamma.

        Parameters
        ----------
        slider : matscipy.pressurecoupling.SlideWithNormalPressureCuboidCell
            ASE constraint used for sliding with pressure coupling.
        atoms : ase.Atoms
            Atomic configuration.

        Returns
        -------
        M: float
            Mass parameter.
        gamma : float
            Damping parameter.
        """
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
    """ASE constraint used for sliding with pressure coupling.

    Following L. Pastewka, S. Moser, and M. Moseler,
    Tribol. Lett. 39, 49 (2010).

    Parameters
    ----------
    top_mask : boolean numpy array
        Array a with a[i] == True for each index i of the
        constraint top atoms (the atoms which slide with constant speed).
    bottom_mask : boolean numpy array
        same as top_mask but for completely fixed bottom atoms.
    Pdir : int
        Index of cell axis (0, 1, 2) along which normal pressure is applied.
    P : int
        Normal pressure in ASE units (e.g. 10.0 * ase.units.GPa).
    vdir : int
        Index of cell axis (0, 1, 2) along which to slide.
    v : float
        Constant sliding speed in ASE units (e.g. 100.0 * ase.units.m / ase.units.s).
    damping :
        Damping object (e.g. matscipy.pressurecoupling.AutoDamping instance).
    """

    def __init__(self, top_mask, bottom_mask, Pdir, P, vdir, v, damping):
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
        """Direction used for thermostatting.

        Thermostat direction is normal to the sliding direction and the direction of the
        applied load direction.

        Returns
        -------
        int
            Direction used for thermostatting.
        """
        all_dirs = {0, 1, 2}
        all_dirs.remove(self.Pdir)
        all_dirs.remove(self.vdir)
        return all_dirs.pop()

    @property
    def middle_mask(self):
        """Mask of free atoms.

        Returns
        -------
        numpy boolean array
            Array a with a[i] == True for each index i of the atoms
            not being part of lower or upper rigid group.
        """
        return np.logical_not(np.logical_or(self.top_mask, self.bottom_mask))

    def adjust_positions(self, atoms, positions):
        """Do not adjust positions."""
        pass

    def get_A(self, atoms):
        """Calculate cell area normal to applied load.

        Returns
        -------
        float
            Cell area normal to applied load.

        Raises
        ------
        NotImplementedError
            If atoms.get_cell() is non-orthogonal,
            SlideWithNormalPressureCuboidCell only works for orthogonal cells.
        """
        if np.abs(atoms.get_cell().sum() - atoms.get_cell().trace()) > 0:
            raise NotImplementedError("Can't do non-orthogonal cell!")
        A = 1.0
        for c in (0, 1, 2):
            if c != self.Pdir:
                A *= atoms.cell[c, c]
        return A

    def adjust_forces(self, atoms, forces):
        """Adjust forces of upper and lower rigid atoms.

        Raises
        ------
        NotImplementedError
            If atoms.get_cell() is non-orthogonal,
            SlideWithNormalPressureCuboidCell only works for orthogonal cells.
        """
        if np.abs(atoms.get_cell().sum() - atoms.get_cell().trace()) > 0:
            raise NotImplementedError("Can't do non-orthogonal cell!")
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
        """Adjust momenta of upper and lower rigid atoms.

        Raises
        ------
        NotImplementedError
            If atoms.get_cell() is non-orthogonal,
            SlideWithNormalPressureCuboidCell only works for orthogonal cells.
        """
        if np.abs(atoms.get_cell().sum() - atoms.get_cell().trace()) > 0:
            raise NotImplementedError("Can't do non-orthogonal cell!")
        top_masses = atoms.get_masses()[self.top_mask]
        vtop = (momenta[self.top_mask, self.Pdir] / top_masses).sum() / self.Ntop
        momenta[self.bottom_mask, :] = 0.0
        momenta[self.top_mask, :] = 0.0
        momenta[self.top_mask, self.vdir] = self.v * top_masses
        momenta[self.top_mask, self.Pdir] = vtop * top_masses

    def adjust_potential_energy(self, atoms):
        """Do not adjust energy."""
        return 0.0


class SlideLogger(object):
    """Logger to be attached to an ASE integrator.

    For new files (not restart jobs), the write_header method should
    be called once in order to write the header to the file. Also note
    that ASE does not write the time step 0 automatically.
    You can do so by calling the SlideLogger instance once
    before starting the integration as in example 1 below.

    Parameters
    ----------
    handle : filehandle
       Filehandle e.g. pointing to a file opened in w or a mode.
    atoms : ase.Atoms
        Atomic configuration.
    slider : slider object
        Instance of SlideWithNormalPressureCuboidCell.
    integrator : ASE integrator object,
        Instance of ASE integrator e.g. ase.md.langevin.Langevin.
    step_offset : int
        Last step already written to log file, useful for restarts.

    Examples
    --------
    1. For new runs:
        log_handle = open(logfn, 'w', 1)  # line buffered
        logger = SlideLogger(log_handle, ...)
        logger.write_header()
        logger()
        integrator.attach(logger)
        integrator.run(steps_integrate)
        log_handle.close()

    2. For restarts:
        with open(logfn, 'r') as log_handle:
            step_offset = SlideLog(log_handle).step[-1]
        log_handle = open(logfn, 'a', 1)  # line buffered append
        logger = SlideLogger(log_handle, ..., step_offset=step_offset)
        integrator.attach(logger)
        integrator.run(steps_integrate)
        log_handle.close()
    """

    def __init__(self, handle, atoms, slider, integrator, step_offset=0):
        self.handle = handle
        self.atoms = atoms
        self.slider = slider
        self.integrator = integrator
        self.step_offset = step_offset

    def write_header(self):
        """Write header of log-file."""
        self.handle.write('# step | time / fs | T_thermostat / K | P_top / GPa | P_bottom / GPa | h / Ang | v / Ang * fs | a / Ang * fs ** 2 | tau_top / GPa | tau_bottom / GPa\n')

    def __call__(self):
        """Write current status (time, T, P, ...) to log-file."""
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

    Parameters
    ----------
    handle matscipy.pressurecoupling.SlideLogger instance
        Handle or filename pointing to log file.

    Attributes
    ----------
    step : ndarray
        Step indices 0, 1, 2, ....
    time : ndarray
        Simulation time in fs at step.
    T_thermostat : ndarray
        Instantantaneous temperature in K from thermostat-region
        only from degrees of freedom along thermalized direction.
    P_top : ndarray
        Normal pressure on lid in GPa.
    P_bottom : ndarray
        Normal pressure on base in GPa.
    h : ndarray
        Separation of lid and base in Ang.
    v : ndarray
        Normal speed of lid in Ang / fs.
    a : ndarray
        Normal acceleration of lid in Ang / fs ** 2.
    tau_top : ndarray
        Shear stress on lid in GPa.
    tau_bottom : ndarray
        Shear stress on base in GPa.
    rows : ndarray
        All data in a 2d array with axis 0 step and axis 1
        the values ordered as above.
    """

    def __init__(self, handle):
        self.rows = np.loadtxt(handle)
        self.step, self.time, self.T_thermostat, self.P_top, self.P_bottom, self.h, self.v, self.a, self.tau_top, self.tau_bottom = self.rows.T
        self.step = self.step.astype(int)
