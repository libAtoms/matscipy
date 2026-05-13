

"""
Peters DPD thermostat.

Reference:
    E. A. J. F. Peters, Europhys. Lett. 66, 311 (2004).
"""

import numpy as np
from ase.md.verlet import VelocityVerlet
from ase.neighborlist import NeighborList
from ase.neighborlist import neighbor_list as ase_neighbor_list
from ase.units import kB


class DPDThermostat(VelocityVerlet):
    """Peters DPD thermostat with time-step-independent equilibrium statistics.

    Combines velocity Verlet integration for conservative forces with the
    pairwise momentum re-equilibration scheme (Scheme II) of Peters (2004).
    The thermostat leaves the Maxwell-Boltzmann distribution invariant for any
    finite time step; residual time-step error comes solely from the
    discretisation of the conservative (Verlet) part.

    After each Verlet step all pairs within *cutoff* are processed
    simultaneously in a vectorised manner. Note that this deviates
    from the approach in Peters, where each pair is updated
    sequentially in a randomized manner

    Parameters
    ----------
    atoms : ase.Atoms
        The Atoms object to operate on.
    timestep : float
        Time step in ASE time units.  Use ``ase.units.fs`` to convert from fs.
    T : float
        Target temperature in Kelvin.
    gamma : float
        Friction coefficient in ASE units (amu / ASE_time_unit).
    cutoff : float
        Pairwise interaction cutoff radius in Angstrom used for the neighbor list.
    weight_function : callable, optional
        ``omega(r)`` — weight function of pair distance *r* (array in).
        Defaults to ``max(1 - r / cutoff, 0) ** 2`` as in the original DPD
        formulation (eq. 2 of Peters 2004).
    rng : numpy.random.Generator, optional
        Random number generator.  Pass ``numpy.random.default_rng(seed)``
        for reproducible runs.  Defaults to a fresh (non-seeded) generator.
    **kwargs
        Extra arguments forwarded to :class:`ase.md.md.MolecularDynamics`
        (e.g. ``trajectory``, ``logfile``, ``loginterval``).
    """

    def __init__(self, atoms, timestep, T, gamma, cutoff,
                 weight_function=None, rng=None, neighbor_list=ase_neighbor_list , **kwargs):
        super().__init__(atoms, timestep, **kwargs)
        self.rng = np.random.default_rng() if rng is None else rng
        self.T = T
        self.gamma = gamma
        self.cutoff = cutoff
        self.kT = T * kB

        if weight_function is None:
            self._weight = lambda r: np.maximum(1.0 - r / cutoff, 0.0) ** 2
        else:
            self._weight = weight_function

        self._nl = neighbor_list

    def step(self, forces=None):
        forces = super().step(forces)
        self._apply_thermostat()
        return forces

    def _apply_thermostat(self):
        """Apply one simultaneous pairwise thermostat step (Scheme II)."""
        atoms = self.atoms
        positions = atoms.get_positions()
        momenta = atoms.get_momenta()
        masses = atoms.get_masses()
        cell = atoms.get_cell()

        i, j, dist, dr = self._nl('ijdD', atoms, cutoff=self.cutoff)

        # Avoid double counting of pairs
        mask = j < i
        i = i[mask]
        j = j[mask]
        dist = dist[mask]
        dr = dr[mask]

        # i_list, j_list, dr_list = [], [], []
        # for i in range(len(atoms)):
        #     indices, offsets = self._nl.get_neighbors(i)
        #     for j, offset in zip(indices, offsets):
        #         i_list.append(i)
        #         j_list.append(j)
        #         dr_list.append(positions[j] + offset @ cell - positions[i])
        #
        # if not i_list:
        #     return
        #
        # i_arr = np.array(i_list)
        # j_arr = np.array(j_list)
        # dr = np.array(dr_list)           # (N_pairs, 3)
        #
        # r = np.linalg.norm(dr, axis=1)  # (N_pairs,)
        r_hat = dr / dist[:, None]

        mi = masses[i]
        mj = masses[j]
        mu = mi * mj / (mi + mj)        # reduced mass

        # Scheme II (Table I of Peters 2004): exact integration of the
        # irreversible pair dynamics over one time step dt.
        W = self.gamma * self._weight(dist) * self.dt / mu  # dimensionless
        a_dt = mu * (1.0 - np.exp(-W))
        b_sqdt = np.sqrt(self.kT * mu * (1.0 - np.exp(-2.0 * W)))

        vi = momenta[i] / mi[:, None]
        vj = momenta[j] / mj[:, None]
        v_proj = np.einsum('ij,ij->i', vi - vj, r_hat)  # (v_i - v_j) . r_hat

        xi = self.rng.standard_normal(size=len(i))
        dp_mag = -a_dt * v_proj + b_sqdt * xi
        dp = dp_mag[:, None] * r_hat    # (N_pairs, 3)

        # Simultaneous update: accumulate all momentum changes, then apply.
        np.add.at(momenta, i, dp)
        np.add.at(momenta, j, -dp)

        atoms.set_momenta(momenta, apply_constraint=False)
