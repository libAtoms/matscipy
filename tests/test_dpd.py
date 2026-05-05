"""Tests for the Peters DPD thermostat (matscipy.dpd.DPDThermostat)."""

import numpy as np
import pytest
from ase import Atoms
from ase.units import kB, fs
from ase.calculators.calculator import Calculator, all_changes

from matscipy.dpd import DPDThermostat


class ZeroForceCalculator(Calculator):
    """Returns zero forces, isolating the thermostat from conservative dynamics."""
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results = {
            'energy': 0.0,
            'forces': np.zeros((len(atoms), 3)),
        }


def _make_gas(n=200, density=0.07, symbols='Ar', seed=42):
    """Random positions in a cubic PBC box at given number density (1/Ang^3)."""
    rng = np.random.default_rng(seed)
    L = (n / density) ** (1 / 3)
    positions = rng.uniform(0, L, size=(n, 3))
    if isinstance(symbols, str):
        symbols = [symbols] * n
    atoms = Atoms(symbols, positions=positions, cell=[L, L, L], pbc=True)
    atoms.set_momenta(np.zeros((n, 3)))
    atoms.calc = ZeroForceCalculator()
    return atoms


def _kinetic_temperature(atoms):
    p = atoms.get_momenta()
    m = atoms.get_masses()
    ekin = 0.5 * np.sum(p ** 2 / m[:, None])
    return 2 * ekin / (3 * len(atoms) * kB)


# density=0.07 atoms/Ang^3 gives ~13 neighbors per atom (enough to span
# all velocity directions efficiently) and gamma=200 gives W~0.1 per pair.
_COMMON = dict(T=300.0, gamma=200.0, cutoff=3.5, timestep=1 * fs)


class TestDPDThermostat:

    def test_temperature_convergence(self):
        """Kinetic temperature must converge to T_target within 5%."""
        atoms = _make_gas(n=200)
        dyn = DPDThermostat(atoms, **_COMMON)

        for _ in range(200):   # equilibration
            dyn.step()

        T_samples = []
        for _ in range(500):   # production
            dyn.step()
            T_samples.append(_kinetic_temperature(atoms))

        T_mean = np.mean(T_samples)
        assert abs(T_mean - _COMMON['T']) / _COMMON['T'] < 0.05, (
            f"Temperature {T_mean:.1f} K deviates more than 5% from "
            f"target {_COMMON['T']} K"
        )

    def test_momentum_conservation(self):
        """Total momentum must be conserved exactly (Newton's third law)."""
        atoms = _make_gas(n=50, seed=1)
        rng = np.random.default_rng(99)
        atoms.set_momenta(rng.normal(size=(50, 3)))
        p0 = atoms.get_momenta().sum(axis=0).copy()

        dyn = DPDThermostat(atoms, **_COMMON)
        for _ in range(100):
            dyn.step()
            p = atoms.get_momenta().sum(axis=0)
            np.testing.assert_allclose(p, p0, atol=1e-10,
                                       err_msg="Total momentum not conserved")

    def test_temperature_convergence_unequal_masses(self):
        """Temperature must converge with a mixture of two species."""
        n = 200
        symbols = ['Ar'] * (n // 2) + ['Ne'] * (n // 2)
        atoms = _make_gas(n=n, symbols=symbols, seed=7)

        dyn = DPDThermostat(atoms, **_COMMON)
        for _ in range(500):   # longer equilibration: unequal masses need more steps
            dyn.step()

        T_samples = []
        for _ in range(1000):
            dyn.step()
            T_samples.append(_kinetic_temperature(atoms))

        T_mean = np.mean(T_samples)
        assert abs(T_mean - _COMMON['T']) / _COMMON['T'] < 0.05, (
            f"Temperature {T_mean:.1f} K deviates more than 5% from "
            f"target {_COMMON['T']} K (unequal masses)"
        )
