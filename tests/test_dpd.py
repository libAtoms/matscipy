"""Tests for the Peters DPD thermostat (matscipy.dpd.DPDThermostat)."""

import numpy as np
import pytest
from ase import Atoms
from ase.units import kB, fs
import ase.units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import os

from matscipy.dpd import DPDThermostat
from matscipy.neighbours import neighbour_list as matscipy_neighbor_list


class ZeroForceCalculator(Calculator):
    """Returns zero forces, isolating the thermostat from conservative dynamics."""
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results = {
            'energy': 0.0,
            'forces': np.zeros((len(atoms), 3)),
        }


def kinetic_temperature(atoms):
    """Return kBT_kin in DPD units using 3N-3 DOF (momentum conserved)."""
    p = atoms.get_momenta()
    m = atoms.get_masses()
    ekin = 0.5 * np.sum(p ** 2 / m[:, None])
    n = len(atoms)
    return 2.0 * ekin / (3 * n - 3)



# Defining unit system
kBT = 1.
MASS = 1.
RC = 1. # length unit


def _make_noninteracting_gas(n=200, density=4,  seed=42, initialize_velocities=True, masses=None):
    """Random positions in a cubic PBC box at given number density (1/Ang^3)."""
    rng = np.random.default_rng(seed)
    L = (n / density) ** (1 / 3)
    positions = rng.uniform(0, L, size=(n, 3))
    atoms = Atoms(['X'] * n, positions=positions, cell=[L, L, L],
                  masses = masses if masses is not None else [MASS] * n ,  pbc=True)
    atoms.calc = ZeroForceCalculator()

    if initialize_velocities:
        MaxwellBoltzmannDistribution(atoms, temperature_K=kBT / ase.units.kB,
                                     rng=np.random.default_rng(seed))
        Stationary(atoms)
    else:
        atoms.set_momenta(np.zeros((n, 3)))


    return atoms



def _kinetic_kBT(atoms):
    p = atoms.get_momenta()
    m = atoms.get_masses()
    ekin = 0.5 * np.sum(p ** 2 / m[:, None])
    return 2 * ekin / (3 * len(atoms))


class TestDPDThermostat:
    THERMOSTAT_PARAMETERS = dict(
        gamma=4.5,
        cutoff=RC,
        neighbor_list=matscipy_neighbor_list,
        T = kBT / ase.units.kB,
    )

    @pytest.mark.parametrize("seed", [0,1])
    def test_temperature_convergence(self, seed, verbose=True):
        """Kinetic temperature must converge to T_target within 5%."""

        dt = 0.05
        t_equil = 3.
        t_sample = 7.

        n_equil = max(1, round(t_equil / dt))
        n_sample = max(1, round(t_sample / dt))

        atoms = _make_noninteracting_gas(n=200, density=4, initialize_velocities=False)
        dyn = DPDThermostat(atoms, dt,
                            rng = np.random.default_rng(seed),
                            **self.THERMOSTAT_PARAMETERS
                            )
        for _ in range(n_equil):   # equilibration
            dyn.step()

        kbT_samples = []
        for _ in range(n_sample):   # production
            dyn.step()
            kbT_samples.append(_kinetic_kBT(atoms))
        kBT_mean = np.mean(kbT_samples)

        if verbose:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(np.arange(n_sample) * dt, kbT_samples, 's-', label='model B')
            ax.axhline(kBT, ls='--', color='gray', lw=0.8, label='target $k_BT=1$')
            ax.set_xlabel(r'time')
            # ax.set_xscale('log')
            ax.set_ylabel(r'$k_B T$')
            ax.axhline(kBT_mean)
            ax.legend()
            fig.tight_layout()
            out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'noninteracting_gas.png')
            fig.savefig(out_path, dpi=300)

        print(f'seed {seed}, dt = {dt:.3f}  , kBT {kBT_mean:.5f}')
        assert abs(kBT_mean - kBT) / kBT < 0.05, (
            f"Temperature kBT = {kBT_mean:.1f} K deviates more than 5% from "
            f"target 1.0 ")


    @pytest.mark.parametrize("seed", [0, 1])
    def test_momentum_conservation(self, seed):
        """Total momentum must be conserved exactly (Newton's third law)."""
        atoms = _make_noninteracting_gas(n=50, seed=1, density = 4, initialize_velocities=True)
        p0 = atoms.get_momenta().sum(axis=0).copy()

        dyn = DPDThermostat(atoms, 0.1,
                            **self.THERMOSTAT_PARAMETERS,
                            rng=np.random.default_rng(seed))
        for _ in range(100):
            dyn.step()
            p = atoms.get_momenta().sum(axis=0)
            np.testing.assert_allclose(p, p0, atol=1e-10,
                                       err_msg="Total momentum not conserved")

    @pytest.mark.parametrize("seed", [0, 1,])
    def test_temperature_convergence_unequal_masses(self, seed):
        """Temperature must converge with a mixture of two species."""
        n = 200
        symbols = ['Ar'] * (n // 2) + ['Ne'] * (n // 2)
        atoms = _make_noninteracting_gas(n=n, density=4,
                                         masses =[1.] * (n // 2) + [4.] * (n // 2),
                                         seed=seed)


        dt = 0.05

        t_equil = 3.
        n_sample = 500

        n_equil = max(1, round(t_equil / dt))

        dyn = DPDThermostat(atoms, dt,
                            **self.THERMOSTAT_PARAMETERS,
                            rng=np.random.default_rng(seed))
        for _ in range(n_equil):   # longer equilibration: unequal masses need more steps
            dyn.step()

        kBT_samples = []
        for _ in range(n_sample):
            dyn.step()
            kBT_samples.append(_kinetic_kBT(atoms))

        kBT_mean = np.mean(kBT_samples)
        print(f'seed {seed}, dt = {dt:.3f}  , kBT {kBT_mean:.5f}')

        assert abs(kBT_mean - kBT) / kBT < 0.05, (
            f"Temperature {kBT_mean:.4f} 1 deviates more than 5% from "
            f"target 1 "
        )
