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
from ase.neighborlist import neighbor_list as ase_neighbor_list


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


@pytest.mark.parametrize("neighbor_list", [matscipy_neighbor_list, ase_neighbor_list])
def test_neighbors_cutoff(neighbor_list):
    """

    Asserts the cutoff passed to neighbor list functions represents the distance between the atoms.

    """

    atoms = Atoms(['H', 'H'], positions=[[0,0,0], [0,0,1]], pbc=False, cell=[2,2,2])

    i,j,d = neighbor_list('ijd', atoms, cutoff=1.01)
    assert len(i) == 2
    assert len(j) == 2
    assert len(d) == 2


    i,j,d = neighbor_list('ijd', atoms, cutoff=0.6)
    assert len(i) == 0
    assert len(j) == 0
    assert len(d) == 0


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
        t_equil = 5.
        t_sample = 20.

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




# --- Conservative force calculator -------------------------------------------

class DPDRepulsiveCalculator(Calculator):
    """Pairwise repulsive force F_ij = a*(1-r/rc)*r_hat_ij.

    matscipy.neighbour_list returns both (i,j) and (j,i), so forces are
    accumulated only on i.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, a=25., rc=RC):
        super().__init__()
        self.a = a
        self.rc = rc

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        n = len(atoms)
        forces = np.zeros((n, 3))
        energy = 0.0

        i, j, D, d = matscipy_neighbor_list('ijDd', atoms, cutoff=self.rc)

        if len(i) > 0:
            w = 1.0 - d / self.rc                        # (1-r/rc), shape (P,)
            # D = r_j - r_i; repulsive force on i points from j towards i = -D/d
            f_vec = -(self.a * w / d)[:, None] * D       # (P, 3)
            np.add.at(forces, i, f_vec)
            # pair potential U(r) = (a*rc/2)*(1-r/rc)^2; each pair counted twice
            energy = np.sum(0.5 * self.a * self.rc * w ** 2) / 2

        self.results = {'energy': energy, 'forces': forces}


@pytest.mark.parametrize("seed", [0, 1])
def test_repulsive_gas(seed):
    """

    Fluid with smooth repulsive forces between the particles

    This is the same system as model B in Peters Europhys. Lett. (2004)

    """

    REP_AMPLITUDE = 25.0  # repulsive force amplitude
    N_ATOMS = 500
    BOX = (N_ATOMS / 4) ** (1 / 3)  # keeps density = 4



    # --- System setup -------------------------------------------------------------
    def make_atoms(seed=42):
        """N_ATOMS unit-mass particles in a cubic box at density 4 with Maxwell-Boltzmann momenta."""
        rng = np.random.default_rng(seed)
        positions = rng.uniform(0, BOX, size=(N_ATOMS, 3))
        atoms = Atoms(
            ['H'] * N_ATOMS,
            masses=[MASS] * N_ATOMS,
            positions=positions,
            cell=[BOX, BOX, BOX],
            pbc=True,
        )
        atoms.calc = DPDRepulsiveCalculator(a=REP_AMPLITUDE, rc=RC)
        MaxwellBoltzmannDistribution(atoms, temperature_K=kBT / ase.units.kB,
                                      rng=np.random.default_rng(seed + 1))
        Stationary(atoms)
        return atoms

    def run(atoms, dt, n_eq, n_prod, rng_seed):
        """Standard Peters thermostat: 1 VV step + 1 thermostat per dt."""
        dyn = DPDThermostat(
            atoms, dt,
            gamma=4.5,
            cutoff=RC,
            neighbor_list=matscipy_neighbor_list,
            T=kBT / ase.units.kB,
            rng=np.random.default_rng(rng_seed),
        )
        for _ in range(n_eq):
            dyn.step()
        T_acc = 0.0
        for _ in range(n_prod):
            dyn.step()
            T_acc += kinetic_temperature(atoms)
        return T_acc / n_prod

    # equally well equilibrated and sampled.
    T_EQ = 5.0  # DPD time units of equilibration
    T_PROD = 20.0  # DPD time units of production

    dt = 0.05

    n_eq = max(1, round(T_EQ / dt))
    n_prod = max(1, round(T_PROD / dt))

    atoms = make_atoms(seed=seed)
    kT_meas = run(atoms, dt, n_eq, n_prod, rng_seed=seed)
    print(f"  measured:    kBT = {kT_meas:.4f}", flush=True)


    assert abs(kT_meas - kBT) / kBT < 0.05, (
        f"Temperature kBT = {kT_meas:.1f} K deviates more than 5% from "
        f"target 1.0 ")
