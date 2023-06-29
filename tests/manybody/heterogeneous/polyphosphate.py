import pytest
import numpy as np

from ase.io import read
from ase.optimize import FIRE
from ase.calculators.mixing import SumCalculator
from ase.calculators.calculator import PropertyNotImplementedError
from matscipy.numerical import (numerical_forces, numerical_stress,
                                numerical_nonaffine_forces, numerical_hessian)
from matscipy.elasticity import \
    measure_triclinic_elastic_constants as numerical_birch
from matscipy.molecules import Molecules
from matscipy.neighbours import MolecularNeighbourhood
from matscipy.calculators.pair_potential import PairPotential, LennardJonesCut
from matscipy.calculators.ewald import Ewald
from matscipy.calculators.manybody.newmb import Manybody
from matscipy.calculators.manybody.potentials import \
    ZeroPair, HarmonicAngle


NUM_PROPERTIES = {
    "forces": numerical_forces,
    "stress": numerical_stress,
    "nonaffine_forces": lambda a: numerical_nonaffine_forces(a, d=1e-8),
    "hessian": numerical_hessian,
    "birch_coefficients": numerical_birch,
}


def here(path):
    from pathlib import Path
    return Path(__file__).parent / path


def set_lammps(atoms, lammps_datafile):
    """Set LAMMPS calculator to Atoms object."""
    from ase.calculators.lammpslib import LAMMPSlib
    from ase.geometry import wrap_positions

    atom_symbol_to_lammps = {
        "O": 1,
        "P": 2,
        "Zn": 3,
    }

    atoms.positions = wrap_positions(atoms.positions, atoms.cell, atoms.pbc)

    header = f"""
    boundary p p p
    units        metal
    atom_style   full
    pair_style   lj/cut/coul/long 10 10
    bond_style   zero
    angle_style  harmonic
    special_bonds lj/coul 1 1 1
    read_data {lammps_datafile}

    # These commands to accomodate for ASE
    change_box all triclinic
    kspace_style ewald 1e-12

    # Deactivate pair and coulomb
    kspace_style none
    pair_style lj/cut 10
    pair_coeff * * 0 1
    """.split("\n")

    calc = LAMMPSlib(
        lmpcmds=[],
        atom_types=atom_symbol_to_lammps,
        lammps_header=header,
        create_atoms=False,
        create_box=False,
        boundary=False,
        keep_alive=True,
        log_file='log.lammps')
    atoms.calc = calc


def set_legacy_manybody(atoms, molecules):
    """Set matscipy calculators for system."""
    from matscipy.calculators.manybody.calculator import NiceManybody
    from matscipy.calculators.manybody.explicit_forms import \
        HarmonicAngle, ZeroPair

    lj_interactions = {
        (8, 8): LennardJonesCut(0.012185, 2.9170696728, 10),
        (8, 15): LennardJonesCut(0.004251, 1.9198867376, 10),
        (8, 30): LennardJonesCut(8.27e-4, 2.792343852, 10),
    }

    pair = PairPotential(lj_interactions)
    triplet = NiceManybody(
        ZeroPair(), HarmonicAngle(np.radians(109.47), 2 * 1.77005, atoms),
        MolecularNeighbourhood(molecules))
    ewald = Ewald()
    ewald.set(accuracy=1e-4, cutoff=10., verbose=False)
    atoms.arrays['charge'] = atoms.get_initial_charges()

    atoms.calc = SumCalculator([triplet])


def set_manybody(atoms, molecules):
    lj_interactions = {
        (8, 8): LennardJonesCut(0.012185, 2.9170696728, 10),
        (8, 15): LennardJonesCut(0.004251, 1.9198867376, 10),
        (8, 30): LennardJonesCut(8.27e-4, 2.792343852, 10),
    }

    neigh = MolecularNeighbourhood(molecules)
    pair = PairPotential(lj_interactions)
    triplet = Manybody({1: ZeroPair()}, {
        1: HarmonicAngle(2 * 1.77005, np.radians(109.47)),
        2: HarmonicAngle(2 * 10.4663, np.radians(135.58)),
    }, neigh)

    ewald = Ewald()
    ewald.set(accuracy=1e-12, cutoff=10., verbose=False)
    atoms.arrays['charge'] = atoms.get_initial_charges()

    atoms.calc = SumCalculator([pair, ewald, triplet])


def map_ase_types(atoms):
    """Convert atom types to atomic numbers."""
    lammps_to_atom_num = {
        1: 8,  # O
        2: 15,  # P
        3: 30,  # Zn
    }

    for lammps_type, atom_num in lammps_to_atom_num.items():
        atoms.numbers[atoms.numbers == lammps_type] = atom_num

    return atoms


@pytest.fixture
def polyphosphate():
    atoms = read(
        here('polyphosphate.data'),
        format='lammps-data',
        style='full',
        units='metal')

    atoms = map_ase_types(atoms)
    mol = Molecules.from_atoms(atoms)
    atoms.calc = Manybody({1: ZeroPair()}, {
        1: HarmonicAngle(2 * 1.77005, np.radians(109.47)),
        2: HarmonicAngle(2 * 10.4663, np.radians(135.58)),
    }, MolecularNeighbourhood(mol))

    # So that Cauchy stress is non-zero
    atoms.cell *= 0.8
    atoms.positions *= 0.8
    return atoms, mol


def lammps_prop(atoms, prop):
    """Return property computed with LAMMPS if installed."""
    atoms = atoms.copy()

    try:
        set_lammps(atoms, here('polyphosphate.data'))
        return atoms.calc.get_property(prop, atoms)
    except PropertyNotImplementedError:
        return NUM_PROPERTIES[prop](atoms)
    except Exception as e:
        print(type(e), e)
        return None


def test_angles_energy(polyphosphate):
    atoms, mol = polyphosphate
    epot = atoms.get_potential_energy()

    epot_ref = 0
    for t, k, theta in zip([1, 2], [1.77005, 10.4663], [109.47, 135.58]):
        angles = mol.get_angles(atoms)[mol.angles['type'] == t]
        epot_ref += sum(k * np.radians(angles - theta)**2)

    assert np.abs(epot_ref - epot) / epot_ref < 1e-14

    epot_ref = lammps_prop(atoms, 'energy')

    if epot_ref is not None:
        assert np.abs(epot_ref - epot) / epot_ref < 1e-13


PROPERTIES = [
    "forces",
    "stress",
    "nonaffine_forces",
    "birch_coefficients",
    "hessian",
]


@pytest.mark.parametrize("prop", PROPERTIES)
def test_properties(polyphosphate, prop):
    atoms, _ = polyphosphate
    atol, rtol = 4e-6, 1e-7

    data = atoms.calc.get_property(prop, atoms)
    ref = NUM_PROPERTIES[prop](atoms)
    lref = lammps_prop(atoms, prop)

    def dense_cast(x):
        from scipy.sparse import issparse
        return x.todense() if issparse(x) else x

    data = dense_cast(data)
    ref = dense_cast(data)
    lref = dense_cast(lref)

    if lref is not None:
        np.testing.assert_allclose(data, lref, atol=atol, rtol=rtol)
    np.testing.assert_allclose(data, ref, atol=atol, rtol=rtol)
