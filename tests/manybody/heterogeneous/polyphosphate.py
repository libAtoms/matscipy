import pytest
import numpy as np

from numpy.linalg import norm
from ase.io import read
from ase.calculators.mixing import SumCalculator
from matscipy.molecules import Molecules
from matscipy.neighbours import MolecularNeighbourhood
from matscipy.calculators.pair_potential import PairPotential, LennardJonesCut
from matscipy.calculators.ewald import Ewald
from matscipy.calculators.manybody.newmb import Manybody
from matscipy.calculators.manybody.potentials import \
    ZeroPair, HarmonicAngle


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

    calc = LAMMPSlib(lmpcmds=[],
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
    from matscipy.calculators.manybody.explicit_forms import HarmonicAngle, ZeroPair

    lj_interactions = {
        (8, 8): LennardJonesCut(0.012185, 2.9170696728, 10),
        (8, 15): LennardJonesCut(0.004251, 1.9198867376, 10),
        (8, 30): LennardJonesCut(8.27e-4, 2.792343852, 10),
    }

    pair = PairPotential(lj_interactions)
    triplet = NiceManybody(ZeroPair(),
                           HarmonicAngle(
                               np.radians(109.47), 2 * 1.77005, atoms
                           ),
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
        1: 8,   # O
        2: 15,  # P
        3: 30,  # Zn
    }

    for lammps_type, atom_num in lammps_to_atom_num.items():
        atoms.numbers[atoms.numbers == lammps_type] = atom_num

    return atoms


@pytest.fixture
def polyphosphate():
    atoms = read('polyphosphate.data',
                 format='lammps-data',
                 style='full',
                 units='metal')

    atoms = map_ase_types(atoms)
    mol = Molecules.from_atoms(atoms)
    atoms.calc = Manybody({1: ZeroPair()}, {
        1: HarmonicAngle(2 * 1.77005, np.radians(109.47)),
        2: HarmonicAngle(2 * 10.4663, np.radians(135.58)),
    }, MolecularNeighbourhood(mol))
    return atoms, mol


def test_angles_energy(polyphosphate):
    atoms, mol = polyphosphate
    epot = atoms.get_potential_energy()

    epot_ref = 0
    for t, k, theta in zip([1, 2], [1.77005, 10.4663], [109.47, 135.58]):
        angles = mol.get_angles(atoms)[mol.angles['type'] == t]
        epot_ref += sum(k * np.radians(angles - theta)**2)

    assert np.abs(epot_ref - epot) / epot_ref < 1e-14


def test_affine_elastic_constants(polyphosphate):
    from matscipy.elasticity import \
        measure_triclinic_elastic_constants as num_constants

    atoms, _ = polyphosphate

    # So that Cauchy stress is non-zero
    atoms.cell *= 0.8
    atoms.positions *= 0.8

    stress = atoms.get_stress()

    # Numerical Birch
    C_many_ref = num_constants(atoms, delta=1e-2)

    # Consistency test if LAMMPS is installed
    try:
        latoms = atoms.copy()
        set_lammps(latoms, 'polyphosphate.data')
        C_lammps = num_constants(latoms, delta=1e-2, verbose=False)
        lstress = latoms.get_stress()
        assert norm(C_lammps - C_many_ref) / norm(C_lammps) < 1e-7
        assert norm(lstress - stress) / norm(lstress) < 1e-7
    except AssertionError as e:
        raise(e)
    except Exception as e:
        print(e)

    C_birch = atoms.calc.get_property('birch_coefficients', atoms)
    assert norm(C_birch - C_many_ref) / norm(C_many_ref) < 1e-2
    np.testing.assert_allclose(C_birch, C_many_ref, atol=3e-5, rtol=1e-3)
