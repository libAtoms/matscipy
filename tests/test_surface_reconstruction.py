import numpy as np
import pytest
from ase.constraints import ExpCellFilter
from ase.lattice.cubic import Diamond, FaceCenteredCubic
from ase.optimize.precon import PreconLBFGS

from matscipy.calculators.eam import EAM
from matscipy.calculators.manybody import Manybody
from matscipy.calculators.manybody.explicit_forms.tersoff_brenner import (
    Erhart_PRB_71_035211_Si, TersoffBrenner)
from matscipy.cauchy_born import CubicCauchyBorn
from matscipy.surface_reconstruction import SurfaceReconstruction


@pytest.fixture
def multilattice_system():
    """Set up multilattice system (Diamond Si with Tersoff-Brenner potential)."""
    el = "Si"
    unitcell = Diamond(
        el,
        latticeconstant=5.43,
        size=[1, 1, 1],
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )

    calc = Manybody(**TersoffBrenner(Erhart_PRB_71_035211_Si))
    unitcell.calc = calc
    ecf = ExpCellFilter(unitcell)
    opt = PreconLBFGS(ecf)
    opt.run(fmax=0.0001, smax=0.0001)
    a0 = unitcell.cell[0, 0]
    cb = CubicCauchyBorn(el, a0, calc, lattice=Diamond)

    return {"el": el, "a0": a0, "calc": calc, "cb": cb}


@pytest.fixture
def single_lattice_system(datafile_directory):
    """Set up single lattice system (FCC Au with EAM potential)."""
    el = "Au"
    unitcell = FaceCenteredCubic(
        el,
        latticeconstant=4.07,
        size=[1, 1, 1],
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    calc = EAM(f"{datafile_directory}/Au-Grochola-JCP05.eam.alloy")
    unitcell.calc = calc
    ecf = ExpCellFilter(unitcell)
    opt = PreconLBFGS(ecf)
    opt.run(fmax=0.0001, smax=0.0001)
    a0 = unitcell.cell[0, 0]

    return {"el": el, "a0": a0, "calc": calc}


def test_surface_mapping_multilattice(multilattice_system):
    """Test surface reconstruction mapping for multilattice (Diamond) systems.

    Tests multiple crystal orientations and surface directions to verify
    that the reconstruction reduces forces on surface atoms.
    """
    dir_vals = [[[1, 1, 0], [-1, 1, 0], [0, 0, 1]], [[1, 1, 1], [-2, 1, 1], [0, -1, 1]]]
    surf_dirs = [0, 1, 2]

    for dirs in dir_vals:
        for surf_dir in surf_dirs:
            sr = SurfaceReconstruction(
                multilattice_system["el"],
                multilattice_system["a0"],
                multilattice_system["calc"],
                dirs,
                surf_dir,
                lattice=Diamond,
                multilattice=True,
            )
            sr.map_surface(fmax=0.0005, shift=0)
            _check_surface_mapping_multilattice(
                sr, dirs, Diamond, surf_dir, multilattice_system
            )


def _check_surface_mapping_multilattice(
    sr, directions, lattice, surf_dir, system, shift=0
):
    """Verify surface reconstruction reduces forces for multilattice systems.

    Creates a slab with vacuum, applies surface reconstruction, and checks
    that maximum force on surface atoms is reduced by at least a factor of 10.
    """
    cb = system["cb"]
    el = system["el"]
    a0 = system["a0"]
    calc = system["calc"]

    size = [1, 1, 1]
    size[surf_dir] *= 20

    shift_vec = np.zeros([3])
    shift_vec[surf_dir] += 0.01 + shift

    bulk = lattice(
        directions=directions, size=size, symbol=el, latticeconstant=a0, pbc=(1, 1, 1)
    )
    bulk.positions += 0.01 + shift
    bulk.wrap()

    # Create slab with vacuum
    cell = bulk.get_cell()
    cell[surf_dir, :] *= 2
    slab = bulk.copy()
    slab.set_cell(cell, scale_atoms=False)
    pos = slab.get_positions()

    sorted_surf_indices = np.argsort(pos[:, surf_dir])

    # Apply surface reconstruction to top surface
    surface_atom = np.argmax(pos[:, surf_dir])
    surface_coords = pos[surface_atom, :] + np.array([0.01, 0.01, 0.01])

    slab.set_calculator(calc)
    initial_force = slab.get_forces()

    sr.apply_surface_shift(
        slab,
        surface_coords,
        cb=cb,
        xlim=None,
        ylim=None,
        zlim=None,
        search_dir=-1,
        atoms_for_cb=bulk,
    )
    final_force = slab.get_forces()

    # Check forces on top half of atoms (near surface)
    natoms = len(slab)
    maxfbefore = np.max(
        np.abs(initial_force[sorted_surf_indices[int(natoms / 2): natoms]])
    )
    maxfafter = np.max(
        np.abs(final_force[sorted_surf_indices[int(natoms / 2): natoms]])
    )

    assert maxfafter < (
        maxfbefore / 10
    ), f"Force reduction insufficient: {maxfafter} vs {maxfbefore}"


def test_surface_mapping_single_lattice(single_lattice_system):
    """Test surface reconstruction mapping for single lattice (FCC) systems.

    Tests multiple crystal orientations and surface directions to verify
    that the reconstruction reduces forces on surface atoms.
    """
    dir_vals = [[[1, 1, 0], [-1, 1, 0], [0, 0, 1]], [[1, 1, 1], [-2, 1, 1], [0, -1, 1]]]
    surf_dirs = [0, 1, 2]

    for dirs in dir_vals:
        for surf_dir in surf_dirs:
            print(f"testing relaxation of {dirs[surf_dir]} surface")
            sr = SurfaceReconstruction(
                single_lattice_system["el"],
                single_lattice_system["a0"],
                single_lattice_system["calc"],
                dirs,
                surf_dir,
                lattice=FaceCenteredCubic,
            )
            sr.map_surface(fmax=0.0005, shift=0, layers=20)
            _check_surface_mapping_single_lattice(
                sr, dirs, FaceCenteredCubic, surf_dir, single_lattice_system
            )


def _check_surface_mapping_single_lattice(
    sr, directions, lattice, surf_dir, system, shift=0
):
    """Verify surface reconstruction reduces forces for single lattice systems.

    Creates a slab with vacuum, applies surface reconstruction, and checks
    that maximum force on surface atoms is reduced by at least a factor of 10.
    """
    el = system["el"]
    a0 = system["a0"]
    calc = system["calc"]

    size = [1, 1, 1]
    size[surf_dir] *= 20

    shift_vec = np.zeros([3])
    shift_vec[surf_dir] += 0.01 + shift

    bulk = lattice(
        directions=directions, size=size, symbol=el, latticeconstant=a0, pbc=(1, 1, 1)
    )
    bulk.positions += 0.01 + shift
    bulk.wrap()

    # Create slab with vacuum
    cell = bulk.get_cell()
    cell[surf_dir, :] *= 2
    slab = bulk.copy()
    slab.set_cell(cell, scale_atoms=False)
    pos = slab.get_positions()

    sorted_surf_indices = np.argsort(pos[:, surf_dir])

    # Apply surface reconstruction to top surface
    surface_atom = np.argmax(pos[:, surf_dir])
    surface_coords = pos[surface_atom, :] + np.array([0.01, 0.01, 0.01])

    slab.set_calculator(calc)
    initial_force = slab.get_forces()

    sr.apply_surface_shift(
        slab, surface_coords, xlim=None, ylim=None, zlim=None, search_dir=-1
    )
    final_force = slab.get_forces()

    # Check forces on top half of atoms (near surface)
    natoms = len(slab)
    maxfbefore = np.max(
        np.abs(initial_force[sorted_surf_indices[int(natoms / 2): natoms]])
    )
    print("max force before:", maxfbefore)
    maxfafter = np.max(
        np.abs(final_force[sorted_surf_indices[int(natoms / 2): natoms]])
    )
    print("max force after:", maxfafter)

    assert maxfafter < (
        maxfbefore / 10
    ), f"Force reduction insufficient: {maxfafter} vs {maxfbefore}"
