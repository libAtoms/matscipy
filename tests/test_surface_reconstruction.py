import unittest
import numpy as np
from matscipy.cauchy_born import CubicCauchyBorn
from ase.build import bulk
from ase.lattice.cubic import Diamond, FaceCenteredCubic
from ase.optimize.precon import PreconLBFGS
from ase.constraints import ExpCellFilter
from matscipy.cauchy_born import CubicCauchyBorn
from matscipy.calculators.manybody.explicit_forms.stillinger_weber import StillingerWeber, Holland_Marder_PRL_80_746_Si
from matscipy.calculators.manybody.explicit_forms.tersoff_brenner import TersoffBrenner, Erhart_PRB_71_035211_Si
from matscipy.calculators.manybody import Manybody
from matscipy.surface_reconstruction import SurfaceReconstruction
import ase
from matscipy.calculators.eam import EAM
import matscipytest


class TestSurfaceReconstructionMap(matscipytest.MatSciPyTestCase):
    """
    Tests of Cauchy-Born shift prediction in the case of multilattices.

    Test uses EAM for gold (Au-Grochola-JCP05.eam.alloy)
    and 
    """

    def setupmultilattice(self):
        el = 'Si'  # chemical element
        unitcell = Diamond(el, latticeconstant=5.43, size=[1, 1, 1],
                           directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        calc = Manybody(**TersoffBrenner(Erhart_PRB_71_035211_Si))
        unitcell.calc = calc
        ecf = ExpCellFilter(unitcell)
        opt = PreconLBFGS(ecf)
        opt.run(fmax=0.0001, smax=0.0001)
        a0 = unitcell.cell[0, 0]
        self.cb = CubicCauchyBorn(el, a0, calc, lattice=Diamond)
        self.el = el
        self.a0 = a0
        self.calc = calc

    def test_surface_mapping_multilattice(self):
        self.setupmultilattice()
        dir_vals = [[[1, 1, 0], [-1, 1, 0], [0, 0, 1]],
                    [[1, 1, 1], [-2, 1, 1], [0, -1, 1]]]
        surf_dirs = [0, 1, 2]
        for dirs in dir_vals:
            for surf_dir in surf_dirs:
                # print(f'testing relaxation of {dirs[surf_dir]} surface')
                sr = SurfaceReconstruction(
                    self.el, self.a0, self.calc, dirs, surf_dir, lattice=Diamond, multilattice=True)
                sr.map_surface(fmax=0.0005, shift=0)
                self.check_surface_mapping_multilattice(
                    sr, dirs, Diamond, surf_dir)

    def check_surface_mapping_multilattice(self, sr, directions, lattice, surf_dir, shift=0):
        cb = CubicCauchyBorn(self.el, self.a0, self.calc, lattice=lattice)

        size = [1, 1, 1]
        size[surf_dir] *= 20

        shift_vec = np.zeros([3])
        shift_vec[surf_dir] += 0.01 + shift

        bulk = lattice(directions=directions, size=size,
                       symbol=self.el, latticeconstant=self.a0, pbc=(1, 1, 1))
        calc = self.calc
        bulk.positions += 0.01+shift
        bulk.wrap()

        cell = bulk.get_cell()
        cell[surf_dir, :] *= 2  # vacuum along surface axis (surface normal)
        slab = bulk.copy()
        slab.set_cell(cell, scale_atoms=False)
        pos = slab.get_positions()

        sorted_surf_indices = np.argsort(pos[:, surf_dir])

        # apply the surface reconstruction to the top surface only
        surface_atom = np.argmax(pos[:, surf_dir])
        surface_coords = pos[surface_atom, :]+np.array([0.01, 0.01, 0.01])

        slab.set_calculator(calc)
        initial_force = slab.get_forces()
        # ase.io.write(f'{directions[surf_dir]}_before.xyz',slab)
        sr.apply_surface_shift(slab, surface_coords, cb=cb, xlim=None,
                               ylim=None, zlim=None, search_dir=-1, atoms_for_cb=bulk)
        final_force = slab.get_forces()
        # ase.io.write(f'{directions[surf_dir]}_after.xyz',slab)

        # get the atoms nearest to the top surface
        # apply the surface reconstruction to the bottom surface
        natoms = len(slab)
        maxfbefore = np.max(
            np.abs(initial_force[sorted_surf_indices[int(natoms/2):natoms]]))
        # print('max force before:',maxfbefore)
        maxfafter = np.max(
            np.abs(final_force[sorted_surf_indices[int(natoms/2):natoms]]))
        # print('max force after:',maxfafter)
        assert maxfafter < (maxfbefore/10)

    def setupsinglelattice(self):
        el = 'Au'  # chemical element
        unitcell = FaceCenteredCubic(el, latticeconstant=4.07, size=[1, 1, 1],
                                     directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        calc = EAM('Au-Grochola-JCP05.eam.alloy')
        unitcell.calc = calc
        ecf = ExpCellFilter(unitcell)
        opt = PreconLBFGS(ecf)
        opt.run(fmax=0.0001, smax=0.0001)
        a0 = unitcell.cell[0, 0]
        self.el = el
        self.a0 = a0
        self.calc = calc

    def test_surface_mapping_single_lattice(self):
        self.setupsinglelattice()
        dir_vals = [[[1, 1, 0], [-1, 1, 0], [0, 0, 1]],
                    [[1, 1, 1], [-2, 1, 1], [0, -1, 1]]]
        surf_dirs = [0, 1, 2]
        for dirs in dir_vals:
            for surf_dir in surf_dirs:
                print(f'testing relaxation of {dirs[surf_dir]} surface')
                sr = SurfaceReconstruction(
                    self.el, self.a0, self.calc, dirs, surf_dir, lattice=FaceCenteredCubic)
                sr.map_surface(fmax=0.0005, shift=0, layers=20)
                self.check_surface_mapping_single_lattice(
                    sr, dirs, FaceCenteredCubic, surf_dir)

    def check_surface_mapping_single_lattice(self, sr, directions, lattice, surf_dir, shift=0):
        size = [1, 1, 1]
        size[surf_dir] *= 20

        shift_vec = np.zeros([3])
        shift_vec[surf_dir] += 0.01 + shift

        bulk = lattice(directions=directions, size=size,
                       symbol=self.el, latticeconstant=self.a0, pbc=(1, 1, 1))
        calc = self.calc
        bulk.positions += 0.01+shift
        bulk.wrap()

        cell = bulk.get_cell()
        cell[surf_dir, :] *= 2  # vacuum along surface axis (surface normal)
        slab = bulk.copy()
        slab.set_cell(cell, scale_atoms=False)
        pos = slab.get_positions()

        sorted_surf_indices = np.argsort(pos[:, surf_dir])

        # apply the surface reconstruction to the top surface only
        surface_atom = np.argmax(pos[:, surf_dir])
        surface_coords = pos[surface_atom, :]+np.array([0.01, 0.01, 0.01])

        slab.set_calculator(calc)
        initial_force = slab.get_forces()
        # ase.io.write(f'{directions[surf_dir]}0.xyz',slab)
        sr.apply_surface_shift(slab, surface_coords,
                               xlim=None, ylim=None, zlim=None, search_dir=-1)
        final_force = slab.get_forces()
        # ase.io.write(f'{directions[surf_dir]}1.xyz',slab)

        # get the atoms nearest to the top surface
        # apply the surface reconstruction to the bottom surface
        natoms = len(slab)
        maxfbefore = np.max(
            np.abs(initial_force[sorted_surf_indices[int(natoms/2):natoms]]))
        print('max force before:', maxfbefore)
        maxfafter = np.max(
            np.abs(final_force[sorted_surf_indices[int(natoms/2):natoms]]))
        print('max force after:', maxfafter)
        assert maxfafter < (maxfbefore/10)


if __name__ == '__main__':
    unittest.main()
