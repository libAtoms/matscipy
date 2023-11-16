import unittest
import numpy as np
from ase.build import bulk
from ase.lattice.cubic import Diamond
from ase.optimize.precon import PreconLBFGS
from ase.constraints import ExpCellFilter
from matscipy.cauchy_born import CubicCauchyBorn
from matscipy.calculators.manybody.explicit_forms.stillinger_weber import StillingerWeber, Holland_Marder_PRL_80_746_Si
from matscipy.calculators.manybody import Manybody
from matscipy.fracture_mechanics.clusters import generate_3D_cubic_111, diamond, set_regions, generate_3D_structure
import matscipytest
import ase.io


class TestBuild3DPeriodicFractureCell(matscipytest.MatSciPyTestCase):
    """
    Test to build 3D kink-periodic fracture cell. 
    """

    def setUp(self):
        el = 'Si'  # chemical element
        si = Diamond(el, latticeconstant=5.43, size=[1, 1, 1],
                     directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        calc = Manybody(**StillingerWeber(Holland_Marder_PRL_80_746_Si))
        si.calc = calc
        self.calc = calc
        ecf = ExpCellFilter(si)
        opt = PreconLBFGS(ecf)
        opt.run(fmax=1e-6, smax=1e-6)
        # print(si.get_positions())
        a0 = si.cell[0, 0]
        # print(a0)
        self.cb = CubicCauchyBorn(el, a0, calc, lattice=Diamond)
        self.unit_cell = si
        self.a0 = a0
        self.el = el

    def build_3D_cluster(self, build_func, nzlayer, crack_surface, crack_front):

        A = np.zeros([3, 3])
        directions = [np.cross(crack_surface, crack_front),
                      crack_surface, crack_front]
        for i, direction in enumerate(directions):
            direction = np.array(direction)
            A[:, i] = direction / np.linalg.norm(direction)

        r_III = 40
        cutoff = 6
        padding = 3
        r_I = 20
        [ax, ay, az] = diamond(self.el, self.a0, [1, 1, 1],
                               crack_surface, crack_front).cell.lengths()
        n = [2 * (int(np.ceil(((r_III + 2*cutoff) / ax)))+padding)+1,
             2 * (int(np.ceil((r_III + 2*cutoff) / ay))+padding+1), 1]

        cryst = diamond(self.el, self.a0, n, crack_surface, crack_front,
                        cb=self.cb, shift=np.array([0.0, 0.25, 0.0]), switch_sublattices=True)
        self.cb.set_sublattices(cryst, A, read_from_atoms=True)

        cryst, theta = build_func(cryst, nzlayer, self.el, self.a0, diamond, crack_surface,
                                  crack_front, cb=self.cb, shift=np.array([0.0, 0.25, 0.0]), switch_sublattices=True)

        # create cluster from this surface relaxed version
        # carve circular cluster
        cluster = set_regions(cryst, r_I, cutoff, r_III)

        # assert that forces between 5 and 10 angstroms are all small
        sx, sy, sz = cluster.cell.diagonal()
        x, y = cluster.positions[:, 0], cluster.positions[:, 1]
        cx, cy = sx/2, sy/2
        r = np.sqrt((x - cx)**2 + (y - cy)**2)

        mask = (r > 5) & (r < 10)
        cluster.calc = self.calc
        forces = cluster.get_forces()
        # ase.io.write('cluster.xyz',cluster)
        assert np.max(np.linalg.norm(forces[mask], axis=1)) < 0.5

    def test_build_cubic_111_periodic_cell(self):
        nzlayer = 5
        crack_surface = np.array([1, 1, 1])
        crack_front = np.array([1, -1, 0])
        self.build_3D_cluster(generate_3D_cubic_111,
                              nzlayer, crack_surface, crack_front)

    def test_build_3D_kink_periodic_cell(self):
        nzlayer = 5
        crack_surface = np.array([1, 1, 1])
        crack_front = np.array([1, -1, 0])
        self.build_3D_cluster(generate_3D_structure,
                              nzlayer, crack_surface, crack_front)
