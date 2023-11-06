# write a unit test which tests the r_theta_z sort in cluster.py

import unittest
import numpy as np
from ase.build import bulk
from ase.lattice.cubic import Diamond
from ase.optimize.precon import PreconLBFGS
from ase.constraints import ExpCellFilter
from scipy.linalg import sqrtm
import ase.io
import matscipytest
from matscipy.fracture_mechanics.clusters import diamond, set_groups, set_regions, get_alpha_period, bcc


class TestStableSortCluster(matscipytest.MatSciPyTestCase):
    """
    Tests of stable sort order when building clusters for fracture
    testing.
    """

    def build_cluster(self, lattice, el, a0, rI, rIII, cutoff, crack_surface, crack_front, shift=np.array([0.0, 0.0, 0.0])):
        padding = 3
        [ax, ay, az] = lattice(
            el, a0, [1, 1, 1], crack_surface, crack_front).cell.lengths()
        n = [2 * (int(np.ceil(((rIII + 2*cutoff) / ax)))+padding)+1,
             2 * (int(np.ceil((rIII + 2*cutoff) / ay))+padding+1), 1]
        print('N VALS', n)
        cryst = lattice(el, a0, n, crack_surface, crack_front, shift=shift)
        cluster = set_regions(cryst, rI, cutoff, rIII, sort_type='r_theta_z')
        print(cluster)
        # ase.io.write(f'{el}_cluster.xyz',cluster)
        return cluster

        """    def check_theta_close(self,arr1,arr2,tol):
        #test if two theta arrays are close, including when theta jumps by pi
        diff_arr = arr2-arr1
        wrong_indices = np.where(np.abs(diff_arr)>tol)[0]
        pi_indices = np.where(np.abs((np.abs(diff_arr))-np.pi)<tol)[0]
        if len(pi_indices)>0:
            arr1 = np.delete(arr1, pi_indices)
            arr2 = np.delete(arr2, pi_indices)
        self.assertArrayAlmostEqual(arr1,arr2,tol=tol)"""

    def stable_sort_test(self, lattice, el, a0, crack_surface, crack_front, shift=np.array([0.0, 0.0, 0.0])):
        # set r_I vals between 24 and 64
        r_I_vals = np.arange(24, 64, 10)
        # set r_III vals between 44 and 84
        r_III_vals = np.arange(44, 84, 10)
        # set cutoff to 6
        cutoff = 6
        # set up largest cluster
        big_cluster = self.build_cluster(
            lattice, el, a0, r_I_vals[-1], r_III_vals[-1], cutoff, crack_surface, crack_front, shift=shift)
        big_cluster_pos = big_cluster.positions
        # get radial big cluster positions, as well as theta and z
        sx, sy, sz = big_cluster.cell.diagonal()
        x, y, z_big = big_cluster_pos[:,
                                      0], big_cluster_pos[:, 1], big_cluster_pos[:, 2]
        cx, cy = sx/2, sy/2
        big_cluster_r = np.sqrt((x - cx)**2 + (y - cy)**2)
        big_cluster_theta = np.arctan2(y-cy, x-cx)
        # big_cluster.new_array('r', np.zeros(len(big_cluster), dtype=float))
        # big_cluster.new_array('theta',np.zeros(len(big_cluster)),dtype=float)
        # cluster.new_array('index',np.zeros(len(cluster)),dtype=float)
        # r_array = big_cluster.arrays['r']
        # theta_array = big_cluster.arrays['theta']
        # r_array[:]=big_cluster_r[:]
        # theta_array[:]=big_cluster_theta[:]
        # ase.io.write(f'{el}_big_cluster.xyz',big_cluster)

        # now set up each smaller cluster, and check the r, theta, z of the
        # first atoms of big_cluster match those of each small cluster
        for i in range(len(r_I_vals)-1):
            cluster = self.build_cluster(
                lattice, el, a0, r_I_vals[i], r_III_vals[i], cutoff, crack_surface, crack_front, shift=shift)
            small_cluster_pos = cluster.positions
            # get radial small cluster positions (from x and y)
            sx, sy, sz = cluster.cell.diagonal()
            x, y, z_small = small_cluster_pos[:,
                                              0], small_cluster_pos[:, 1], small_cluster_pos[:, 2]
            cx, cy = sx/2, sy/2
            small_cluster_r = np.sqrt((x - cx)**2 + (y - cy)**2)
            small_cluster_theta = np.arctan2(y-cy, x-cx)
            cluster.new_array('r', np.zeros(len(cluster), dtype=float))
            cluster.new_array('theta', np.zeros(len(cluster)), dtype=float)
            # cluster.new_array('index',np.zeros(len(cluster)),dtype=float)
            # r_array = cluster.arrays['r']
            # theta_array = cluster.arrays['theta']
            # r_array[:]=small_cluster_r[:]
            # theta_array[:]=small_cluster_theta[:]
            # ase.io.write(f'{el}_small_cluster.xyz',cluster)
            # assert that the first atoms of the big cluster match those of the small cluster
            self.assertArrayAlmostEqual(
                big_cluster_r[:len(cluster)], small_cluster_r[:], tol=1e-5)
            # print(big_cluster_theta[7],small_cluster_theta[7])
            self.assertArrayAlmostEqual(
                big_cluster_theta[:len(cluster)], small_cluster_theta[:], tol=1e-5)
            self.assertArrayAlmostEqual(
                z_small, z_big[:len(cluster)], tol=1e-5)

    def test_stable_sort_silicon(self):
        # first, test silicon
        el = 'Si'
        a0 = 5.43
        crack_surface = np.array([1, 1, 1])
        crack_front = np.array([1, -1, 0])
        self.stable_sort_test(diamond, el, a0, crack_surface, crack_front)

        crack_surface = np.array([1, 1, 0])
        crack_front = np.array([0, 0, 1])
        self.stable_sort_test(diamond, el, a0, crack_surface,
                              crack_front, shift=np.array([0.0, 0.25, 0.0]))

    def test_stable_sort_iron(self):
        # next, test iron
        el = 'Fe'
        a0 = 2.87
        crack_surface = np.array([1, 0, 0])
        crack_front = np.array([0, 0, 1])
        self.stable_sort_test(bcc, el, a0, crack_surface, crack_front)

        crack_surface = np.array([1, 1, 0])
        crack_front = np.array([0, 0, 1])
        self.stable_sort_test(bcc, el, a0, crack_surface, crack_front)

        crack_surface = np.array([1, 1, 0])
        crack_front = np.array([1, -1, 0])
        self.stable_sort_test(bcc, el, a0, crack_surface, crack_front)

        crack_surface = np.array([1, 0, 0])
        crack_front = np.array([0, 1, 1])
        self.stable_sort_test(bcc, el, a0, crack_surface, crack_front)
