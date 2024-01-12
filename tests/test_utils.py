import os
import unittest
import matscipytest
import sys

import matscipy.utils as utils_mod
import numpy as np


test_dir = os.path.dirname(os.path.realpath(__file__))


class TestUtils(matscipytest.MatSciPyTestCase):
    """Class to store unittests of utils.py module."""

    def test_validate_cubic_cell(self):
        from ase.build import bulk as build_bulk
        import warnings

        warnings.simplefilter("always")
        
        bulk_fcc = build_bulk("Cu", crystalstructure="fcc", cubic=True)
        bulk_bcc = build_bulk("W", crystalstructure="bcc", cubic=True)
        bulk_dia = build_bulk("C", crystalstructure="diamond", cubic=True)

        bulks = {
            "fcc": bulk_fcc,
            "bcc": bulk_bcc,
            "diamond": bulk_dia
        }

        axes = np.eye(3).astype(int)

        # Check legacy behaviour works
        for structure in bulks.keys():
            bulk = bulks[structure]
            alat = bulk.cell[0, 0]

            alat_valid, struct = utils_mod.validate_cubic_cell(alat, symbol="Cu", axes=axes, crystalstructure=structure)

            assert alat_valid == alat
            bpos = np.sort(bulk.get_scaled_positions(), axis=0)
            spos = np.sort(struct.get_scaled_positions(), axis=0)
            assert np.allclose(struct.cell[:, :], bulk.cell[:, :])
            assert np.allclose(bpos, spos)

        # Check new functionality accepted
        for structure in bulks.keys():
            bulk = bulks[structure]
            alat = bulk.cell[0, 0]
            alat_valid, struct = utils_mod.validate_cubic_cell(bulk, symbol="Cu", axes=axes, crystalstructure=structure)

            assert alat_valid == alat
            bpos = np.sort(bulk.get_scaled_positions(), axis=0)
            spos = np.sort(struct.get_scaled_positions(), axis=0)
            assert np.allclose(struct.cell[:, :], bulk.cell[:, :])
            assert np.allclose(bpos, spos)

        # Check Fail/Warn states
        # "a" validation
        self.assertRaises(TypeError, utils_mod.validate_cubic_cell, "Not a bulk or lattice constant", symbol="Cu", axes=axes, crystalstructure="fcc")
        
        # Crystalstructure validation
        self.assertRaises(ValueError, utils_mod.validate_cubic_cell, bulk_fcc, symbol="Cu", axes=axes, crystalstructure="made up crystalstructure")
        self.assertRaises(TypeError, utils_mod.validate_cubic_cell, bulk_fcc, symbol="Cu", axes=axes, crystalstructure=["not the ", "right datatype"])

        # Bulk structure validation
        self.assertWarns(UserWarning, utils_mod.validate_cubic_cell, bulk_fcc, symbol="Cu", axes=axes, crystalstructure="bcc")
        self.assertWarns(UserWarning, utils_mod.validate_cubic_cell, bulk_fcc, symbol="Cu", axes=axes, crystalstructure="diamond")
        self.assertWarns(UserWarning, utils_mod.validate_cubic_cell, bulk_bcc, symbol="Cu", axes=axes, crystalstructure="fcc")

        ats = bulk_fcc.copy()
        del ats[0]
        # FCC with vacancy is not bulk FCC!
        self.assertWarns(UserWarning, utils_mod.validate_cubic_cell, ats, symbol="Cu", axes=axes, crystalstructure="fcc")

        
        # Test w/ supercells of bulk
        utils_mod.validate_cubic_cell(bulk_fcc * (2, 2, 2), symbol="Cu", axes=axes, crystalstructure="fcc")

        # Test w/ more complex crystal
        nacl = build_bulk("NaCl", crystalstructure="zincblende", a=5.8, cubic=True)

        utils_mod.validate_cubic_cell(nacl, symbol="Cu", axes=axes, crystalstructure="diamond")

        warnings.simplefilter("default")


    def test_complete_basis(self):

        def test_case(vecs, check=True):
            tol = 1e-6

            v1, v2, v3 = utils_mod.complete_basis(*vecs, normalise=True)
            
            if check:
                # Check orthogonal basis
                assert v1.T @ v2 < tol
                assert v2.T @ v3 < tol
                assert v3.T @ v1 < tol

                # Check norms of vecs
                assert np.linalg.norm(v1) - 1 < tol
                assert np.linalg.norm(v2) - 1 < tol
                assert np.linalg.norm(v3) - 1 < tol

                # Check handedness
                assert np.cross(v1, v2).T @ v3 > 0

                # Check dtype
                assert v1.dtype == v2.dtype == v3.dtype
                assert np.issubdtype(v1.dtype, np.floating)

            v1, v2, v3 = utils_mod.complete_basis(*vecs, normalise=False)

            if check:
                # Check dtype for non-normalised mode
                assert v1.dtype == v2.dtype == v3.dtype
                assert np.issubdtype(v1.dtype, np.integer)


        v1 = np.array([1, 1, 1])
        v2 = np.array([1, 1, -2])

        # Single vec given
        test_case([v1])

        # Two vecs
        test_case([v1, v2])

        v1 = np.array([1, 0, 0])
        
        # Non-orthoganal pair
        # Will fail assertions, so bypass
        self.assertWarns(RuntimeWarning, test_case, [v1, v2], check=False)


        v1 = np.array([13, 19, 37])
        # v1 is three prime numbers, unlikely to have a "neat" orthogonal vector
        # with max value less than the default nmax
        self.assertRaises(RuntimeError, test_case, [v1])

    def test_line_intersect_2d(self):

        p1 = np.array([0, 0])
        p2 = np.array([2, 0])

        # "Plus" pattern
        x1 = np.array([1, 1])
        x2 = np.array([1, -1])

        assert utils_mod.line_intersect_2D(p1, p2, x1, x2) == True

        # Lines don't cross ever
        x2 = np.array([2, 1])
        assert utils_mod.line_intersect_2D(p1, p2, x1, x2) == False

        # Finite lines don't meet, but infinite ones would
        x2 = np.array([2, 0.5])
        assert utils_mod.line_intersect_2D(p1, p2, x1, x2) == False

        # Lines meet at one of the points
        x1 = np.array([0, 1])
        x2 = np.array([0, -1])
        assert utils_mod.line_intersect_2D(p1, p2, x1, x2) == True

        # Lines end at the same point
        x2 = p1
        assert utils_mod.line_intersect_2D(p1, p2, x1, x2) == True


    def test_points_in_polygon2D(self):

        # Simple square shape
        polygon = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])

        inside = np.array([0.5, 0.5])
        outside = np.array([0.5, 4])

        assert utils_mod.points_in_polygon2D(inside, polygon) == True
        assert utils_mod.points_in_polygon2D(outside, polygon) == False

        # Complex polygon

        polygon = np.array([
            [0, 0],
            [5, 0],
            [5, 5],
            [3, 5],
            [3, 4],
            [4, 4],
            [4, 1],
            [1, 1],
            [1, 4],
            [2, 4],
            [2, 5],
            [0, 5]
        ])

        inside = np.array([
            [0.5, 0.5],
            [1.5, 4.5]
        ])


        outside = np.array([
            [6, 0],
            [0, 6],
            [2.5, 4.5],
            [2.5, 2.5],
            [3.5, 3.5]
        ])

        assert np.all(utils_mod.points_in_polygon2D(inside, polygon) == True)
        assert np.all(utils_mod.points_in_polygon2D(outside, polygon) == False)

    def test_get_distance_from_polygon(self):
        polygon = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])

        points = np.array([
            [2, 0], # 1 away from point
            [0.5, 0], # touching an edge of the polygon
            [0.5, 0.5] # In the centre, 0.5 from a line
        ])

        expected_dists = np.array([1, 0, 0.5])

        dists = utils_mod.get_distance_from_polygon2D(points, polygon)

        assert np.allclose(dists, expected_dists)

    def test_radial_mask_from_polygon2D(self):
        polygon = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        
        # Get 100 random points
        points = np.random.rand(100, 2)

        # Check that points_in_polygon2D is reproduced
        assert np.all(utils_mod.radial_mask_from_polygon2D(points, polygon, radius=0.0, inner=True) ==
                      utils_mod.points_in_polygon2D(points, polygon))
        

        radius = 0.2
        inner = False

        points = np.array([
            [0.5, 0.5], # Inside polygon, outside radius from points/lines
            [0.5, 0.1], # Close to a line
            [2, 0.5] # Outside polygon, far from lines
        ])

        target_mask = np.array([False, True, False])

        assert np.all(utils_mod.radial_mask_from_polygon2D(points, polygon, radius=radius, 
                                                           inner=inner) == target_mask)
        
        inner = True
        target_mask = np.array([True, True, False])
        assert np.all(utils_mod.radial_mask_from_polygon2D(points, polygon, radius=radius, 
                                                           inner=inner) == target_mask)



if __name__ == '__main__':
    unittest.main()
