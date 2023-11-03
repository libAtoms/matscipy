import os
import unittest
import matscipytest
import sys

import matscipy.utils as utils_mod
import numpy as np


test_dir = os.path.dirname(os.path.realpath(__file__))


class TestDislocation(matscipytest.MatSciPyTestCase):
    """Class to store test for utils.py module."""

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

if __name__ == '__main__':
    unittest.main()
