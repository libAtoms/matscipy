import unittest
import dislocation as sd
import numpy as np
from scipy.optimize import minimize

import matscipytest


class TestDislocation(matscipytest.MatSciPyTestCase):
    
    
    def test_core_position(self):
        
        dft_alat = 3.19

        dft_C11 = 488
        dft_C12 = 200
        dft_C44 = 137


        dft_elastic_param = [dft_alat, 
                             dft_C11, 
                             dft_C12, 
                             dft_C44]


        cent_x = np.sqrt(6.0)*dft_alat/3.0
        center = (cent_x, 0.0, 0.0)

        # make othe cell with dislocation core not in center

        disloc, bulk, u = sd.make_screw_cyl(dft_alat, dft_C11, dft_C12, dft_C44,
                                            cylinder_r=40,
                                            center=center)
        res = minimize(sd.cost_function,
                       (1.0, 0.5),
                       args=(disloc, 
                             bulk, 
                             40, 
                             dft_elastic_param,
                             False), 
                       method='Nelder-Mead')

        self.assertArrayAlmostEqual(res.x, center[:2], tol=1e-4)

if __name__ == '__main__':
    unittest.main()
