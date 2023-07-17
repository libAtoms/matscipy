import unittest
import numpy as np
from matscipy.cauchy_born import CubicCauchyBorn
from ase.build import bulk
from ase.lattice.cubic import Diamond
from ase.optimize.precon import PreconLBFGS
from ase.constraints import ExpCellFilter
from matscipy.cauchy_born import CubicCauchyBorn
from matscipy.calculators.manybody.explicit_forms.stillinger_weber import StillingerWeber, Holland_Marder_PRL_80_746_Si
from matscipy.calculators.manybody import Manybody
from scipy.linalg import sqrtm
import ase.io
import matscipytest


class TestPredictCauchyBornShifts(matscipytest.MatSciPyTestCase):
    """
    Tests of Cauchy-Born shift prediction in the case of multilattices.

    We test with a cubic Silicon diamond multi-lattice
    """
    def setUp(self):
        el = 'Si' # chemical element
        si = Diamond(el, latticeconstant=5.43, size=[1, 1, 1], 
                       directions=[[1,0,0],[0,1,0],[0,0,1]])
        calc = Manybody(**StillingerWeber(Holland_Marder_PRL_80_746_Si))
        si.calc = calc
        ecf = ExpCellFilter(si)
        opt = PreconLBFGS(ecf)
        opt.run(fmax=1e-6, smax=1e-6)
        #print(si.get_positions())
        a0 = si.cell[0, 0]
        #print(a0)
        self.cb = CubicCauchyBorn(el,a0,calc,lattice=Diamond)
        self.unit_cell = si


    def test_set_sublattice(self):
        A = np.eye(3)
        self.cb.set_sublattices(self.unit_cell,A)
        assert np.array_equal(self.cb.lattice1mask,np.array([True,False,True,False,True,False,True,False],dtype=bool))
        assert np.array_equal(self.cb.lattice2mask,np.array([False,True,False,True,False,True,False,True],dtype=bool))

    def test_fit_taylor_model(self):
        self.cb.fit_taylor()
        #print(self.cb.grad_f)
        #print(self.cb.hess_f)
        grad_f = np.array([0.00000000e+00,0.00000000e+00,0.00000000e+00,-1.09995082e+00,-2.22044605e-12,-3.33066907e-12])
        hess_f = np.array([[0.00000000e+00,0.00000000e+00,0.00000000e+00,2.03958794e-01,0.00000000e+00,1.11022302e-08],
                        [0.00000000e+00,0.00000000e+00,0.00000000e+00,9.12311482e-01,-5.55111512e-09,0.00000000e+00],
                        [0.00000000e+00,0.00000000e+00,0.00000000e+00,9.12311465e-01,0.00000000e+00,0.00000000e+00],
                        [2.03958794e-01,9.12311493e-01,9.12311465e-01,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                        [0.00000000e+00,-5.55111512e-09,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                        [1.11022302e-08,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00]])
        assert np.allclose(self.cb.grad_f,grad_f,atol=1e-8)
        assert np.allclose(self.cb.hess_f,hess_f,atol=1e-8)

    def E_cart3D(self,x,y,z,eps=None):
        E = np.zeros([len(x),3,3])
        E[:,0,0] = eps[0]
        E[:,1,1] = eps[1]
        E[:,2,2] = eps[2]
        E[:,1,2], E[:,2,1] = eps[3],eps[3]
        E[:,0,2], E[:,2,0] = eps[4],eps[4]
        E[:,0,1], E[:,1,0] = eps[5],eps[5]
        return E

    def E_cylind3D(self,r,theta,z,eps=None):
        E = np.zeros([len(r),3,3])
        E[:,0,0] = eps[0]
        E[:,1,1] = eps[1]
        E[:,2,2] = eps[2]
        E[:,1,2], E[:,2,1] = eps[3],eps[3]
        E[:,0,2], E[:,2,0] = eps[4],eps[4]
        E[:,0,1], E[:,1,0] = eps[5],eps[5]
        return E
        
    def F_cart3D(self,x,y,z,eps=None):
        E = np.zeros([len(x),3,3])
        E[:,0,0] = eps[0]
        E[:,1,1] = eps[1]
        E[:,2,2] = eps[2]
        E[:,1,2], E[:,2,1] = eps[3],eps[3]
        E[:,0,2], E[:,2,0] = eps[4],eps[4]
        E[:,0,1], E[:,1,0] = eps[5],eps[5]
        U = sqrtm((2*E[i,:,:])+np.eye(3))
        return U #with no rigid rotation, U is F
        
    def F_cylind3D(self,r,theta,z,eps=None):
        E = np.zeros([len(r),3,3])
        E[:,0,0] = eps[0]
        E[:,1,1] = eps[1]
        E[:,2,2] = eps[2]
        E[:,1,2], E[:,2,1] = eps[3],eps[3]
        E[:,0,2], E[:,2,0] = eps[4],eps[4]
        E[:,0,1], E[:,1,0] = eps[5],eps[5]
        U = sqrtm((2*E[i,:,:])+np.eye(3))
        return U #with no rigid rotation, U is F
       

    def model_prediction(self,dirs,eps,method,E_func=None,F_func=None,coordinates='cart3D',atol=1.5e-5,returnvals=False):
        #print(dirs,coordinates)
        #Get rotation matrix from dirs (A)
        A = np.zeros([3,3])
        for i, direction in enumerate(dirs):
            direction = np.array(direction)
            direction = (1/np.linalg.norm(direction))*direction
            A[:,i] = direction
        
        #generate rotated unitcell (according to dirs)
        atoms = Diamond(self.cb.el, latticeconstant=self.cb.a0, size=[1, 1, 1],directions=dirs)
        atoms.set_pbc([True,True,True])
        atoms.calc = self.cb.calc
        atoms_copy = atoms.copy()
        self.cb.set_sublattices(atoms,A)
        shift_diff_true = self.cb.eval_shift(eps,atoms) #lab frame shift diff
        if E_func is not None:
            shifts = self.cb.predict_shifts(A,atoms,E_func=E_func,eps=eps,method=method,coordinates=coordinates)
            shift_diff_predicted = shifts[0,:]
        elif F_func is not None:
            shifts = self.cb.predict_shifts(A,atoms,F_func=F_func,eps=eps,method=method,coordinates=coordinates)
            shift_diff_predicted = shifts[0,:]
        #print(shift_diff_predicted,shift_diff_true)
        #print(shift_diff_predicted-shift_diff_true)
        assert np.allclose(shift_diff_predicted,shift_diff_true,atol=atol)
        if returnvals:
            return atoms_copy, shifts,np.linalg.norm(shift_diff_predicted-shift_diff_true),A


    def test_taylor_model(self):
        self.cb.fit_taylor()
        eps_vals = [np.array([0,0,0,0.0001,0,0]),
                    np.array([0.0001,0,0.0001,0.0001,0.0001,0]),
                    np.array([0,0,0,0.0001,0,0]),
                    np.array([0,0,0,0.0001,0,0])]

        dir_vals = [[[1,0,0],[0,1,0],[0,0,1]],
                    [[1,1,0],[-1,1,0],np.cross([1,1,0],[-1,1,0])],
                    [[1,1,1],[-2,1,1],np.cross([1,1,1],[-2,1,1])],
                    [[1,1,0],[-1,1,0],np.cross([1,1,0],[-1,1,0])]]
        
        funcs = [self.E_cart3D,self.E_cart3D,self.E_cart3D,self.E_cylind3D]
        coords = ['cart3D','cart3D','cart3D','cylind3D']
        for i in range(len(funcs)):
            self.model_prediction(dir_vals[i],eps_vals[i],method='taylor',E_func=funcs[i],coordinates=coords[i],atol=1e-7)

    def test_fit_regression_model(self):
        self.cb.initial_regression_fit()

    def test_regression_model(self):
        self.cb.initial_regression_fit()
        eps_vals = [np.array([0,0,0,0.0001,0,0]),
                    np.array([0.0001,0,0.0001,0.0001,0.0001,0]),
                    np.array([0,0,0,0.0001,0,0]),
                    np.array([0,0,0,0.0001,0,0])]

        dir_vals = [[[1,0,0],[0,1,0],[0,0,1]],
                    [[1,1,0],[-1,1,0],np.cross([1,1,0],[-1,1,0])],
                    [[1,1,1],[-2,1,1],np.cross([1,1,1],[-2,1,1])],
                    [[1,1,0],[-1,1,0],np.cross([1,1,0],[-1,1,0])]]
        
        funcs = [self.E_cart3D,self.E_cart3D,self.E_cart3D,self.E_cylind3D]
        coords = ['cart3D','cart3D','cart3D','cylind3D']
        for i in range(len(funcs)):
            self.model_prediction(dir_vals[i],eps_vals[i],method='regression',E_func=funcs[i],coordinates=coords[i],atol=1e-6)


    def test_regression_model_refit(self):
        self.cb.initial_regression_fit()
        eps = np.array([0.01,0,0.01,0.01,0,-0.01])
        dirs = [[1,1,1],[-2,1,1],np.cross([1,1,1],[-2,1,1])]
        func = self.E_cart3D
        coordinates = 'cart3D'
        atoms,shifts,shift_err_before,A = self.model_prediction(dirs,eps,method='regression',E_func=func,coordinates=coordinates,atol=1e-4,returnvals=True)
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.cb.calc
        self.cb.apply_shifts(atoms_copy,shifts)
        forces_after = atoms_copy.get_forces()
        converged,err = self.cb.check_for_refit(A,atoms_copy,forces_after,E_func=func,eps=eps,tol=1e-4,refit_points=2)
        atoms,shifts,shift_err_after,A = self.model_prediction(dirs,eps,method='regression',E_func=func,coordinates=coordinates,atol=1e-4,returnvals=True)
        assert shift_err_after<shift_err_before
        #print(shift_err_before,shift_err_after)

if __name__ == '__main__':
    unittest.main()

