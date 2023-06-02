import numpy as np
from ase.lattice.cubic import Diamond

from ase.optimize import LBFGS

from scipy.linalg import polar, sqrtm

from itertools import permutations

class CubicCauchyBorn:
    def __init__(self,el,a0,calc,lattice=Diamond,term=2):
        self.el = el
        self.a0 = a0
        self.calc = calc
        self.lattice = lattice
        self.num_terms = term
    

    def fit_taylor(self,de=1e-4):
        calc = self.calc
        def f_gl(E_vec,calc,unitcell):
            #green lagrange version of function
            #turn E into matrix
            E = np.zeros([3,3])
            E[0,0] = E_vec[0]
            E[1,1] = E_vec[1]
            E[2,2] = E_vec[2]
            E[2,1],E[1,2] = E_vec[3],E_vec[3]
            E[2,0],E[0,2] = E_vec[4],E_vec[4]
            E[0,1],E[1,0] = E_vec[5],E_vec[5]
            # get U^2
            Usqr = 2*E + np.eye(3)
            #square root matrix
            U = sqrtm(Usqr,disp=True)

            #this is just the symmetric stretch tensor, exactly what we need.
            x = U


            initial_shift = np.zeros([3])
            relaxed_shift = np.zeros([3])

            #build cell
            unitcell_copy = unitcell.copy()
            unitcell_copy.calc = self.calc
            cell = unitcell_copy.get_cell()
            cell_rescale = x[:,:]@cell
            unitcell_copy.set_cell(cell_rescale,scale_atoms=True)
            initial_shift[:] = unitcell_copy.get_positions()[1] - unitcell_copy.get_positions()[0]

            #relax cell
            opt = LBFGS(unitcell_copy,logfile=None)
            opt.run(fmax = 1e-10)
            relaxed_shift[:] = unitcell_copy.get_positions()[1] - unitcell_copy.get_positions()[0]

            shift_diff = relaxed_shift-initial_shift
            #print(shift_diff)
            # THIS CAN BE SPED UP
            # TECHNICALLY, EACH CALL OF THIS FUNCTION CAN BE USED
            # TO COMPUTE 3 DERIVATIVES RATHER THAN JUST 1
            # IF YOU EXPLOIT THE SYMMETRY OF CUBIC SYSTEMS
            # WOULD MAKE THINGS 3x FASTER, BUT WOULD REQUIRE 
            # *SOME* CODING
            return shift_diff[0] #return 1 component

        
        def find_grad(eps,i,*args):
            #compute gradient of function by varying a single component of eps
            eps_up = eps.copy()
            eps_down = eps.copy()
            eps_up[i]+=self.de
            eps_down[i]-=self.de
            return (f_gl(eps_up,*args) - f_gl(eps_down,*args))/(2*self.de)

        def find_second_deriv(eps,i,j,*args):
            #find the second derivative of f
            #by varying 2 components of eps
            eps_up = eps.copy()
            eps_down = eps.copy()
            eps_up[j]+=self.de
            eps_down[j]-=self.de
            return (find_grad(eps_up,i,calc,unitcell)-find_grad(eps_down,i,calc,unitcell))/(2*self.de)


        def find_third_deriv(eps,i,j,k,*args):
            eps_up = eps.copy()
            eps_down = eps.copy()
            eps_up[k]+=self.de
            eps_down[k]-=self.de
            return (find_second_deriv(eps_up,i,j,calc,unitcell)-find_second_deriv(eps_down,i,j,calc,unitcell))/(2*self.de)

        def find_fourth_deriv(eps,i,j,k,l,*args):
            eps_up = eps.copy()
            eps_down = eps.copy()
            eps_up[l]+=self.de
            eps_down[l]-=self.de
            return (find_third_deriv(eps_up,i,j,k,calc,unitcell)-find_third_deriv(eps_down,i,j,k,calc,unitcell))/(2*self.de)


        self.de = de
        unitcell = self.lattice(self.el, latticeconstant=self.a0, size=[1, 1, 1])
        unitcell.set_pbc([True,True,True])

        #compute all the terms we need:
        grad_f = np.zeros([6])
        hess_f = np.zeros([6,6])
        third_deriv_f = np.zeros([6,6,6])
        fourth_deriv_f = np.zeros([6,6,6,6])
        eps_vec = np.zeros([6])
        for i in range(6):
            grad_f[i] = find_grad(eps_vec,i,calc,unitcell)
            #finds the gradient of the gradient of f with e[i] with respect to each strain component e[j]
            for j in range(6-i):
                hess_f[i,j] = find_second_deriv(eps_vec,i,j,calc,unitcell)
                #include relevant symmetries
                hess_f[j,i] = hess_f[i,j]
                for k in range(6-j):
                    third_deriv_f[i,j,k] = find_third_deriv(eps_vec,i,j,k,calc,unitcell)
                    third_deriv_f[i,k,j] = third_deriv_f[i,j,k]
                    third_deriv_f[j,i,k] = third_deriv_f[i,j,k]
                    third_deriv_f[j,k,i] = third_deriv_f[i,j,k]
                    third_deriv_f[k,i,j] = third_deriv_f[i,j,k]
                    third_deriv_f[k,j,i] = third_deriv_f[i,j,k]
                    for l in range(6-k):
                        fourth_deriv_f[i,j,k,l] = find_fourth_deriv(eps_vec,i,j,k,l,calc,unitcell)
                        a = [i,j,k,l]
                        perms = set(permutations(a))
                        for p in perms:
                            (m,n,o,p) = p
                            fourth_deriv_f[m,n,o,p] = fourth_deriv_f[i,j,k,l]

        self.grad_f = grad_f
        self.hess_f = hess_f
        self.third_deriv_f = third_deriv_f
        self.fourth_deriv_f = fourth_deriv_f

    def save_taylor(self):
        try:
            np.save('grad_f.npy',self.grad_f)
            np.save('hess_f.npy',self.hess_f)
            np.save('third_deriv_f.npy',self.third_deriv_f)
            np.save('fourth_deriv_f.npy',self.fourth_deriv_f)
        except AttributeError:
            print('Need to fit a model with fit taylor before it can be saved!')

    def load_taylor(self):
        try:
            self.grad_f = np.load('grad_f.npy')
            self.hess_f = np.load('hess_f.npy')
            self.third_deriv_f = np.load('third_deriv_f.npy')
            self.fourth_deriv_f = np.load('fourth_deriv_f.npy')
        except FileNotFoundError:
            print('No files found containing model parameters - make sure to call save_taylor() to save the model.')
        
        
        

    
    def predict_shifts(self,A,atoms,F_func=None,E_func=None,coordinates='cart3D',*args,**kwargs):
        '''
        F_func - function to calculate the deformation field tensors at
        a vector of x,y,z, r,theta,(z), or r, theta, phi based on what
        is passed to coordinates. Note that F must return a vector of shape
        [number of atoms, 2, 2] for 2D, and [number of atoms, 3, 3] for 3D,
        and the returned matrix must be expressed in cartesian coordinates. Optional, but must be 
        given if E is not specified.

        E_func - function to directly calculate the get the strain tensors applied to each atom
        in the case where this is easier than F. Strain tensors must be returned of the form
        [number of atoms, 3, 3]. Optional, but must be given if F is not specified. 

        A - a matrix which holds the axes of the lab frame described in
        the lattice frame, of the form (x^T,y^T,z^T), where x,y and z are the
        normalised lab frame axes defined with respect to the cubic lattice basis.
        This acts as a rotation matrix. For example
        if our lab frame axis were x, y, z, which were defined
        by x = 1/sqrt(2)[110], y = 1/sqrt(2)[-110], z = [001] with respect to our crystal lattice
        A would be:
        [[1/sqrt(2)   -1/sqrt(2)     0]
         [1/sqrt(2)    1/sqrt(2)     0]
         [0            0             1]]

        coordinates - the form of atomic coordinates that need to be passed
        to the function F. Can be cart2D, cart3D, cylind2D, cylind3D, spherical.
        
        *args, **kwargs - anything to additionally be passed to F_func or E_func
        '''
        natoms = len(atoms)
        if F_func is not None:
            if E_func is not None:
                print('Need to only provide one of E or F, not both')
                raise ValueError
    
            if coordinates=='cylind2D':
                sx, sy, sz = atoms.cell.diagonal()
                x, y = atoms.positions[:, 0], atoms.positions[:, 1]
                cx, cy = sx/2, sy/2
                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                theta = np.arctan2((y-cy),(x-cx))
                F_2D = F_func(r,theta,*args,**kwargs)
                #pad F into a 3x3 matrix
                F_3D = (np.zeros([natoms,3,3]))
                F_3D[:,0:2,0:2] = F_2D[:,:,:]
                F_3D[:,2,2] = 1

            elif coordinates=='cylind3D':
                sx, sy, sz = atoms.cell.diagonal()
                x, y, z = atoms.positions[:,0],atoms.positions[:,1],atoms.positions[:,2]
                cx, cy, cz = sx/2, sy/2, sz/2
                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                theta = np.arctan2((y-cy),(x-cx))
                F_3D = F_func(r,theta,z,*args,**kwargs)

            elif coordinates=='spherical':
                raise NotImplementedError

            elif coordinates=='cart2D':
                x, y = atoms.positions[:, 0], atoms.positions[:, 1]
                F_2D = F_func(x,y,*args,**kwargs)            
                F_3D = (np.zeros([natoms,3,3]))
                F_3D[:,0:2,0:2] = F_2D[:,:,:]
                F_3D[:,2,2] = 1
            
            elif coordinates=='cart3D':
                x, y, z = atoms.positions[:, 0], atoms.positions[:, 1], atoms.positions[:,2]
                F_3D = F_func(x,y,z,*args,**kwargs)
                
            else:
                print('Coordinate system should be cart, cylind2D, cylind3D, spherical')
                raise NotImplementedError


            #transform F from lab frame to lattice frame,
            #find right handed stretch tensor U and rotation matrix
            #R via polar decomposition, then use these to find 
            #green-lagrange strain tensors E.

            Fprime = np.zeros_like(F_3D)
            R = np.zeros_like(F_3D)
            U = np.zeros_like(F_3D)
            E = np.zeros_like(F_3D)
            for i in range(natoms):
                Fprime[i,:,:] = A@F_3D[i,:,:]@np.transpose(A)
                R[i,:,:],U[i,:,:] = polar(Fprime[i,:,:])
                E[i,:,:] = 0.5*(U[i,:,:]@(np.transpose(U[i,:,:])) - np.eye(3))
            
        elif E_func is not None:
            x, y, z = atoms.positions[:, 0], atoms.positions[:, 1], atoms.positions[:,2]
            E_3D_lab = E_func(x,y,z,*args,**kwargs)
            E = np.zeros_like(E_3D_lab)
            R = np.zeros_like(E_3D_lab)
            #for a specified strain, R is just I for all atoms
            for i in range(natoms):
                E[i,:,:] = A@E_3D_lab[i,:,:]@np.transpose(A)
                R[i,:,:] = np.eye(3)
            

        
        else:
            print('Error, please provide one of E or F')
            raise ValueError

        #get the cauchy born shifts unrotated
        shifts_no_rr = self.evaluate_shift_model(E)

        shifts = np.zeros_like(shifts_no_rr)

        #rotate the cauchy shifts both by the rotation induced by F
        #and to get them back into the lab frame
        for i in range(natoms):
            shifts[i,:] = np.transpose(A)@R[i,:,:]@shifts_no_rr[i,:]
        

        #FUDGE, FIX THIS
        #shifts[:,1] = -shifts[:,1] #invert the y component ??? WHY


        return shifts

    def evaluate_shift_model(self,E):
        #takes E in the form of a list of matrices [natoms, :, :]
    
        #define function for evaluating model
        def eval_tay_model(self,eps):
            grad_f = self.grad_f
            hess_f = self.hess_f
            third_deriv_f = self.third_deriv_f
            fourth_deriv_f = self.fourth_deriv_f
            #takes E as a 2D array of voigt vectors, with number of rows=number atoms
            # Note that the permutation is computed from observing how the matrix components permute
            # when transforming the strain matrix by the 120 degree rotation matrices about [111],
            # such that the axis are permuted in the order
            # 1 ---> 2 ---> 3
            def permutation(strain_vec,perm_shift):
                strain_vec_perm = np.zeros_like(strain_vec)
                for i in range(3):
                    perm_num = (i-perm_shift)%3
                    strain_vec_perm[:,perm_num] = strain_vec[:,i]
                    strain_vec_perm[:,perm_num+3] = strain_vec[:,i+3]
                return strain_vec_perm

            def eval_eps(eps_vec,grad_f,hess_f,third_deriv_f):
                term_1 = grad_f@eps_vec
                term_2 = (1/2)*(np.transpose(eps_vec)@hess_f@eps_vec)
                term_3 = (1/6)*(np.einsum('ijk,i,j,k',third_deriv_f,eps_vec,eps_vec,eps_vec))
                term_4 = (1/24)*(np.einsum('ijkl,i,j,k,l',fourth_deriv_f,eps_vec,eps_vec,eps_vec,eps_vec))

                if self.num_terms == 1:
                    return term_1
                elif self.num_terms == 2:
                    return term_1 + term_2
                elif self.num_terms == 3:
                    return term_1 + term_2 + term_3
                elif self.num_terms == 4:
                    return term_1 + term_2 + term_3 + term_4
                else:
                    print('Error, can only return 1, 2, 3 or 4 terms')
                    raise ValueError

            epsx = eps
            epsy = permutation(eps,1)
            epsz = permutation(eps,2)
            predictions = np.zeros([np.shape(eps)[0],3])
            for i in range(np.shape(eps)[0]):
                predictions[i,0] = eval_eps(epsx[i,:],grad_f,hess_f,third_deriv_f)
                predictions[i,1] = eval_eps(epsy[i,:],grad_f,hess_f,third_deriv_f)
                predictions[i,2] = eval_eps(epsz[i,:],grad_f,hess_f,third_deriv_f)
            
            return predictions


        #turn E into voigt vector
        E_voigt = np.zeros([np.shape(E)[0],6])
        E_voigt[:,0] = E[:,0,0]
        E_voigt[:,1] = E[:,1,1]
        E_voigt[:,2] = E[:,2,2]
        E_voigt[:,3] = E[:,1,2]
        E_voigt[:,4] = E[:,0,2]
        E_voigt[:,5] = E[:,0,1]


        #return the shift vectors
        return eval_tay_model(self,E_voigt)