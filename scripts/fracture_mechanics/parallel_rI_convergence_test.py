import matscipy; print(matscipy.__file__)
from matscipy import parameter

from mpi4py import MPI
from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack

from ase.units import GPa
import numpy as np

import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import params
import os

import time

num_processors = parameter('num_processors')
#read parameters in from params file
calc = parameter('calc')
fmax = parameter('fmax', 1e-3)
cb = parameter('cb', 'None')
r_I_vals = parameter('r_I_vals')
r_III_vals = parameter('r_III_vals')
cutoff = parameter('cutoff')
k1_curve_file = parameter('k1_curve_file')
num_data_points = parameter('num_data_points')
follow_G_contour = parameter('follow_G_contour',False)
folder_name = parameter('folder_name','data')
a0 = parameter('a0') # lattice constant
k0 = parameter('k0', 1.0)
alpha0 = parameter('alpha0', 0.0) # initial guess for crack position
traj_file = parameter('traj_file', 'x_traj.h5')
extended_far_field = parameter('extended_far_field', False)
vacuum = parameter('vacuum', 10.0)
flexible = parameter('flexible', True)
cont_k = parameter('cont_k','k1')
clusters = parameter('clusters')#get full set of clusters being used

if cb == 'None':
    cb = None
    
if cb is not None:
    if rank == 0:
        #cb.initial_regression_fit()
        #cb.save_regression_model()
        cb.load_regression_model()
        cb_mod = cb.get_model()
        #communicate cb model to all other processors using mpi broadcast
    else:
        cb_mod = 0
    cb_mod = comm.bcast(cb_mod, root=0)
    cb.set_model(cb_mod)

crk = CubicCrystalCrack(parameter('crack_surface'),
                parameter('crack_front'),
                C=parameter('C')/GPa,cauchy_born=cb)

k1g = crk.k1g(parameter('surface_energy'))
print('griffthk1,',k1g)

if rank == 0:
    #now read in the initial data file and start to send over data
    hf = h5py.File(k1_curve_file, 'r')
    x_traj = hf['x']
    x = x_traj[:,:]
    hf.close()

    total_data_points = np.shape(x)[0]
    sample_frac = int(total_data_points/num_data_points)
    x = x[::sample_frac,:]

    data_points_per_proc = int(np.floor(np.shape(x)[0]/num_processors))
    #k2 distance to walk size

    num_data_point_array = [data_points_per_proc for i in range(num_processors)]
    sum_diff = np.shape(x)[0] - np.sum(num_data_point_array)
    for i in range(sum_diff):
        num_data_point_array[i] += 1
    # print(num_data_point_array)
    time.sleep(3)
    #now communicate over all data
    for proc_num in range(1,num_processors):
        # print(proc_num)
        comm.send(num_data_point_array[proc_num], dest=proc_num, tag=12)
        comm.send(x[sum(num_data_point_array[:proc_num]):sum(num_data_point_array[:proc_num+1]),:], dest=proc_num, tag=10)
        comm.send(os.getpid(),dest=proc_num,tag=13)
    #the first set of data (proc_num 0) is assigned to rank 0 (this processor)
    num_data_points = num_data_point_array[0]
    x = x[0:num_data_points,:]
    pid = os.getpid()
else:
    num_data_points = comm.recv(source=0, tag=12)
    x = comm.recv(source=0, tag=10)

    #recieve pid
    pid = comm.recv(source=0, tag=13)

for dp in range(num_data_points):
    #each process creates a file titled f'{pid}_{x[dp][-1]}'
    #this file contains the data for the current data point
    #the file is created in the folder f'{folder_name}'

    #create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    #create file, with names rouneded to 3dp

    file_name = f'{folder_name}/{pid}_rank_{rank}_point_{dp}.h5'
    #create file
    hf = h5py.File(file_name,'w')
    for i, curr_cluster in enumerate(clusters):
        # print(r_I_vals[i],r_III_vals[i])
        cluster = curr_cluster.copy()
        if crk.cauchy_born is not None:
            crk.cauchy_born.set_sublattices(cluster,np.transpose(crk.RotationMatrix),read_from_atoms=True)

        sc = SinclairCrack(crk, cluster, calc, k0 * k1g,
                    alpha=alpha0,
                    vacuum=vacuum,
                    variable_alpha=flexible,
                    extended_far_field=extended_far_field,rI=r_I_vals[i],
                    rIII=r_III_vals[i],cutoff=cutoff,incl_rI_f_alpha=True)
        
        sc.write_atoms_to_file()
        sc.k1g = k1g
        sc.variable_alpha = True #True
        sc.variable_k=True
        sc.cont_k = cont_k

        if i == 0:
            #on the first iteration, just set prev_relaxed to the initial config
            #and pass
            x_initial = x[dp]
            sc.set_dofs(x_initial[:])
            #sc.write_atoms_to_file()
            #check if both kI and kII are in x[dp], if they are then
            #remove whichever is involved in continuation
            if len(x_initial) == len(sc)+1:
                if sc.cont_k == 'k1':
                    idx = -1
                    sc.kII = x_initial[idx]
                    kII0 = sc.kII
                elif sc.cont_k == 'k2':
                    idx = -2
                    sc.kI = x_initial[idx]
                    kI0 = sc.kI
                x_initial = np.delete(x_initial,idx)
            
            prev_u, prev_alph, prev_k = sc.unpack(x_initial[:],reshape=True)
            continue

        #now take sc.atoms, mask out the inner r_I_vals[i-1] atoms, and set to
        sx, sy, sz = cluster.cell.diagonal()
        pos_rI = cluster.get_positions()[sc.regionI]
        xcoord, ycoord = pos_rI[:, 0], pos_rI[:, 1]
        cx, cy = sx/2, sy/2
        r = np.sqrt((xcoord - cx)**2 + (ycoord - cy)**2)
        mask = r<r_I_vals[i-1]
        sc.alpha = prev_alph
        if sc.cont_k == 'k1':
            sc.kI = prev_k
            sc.kII = kII0
        elif sc.cont_k == 'k2':
            sc.kII = prev_k
            sc.kI = kI0
        
        sc.u[mask] = prev_u
        sc.update_atoms()
        
        #sc.write_atoms_to_file()
        sc.variable_k=False
        #now relax
        try:
            sc.optimize(ftol=0.0001, steps=100,method='krylov')
        except RuntimeError:
            #proceed to next data point
            break
        sc.write_atoms_to_file()
        #get out data
        sc.variable_k = True

        x_new = sc.get_dofs()
        prev_u, prev_alph, prev_k = sc.unpack(x_new,reshape=True)

        #write x_new to a new file dataset called f'r_I_vals[i]'
        with h5py.File(file_name,'a') as hf:
            hf.create_dataset(f'r_I_{r_I_vals[i]}',data=x_new, compression='gzip')
        