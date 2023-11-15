import matscipy; print(matscipy.__file__)
from matscipy import parameter

from mpi4py import MPI
from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack

from ase.units import GPa
import numpy as np

from copy import deepcopy

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
max_opt_steps = parameter('max_opt_steps', 100)
max_steps = parameter('max_arc_steps', 10)
vacuum = parameter('vacuum', 10.0)
flexible = parameter('flexible', True)
continuation = parameter('continuation', False)
ds = parameter('ds', 1e-2)
nsteps = parameter('nsteps', 10)
a0 = parameter('a0') # lattice constant
k0 = parameter('k0', 1.0)
extended_far_field = parameter('extended_far_field', False)
alpha0 = parameter('alpha0', 0.0) # initial guess for crack position
dump = parameter('dump', False)
precon = parameter('precon', False)
traj_file = parameter('traj_file', 'x_traj.h5')
restart_file = parameter('restart_file', traj_file)
traj_interval = parameter('traj_interval', 1)
direction = parameter('direction', +1)
ds_max = parameter('ds_max', 0.1)
ds_min = parameter('ds_min', 1e-6)
ds_aggressiveness=parameter('ds_aggressiveness', 1.25)
opt_method=parameter('opt_method', 'krylov')
cb = parameter('cb', 'None')
rI = parameter('r_I')
rIII = parameter('r_III')
cutoff = parameter('cutoff')
dk = parameter('dk', 1e-4)
dalpha = parameter('dalpha', 1e-1)
explore_direction = parameter('explore_direction', 0)
allow_alpha_backtracking = parameter('allow_alpha_backtracking',False)
k1_curve_file = parameter('k1_curve_file')
num_data_points = parameter('num_data_points')
follow_G_contour = parameter('follow_G_contour',False)
folder_name = parameter('folder_name','data')
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

cryst = params.cryst.copy()

pos = cryst.get_positions()
sx, sy, sz = cryst.cell.diagonal()
xpos = pos[:,0] - sx/2
ypos = pos[:,1] - sy/2

crk = CubicCrystalCrack(parameter('crack_surface'),
                parameter('crack_front'),
                C=parameter('C')/GPa,cauchy_born=cb)


k1g = crk.k1g(parameter('surface_energy'))
print('griffthk1,',k1g)
cluster = params.cluster.copy() 
if crk.cauchy_born is not None:
    crk.cauchy_born.set_sublattices(cluster,np.transpose(crk.RotationMatrix),read_from_atoms=True)


sc = SinclairCrack(crk, cluster, calc, k0 * k1g,
                alpha=alpha0,
                vacuum=vacuum,
                variable_alpha=flexible,
                extended_far_field=extended_far_field,rI=rI,
                rIII=rIII,cutoff=cutoff,incl_rI_f_alpha=True)
sc.k1g = k1g
sc.variable_alpha = True #True
sc.variable_k = True
sc.cont_k = 'k2'


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
    
for i in range(num_data_points):
    assert sc.cont_k == 'k2'
    #pull kII0 from x
    sc.variable_k = True
    x0 = x[i,:]
    sc.set_dofs(x0)
    sc.variable_k = False
    converged=False
    dkcurr = dk
    for attempt in range(10):
        kII0 = x[i,-1]
        k_x0 = kII0/sc.k1g
        alpha_x0 = x[i,-3]
        print(f'Rescaling K_II from {kII0} to {kII0 + dkcurr}')
        k_x1 = k_x0 + dkcurr
        sc.kII = k_x1*sc.k1g
        if follow_G_contour:
            # print('before',sc.kI)
            sc.kI = np.sqrt(((k_x0*sc.k1g)**2 + sc.kI**2) - (k_x1*sc.k1g)**2)
            # print('after',sc.kI)
            time.sleep(5)
        
        sc.update_atoms()
        print('starting search...')
        try:
            sc.optimize(ftol=0.0001, steps=100,method='krylov')
            converged=True
            break
        except RuntimeError:
            print('No convergence, retrying with smaller k step')
            dkcurr = dkcurr/2
    
    if converged == False:
        print('No convergence, skipping this point')
        continue
    x1 = np.r_[sc.get_dofs(), k_x1 * sc.k1g]
    alpha_x1 = sc.alpha
    print(f'k0={k_x0}, k1={k_x1} --> alpha0={alpha_x0}, alpha1={alpha_x1}')

    #start that continuation
    sc.variable_k = True
    traj_file_name = f'{folder_name}/main_process_{pid}_sub_process_{rank}_point_{i}.h5'
    print('starting ncflex...')
    sc.arc_length_continuation(x0, x1, N=nsteps,
                    ds=ds, ds_max=0.05, ftol=fmax, max_steps=10,
                    direction=explore_direction,
                    continuation=continuation,
                    traj_file=traj_file_name,
                    traj_interval=traj_interval,
                    precon=precon,opt_method='krylov',parallel=False,
                    allow_alpha_backtracking=allow_alpha_backtracking,
                    follow_G_contour=follow_G_contour)

