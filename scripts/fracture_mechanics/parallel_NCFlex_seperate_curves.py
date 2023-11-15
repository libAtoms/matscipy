import matscipy; print(matscipy.__file__)

import os
import numpy as np

import h5py
import ase.io
from ase.units import GPa
from ase.constraints import FixAtoms
#from ase.optimize import LBFGSLineSearch
from ase.optimize.precon import PreconLBFGS
from ase.optimize import LBFGS

from matscipy import parameter
from matscipy.elasticity import  fit_elastic_constants
from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack

from scipy.optimize import fsolve, fminbound, minimize
from scipy.optimize.nonlin import NoConvergence

import sys
sys.path.insert(0, '.')

import multiprocessing
import params
import time
from ase.optimize.sciopt import OptimizerConvergenceError

from queue import Empty
from copy import deepcopy



def init_worker():
    global data_queue
    print('worker initialised')
    data_queue.put(os.getpid())

def assign_sc(sc):
    return [os.getpid(),sc]

def assign_calc(sc):
    #originally was passing around sc objects, but some
    #calculators aren't pickleable
    calc = params.new_calculator_instance()
    sc.calc = calc
    sc.atoms.calc = calc
    sc.cryst.calc = calc
    if sc.crk.cauchy_born is not None:
        sc.crk.cauchy_born.calc = calc

def delete_calc(sc):
    del sc.calc
    del sc.atoms.calc
    del sc.cryst.calc
    if sc.crk.cauchy_born is not None:
        del sc.crk.cauchy_born

def search(K0,alpha0,sc_dict):
    global status_queue, search_queue
    #set status as 'searching'

    status_queue.put([os.getpid(),'searching'],block=False)
    sc = sc_dict[os.getpid()]
    assign_calc(sc)

    # print(f'pid {os.getpid()} sc is {sc} sc calc is {sc.calc}')
    sc.variable_k = False
    k1g = sc.k1g
    sc.k = K0*k1g
    sc.alpha = alpha0
    sc.update_atoms()

    converged = False
    try:
        sc.optimize(ftol=0.0005, steps=1000,method='ode12r',precon=True)
    except OptimizerConvergenceError:
        print('did not converge fully in ode12r section, proceeding with krylov')

    try:
        sc.optimize(ftol=fmax, steps=100,method='krylov')
        converged = True
    except RuntimeError:
        print('No convergence')

    if converged:
        k_x0 = sc.k/k1g
        x0 = np.r_[sc.get_dofs(), k_x0 * k1g]
        alpha_x0 = sc.alpha

        print(f'Rescaling K_I from {sc.k} to {sc.k + dk * k1g}')
        k_x1 = k_x0 + dk
        # print(k0)
        sc.k = k_x1*k1g
        sc.update_atoms()
        sc.optimize(ftol=0.0001, steps=100,method='krylov')
        x1 = np.r_[sc.get_dofs(), k_x1 * k1g]
        alpha_x1 = sc.alpha
        print(f'k0={k_x0}, k1={k_x1} --> alpha0={alpha_x0}, alpha1={alpha_x1}')
        
    else:
        #return sentinal values
        #FIXME really this should do something more drastic
        x0 = [-10000,-10000]
        x1 = [-10000,-10000]

    #place result in search queue (blocking)
    print('Braodcasting search result')
    search_queue.put([os.getpid(),[x0,x1]])

    delete_calc(sc)
    #at end,
    #status_queue.put([os.getpid(),'idle'],block=False)

def walk(x0,x1,direction,sc_dict):
    global data_queue, status_queue

    status_queue.put([os.getpid(),'walking'],block=False)
    sc = sc_dict[os.getpid()]
    assign_calc(sc)
    #set status as 'walking'
    # print('here1')
    # print(f'pid {os.getpid()} sc is {sc} sc calc is {sc.calc}')

    #start NCFlex, passing the data queue and the pipe output
    # data queue sends back data to main
    # pipe output allows main to kill the process
    print('starting ncflex...')
    sc.variable_k = True
    precon = False
    # print('lammps lib process',os.getpid(),sc.calc)
    traj_file_name = f'curve_alph_{np.round(x0[-2],decimals=3)}_K_{np.round(x0[-1]/sc.k1g,decimals=3)}_dir_{direction}.h5'
    sc.arc_length_continuation(x0, x1, N=nsteps,
                            ds=ds, ds_max=0.05, ftol=fmax, max_steps=10,
                            direction=direction,
                            continuation=continuation,
                            traj_file=traj_file_name,
                            traj_interval=traj_interval,
                            precon=precon,opt_method='krylov',parallel=False,
                            allow_alpha_backtracking=allow_alpha_backtracking)

    #time.sleep(10)
    #go idle
    status_queue.put([os.getpid(),'idle'],block=False)
    delete_calc(sc)


def main(K_range,alpha_range):
    global data_queue, sc, status_queue,search_queue

    #kill old sc calculator
    del sc.calc
    del sc.atoms.calc
    del sc.cryst.calc
    if sc.crk.cauchy_born is not None:
        del sc.crk.cauchy_born.calc

    #create copies of sc for each process
    sc_array = [deepcopy(sc) for i in range(num_processors)]
    # print(sc_array)

    #now, start worker pool
    worker_pool = multiprocessing.Pool(num_processors,initializer=init_worker)

    #set up dictionary to hold each of the sc objects that workers need
    sc_dict = {}

    #set up dictionary to hold worker status
    worker_status = {}

    #get PID list
    queue_empty = False
    while not queue_empty:
        try:
            pid = data_queue.get(block=True,timeout=5)
            worker_status[pid] = 'idle'
        except Empty:
            queue_empty = True

    #assign scs
    # print('assigning SCs')
    for i in range(num_processors):
        #make a new calculator instance
        calc = params.new_calculator_instance()
        [pid,sc] = worker_pool.apply(assign_sc, args=(tuple([sc_array[i]])))
        sc_dict[pid] = sc
    # print(sc_dict)

    #launch workers on a non-blocking initial search
    #depending on search direction
    #generate the grid of initial points for exploration
    # print('explore direction',explore_direction)
    if (explore_direction == 1) or (explore_direction ==-1):
        launch_num = num_processors
    elif (explore_direction == 0):
        launch_num = int(num_processors/2)

    num_branches = int(np.floor((2*(alpha_range[1]-alpha_range[0]))/alpha_period))
    if num_branches == 0:
        num_K_vals = launch_num
    else:
        num_K_vals = int(np.floor(launch_num/num_branches))
    
    num_K_val_array = [num_K_vals for i in range(num_branches)]
    sum_diff = launch_num - np.sum(num_K_val_array)
    for i in range(sum_diff):
        num_K_val_array[i] += 1
    
    #FIXME this assumes that alpha_range > alpha_period/4 (should be usually)

    init_alpha_vals = np.array([alpha_range[0] + (alpha_period/4) + ((alpha_period/2)*i) for i in range(num_branches)])

    #alpha_range_buffer = (alpha_range[1]-alpha_range[0])*(1/(num_processors/2))
    #alphas = np.linspace(alpha_range[0]+alpha_range_buffer,alpha_range[1]-alpha_range_buffer,int(num_processors/2))
    #Ks = np.random.uniform(K_range[0],K_range[1],int(num_processors/2))

    #launch searches
    for i, alpha in enumerate(init_alpha_vals):
        if num_K_val_array[i] == 0:
            continue
        else:
            K_step = (K_range[1]-K_range[0])/num_K_val_array[i]
            initial_K_step = np.random.uniform(0,K_step)
            init_K_vals = np.array([K_range[0] + initial_K_step + K_step*p for p in range(num_K_val_array[i])])
            for K in init_K_vals:
                worker_pool.apply_async(search, args=(K,alpha,sc_dict))

    it_num = 0
    killed_num = 0
    curve_explored = False
    percentage_covered = 0
    time.sleep(5)
    while not curve_explored:
        it_num += 1
        #first, check the status queue to update any worker statuses. 
        #get PID list
        queue_empty = False
        #print('checking status queue')
        while not queue_empty:
            try:
                [pid,status] = status_queue.get(block=False)
                worker_status[pid] = status
            except Empty:
                queue_empty = True
        
        #Write worker status to file and then launch any new workers if necessary
        #print('writing to file')
        if it_num%1 == 0:
            with open('worker_status.txt', 'w') as f:
                for pid in worker_status:
                    f.write(f'{pid} : {worker_status[pid]}\n')
        #print('checking worker status')
        #check what workers are doing
        search_num = 0
        walk_num = 0
        idle_num = 0
        for pid in worker_status:
            if worker_status[pid] == 'idle':
                idle_num += 1
            elif worker_status[pid] == 'searching':
                search_num += 1
            elif worker_status[pid] == 'walking':
                walk_num += 1
        if it_num % 100 == 0:
            print(f'idle: {idle_num}, search, {search_num}, walk {walk_num}')
        
        #print('checking to launch new searches')
        #if there's unnaccounted for idle processes, launch new searches
        if explore_direction == 0:
            num_new_searches = int(np.floor(idle_num-search_num))
        elif (explore_direction == 1) or (explore_direction ==-1):
            num_new_searches = idle_num
        if num_new_searches>0:
            for i in range(num_new_searches):
                # print('LAUNCHING A NEW SEARCH')
                new_K = np.random.uniform(K_range[0],K_range[1])
                new_alpha = np.random.uniform(alpha_range[0],alpha_range[1])
                print('INITIAL K, ALPHA OF NEW SEARCH:',new_K,new_alpha)
                worker_pool.apply_async(search, args=(new_K,new_alpha,sc_dict))
                time.sleep(1)
            

        #print('checking for finished searches')
        #check for any finished searches in the search_queue
        #for any finished searches, start two walk jobs from the found position in each
        #direction (if the search position lies in the alpha range). If it doesn't, start a new search.
        queue_empty = False
        search_results = []
        while not queue_empty:
            try:
                item = search_queue.get(block=False)
                search_results.append(item)
            except Empty:
                queue_empty = True

        if len(search_results)>0:
            print('found finished search!')
            # print(search_results)
            for res in search_results:
                [pid,[x0,x1]] = res

                if len(x0) == 2:
                    #this indicates that the search failed. Immediately restart it
                    print('search failed, restarting')
                    new_K = np.random.uniform(K_range[0],K_range[1])
                    new_alpha = np.random.uniform(alpha_range[0],alpha_range[1])
                    worker_pool.apply_async(search, args=(new_K,new_alpha,sc_dict))
                    continue

                found_alpha = x0[-2]

                #set worker status to idle here rather than in function to prevent
                #new searches being started erroneously
                worker_status[pid] = 'idle'

                #if found alpha is out of range, start a new search
                if (found_alpha<alpha_range[0] or found_alpha>alpha_range[1]):
                    new_K = np.random.uniform(K_range[0],K_range[1])
                    new_alpha = np.random.uniform(alpha_range[0],alpha_range[1])
                    worker_pool.apply_async(search, args=(new_K,new_alpha,sc_dict))
                    continue
                
                if explore_direction == 0:
                    #start walk jobs
                    worker_pool.apply_async(walk,args=(x0,x1,-1,sc_dict))
                    worker_pool.apply_async(walk,args=(x0,x1,1,sc_dict))


                elif (explore_direction == 1) or (explore_direction ==-1):
                    #start walk job
                    worker_pool.apply_async(walk,args=(x0,x1,explore_direction,sc_dict))

    
    #kill all workers outside of while loop
    worker_pool.close()



if __name__ == '__main__':
    
    #multiprocessing configuration
    num_processors = parameter('num_processors')
    K_range = parameter('K_range')
    alpha_range = parameter('alpha_range')
    
    # Set start method to fork so 
    # each process inherits information from the parent process
    # this includes the queue through which information is passed
    multiprocessing.set_start_method('fork')
    global data_queue, pipe_setup_queue, status_queue, search_queue, kill_confirm_queue

    data_queue = multiprocessing.Queue()
    status_queue = multiprocessing.Queue()
    search_queue = multiprocessing.Queue()

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

    if cb == 'None':
        cb = None
    
    cryst = params.cryst.copy()

    pos = cryst.get_positions()
    sx, sy, sz = cryst.cell.diagonal()
    xpos = pos[:,0] - sx/2
    ypos = pos[:,1] - sy/2

    #find the closest y atoms by finding which atoms lie within 1e-2 of the min
    #filter out all atoms with x less than 0
    xmask = xpos>0
    closest_y_mask = np.abs(ypos)<(np.min(np.abs(ypos[xmask]))+(1e-2))


    #find the x positions of these atoms
    closest_x = xpos[closest_y_mask&xmask]

    #sort these atoms and find the largest x gap
    sorted_x = np.sort(closest_x)
    diffs = np.diff(sorted_x)
    alpha_period = np.sum(np.unique(np.round(np.diff(sorted_x),decimals=4)))
    # print('alpha_period',alpha_period)
    # setup the crack
    crk = CubicCrystalCrack(parameter('crack_surface'),
                    parameter('crack_front'),
                    C=parameter('C')/GPa,cauchy_born=cb) #TODO change back to cb
    
    #get k1g
    k1g = crk.k1g(parameter('surface_energy'))
    print('griffthk1,',k1g)
    cluster = params.cluster.copy() 
    if crk.cauchy_born is not None:
        crk.cauchy_born.set_sublattices(cluster,np.transpose(crk.RotationMatrix),read_from_atoms=True)

    global sc

    sc = SinclairCrack(crk, cluster, calc, k0 * k1g,
                    alpha=alpha0,
                    vacuum=vacuum,
                    variable_alpha=flexible,
                    extended_far_field=extended_far_field,rI=rI,
                    rIII=rIII,cutoff=cutoff,incl_rI_f_alpha=True)
    sc.k1g = k1g

    sc.variable_alpha = True #True
    sc.variable_k = True

    main(K_range,alpha_range)

