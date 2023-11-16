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

from queue import Empty
from copy import deepcopy

from ase.optimize.sciopt import OptimizerConvergenceError


class RegressionModel:
    def fit(self,alphas,Ks):
        phi = self.basis_function_evaluation(alphas)
        self.model = np.linalg.lstsq(phi,Ks,rcond=None)[0]
        #print(phi,self.model)
        
    def predict(self,alphas):
        phi = self.basis_function_evaluation(alphas)
        #print(phi,self.model)
        return phi @ self.model

    def basis_function_evaluation(self,alphas):
        if isinstance(alphas,np.ndarray):
            xdim = len(alphas)
        else:
            xdim = 1
        phi = np.zeros([xdim,3])
        phi[:,0] = 1.0
        phi[:,1] = np.sin(((2*np.pi)/alpha_period)*alphas)
        phi[:,2] = np.cos(((2*np.pi)/alpha_period)*alphas)
        return phi



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

def walk(x0,x1,direction,pipe_output,sc_dict):
    global data_queue, pipe_setup_queue, status_queue, kill_confirm_queue

    status_queue.put([os.getpid(),'walking'],block=False)
    sc = sc_dict[os.getpid()]
    assign_calc(sc)
    #set status as 'walking'
    status_queue.put([os.getpid(),'walking'],block=False)
    print(f'pid {os.getpid()} sc is {sc} sc calc is {sc.calc}')

    #set up pipe
    print('setting up pipe')
    pipe_id = pipe_output.recv()
    pipe_setup_queue.put([pipe_id,os.getpid()])
    print('pipe set up!')
    #start NCFlex, passing the data queue and the pipe output
    # data queue sends back data to main
    # pipe output allows main to kill the process
    print('starting ncflex...')
    sc.variable_k = True
    precon = False
    # print('lammps lib process',os.getpid(),sc.calc)
    sc.arc_length_continuation(x0, x1, N=nsteps,
                            ds=ds, ds_max=0.05, ftol=fmax, max_steps=10,
                            direction=direction,
                            continuation=continuation,
                            traj_file=traj_file,
                            traj_interval=traj_interval,
                            precon=precon,opt_method='krylov',parallel=True,
                            pipe_output=pipe_output,data_queue=data_queue,
                            kill_confirm_queue=kill_confirm_queue)

    #time.sleep(10)
    #go idle
    status_queue.put([os.getpid(),'idle'],block=False)
    delete_calc(sc)

def get_opt_K_alpha(walk_positions,trail_positions,currently_walking_pids):    
    print('GETTING NEW OPT K ALPHA')
    #get a random value of alpha that's not yet been 
    #searched, as well as corresponding estimate for K from
    #a fitted sine curve
    starts = []
    ends = []
    pids = []

    alpha_covered = 0
    for pid in walk_positions:
        if pid in currently_walking_pids:
            [[k_lead,alpha_lead],direction] = walk_positions[pid]
            [k_trail,alpha_trail] = trail_positions[pid]
            alpha_covered += (alpha_lead-alpha_trail)*direction
            if direction == 1:
                pids.append(pid)
                starts.append(alpha_trail)
                ends.append(alpha_lead)
            elif direction == -1:
                pids.append(pid)
                starts.append(alpha_lead)
                ends.append(alpha_trail)

    # print('alpha covered',alpha_covered)
    alpha_not_covered = (alpha_range[1]-alpha_range[0])-alpha_covered
    # print('alpha not covered',alpha_not_covered)
    random_alph_start = np.random.uniform(0,alpha_not_covered)
    # print('random alpha start', random_alph_start)

    if len(starts) == 0:
        #if nothing has been started yet, just guess alpha as random_alph_start
        alpha=random_alph_start
        K = np.random.uniform(K_range[0],K_range[1])
        return [K,alpha]

    #now sort the starts
    si = np.argsort(starts)
    alpha_assigned = False
    i = 0
    #first gap
    # print('first start:', starts[si[0]])
    if alpha_range[0] < np.min(starts):
        random_alph_start -= (starts[si[0]]-alpha_range[0])
        # print(f'subtract_amount{i}:', (starts[si[0]]-alpha_range[0]))
        # print(f'random_alph_start_subtract_{i}:', random_alph_start)
        if random_alph_start < 0:
            alpha = starts[si[0]] + random_alph_start
            alpha_assigned = True
            
    while not alpha_assigned:
        i += 1
        #if at the end:
        if i == (len(si)):
            # print('at the end')
            # print(f'final end is:{ends[si[i-1]]}')
            # print(f'remaining alpha to spend is',{random_alph_start})
            alpha = ends[si[i-1]] + random_alph_start
            # print(f'alpha is{alpha}')
            alpha_assigned = True
            break
        gap = starts[si[i]]-ends[si[i-1]]
        # print(f'next start{i}:',starts[si[i]])
        # print(f'next end{i}',ends[si[i-1]])
        # print(f'subtract_amount{i}:', (starts[si[i]]-ends[si[i-1]]))
        if gap<0:
            # print('gap zeroed')
            gap = 0
        random_alph_start -= gap
        # print(f'random_alph_start_subtract_{i}:', random_alph_start)
        if random_alph_start < 0:
            # print('found right gap')
            # print(f'right hand edge of gap, {starts[si[i]]}')
            # print(f'subtraction amount {random_alph_start}')
            alpha = starts[si[i]] + random_alph_start
            # print(f'alpha assigned as {alpha}')
            alpha_assigned = True
    
    opt_alpha = alpha
    #fit regression model and predict K
    for nattempt in range(10000):
        try:
            hf = h5py.File(traj_file, 'r')
            x_traj = hf['x']
            K_vals = x_traj[:,-1]
            alpha_vals = x_traj[:,-2]
            hf.close()
            K_vals_reduced = K_vals/sc.k1g
            break
        except OSError:
            print('hdf5 file not accessible, trying again in 1s')
            print('failed to access file')
            time.sleep(1.0)
            continue
    
    RM = RegressionModel()
    RM.fit(alpha_vals,K_vals_reduced)
    opt_K = RM.predict(opt_alpha)

    print(f'from opt alpha {alpha}, opt k has been predicted as {opt_K} from sine curve')
    maxk = max(np.max(K_vals_reduced),K_range[1])
    mink = min(np.min(K_vals_reduced),K_range[0])
    if opt_K>maxk:
        print(f'k out of upper range, setting to {maxk}')
        opt_K = maxk
    elif opt_K<mink:
        print(f'k out of lower range, setting to {mink}')
        opt_K = mink

    return [opt_K,opt_alpha]



def main(K_range,alpha_range):
    global data_queue, sc, pipe_setup_queue, status_queue,search_queue

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

    #set up a dictionary to hold walker locations
    walk_positions = {}

    #set up a dictionary to hold walker initial positions ('tails')
    trail_positions = {}

    #set up dictionary to hold worker kill pipes
    pipe_dict = {}

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
    currently_walking_pids = []
    curve_explored = False
    percentage_covered = 0
    time.sleep(5)
    while not curve_explored:
        # print(f'currently walking pids, {currently_walking_pids}')
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
                f.write(f'Completion percentage : {np.round((100*percentage_covered),decimals=2)}%')
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
        # print(f'idle: {idle_num}, search, {search_num}, walk {walk_num}')
        
        #print('checking to launch new searches')
        #if there's unnaccounted for idle processes, launch new searches
        if explore_direction == 0:
            num_new_searches = int(np.floor(idle_num-search_num))
        elif (explore_direction == 1) or (explore_direction ==-1):
            num_new_searches = idle_num
        if num_new_searches>0:
            for i in range(num_new_searches):
                print('LAUNCHING A NEW SEARCH')
                new_K, new_alpha = get_opt_K_alpha(walk_positions,trail_positions,currently_walking_pids)
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
                    new_K, new_alpha = get_opt_K_alpha(walk_positions,trail_positions,currently_walking_pids)
                    worker_pool.apply_async(search, args=(new_K,new_alpha,sc_dict))
                    continue

                found_alpha = x0[-2]

                #set worker status to idle here rather than in function to prevent
                #new searches being started erroneously
                worker_status[pid] = 'idle'

                #if found alpha is out of range, start a new search
                if (found_alpha<alpha_range[0] or found_alpha>alpha_range[1]):
                    new_K, new_alpha = get_opt_K_alpha(walk_positions,trail_positions,currently_walking_pids)
                    worker_pool.apply_async(search, args=(new_K,new_alpha,sc_dict))
                    continue

                #code for when found alpha is already within a searched range
                search_restarted=False
                for other_pid in walk_positions:
                    if other_pid != pid:
                        [[comp_k_lead,comp_alpha_lead],comp_direction] = walk_positions[other_pid]
                        [comp_k_trail,comp_alpha_trail] = trail_positions[other_pid]
                        #cases when we want to kill walker:
                        comp_alpha_range = np.sort([comp_alpha_lead,comp_alpha_trail])
                        #if the alpha lies within the comparison range, and the comparision range
                        #is at least 0.1 big. 
                        if (alpha_lead>comp_alpha_range[0]) and (alpha_lead<comp_alpha_range[1]):
                            print('alpha detected in already searched range, starting new search')
                            new_K, new_alpha = get_opt_K_alpha(walk_positions,trail_positions,currently_walking_pids)
                            worker_pool.apply_async(search, args=(new_K,new_alpha,sc_dict))
                            search_restarted=True
                            break

                if search_restarted:
                    continue
                
                if explore_direction == 0:
                    #create two new pipes
                    pipes = []
                    pipes.append(multiprocessing.Pipe())
                    pipes.append(multiprocessing.Pipe())

                    #start walk jobs
                    worker_pool.apply_async(walk,args=(x0,x1,-1,pipes[0][0],sc_dict))
                    worker_pool.apply_async(walk,args=(x0,x1,1,pipes[1][0],sc_dict))

                    #send pipeIDs down pipes
                    pipes[0][1].send(0)
                    pipes[1][1].send(1)

                    #assign resulting pipe communication to correct pid
                    queue_empty = False
                    while not queue_empty:
                        try:
                            [pipe_id,pid] = pipe_setup_queue.get(block=True,timeout=3)
                            pipe_dict[pid] = pipes[pipe_id][1]
                            #add initial positions as trails to dict
                            trail_positions[pid] = [x0[-1],x0[-2]] #[K,alpha]
                            currently_walking_pids.append(pid)
                        except Empty:
                            queue_empty = True
                elif (explore_direction == 1) or (explore_direction ==-1):
                    #create new pipe
                    pipes = []
                    pipes.append(multiprocessing.Pipe())

                    #start walk job
                    worker_pool.apply_async(walk,args=(x0,x1,explore_direction,pipes[0][0],sc_dict))

                    #send pipeID down pipe
                    pipes[0][1].send(0)

                    #assign resulting pipe communication to correct pid
                    queue_empty = False
                    while not queue_empty:
                        try:
                            [pipe_id,pid] = pipe_setup_queue.get(block=True,timeout=3)
                            pipe_dict[pid] = pipes[pipe_id][1]
                            #add initial positions as trails to dict
                            trail_positions[pid] = [x0[-1],x0[-2]] #[K,alpha]
                            currently_walking_pids.append(pid)
                        except Empty:
                            queue_empty = True


                


        #print('checking for results from ongoing walk jobs')
        #next, check for any items in the queue from already going walk jobs
        #if any data is available - write this data to a file, and update the stored
        #start and end positions of each PID 
        items = []
        queue_empty = False
        while not queue_empty:
            try:
                item = data_queue.get(block=True,timeout=0.2)
                items.append(item)
                # print('got data,', item)
            except Empty:
                queue_empty = True
        if it_num%100 == 0:
            print('current walk positions are:')
            print(walk_positions)
        #print('checking walker encroachment')
        #check if any walker is encroaching on another, if so, kill the job(s)
        if len(items) > 0:
            kill_pids = []
            for item in items:
                [pid, x, direction] = item
                if pid not in currently_walking_pids:
                    #if the pid that sent the message
                    #has already had a command sent to kill
                    #but has not yet been killed
                    #ignore the message
                    continue
                # write x to h5 traj file
                # print('writing to file!')
                for nattempt in range(1000):
                    try:
                        with h5py.File(traj_file, 'a') as hf:
                            x_traj = hf['x']
                            x_traj.resize((x_traj.shape[0] + 1, x_traj.shape[1]))
                            x_traj[-1, :] = x
                            # print('written')
                            break
                    except OSError:
                        print('hdf5 file not accessible, trying again in 1s')
                        print('failed to access file')
                        time.sleep(1.0)
                        continue
                else:
                    raise IOError("ran out of attempts to access trajectory file")
                #update dictionary
                k_lead = x[-1]
                alpha_lead = x[-2]
                walk_positions[pid] = [[k_lead, alpha_lead], direction]
                #check for encroachment
                # print('LEAD POSITIONS:', walk_positions)
                # print('TRAIL POSITIONS:',trail_positions)
                for other_pid in walk_positions:
                    if other_pid != pid:
                        [[comp_k_lead,comp_alpha_lead],comp_direction] = walk_positions[other_pid]
                        [comp_k_trail,comp_alpha_trail] = trail_positions[other_pid]
                        if direction != 0: #if walker has started search
                            #cases when we want to kill walker:
                            comp_alpha_range = np.sort([comp_alpha_lead,comp_alpha_trail])
                            
                            #if the alpha lies within the comparison range, and the comparision range
                            #is at least 0.1 big. 
                            if (alpha_lead>comp_alpha_range[0]) and (alpha_lead<comp_alpha_range[1]) and ((comp_alpha_range[1]-comp_alpha_range[0])>1e-2):
                                # print('alpha detected in range!')
                                #if it's more than 1e-3 away from one of the ends (some serious overlap)
                                if (abs(alpha_lead-comp_alpha_lead)>1e-3) and (abs(alpha_lead-comp_alpha_trail)>1e-3):
                                    # print('encroachment detected!')
                                    if pid not in kill_pids:
                                        kill_pids.append(pid)
                            
                if alpha_lead > alpha_range[1]:
                    # print('alpha detected out of upper range')
                    if pid not in kill_pids:
                        kill_pids.append(pid)
                elif alpha_lead < alpha_range[0]:
                    # print('alpha detected out of lower range')
                    if pid not in kill_pids:
                        kill_pids.append(pid)
            
            # print('kill_pids:,', kill_pids)
            #kill any pids that need killing
            if len(kill_pids)>0:
                for pid in kill_pids:
                    pipe_dict[pid].send(1) #send kill command
                    currently_walking_pids.remove(pid)
                    killed_num += 1
                    trail_positions[f'killed{killed_num}'] = trail_positions[pid]
                    walk_positions[f'killed{killed_num}'] = walk_positions[pid]
                    #walk_positions[pid] = [[0,0],0]
                    #trail_positions[pid] = [[0,0]]
                # print(kill_pids)
                #worker_pool.close()
                #exit()
        
        #check for confirmations that workers have been killed
        queue_empty = False
        while not queue_empty:
            try:
                pid = kill_confirm_queue.get(block=False)
                del trail_positions[pid]
                del walk_positions[pid]
            except Empty:
                queue_empty = True

        #check if the search is finished and calculate %
        total_alpha_covered = 0
        for pid in walk_positions:
            if pid not in currently_walking_pids:
                #need to avoid double counting when a process terminates
                continue
            [[k_lead,alpha_lead],direction] = walk_positions[pid]
            [k_trail,alpha_trail] = trail_positions[pid]
            contribution = (alpha_lead-alpha_trail)*direction
            total_alpha_covered += contribution
            # print('CONTRIBUTION:',pid,contribution)
        percentage_covered = total_alpha_covered/(alpha_range[1]-alpha_range[0])
        if percentage_covered > 1:
            curve_explored=True
    
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
    pipe_setup_queue = multiprocessing.Queue()
    status_queue = multiprocessing.Queue()
    search_queue = multiprocessing.Queue()
    kill_confirm_queue = multiprocessing.Queue()

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
    # save the cluster used for NCFlex, to avoid sort-index inconsistencies
    ase.io.write('ncflex_cluster.xyz',cluster) 

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


    #open h5 file
    with h5py.File(traj_file, 'a') as hf:
        if 'x' in hf.keys():
            x_traj = hf['x']
        else:
            x_traj = hf.create_dataset('x', (0, len(sc)),
                                    maxshape=(None, len(sc)),
                                    compression='gzip')
    main(K_range,alpha_range)

