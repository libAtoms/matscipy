from matscipy.fracture_mechanics.thin_strip_utils import ThinStripBuilder, set_up_simulation_lammps, write_potential_and_buffer
from matscipy import parameter
from matscipy.fracture_mechanics.crack import G_to_strain
import ase.io.lammpsdata
import ase.io
import numpy as np
from mpi4py import MPI
from lammps import lammps
from matscipy.fracture_mechanics.crack import find_tip_coordination
from matscipy.fracture_mechanics.clusters import set_groups
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
import os 
import params
import ase.units as units
import warnings


me = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()
strip_height = parameter('strip_height')
strip_width = parameter('strip_width')
strip_thickness = parameter('strip_thickness')
vacuum = parameter('vacuum')
crack_seed_length = parameter('crack_seed_length')
strain_ramp_length = parameter('strain_ramp_length')
C = parameter('C')
a0 = parameter('a0')
directions = [parameter('crack_direction'),
              parameter('cleavage_plane'),
              parameter('crack_front')]
el = parameter('el')
bondlength = parameter('bondlength')
bulk_nn = parameter('bulk_nn')
paste_threshold = parameter('paste_threshold')
track_spacing = parameter('track_spacing',150) #distance between tracked atoms
track_tstep = parameter('track_tstep',5) #number of timesteps between tracked atom dumping
restart = parameter('restart', False)
checkpointing = parameter('checkpointing',False)
checkpoint_filename = parameter('checkpoint_filename','thin_strip')
initial_checkpoint_num = parameter('initial_checkpoint_num',1)
results_path = parameter('results_path','./results')
checkpoint_interval = parameter('checkpoint_interval',1)
kvals = parameter('kvals') #initial stress intensity factor (in MPa sqrt(m))
cb = parameter('cb','None')
multilattice = parameter('multilattice',False)
right_hand_edge_dist = parameter('right_hand_edge_dist',0.0)
crack_reset_pos = parameter('crack_reset_pos',strip_width/2)
lattice = parameter('lattice')
calc = parameter('calc')
mass = parameter('mass') #atomic mass
cmds = parameter('cmds')
sim_tstep = parameter('sim_tstep',0.001) #ps
temp_path = parameter('temp_path','./temp')
damping_strength_right = parameter('damping_strength_right', 0.1)
damping_strength_left = parameter('damping_strength_left', 0.1)
dump_freq = parameter('dump_freq',100)
dump_name = parameter('dump_name','dump.lammpstrj')
thermo_freq = parameter('thermo_freq',100)
initial_damp = parameter('initial_damp', True)
initial_damping_time = parameter('initial_damping_time',10)
initial_damping_strengths = parameter('initial_damping_strengths',[0.1*(i+1) for i in range(initial_damping_time)])
step_tolerant = parameter('step_tolerant',False)

assert len(initial_damping_strengths) == initial_damping_time, 'initial_damping_strengths must be a list of length initial_damping_time'

n_steps_per_loop = parameter('n_steps_per_loop',1000)
steady_state_check_dist = parameter('steady_state_check_dist',strip_width/5)
max_loop_number = parameter('max_loop_number',500)
left_damp_thickness = parameter('left_damp_thickness',60.0)
right_damp_thickness = parameter('right_damp_thickness',60.0)
n_v_compare = parameter('n_v_compare',10) # number of velocity values to compare to check for steady state
ss_tol = parameter('ss_tol',0.01) # user defined tolerance for steady state around velocity mean
dump_files = parameter('dump_files', True)
initial_kick = parameter('initial_kick', False) #give the crack an initial kick at the end of the damping time
kick_timestep = parameter('kick_timestep', 9) # timestep to give the crack an initial kick

y_buffer_unit_cells = parameter('y_buffer_unit_cells',0) #added additional unit cells of atoms in y to be fixed 
# important if you don't want potential to see outide edge
approximate_strain = parameter('approximate_strain',False) #if True, use approximate strain calculation

multi_potential = parameter('multi_potential',False) #if True, use dual potentials, for different regions of the crack
if multi_potential:
    partition_type = parameter('partition_type') #get the type of partitioning to use between potentials
    #if the partition type is strip, then get the strip width
    buffer_thickness = parameter('buffer_thickness',6.0)
    if partition_type == 'strip':
        partition_width = parameter('partition_width')
    else:
        partition_width = None
else:
    partition_type = None
    partition_width = None

y_threshold = parameter('y_threshold',1)
cpnum = initial_checkpoint_num

if restart:
    restart_path = parameter('restart_path')

if int(strip_width/track_spacing) > 25:
    raise ValueError('LAMMPS only allows 32 groups total, reduce track spacing')

if restart and initial_damp:
    initial_damp = False
    warnings.warn('Restarting from checkpoint, initial damping will be skipped')

if restart and initial_kick:
    initial_kick = False
    warnings.warn('Restarting from checkpoint, initial kick will be skipped')

lmp = lammps()

initial_K = kvals[0]
if cb == 'None':
    cb = None

sim_time = 0
plot_num = 0

######################SET UP SIMULATION OR RESTART FROM CHECKPOINT FILES##############################
if me == 0:
    tsb = ThinStripBuilder(el,a0,C,calc,lattice,directions,multilattice=multilattice,cb=cb,switch_sublattices=True)
    tsb.y_buffer_unit_cells = int(y_buffer_unit_cells)
    if not approximate_strain:
        tsb.measure_energy_strain_relation(resolution=1000)
    
    crack_slab = tsb.build_thin_strip_with_crack(initial_K,strip_width,strip_height,strip_thickness\
                                               ,vacuum,crack_seed_length,strain_ramp_length,track_spacing=track_spacing,apply_x_strain=False,approximate=approximate_strain)

    os.makedirs(temp_path,exist_ok=True)
    if not restart:
        os.makedirs(results_path, exist_ok=False)
        tracked_array = crack_slab.arrays['tracked']
        print('Writing crack slab to file "crack.xyz"')
        crack_slab.set_velocities(np.zeros([len(crack_slab),3]))

        #--------------- if multi potential, draw boundaries ----------------#
        if multi_potential:
            if partition_type == 'strip':
                tsb.draw_strip_potential_boundary(crack_slab,partition_width,buffer_thickness,
                                                    strip_width,strip_height,strip_thickness,vacuum)
            else:
                raise ValueError('Partition type not recognised')

        ase.io.write(f'{temp_path}/crack.xyz', crack_slab, format='extxyz')
        ase.io.lammpsdata.write_lammps_data(f'{temp_path}/crack.lj',crack_slab,velocities=True)
        
        if multi_potential:
            write_potential_and_buffer(crack_slab,f'{temp_path}/crack.lj')

        tip_pos = crack_seed_length+strain_ramp_length+np.min(crack_slab.get_positions()[:,0])
        K_curr = initial_K
    else:
        #if restarting
        #if results_path does not exist, create it and raise warning
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=False)
        else:
            warnings.warn('Results path already exists, overwriting files')
        
        #copy checkpointed files to results path
        os.system(f'cp {restart_path}/tracked_motion_*.txt {results_path}')
        os.system(f'cp {restart_path}/steady_state.txt {results_path}')
        os.system(f'cp {restart_path}/crack.lj {temp_path}/crack.lj')
        tracked_array = np.loadtxt(f'{restart_path}/tracked_array.txt').astype(int)
        crack_slab.arrays['tracked'] = tracked_array
        [restart_knum,K_restart,sim_time,plot_num,cpnum,total_added_dist,tip_pos] = np.loadtxt(f'{restart_path}/simulation_restart_params.txt')
        tsb.total_added_dist = total_added_dist
        restart_knum = int(restart_knum)
        plot_num = int(plot_num)
        cpnum = int(cpnum)+1
        assert (kvals[restart_knum] == K_restart), 'K value in restart file does not match K value in kvals list'
        #assert that cpdir/cpnum does not exist
        assert not os.path.exists(f'./{checkpoint_filename}/{cpnum}'), f'The next checkpoint directory ./{checkpoint_filename}/{cpnum} already exists, please change checkpoint directory'
        #shorten kvals list to start from the restart_knum entry
        kvals = kvals[restart_knum:]
        K_curr = K_restart
    
    y_fixed_length = y_buffer_unit_cells*tsb.single_cell_height + 1
else:
    tracked_array = None
    y_fixed_length = 0

######################################################################################################

###########################MAIN LOOP#################################

y_fixed_length = MPI.COMM_WORLD.bcast(y_fixed_length,root=0)
for knum,K in enumerate(kvals):
    if me == 0:
        crack_tip_positions = np.array([])
        prev_v = 0
        unique_v_vals = 0
        if K != K_curr: #we don't need to rescale if K is K_curr
            #unscaled_crack = ase.io.lammpsdata.read_lammps_data(f'{temp_path}/crack.lj',atom_style='atomic')
            unscaled_crack = ase.io.read(f'{temp_path}/crack.xyz',format='extxyz',parallel=False)
            unscaled_crack.set_pbc([False,False,True])
            rescale_crack = tsb.rescale_K(unscaled_crack,K_curr,K,strip_height,tip_pos,approximate=approximate_strain)
            #re-write lammps data file
            ase.io.lammpsdata.write_lammps_data(f'{temp_path}/crack.lj',rescale_crack,velocities=True,masses=True)
            if multi_potential:
                write_potential_and_buffer(crack_slab,f'{temp_path}/crack.lj')

    K_curr = K
    if (knum == 0) and initial_damp:
        initial_damp = True
    else:
        initial_damp = False
    
    if (knum == 0) and initial_kick:
        initial_kick = True
    else:
        initial_kick = False

    at_steady_state = False
    i = -1
    final_v = -1 #set to -1 so that if steady state is not reached, it is obvious
    final_v_std = 100 #set to 100 so that if steady state is not reached, it is obvious
    while not at_steady_state:
        i += 1
        if i>max_loop_number:
            break
        #communicate the tracked_array numpy array to all processes
        tracked_array = MPI.COMM_WORLD.bcast(tracked_array,root=0)
        #sychronise all processes
        MPI.COMM_WORLD.Barrier()
        #now set up simulation (but do not run)
        set_up_simulation_lammps(lmp,temp_path,mass,cmds,sim_tstep=sim_tstep,damping_strength_right=damping_strength_right,damping_strength_left=damping_strength_left
                                 , dump_freq=dump_freq, dump_name=dump_name, thermo_freq=thermo_freq, dump_files=dump_files,
                                 left_damp_thickness=left_damp_thickness, right_damp_thickness=right_damp_thickness,multi_potential=multi_potential,
                                 y_fixed_length=y_fixed_length)
        if (i < initial_damping_time) and (initial_damp):
            #add a lammps command to set a thermostat for all atoms initially
            lmp.command(f'fix therm2 nve_atoms langevin 0.0 0.0 {initial_damping_strengths[i]} 1029')
        atom_ids = np.where(tracked_array>0)[0]
        #make a string of atom ids to pass to lammps
        #atom_id_string = ' '.join1([str(x) for x in atom_ids])
        fnames = []
        for track_group,id in enumerate(atom_ids):
            # need to add 1 to id, as LAMMPS indexing starts from 1
            lmp.command(f'group tracked_{track_group} id {id+1}')
            lmp.command(f'dump track_dump_{track_group} tracked_{track_group} atom {track_tstep} {temp_path}/tracked_{track_group}.lammpstrj')
            fnames.append(f'{temp_path}/tracked_{track_group}.lammpstrj')
        #run for 1000 timesteps
        lmp.command(f'run {n_steps_per_loop}')
        #write temp output file
        lmp.command(f'write_data {temp_path}/simulation_output.temp nocoeff nofix nolabelmap')

        #at the end of 1000 timesteps, read the output files and find the crack tip
        if me == 0:
            #first deal with reading and processing the track_dump file
            tracked_motion_dict = {}
            for fnum, fname in enumerate(fnames):
                traj = ase.io.read(fname,index=':')
                atom_index = atom_ids[fnum]
                tracked_motion_dict[tracked_array[atom_index]] = np.zeros([len(traj)-1,3])
                sim_time_tmp = sim_time
                for tstep in range(1,len(traj)):
                    t = traj[tstep]
                    sim_time_tmp += track_tstep*sim_tstep
                    atom = t[0]
                    tracked_motion_dict[tracked_array[atom_index]][tstep-1,:] = [sim_time_tmp,atom.position[1],atom.position[0]+tsb.total_added_dist]
            sim_time=sim_time_tmp
            #now append the numpy arrays to seperate txt files based on key
            #print(tracked_motion_dict.keys())
            for key in tracked_motion_dict.keys():
                #print(key)
                with open(f'{results_path}/tracked_motion_{key}.txt','ab') as f:
                    np.savetxt(f,tracked_motion_dict[key])
            
            # -------------- read in simulation output file --------------- #
            final_crack_state = ase.io.lammpsdata.read_lammps_data(f'{temp_path}/simulation_output.temp',atom_style='atomic')
            final_crack_state.set_pbc([False,False,True])
            final_crack_state.new_array('groups',crack_slab.arrays['groups'])
            final_crack_state.new_array('tracked',tracked_array)
            final_crack_state.new_array('trackable',crack_slab.arrays['trackable'])
            trackable = final_crack_state.arrays['trackable']
            ase.io.write(f'{results_path}/final_crack_state.xyz',final_crack_state,format='extxyz')
            #plot out spacing of trackable atoms
            plt.figure()
            plot_num += 1
            plt.plot(final_crack_state.get_positions()[:,0][trackable][:-1],np.diff(final_crack_state.get_positions()[:,0][trackable]),'o')
            plt.savefig(f'{results_path}/trackable_spacing_{plot_num}.png')
            plt.close()

            # ------------- find crack tip ----------- #
            # tmp_bondlength = bondlength
            # found_tip=False
            # for j in range(10):
            #     try:
            #         # print(tmp_bondlength,bulk_nn)
            #         bond_atoms = find_tip_coordination(final_crack_state,bondlength=tmp_bondlength,bulk_nn=bulk_nn,calculate_midpoint=True)
            #         tip_pos = (final_crack_state.get_positions()[bond_atoms,:][:,0])
            #         #print(tip_pos[0],tip_pos[1])
            #         assert np.abs(tip_pos[0]-tip_pos[1]) < 1
            #         found_tip=True
            #         break
            #     except AssertionError:
            #         tmp_bondlength += 0.01
            #         #keep trying till a crack tip is found
            # if not found_tip:
            #     raise RuntimeError('Lost crack tip!')
            # tip_pos = tip_pos[0]
            # print(f'Found crack tip at position {tip_pos}')
            tip_pos = tsb.find_strip_crack_tip(final_crack_state,bondlength,bulk_nn,calculate_midpoint=True,step_tolerant=step_tolerant)

            # ------------check for an arrested crack ------------ #
            #only start checking after the first initial_damping steps
            if (i >= initial_damping_time) or (not initial_damp):
                crack_tip_positions = np.append(crack_tip_positions,tip_pos+tsb.total_added_dist)
                #if the crack tip has not moved in the last 10 loops, then it is arrested
                if len(crack_tip_positions) > 10:
                    if np.max(crack_tip_positions[-10:]) - np.min(crack_tip_positions[-10:]) < 1:
                        at_steady_state = True
                        final_v = 0
                        final_v_std = 0
                        print('Detected crack arrest!')
            
            # ----------- check to paste atoms ------------ #
            pasted = False
            if tip_pos > (np.max(final_crack_state.get_positions()[:,0])-paste_threshold):
                #if it is, set crop to be the distance between the crack tip and the crack_reset_pos
                crop = tip_pos - (np.min(final_crack_state.get_positions()[:,0]) + crack_reset_pos)
                pasted = True
            else:
                #do no pasting and just create a fresh input file identical to old one
                crop = 0

            # -------------- paste atoms --------------- #
            new_slab = tsb.paste_atoms_into_strip(K_curr,strip_width,strip_height,strip_thickness,vacuum,\
                                            final_crack_state,crop=crop,track_spacing=track_spacing,right_hand_edge_dist=right_hand_edge_dist,
                                            match_cell_length=False,approximate=approximate_strain)
            new_slab.new_array('masses',final_crack_state.arrays['masses'])

            # -------------- if turned on, give atoms an initial kick for the next loop --------------- #
            if initial_kick and (i+1 == kick_timestep):
                #get a mask for the right atoms to kick
                simple_strip = tsb.build_thin_strip(strip_width,strip_height,strip_thickness,vacuum)
                x_pos = simple_strip.get_positions()[:,0]
                y_pos = simple_strip.get_positions()[:,1]
                tip_x_pos = x_pos[bond_atoms[0]]
                tip_y_pos = [y_pos[bond_atoms[0]], y_pos[bond_atoms[1]]]
                full_mask = (x_pos>(tip_x_pos-10)) & (x_pos<(tip_x_pos+40))
                
                #mask is full mask and atoms which have a y position equal to that of the crack tip within numerical resolution
                top_mask = full_mask & (np.round(y_pos,decimals=3) == np.round(tip_y_pos[0],decimals=3))
                bot_mask = full_mask & (np.round(y_pos,decimals=3) == np.round(tip_y_pos[1],decimals=3))

                print('Delivering kick.........')
                #set bond_atoms[0] velocity to -2 and bond_atoms[1] velocity to 2
                velocities = new_slab.get_velocities()
                velocities[:,1][top_mask] += 0.3
                velocities[:,1][bot_mask] += -0.3
                initial_kick = False
                new_slab.set_velocities(velocities)


            #--------------- if multi potential, redraw boundaries ----------------#
            if multi_potential:
                if partition_type == 'strip':
                    tsb.draw_strip_potential_boundary(new_slab,partition_width,buffer_thickness,
                                                      strip_width,strip_height,strip_thickness,vacuum)
                else:
                    raise ValueError('Partition type not recognised')

            # -------------- write file for next loop --------------- #
            #ase.io.write(f'{results_path}/new_slab_{plot_num}.xyz',new_slab,format='extxyz')
            ase.io.lammpsdata.write_lammps_data(f'{temp_path}/crack.lj',new_slab,velocities=True,masses=True)
            if multi_potential:
                write_potential_and_buffer(new_slab,f'{temp_path}/crack.lj')
            #write extxyz file too in case of rescaling
            ase.io.write(f'{temp_path}/crack.xyz', new_slab, format='extxyz')
            
            # --------------- check crack velocity --------------- #
            #finally, get the crack velocity and work out if it's reached steady state         
            mask = (final_crack_state.get_positions()[:,0] < (tip_pos-steady_state_check_dist)) & (tracked_array>0)
            print(f'CRACK TIP POS:{tip_pos}')
            print(f'SHIFTED CRACK TIP POS:{tip_pos-steady_state_check_dist}')
            print(f'MASK LEN:{len(np.where(mask)[0])}')
            if len(np.where(mask)[0])>=2:
                prev_2_tracked_idx = np.argsort(final_crack_state.get_positions()[mask,0])[-2:]
                atom_ids = np.where(mask)[0][prev_2_tracked_idx]
                #print(atom_ids)
                #read numpy arrays from files
                atom_1_traj = np.loadtxt(f'{results_path}/tracked_motion_{tracked_array[atom_ids[0]]}.txt')
                atom_2_traj = np.loadtxt(f'{results_path}/tracked_motion_{tracked_array[atom_ids[1]]}.txt')
                #calculate crack velocity and check if it's steady
                v, ss_c = tsb.check_steady_state(atom_1_traj,atom_2_traj,y_threshold=y_threshold)

                # -------------- if it's a new velocity, write to files --------------- #
                if np.round(v,decimals=6) != np.round(prev_v,decimals=6): 
                    #use numpy to write to file along with current timestep
                    with open(f'{results_path}/steady_state.txt','ab') as f:
                        np.savetxt(f,np.reshape(np.array([sim_time,v,ss_c]),[1,3]))
                        #read the txt and plot
                    
                    steady_state_data = np.reshape(np.loadtxt(f'{results_path}/steady_state.txt'),[-1,3])
                    plt.figure()
                    plt.plot((steady_state_data[:,0]/(10**3)),steady_state_data[:,1],'o')
                    plt.xlabel('Time (ns)')
                    plt.ylabel('Measured crack velocity (km/s)')
                    plt.savefig(f'{results_path}/velocity_steady_state.png')
                    plt.close()
                    plt.figure()
                    plt.plot((steady_state_data[:,0]/(10**3)),steady_state_data[:,2],'o')
                    plt.xlabel('Time (ns)')
                    plt.ylabel('Steady state measure')
                    plt.savefig(f'{results_path}/steady_state_measure.png')
                    plt.close()
                    final_v = np.mean(steady_state_data[-n_v_compare:,1])
                    final_v_std = np.std(steady_state_data[-n_v_compare:,1])
                    # check the range of the last n values of velocity. If it's within user defined tol of mean, then steady state is reached
                    unique_v_vals += 1

                    # -------------- check for steady state --------------- #
                    if unique_v_vals >= n_v_compare:
                        if np.max(steady_state_data[-n_v_compare:,1]) - np.min(steady_state_data[-n_v_compare:,1]) < ss_tol*np.mean(steady_state_data[-n_v_compare:,1]):
                            at_steady_state = True
                            #set final velocity to be the mean of the last n values, with a standard deviation
                            print(f'Final crack velocity: {final_v} +/- {final_v_std} km/s')
                prev_v = v
            #check for copy-pasting

            # -------------- write to checkpoint files --------------- #
            tracked_array = new_slab.arrays['tracked']
            if checkpointing and (i+1)%checkpoint_interval == 0:
                cpdir = f'./{checkpoint_filename}/{cpnum}'
                os.makedirs(cpdir, exist_ok=False)
                #ase.io.lammpsdata.write_lammps_data(f'{cpdir}/crack.lj',new_slab,velocities=True,masses=True)
                os.system(f'cp {temp_path}/crack.lj {cpdir}')
                np.savetxt(f'{cpdir}/simulation_restart_params.txt',np.array([knum,K_curr,sim_time,plot_num,cpnum,tsb.total_added_dist,tip_pos]))
                np.savetxt(f'{cpdir}/tracked_array.txt',tracked_array)
                #copy all files called tracked_motion_*.txt in current directory to cpdir
                os.system(f'cp {results_path}/tracked_motion_*.txt {cpdir}')
                #copy steady_state.txt if it exists
                os.system(f'cp {results_path}/steady_state.txt {cpdir}')
                cpnum += 1

        
        #at the end of each loop, broadcast at_steady_state to all processes
        at_steady_state = MPI.COMM_WORLD.bcast(at_steady_state,root=0)
    
    # -------------- loop broken, store K vs velocity data --------------- #
    if me == 0:
        if not at_steady_state:
            warnings.warn('Steady state not reached in loops given, fracture could be unstable!')
            print(f'Final crack velocity: {final_v} +/- {final_v_std} km/s')

        #write K value, final velocity, velocity std and at_steady_state to file
        with open(f'{results_path}/K_vs_velocity.txt','ab') as f:
            np.savetxt(f,np.reshape(np.array([K,final_v,final_v_std,at_steady_state]),[1,4]))

        #plot K vs velocity with velocity error bars
        #colour points blue if steady state reached, red if not
        #read file
        res = np.reshape(np.loadtxt(f'{results_path}/K_vs_velocity.txt'),[-1,4])
        print('res',res)
        plt.figure()
        mask = np.array(res[:,3].flatten(),dtype=bool)
        plt.errorbar(res[:,0][mask],res[:,1][mask],yerr=res[:,2][mask],fmt='o',color='blue',ecolor='blue',capsize=5,label='Steady state reached')
        plt.errorbar(res[:,0][~mask],res[:,1][~mask],yerr=res[:,2][~mask],fmt='o',color='red',ecolor='red',capsize=5,label='Steady state not reached')
        plt.xlabel('K (MPa sqrt(m))')
        plt.ylabel('Final crack velocity (km/s)')
        plt.legend()
        plt.savefig(f'{results_path}/K_vs_velocity.png')
        plt.close()

print("Finalising on %d out of %d procs" % (me,nprocs),lmp)
MPI.COMM_WORLD.Barrier()
MPI.Finalize()
