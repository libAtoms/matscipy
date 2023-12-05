from matscipy.fracture_mechanics.thin_strip_utils import ThinStripBuilder
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
#print local directory
print(os.getcwd())
#print files in current working directory
print(os.listdir())
import params
import ase.units as units


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
track_tstep = parameter('track_tstep',5) #timestep between tracked atom dumping
lmp = lammps()
restart = parameter('restart', False)
checkpointing = parameter('checkpointing',False)
checkpoint_filename = parameter('checkpoint_filename','thin_strip')
initial_checkpoint_num = parameter('initial_checkpoint_num',1)
lammps_set_up_path = parameter('lammps_set_up_path')
cpnum = initial_checkpoint_num
results_path = parameter('results_path','./results')
checkpoint_interval = parameter('checkpoint_interval',1)
kvals = parameter('kvals') #initial stress intensity factor (in MPa sqrt(m))
cb = parameter('cb','None')
multilattice = parameter('multilattice',False)
right_hand_edge_dist = parameter('right_hand_edge_dist',0.0)
crack_reset_pos = parameter('crack_reset_pos',strip_width/2)
lattice = parameter('lattice')
calc = parameter('calc')

initial_K = kvals[0]
if cb == 'None':
    cb = None

sim_time = 0

plot_num = 0
if me == 0:
    os.makedirs(results_path, exist_ok=False)
    tsb = ThinStripBuilder(el,a0,C,calc,lattice,directions,multilattice=multilattice,cb=cb,switch_sublattices=True)

    crack_slab = tsb.build_thin_strip_with_crack(initial_K,strip_width,strip_height,strip_thickness\
                                               ,vacuum,crack_seed_length,strain_ramp_length,track_spacing=track_spacing)
    if not restart:
        tracked_array = crack_slab.arrays['tracked']
        print('Writing crack slab to file "crack.xyz"')
        crack_slab.set_velocities(np.zeros([len(crack_slab),3]))
        ase.io.write('crack.xyz', crack_slab, format='extxyz')
        ase.io.lammpsdata.write_lammps_data('crack.lj',crack_slab,velocities=True)
    else:
        #if restarting
        tracked_array = np.loadtxt('tracked_array.txt',dtype=int)
        crack_slab.arrays['tracked'] = tracked_array
else:
    tracked_array = None
if me == 0:
    tip_pos = crack_seed_length+strain_ramp_length+np.min(crack_slab.get_positions()[:,0])
    K_curr = initial_K

for knum,K in enumerate(kvals):
    if me == 0:
        unscaled_crack = ase.io.lammpsdata.read_lammps_data('crack.lj',atom_style='atomic')
        unscaled_crack.set_pbc([False,False,True])
        ase.io.write(f'{knum}_rescale0.xyz',unscaled_crack)
        #print out y position of last atom of unscaled crack
        print('in pos',unscaled_crack.get_positions()[-1,1])
        rescale_crack = tsb.rescale_K(unscaled_crack,K_curr,K,strip_height,tip_pos)
        #print out y position of last atom of rescaled crack
        print('out pos',rescale_crack.get_positions()[-1,1])
        #re-write lammps data file
        ase.io.lammpsdata.write_lammps_data('crack.lj',rescale_crack,velocities=True,masses=True)
        ase.io.write(f'{knum}_rescale1.xyz',rescale_crack)
    
    K_curr = K
    if knum == 0:
        intial_damp = True
    else:
        intial_damp = False
    for i in range(500):
        #communicate the tracked_array numpy array to all processes
        tracked_array = MPI.COMM_WORLD.bcast(tracked_array,root=0)
        #sychronise all processes
        MPI.COMM_WORLD.Barrier()
        #now set up simulation (but do not run)
        lmp.file(lammps_set_up_path)
        if (i < 10) and (intial_damp):
            #add a lammps command to set a thermostat for all atoms initially, for the first 10 picoseconds
            lmp.command(f'fix therm2 nve_atoms langevin 0.0 0.0 {0.1*(i+1)} 1029')
        atom_ids = np.where(tracked_array>0)[0]
        #make a string of atom ids to pass to lammps
        #atom_id_string = ' '.join1([str(x) for x in atom_ids])
        fnames = []
        for track_group,id in enumerate(atom_ids):
            # need to add 1 to id, as LAMMPS indexing starts from 1
            lmp.command(f'group tracked_{track_group} id {id+1}')
            lmp.command(f'dump track_dump_{track_group} tracked_{track_group} atom {track_tstep} tracked_{track_group}.lammpstrj')
            fnames.append(f'tracked_{track_group}.lammpstrj')
        #run for 1000 timesteps
        lmp.command('run 1000')
        #write temp output file
        lmp.command('write_data simulation_output.temp nocoeff nofix nolabelmap')

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
                    sim_time_tmp += track_tstep
                    atom = t[0]
                    tracked_motion_dict[tracked_array[atom_index]][tstep-1,:] = [sim_time_tmp,atom.position[1],atom.position[0]+tsb.total_added_dist]
            sim_time=sim_time_tmp
            #now append the numpy arrays to seperate txt files based on key
            print(tracked_motion_dict.keys())
            for key in tracked_motion_dict.keys():
                print(key)
                with open(f'{results_path}/tracked_motion_{key}.txt','ab') as f:
                    np.savetxt(f,tracked_motion_dict[key])
            
            #check for copy-pasting
            final_crack_state = ase.io.lammpsdata.read_lammps_data('simulation_output.temp',atom_style='atomic')
            final_crack_state.set_pbc([False,False,True])
            final_crack_state.new_array('groups',crack_slab.arrays['groups'])
            final_crack_state.new_array('tracked',tracked_array)
            final_crack_state.new_array('trackable',crack_slab.arrays['trackable'])
            trackable = final_crack_state.arrays['trackable']
            ase.io.write('final_crack_state.xyz',final_crack_state,format='extxyz')
            #plot out spacing of trackable atoms
            plt.figure()
            plot_num += 1
            plt.plot(final_crack_state.get_positions()[:,0][trackable][:-1],np.diff(final_crack_state.get_positions()[:,0][trackable]),'o')
            plt.savefig(f'{results_path}/trackable_spacing_{plot_num}.png')
            plt.close()

            tmp_bondlength = bondlength
            found_tip=False
            for j in range(10):
                try:
                    print(tmp_bondlength,bulk_nn)
                    bond_atoms = find_tip_coordination(final_crack_state,bondlength=tmp_bondlength,bulk_nn=bulk_nn,calculate_midpoint=True)
                    tip_pos = (final_crack_state.get_positions()[bond_atoms,:][:,0])
                    #print(tip_pos[0],tip_pos[1])
                    assert tip_pos[0]-tip_pos[1] < 1
                    found_tip=True
                    break
                except AssertionError:
                    tmp_bondlength += 0.01
                    #keep trying till a crack tip is found
            
            if not found_tip:
                raise RuntimeError('Lost crack tip!')
            tip_pos = tip_pos[0]
            print(f'Found crack tip at position {tip_pos}')
            #check if tip is within paste_threshold Angstroms of largest x position
            #print(np.max(final_crack_state.get_positions()[:,0]))
            #print(paste_threshold)
            #get the average diff in the area ahead of the crack
            pasted = False
            if tip_pos > (np.max(final_crack_state.get_positions()[:,0])-paste_threshold):
                #if it is, set crop to be the distance between the crack tip and the crack_reset_pos
                crop = tip_pos - crack_reset_pos
                pasted = True
            else:
                #do no pasting and just create a fresh input file identical to old one
                crop = 0
            new_slab = tsb.paste_atoms_into_strip(K_curr,strip_width,strip_height,strip_thickness,vacuum,\
                                            final_crack_state,crop=crop,track_spacing=track_spacing,right_hand_edge_dist=right_hand_edge_dist,
                                            match_cell_length=True)
            new_slab.new_array('masses',final_crack_state.arrays['masses'])
            #ase.io.write(f'{results_path}/new_slab_{plot_num}.xyz',new_slab,format='extxyz')
            ase.io.lammpsdata.write_lammps_data('crack.lj',new_slab,velocities=True,masses=True)

            #finally, get the crack velocity and work out if it's reached steady state
            #only look at atoms which are at least 100 Angstroms behind the crack tip
            mask = (final_crack_state.get_positions()[:,0] < (tip_pos-120)) & (tracked_array>0)
            print(f'CRACK TIP POS:{tip_pos}')
            print(f'SHIFTED CRACK TIP POS:{tip_pos-120}')
            print(f'MASK LEN:{len(np.where(mask)[0])}')
            if len(np.where(mask)[0])>=2:
                prev_2_tracked_idx = np.argsort(final_crack_state.get_positions()[mask,0])[-2:]
                atom_ids = np.where(mask)[0][prev_2_tracked_idx]
                print(atom_ids)
                #read numpy arrays from files
                atom_1_traj = np.loadtxt(f'{results_path}/tracked_motion_{tracked_array[atom_ids[0]]}.txt')
                atom_2_traj = np.loadtxt(f'{results_path}/tracked_motion_{tracked_array[atom_ids[1]]}.txt')
                #calculate crack velocity and check if it's steady
                v, ss_c = tsb.check_steady_state(atom_1_traj,atom_2_traj)
                #exit()
                #use numpy to write to file along with current timestep
                with open(f'{results_path}/steady_state.txt','ab') as f:
                    np.savetxt(f,np.reshape(np.array([sim_time,v,ss_c]),[1,3]))
                    #read the txt and plot
                steady_state_data = np.reshape(np.loadtxt(f'{results_path}/steady_state.txt'),[-1,3])
                plt.figure()
                plt.plot((steady_state_data[:,0]/(10**6)),steady_state_data[:,1],'o')
                plt.xlabel('Time (ns)')
                plt.ylabel('Measured crack velocity (km/s)')
                plt.savefig(f'{results_path}/velocity_steady_state.png')
                plt.close()
                plt.figure()
                plt.plot((steady_state_data[:,0]/(10**6)),steady_state_data[:,2],'o')
                plt.xlabel('Time (ns)')
                plt.ylabel('Steady state measure')
                plt.savefig(f'{results_path}/steady_state_measure.png')
                plt.close()

            tracked_array = new_slab.arrays['tracked']
            if checkpointing and (i+1)%checkpoint_interval == 0:
                cpdir = f'./{checkpoint_filename}/{cpnum}'
                os.makedirs(cpdir, exist_ok=False)
                ase.io.lammpsdata.write_lammps_data(f'{cpdir}/crack.lj',new_slab,velocities=True,masses=True)
                np.savetxt(f'{cpdir}/tracked_array.txt',tracked_array)
                #copy all files called tracked_motion_*.txt in current directory to cpdir
                os.system(f'cp {results_path}/tracked_motion_*.txt {cpdir}')
                #copy steady_state.txt if it exists
                os.system(f'cp {results_path}/steady_state.txt {cpdir}')
                cpnum += 1
            #if pasted:
            #    exit()

MPI.COMM_WORLD.Barrier()
print("Proc %d out of %d procs has" % (me,nprocs),lmp)
MPI.Finalize()