#this is a function used for deposition of atoms using ASE alongside the python interface to LAMMPS
import numpy as np
from ase.units import fs
from ase import Atom
from ase.io.lammpsdata import write_lammps_data, read_lammps_data
import ase.io

def set_up_lammps(lmps,input_file,mass_cmds,potential_cmds):
    #lmps : LAMMPS object
    #input_file : str, path to input file
    #mass_cmds : commands to set atomic masses in LAMMPS
    #potential_cmds : commands to set potential in LAMMPS
    
    # ---------- Initialize Simulation --------------------- 
    lmps.command('clear') 
    lmps.command('dimension 3')
    lmps.command('boundary p p p')
    lmps.command('atom_style atomic')
    lmps.command('units metal')
    lmps.command('comm_style tiled')
    
    #print(input_file)
    #----------Read atoms------------
    lmps.command(f'read_data {input_file}')
    
    #----------Define masses and Interatomic Potential----------------
    lmps.commands_list(mass_cmds)
    lmps.commands_list(potential_cmds)

    #----------Output thermo data----------------
    lmps.command('thermo 20')

    #----------Create dump file----------------
    lmps.command('dump 1 all custom 50 dump.lammpstrj id type x y z vx vy vz')
    #make sure to append
    lmps.command('dump_modify 1 append yes')

def deposit_single_atom(prev_config,atom_energy,atom_mass):
    #convert atom_energy (in eV) to velocity in ASE units. Atom mass is in atomic mass units
    v_mag = np.sqrt(2*(atom_energy)/atom_mass)

    print(f'VELOCITY MAGNITUDE: {v_mag}')
    #this is a velocity magnitude, we need a vector
    theta = np.random.uniform(-np.pi/12,np.pi/12)
    phi = np.random.uniform(-np.pi,np.pi)
    v = np.array([v_mag*np.sin(theta)*np.cos(phi),v_mag*np.sin(theta)*np.sin(phi),-v_mag*np.cos(theta)])

    print(f'VELOCITY VECTOR: {v}')
    #now we need a random point near the top of the surface, a distance away such that the particle should strike within 5fs
    strike_time = 5*fs
    place_distance = np.abs(v[2])*strike_time

    print(f'PLACE DISTANCE: {place_distance}')
    #get the highest point on the surface
    top_z = np.max(prev_config.positions[:,2])
    place_height = top_z + place_distance

    print(f'PLACE HEIGHT: {place_height}')

    #get a random x and y position from the box dimensions
    cell = prev_config.get_cell()
    x = np.random.uniform(0,cell[0,0])
    y = np.random.uniform(0,cell[1,1])

    #now we have a random point near the top of the surface, we can place the atom there
    new_config = prev_config.copy()
    new_atom = Atom('C')
    new_config.append(new_atom)
    new_config.positions[-1,:] = np.array([x,y,place_height])
    old_velocities = new_config.get_velocities()
    old_velocities[-1,:] = v
    new_config.set_velocities(old_velocities)
    #write to xyz
    #ase.io.write('config_with_new_atom.xyz',new_config,parallel=False)

    return new_config

# function 1: deposit atoms
def deposit_atom_sim(rank,prev_config,atom_energy,atom_mass,lmps,mass_cmds,potential_cmds,separation_tol):
    #this function takes a previous configuration, adds an atom at a random point near the top and gives it a velocity
    #with a magnitude found from atom_energy toward the surface
    #it then runs a short MD simulation in NVE to simulate the atom impact
    #after that, it checks to see if the atom peened off the surface. If it did, it tries again from a different point
    #if it didn't, it runs a longer MD simulation in NVT

    #first, attempt to deposit an atom
    atom_depositied = False
    while not atom_depositied:
        if rank == 0:
            new_config = deposit_single_atom(prev_config,atom_energy,atom_mass)
            #write new_config to lammps file
            write_lammps_data('initial_lammps_cfg.lj', new_config, masses=True,velocities=True)
        #set up LAMMPS sim
        set_up_lammps(lmps,'initial_lammps_cfg.lj',mass_cmds,potential_cmds)
        #set up NVE thermostat initially
        lmps.command('fix 1 all nve')
        particle_peened=False
        loops = 0
        while not particle_peened:
            loops += 1
            if loops == 3:
                if rank == 0:
                    print('DEPOSITION SUCCESSFUL')
                atom_depositied = True
                break
            #in this loop, run NVE simulation for 20fs total, checking twice if the particle has peened off the surface
            #set up timestep based on impact velocity, such that atoms only move 0.1 angstroms per timestep
            if rank == 0:
                vzs = new_config.get_velocities()[:,2]
                particle_vz = np.abs(vzs[-1])
                print(f'PARTICLE Z VELOCITY: {particle_vz}')
                timestep = (0.1/(particle_vz))/(fs*1000) #in ps
                print(f'TIMESTEP: {timestep}')

            else:
                timestep = 0
            #communicate timestep to all ranks
            timestep = lmps.comm.bcast(timestep, root=0)
            
            
            lmps.command(f'timestep {timestep}')
            #now run for twice the amount of time it should take for the atom to strike the surface (10fs, by design)
            n_tsteps = int((10/1000)/(timestep))
            lmps.command(f'run {n_tsteps}')
            # get lammps to write out final file
            lmps.command(f'write_data simulation_output.temp nocoeff nofix nolabelmap')
            #read in final file
            if rank == 0:
                sim_output = read_lammps_data('simulation_output.temp',style='atomic')
                #first, check if the particle has peened off or entered the structure
                particle_peened = check_particle_peen(sim_output)
            else:
                sim_output = prev_config
                particle_peened = False
            #communicate particle_peened and sim_output to all ranks
            particle_peened = lmps.comm.bcast(particle_peened, root=0)


    
    #now we know the atom has successfully deposited, we can run an NVT simulation for 700fs with a strong thermostat
    #each 100fs, we dump the output and check for detached rafts, which we delete, and then reset the timestep based on
    #the max atom velocity


    if rank == 0:
        #write lammps data
        write_lammps_data('initial_lammps_cfg.lj', sim_output, masses=True,velocities=True)

    #set up LAMMPS sim
    set_up_lammps(lmps,'initial_lammps_cfg.lj',mass_cmds,potential_cmds)

    #set up langevin thermostat
    lmps.command('fix 1 all nve')
    lmps.command('fix 2 all langevin 300 300 0.1 12345')

    for i in range(7):
        #set up timestep based on max velocity, such that fastest atom only moves 0.1 angstroms per timestep
        if rank == 0:
            vs = np.linalg.norm(sim_output.get_velocities(),axis=1)
            avg_v = np.mean(vs)
            max_v = np.max(vs)
            print(f'AVERAGE VELOCITY: {avg_v}')
            print(f'MAX VELOCITY: {max_v}')
            #if max velocity is much larger than average velocity, then have small timestep
            if (max_v > 1.3*avg_v) and (i>0):
                print(f'MAX VELOCITY: {max_v}')
                timestep = ((0.1)/(max_v))/(fs*1000) #in ps
                print(f'TIMESTEP: {timestep}')
            else:
                timestep = 0.002 #set timestep to 2fs
        else:
            timestep = 0
        #communicate timestep to all ranks
        timestep = lmps.comm.bcast(timestep, root=0)
        #get number of tsteps from dividing 0.1 by timestep
        n_tsteps = int((0.1)/(timestep))
        lmps.command(f'timestep {timestep}')
        #now run for 100fs
        lmps.command(f'run {n_tsteps}')
        # get lammps to write out final file
        lmps.command(f'write_data simulation_output.temp nocoeff nofix nolabelmap')
        #read in final file
        if rank == 0:
            sim_output = read_lammps_data('simulation_output.temp',style='atomic')
            #check for detached rafts
            sim_output = detect_flakes(sim_output,separation_tol)
            #write lammps data
            write_lammps_data('initial_lammps_cfg.lj', sim_output, masses=True,velocities=True)
        #set up LAMMPS sim
        set_up_lammps(lmps,'initial_lammps_cfg.lj',mass_cmds,potential_cmds)
        #set up langevin thermostat
        lmps.command('fix 1 all nve')
        lmps.command('fix 2 all langevin 300 300 0.1 12345')
        
    if rank != 0:
        sim_output = prev_config
    #communicate final config to all ranks
    sim_output = lmps.comm.bcast(sim_output, root=0)
    return sim_output

    
        
def check_particle_peen(struct):
    #check if the -1 atom (the particle) is above the surface of the structure and has a positive z velocity
    #if it does, return True, else return False
    z_vel = struct.get_velocities()[-1,2]
    z_pos = struct.get_positions()[-1,2]
    
    #get the highest point of all particles that are not impacting
    top_z = np.max(struct.get_positions()[:-1,2])

    #if the particle is above this and has a high z velocity, then we say it has peened off
    if (z_pos > top_z) and (z_vel > 0):
        return True
    else:
        return False



    

# function 2: build substrate
def build_substrate(rank,lattice, directions, el, a0, mass, initial_x,initial_y, initial_z,
                    vacuum,n_added_atoms,atom_energy,lmps,mass_cmds,potential_cmds,separation_tol,shift=0):
    #this function constructs a substrate by first creating a structure, and then depositing atoms on top of it
    #it returns the final structure
    unit_slab = lattice(directions=directions,
                            size=(1, 1, 1),
                            symbol=el,
                            pbc=True,
                            latticeconstant=a0)
    
    #build a substrate of size initial_x*initial_y*initial_z
    ax = unit_slab.get_cell()[0,0]
    ay = unit_slab.get_cell()[1,1]
    az = unit_slab.get_cell()[2,2]
    nx = int(initial_x/ax)
    ny = int(initial_y/ay)
    nz = int(initial_z/az)
    slab = unit_slab*[nx,ny,nz]
    slab.positions[:,2] += shift
    slab.wrap()
    #now turn off PBC in z direction and add vacuum above and below
    slab.set_pbc([True,True,False])
    cell = slab.get_cell()
    cell[2,2] = cell[2,2] + 2*vacuum
    slab.set_cell(cell)
    slab.positions[:,2] += vacuum

    #now deposit n_added_atoms atoms on top of the substrate
    print(f'DEPOSITING {n_added_atoms} ATOMS ONTO SUBSTRATE')
    for i in range(n_added_atoms):
        slab = deposit_atom_sim(rank,slab,atom_energy,mass,lmps,mass_cmds,potential_cmds,separation_tol)
    
    return slab
    

# function 3: detect rogue atoms or sets of rogue atoms by looking atoms outside the bulk region
def detect_flakes(structure,dist_tol):
    #this function takes an atomic structure and finds a histogram of z positions. It checks to see if there are any regions of empty
    #space between lumps of atoms (corresponding to 0s on the histogram with a tunable binsize). If there are, it detects a flake
    #and deletes the atoms, returning the structure with the flake removed

    #take the and find a histogram with a binsize of dist_tol
    z_pos = structure.get_positions()[:,2]
    range = np.max(z_pos) - np.min(z_pos)
    bins = int(range/dist_tol)
    hist,bin_edges = np.histogram(z_pos,bins=bins,density=False)
    
    bin_midpoints = bin_edges[:-1] + np.diff(bin_edges)/2
    #now get the bins with 0 and corresponding z positions
    zero_bins = np.where(hist==0)
    zero_bin_midpoints = bin_midpoints[zero_bins]

    if len(zero_bin_midpoints)>0:
        print('FLAKE DETECTED')
        slice_z_pos = zero_bin_midpoints[-1]
        flake_mask = z_pos > slice_z_pos
        #remove flake_mask from structure
        structure = structure[~flake_mask]


    return structure



