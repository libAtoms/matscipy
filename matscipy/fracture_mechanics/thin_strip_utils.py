import numpy as np
from ase.constraints import ExpCellFilter
from ase.optimize import LBFGS
import ase.io
import ase.io.lammpsdata
import ase.units as units
from matscipy.elasticity import youngs_modulus, poisson_ratio
from matscipy.fracture_mechanics.crack import G_to_strain, thin_strip_displacement_y
from matscipy.fracture_mechanics.clusters import set_groups
from matscipy.fracture_mechanics.crack import find_tip_coordination, find_tip_non_centred
import matplotlib.pyplot as plt
from matscipy.cauchy_born import CubicCauchyBorn
import warnings

class ThinStripBuilder:

    def __init__(self,el,a0,C,calc,lattice,directions,cb=None,multilattice=False,switch_sublattices=False):
        self.el = el
        self.a0 = a0
        self.C = C
        self.directions = directions
        self.E = youngs_modulus(C, directions[1])
        print('youngs modulus', self.E)

        self.multilattice = multilattice
        self.calc = calc
        self.lattice = lattice
        self.y_strain = 0
        self.total_added_dist = 0
        self.y_buffer_unit_cells = 0 #number of unit cells to add to top and bottom in y and fix
        if self.multilattice:
            assert cb is not None, 'Must provide a Cauchy Born object if multilattice is True'
            self.cb = cb
            #build rotation matrix
            self.A = np.zeros([3, 3])
            for i, direc in enumerate(directions):
                self.A[:, i] = direc / np.linalg.norm(direc)
            if switch_sublattices:
                self.switch_sublattices = True
        
        #self.nu = self.measure_poisson_ratio()
        #print('measured poisson ratio',self.nu)
        #get effective modulus from poisson ratios
        #
        nu_yx = poisson_ratio(C, directions[1], directions[0])
        nu_zx = poisson_ratio(C, directions[2], directions[0])
        nu_zy = poisson_ratio(C, directions[2], directions[1])
        nu_xy = poisson_ratio(C, directions[0], directions[1])
        nu_xz = poisson_ratio(C, directions[0], directions[2])
        nu_yz = poisson_ratio(C, directions[1], directions[2])

        nu_prime = (nu_yx + (nu_yz*nu_zx))/(nu_yz + (nu_yx*nu_xz))
        print('nu_prime',nu_prime)
        self.E_effective = self.E/(1-(nu_zy + nu_xy*nu_prime)*((nu_yz)/(1-nu_prime*nu_xz)))
        print('E',self.E)
        print('E effective',self.E_effective)
        self.nu = nu_yx
        self.nu_xy = nu_xy
        self.nu_yx = nu_yx

    def measure_energy_strain_relation(self,resolution=100,return_s_s=False):
        supercell = self.lattice(size=[1,1,1],symbol=self.el,latticeconstant=self.a0,pbc=(1,1,1),directions=self.directions)
        supercell.calc = self.calc
        strains = np.linspace(0,0.3,resolution)
        stresses = []
        running_strains = []
        energies_per_unit_vol = []
        for strain in strains:
            new_cell = supercell.copy()
            new_cell.set_cell(new_cell.get_cell()*[1,1+strain,1],scale_atoms=True)
            new_cell.calc = self.calc
            #now relax
            opt = LBFGS(new_cell)
            opt.run(fmax=1e-3)
            stresses.append(new_cell.get_stress()[1])
            running_strains.append(strain)
            #get energy per unit volume
            energies_per_unit_vol.append(np.trapz(stresses,running_strains))
        #print('plotting graph...')
        #plot stresses and strains
        #plt.plot(running_strains,stresses)
        #plt.xlabel('strain')
        #plt.ylabel('stress')
        #plt.savefig('stress_strain_relation.png')
        #print('done!')
        #now integrate to get energy per unit volume
        print('energies per unit vol',energies_per_unit_vol)
        energy_strain_relation = np.zeros([len(strains),2])
        energy_strain_relation[:,0] = strains
        energy_strain_relation[:,1] = energies_per_unit_vol
        self.energy_strain_relation = energy_strain_relation
        if return_s_s:
            return running_strains,stresses

    def interpolate_energy_strain_relation(self,energy):
        #interpolate the energy strain relation to find the strain corresponding to a given energy per unit vol
        return np.interp(energy,self.energy_strain_relation[:,1],self.energy_strain_relation[:,0])
    

    def K_to_strain(self,K,strip_height,approximate=False):
        """
        convert a stress intensity factor (in MPa sqrt(m)) to a strain for a given unstrained strip height
        """
        print('strip_height',strip_height)
        K_ase = (K/1000) * (units.GPa*np.sqrt(units.m))
        #convert to correct G for plane strain
        initial_G = (((K_ase)**2)*(1-((self.nu_xy)*(self.nu_yx)))/(self.E))

        if approximate:
            #Convert G to strain directly, using the assumption that the stress strain curve is linear
            strain = G_to_strain(initial_G, self.E_effective, self.nu, strip_height,effective_modulus=True)
        else:
            #use the energy strain relation to find the strain corresponding to the energy
            #get energy per unit volume
            energy = initial_G/strip_height
            #interpolate the energy strain relation to find the strain corresponding to the energy
            strain = self.interpolate_energy_strain_relation(energy)
        print('strain',strain)
        return strain

    def E_tensor_from_strain(self,strain_y,strain_x):
        E = np.zeros([3,3])
        E[0,0] = strain_y
        E[1,1] = strain_x
        return E

    def measure_poisson_ratio(self):
        y_strain = 0.01
        x_strain = self.get_equilibrium_x_strain(y_strain)
        return -(x_strain/y_strain)
         
    def get_equilibrium_x_strain(self,y_strain):
        if y_strain == self.y_strain:
            return self.x_strain
        else:
            #build an ASE unit cell with the correct strain
            unitcell = self.lattice(
                self.el, latticeconstant=self.a0, size=[1, 1, 1],directions=self.directions)
            #ase.io.write('unitcell_0.xyz',unitcell)
            #set the strain
            cell = unitcell.get_cell()
            a_x_before = cell[0,0]
            print('a_x_before',a_x_before)
            #multiply the y component of the cell by the strain
            cell[1,1] *= (1+y_strain)
            #set the cell, shifting the atoms
            unitcell.set_cell(cell,scale_atoms=True)
            unitcell.set_pbc((True,True,True))
            #set the calculator
            unitcell.calc = self.calc
            #set the expcellfilter constraint, allowing strain to move in x and z direction
            expcellfilter = ExpCellFilter(unitcell, mask=[1, 0, 1, 0, 0, 0])
            #run the optimisation
            opt = LBFGS(expcellfilter)
            opt.run(fmax=1e-3)
            #ase.io.write('unitcell_1.xyz',unitcell)
            #ge the lattice param in x direction
            a_x = unitcell.get_cell()[0,0]
            print('a_x_after',a_x)
            self.y_strain = y_strain
            self.x_strain = (a_x-a_x_before)/a_x_before
        return self.x_strain

    def build_thin_strip(self,width,height,thickness,vacuum):
        """
        Build a thin strip with no strain and no crack
        """
        print('thickness', thickness)
        # now, we build system aligned with requested crystallographic orientation
        unit_slab = self.lattice(directions=self.directions,
                            size=(1, 1, 1),
                            symbol=self.el,
                            pbc=True,
                            latticeconstant=self.a0)
        if self.multilattice:
            self.cb.set_sublattices(unit_slab, self.A)

        # ase.io.write('unit_slab.xyz',unit_slab)
        # center vertically half way along the vertical bond between atoms 0 and 1
        if len(unit_slab) == 1:
            unit_slab *= (1, 2, 1)
        unit_slab.positions[:, 1] += (unit_slab.positions[1, 1] -
                                    unit_slab.positions[0, 1]) / 2.0

        # map positions back into unit cell
        unit_slab.set_scaled_positions(unit_slab.get_scaled_positions())

        # ***** Setup crack slab supercell *****

        # Now we will build the full crack slab system,
        # approximately matching requested width and height
        nx = int(width / unit_slab.cell[0, 0])
        ny = int(height/ unit_slab.cell[1, 1])
        nz = int(thickness/ unit_slab.cell[2, 2])
        if nz == 0:
            nz = 1
        #add additional unit cells to the top and bottom of the slab
        ny += 2*self.y_buffer_unit_cells
        print('nx ny nz', nx,ny,nz)
        #ase.io.write('unitslab.xyz',unit_slab)
        copy_slab = unit_slab.copy()
        # rectangularise copy_cell
        copy_slab_cell = copy_slab.get_cell()
        if (abs(copy_slab_cell[0, 1]) < 0.01 or abs(copy_slab_cell[1, 0]) < 0.01):
            copy_slab_cell[0, 1] = 0.0
            copy_slab_cell[1, 0] = 0.0
            copy_slab.set_cell(copy_slab_cell)
            copy_slab.wrap()
        
        self.single_cell_width = copy_slab.cell[0,0]
        self.single_cell_height = copy_slab.cell[1,1]
        self.min_x_pos_dist = np.min(copy_slab.get_positions()[:,0])
        self.max_x_pos_dist = self.single_cell_width - np.max(copy_slab.get_positions()[:,0])
        #print('min x pos', self.min_x_pos_dist)
        #print('max x pos', self.max_x_pos_dist)
        # make sure ny is even so slab is centered on a bond
        if ny % 2 == 1:
            ny += 1

        # make a supercell of unit_slab
        crack_slab = unit_slab * (1, ny, nz)
        #add an array 'trackable' to crack_slab which is True for the leftmost atom nearest the centre
        #and False for all other atoms. This is for checking steady state
        xpos = crack_slab.get_positions()[:,0]
        ypos = crack_slab.get_positions()[:,1]
        zpos = crack_slab.get_positions()[:,2]
        shifted_ypos = ypos - (ypos.max() + ypos.min())/2
        ordered_in = np.lexsort((zpos,xpos,np.abs(shifted_ypos)))
        trackable = np.zeros(len(crack_slab),dtype=bool)
        trackable[ordered_in[0]] = True
        crack_slab.new_array('trackable',trackable)

        crack_slab = crack_slab * (nx, 1, 1)
        crack_slab.positions += [1,0.1,0]
        crack_slab.wrap()
        # check if cell is square
        # set any off diagonal components of cell vector to 0
        cell = crack_slab.get_cell()
        if (abs(cell[0, 1]) < 0.01 or abs(cell[1, 0]) < 0.01):
            warnings.warn('Non-square thin strip detected, creating square...')
            cell[0, 1] = 0.0
            cell[1, 0] = 0.0
            crack_slab.set_cell(cell)
            crack_slab.wrap()


        # open up the cell along x and y by introducing some vaccum
        crack_slab.center(vacuum, axis=0)
        crack_slab.center(vacuum, axis=1)

        orig_width = (crack_slab.positions[:, 0].max() -
                    crack_slab.positions[:, 0].min())
        orig_height = (crack_slab.positions[:, 1].max() -
                    crack_slab.positions[:, 1].min())
        orig_thickness = (crack_slab.positions[:, 2].max() -
                        crack_slab.positions[:, 2].min())

        print(('Made slab with %d atoms, original width, height and thickness: %.1f x %.1f x %.1f A^2' %
            (len(crack_slab), orig_width, orig_height, orig_thickness)))
        #resort indicies such that they are stable
        order = self.stable_sort_strip(crack_slab)
        crack_slab = crack_slab[order]

        #set self.trackable_atoms_mask to be the atoms which are in the 'trackable' array of crack_slab
        self.trackable_atoms_mask = crack_slab.arrays['trackable']
        #ase.io.write('1_slab.xyz',crack_slab)
        #set groups (useful later for crack tip determination)
        set_groups(crack_slab,[nx,ny,nz],int(nx/5),int(ny/5))
        self.group_array = crack_slab.arrays['groups']
        crack_slab.set_pbc((False,False,True))
        self.nx,self.ny,self.nz = nx,ny,nz
        return crack_slab

    def build_absorbent_test_strip(self,width):
        """
        build a single unit cell thin strip in y and z which stretches
        along x with periodic boundaries in y and z
        """
        # now, we build system aligned with requested crystallographic orientation
        unit_slab = self.lattice(directions=self.directions,
                            size=(1, 1, 1),
                            symbol=self.el,
                            pbc=True,
                            latticeconstant=self.a0)
        
        nx = int(width / unit_slab.cell[0, 0])

        # make a supercell of unit_slab
        test_slab = unit_slab * (nx, 1, 1)
        test_slab.wrap()
        #set periodic boundaries
        test_slab.set_pbc((False,True,True))
        return test_slab

    def get_energy_accessible_to_crack(self,slab,strip_width,strip_height,strip_thickness,vacuum):
        #build a thin strip with no strain
        crack_slab = self.build_thin_strip(strip_width,strip_height,strip_thickness,vacuum)
        crack_slab.calc = self.calc
        #get a mask for the non fixed atoms
        # get the energy in the non fixed part of the slab
        y_pos = crack_slab.positions[:,1]
        x_pos = crack_slab.positions[:,0]
        #get a mask for the non fixed atoms
        #non_fixed_mask = (y_pos < (y_pos.max() - 10)) & (y_pos > (y_pos.min() + 10)) \
        #    & (x_pos > (x_pos.min() + 10)) & (x_pos < (x_pos.max() - 10))
        non_fixed_mask = np.ones(len(crack_slab),dtype=bool)
        slab.calc = self.calc
        #get the energy of the non fixed atoms
        non_fixed_energy = slab.get_potential_energies()[non_fixed_mask].sum() - crack_slab.get_potential_energies()[non_fixed_mask].sum()
        return non_fixed_energy
            
    def build_thin_strip_with_strain(self,K,width,height,thickness,vacuum,apply_x_strain=False,force_spacing=None,approximate=False):
        #force spacing is the width of a single unit cell to force upon the slab
        crack_slab = self.build_thin_strip(width,height,thickness,vacuum)
        #get true height
        true_height = self.get_true_height(height)
        strain = self.K_to_strain(K,true_height,approximate=approximate)
        #shift crack
        # centre the slab on the origin
        xmean = crack_slab.positions[:, 0].mean()
        ymean = crack_slab.positions[:, 1].mean()
        crack_slab.positions[:, 0] -= xmean
        crack_slab.positions[:, 1] -= ymean

        crack_slab.positions[:, 1] += strain*crack_slab.positions[:, 1]
        #now apply the Poisson strain in the x direction
        #print('NU',self.nu)
        if apply_x_strain:
            if force_spacing is None:
                #x_strain = -self.nu*strain
                x_strain = self.get_equilibrium_x_strain(strain)
            else:
                x_strain = (force_spacing-self.single_cell_width)/self.single_cell_width
                print('effective x strain', x_strain)
        else:
            x_strain = 0

        crack_slab.positions[:, 0] += x_strain*crack_slab.positions[:, 0]
        
        # undo crack shift
        crack_slab.positions[:, 0] += xmean
        crack_slab.positions[:, 1] += ymean

        #add cauchy-born shift correction if crack is a multi-lattice
        if self.multilattice:
            #set sublattices
            self.cb.set_sublattices(crack_slab, self.A, read_from_atoms=True)
            if self.switch_sublattices:
                self.cb.switch_sublattices(crack_slab)
            E_3x3 = self.E_tensor_from_strain(strain,x_strain)
            shifts = self.cb.find_exact_shift_for_homogeneous_field(E_3x3, crack_slab, self.directions)
            #apply shifts
            self.cb.apply_shifts(crack_slab, shifts)

        #ase.io.write('2_slab.xyz',crack_slab)
        #add the x strain as info to crack_slab
        crack_slab.info['x_strain'] = x_strain
        return crack_slab


    def build_thin_strip_with_crack(self,K,width,height,thickness,vacuum,crack_seed_length,strain_ramp_length,track_spacing=0,apply_x_strain=False,approximate=False):
        """
        build a thin strip at some defined strain with a crack of some defined length
        When building the crack, add tracked atoms every track_spacing along x 
        """
        #build a thin strip with no strain
        crack_slab = self.build_thin_strip(width,height,thickness,vacuum)
        #get true height
        true_height = self.get_true_height(height)
        slab_length = crack_slab.positions[:,0].max() - crack_slab.positions[:,0].min()
        crack_length = crack_seed_length + strain_ramp_length
        crack_tip_pos = crack_slab.positions[:,0].min() + crack_length
        strain = self.K_to_strain(K,true_height,approximate=approximate)

        if track_spacing>0:
            atoms_to_track=[]
            #only track atoms if track_spacing is greater than 0
            trackable_atoms = np.where(crack_slab.arrays['trackable'])[0]
            trackable_atom_pos_x = crack_slab[crack_slab.arrays['trackable']].get_positions()[:,0]
            n_atoms_to_track = int(np.floor((slab_length-crack_length)/track_spacing))
            for i in range(n_atoms_to_track):
                #find the closest atom to tip + i*track_spacing
                atoms_to_track.append(trackable_atoms[np.argmin(np.abs(trackable_atom_pos_x-(crack_tip_pos+i*track_spacing)))])
            
            crack_slab.new_array('tracked',np.zeros(len(crack_slab),dtype=int))
            crack_slab.arrays['tracked'][atoms_to_track] = range(1,n_atoms_to_track+1)
            self.tracked_atoms_y0 = crack_slab[crack_slab.arrays['tracked']].get_positions()[0,1]
        else:
            #track no atoms
            crack_slab.new_array('tracked',np.zeros(len(crack_slab),dtype=int))

        #shift crack
        # centre the slab on the origin
        xmean = crack_slab.positions[:, 0].mean()
        ymean = crack_slab.positions[:, 1].mean()
        crack_slab.positions[:, 0] -= xmean
        crack_slab.positions[:, 1] -= ymean

        left = crack_slab.positions[:, 0].min()
        crack_slab.positions[:, 1] += thin_strip_displacement_y(
                                        crack_slab.positions[:, 0],
                                        crack_slab.positions[:, 1],
                                        strain,
                                        left + crack_seed_length,
                                        left + crack_seed_length +
                                                strain_ramp_length)
        
        if apply_x_strain:
            #first part has no associated Poisson strain
            #last part has easy to calculate Poisson strain
            #middle part has a poisson strain gradient
            a = left + crack_seed_length
            b = left + crack_seed_length + strain_ramp_length
            y = crack_slab.positions[:,1]
            x = crack_slab.positions[:,0]
            u_x = np.zeros_like(x)
            #get x_strain
            x_strain = self.get_equilibrium_x_strain(strain)
            u_x[x>b] = x_strain*x[x>b]
            middle = (x >= a) & (x <= b)
            f = (x[middle] - a) / (b - a)
            u_x[middle] = (x_strain * f * x[middle])
            crack_slab.positions[:,0] += u_x


        # undo crack shift
        crack_slab.positions[:, 0] += xmean
        crack_slab.positions[:, 1] += ymean
        #write to file
        #ase.io.write('3_slab.xyz',crack_slab)
        return crack_slab


    def stable_sort_strip(self,atoms):
        #take a set of atom and sort by x, y and z
        pos = atoms.get_positions()
        x, y, z = pos[:,0],pos[:,1], pos[:,2]
        order = np.lexsort((z, x, y))
        #order = np.lexsort((z, y, x))
        # z x y seems to work well as a stable sort order
        # in most cases (including triangle lattice)
        #(z, y, x)
        return order
        
        
        
    def paste_atoms_into_strip(self,K,width,height,thickness,vacuum,old_atoms,crop,track_spacing=0,right_hand_edge_dist=0,match_cell_length=False,approximate=False):
        """
        build a thin strip at some defined strain, and paste in the old atoms at the left hand edge of the strip
        Crop is the amount to be cut and pasted in
        Right hand edge dist is the amount at the right hand edge to be preserved.
        """

        trackable = old_atoms.arrays['trackable']
        #get a mask for between the final 80 and 50 angstroms in x of the old_atoms
        #print('crop and rhed',crop,right_hand_edge_dist)

        mask = (old_atoms.positions[:,0]>(np.max(old_atoms.positions[:,0])-(30+right_hand_edge_dist))) & (old_atoms.positions[:,0]<(np.max(old_atoms.positions[:,0])-(right_hand_edge_dist)))
        #print('mask len',len(np.where(mask)[0]))
        mask = mask & trackable
        avg_cell_length = np.mean(np.diff(old_atoms.get_positions()[:,0][mask]))
        print('avg cell length', avg_cell_length)

        #copy old_atoms to avoid overwriting object
        old_atoms = old_atoms.copy()
        #build a thin strip at some defined strain
        new_strip = self.build_thin_strip_with_strain(K,width,height,thickness,vacuum,apply_x_strain=False,approximate=approximate) #,force_spacing=avg_cell_length)
        #ase.io.write('new_strip_temp.xyz',new_strip)
        paste_num_cells = int(crop/self.single_cell_width)
        right_hand_edge_cells = int(right_hand_edge_dist/self.single_cell_width)
        print(f'Copying and pasting {paste_num_cells} unit cells')
        print(f'Preserving {right_hand_edge_cells} unit cells on the right')
        #strain = self.K_to_strain(K,height)
        right_hand_edge_dist = right_hand_edge_cells*self.single_cell_width
        crop = paste_num_cells*self.single_cell_width
        
        x_strain = new_strip.info['x_strain']
        print('x_strain',x_strain)
        max_x_pos_dist = self.max_x_pos_dist + (self.max_x_pos_dist*x_strain)
        print('max x pos dist',max_x_pos_dist)
        min_x_pos_dist = self.min_x_pos_dist + (self.min_x_pos_dist*x_strain)
        print('min x pos dist',min_x_pos_dist)
        crop += x_strain*crop
        #right_hand_edge_dist -= self.nu*strain*right_hand_edge_dist
        #crop += right_hand_edge_dist

        pos = new_strip.get_positions()

        right_edge_pos = np.max(pos[:,0])-right_hand_edge_dist+max_x_pos_dist-0.05
        
        #align the old_atom positions such that the right most atom aligns with
        #the right most atom of the new strip
        #ase.io.write('new_strip_init.xyz',new_strip)
        #make a copy of the old_atoms
        old_atoms_non_shifted = old_atoms.copy()
        # ase.io.write('old_atoms_1.xyz',old_atoms)
        #shift all the old_atoms positions backwards by crop
        old_atoms.positions[:,0] -= crop
        # ase.io.write('old_atoms_2.xyz',old_atoms)
        diff = (np.max(pos[:,0][pos[:,0]<(right_edge_pos-crop)])-np.max(old_atoms.positions[:,0][pos[:,0]<right_edge_pos]))
        # print('diff',diff)
        old_atoms_non_shifted.positions[:,0] += diff
        old_atoms.positions[:,0] += diff
        # ase.io.write('old_atoms_3.xyz',old_atoms)
        
        print('minmax',min_x_pos_dist, max_x_pos_dist)
        min_crop = np.min(pos[:,0]) + crop - min_x_pos_dist - 0.05
        max_crop = np.max(pos[:,0]) - crop + max_x_pos_dist - 0.05

        print('min_crop', min_crop)
        print('max crop', max_crop)
        new_strip_copy = new_strip.copy()
        new_strip_copy.new_array('crop', np.zeros(len(old_atoms),dtype=int))
        crop_arr = new_strip_copy.arrays['crop']
        crop_arr[pos[:,0]<(max_crop-right_hand_edge_dist)] += 1 
        crop_arr[((pos[:,0]>min_crop)&(pos[:,0]<right_edge_pos))] += 1
        # ase.io.write('crop_arrfile.xyz',new_strip_copy)

        new_pos = new_strip.get_positions()
        old_pos = old_atoms.get_positions()
        old_v = old_atoms.get_velocities()
        old_tracked = old_atoms.arrays['tracked']
        new_strip.new_array('tracked',np.zeros(len(new_strip),dtype=int))
        new_v = np.zeros_like(old_v)
        # ase.io.write('old_atoms_4.xyz',old_atoms)
        #shift the atoms getting tracked backwards
        # new_strip.arrays['pos_less_than_max_crop'] = np.zeros(len(new_strip),dtype=bool)
        # new_strip.arrays['pos_less_than_max_crop'][pos[:,0]<max_crop] = True
        # new_strip.arrays['pos_greater_than_min_crop'] = np.zeros(len(new_strip),dtype=bool)
        # new_strip.arrays['pos_greater_than_min_crop'][pos[:,0]>min_crop] = True
        # new_strip.arrays['old_tracked'] = old_tracked
        new_strip.arrays['tracked'][pos[:,0]<max_crop] = old_tracked[pos[:,0]>min_crop]
        # assert that all tracked atoms are also trackable atoms
        # ase.io.write('new_strip_temp.xyz',new_strip)
        #print(new_strip.arrays['trackable'][new_strip.arrays['tracked']>0])
        assert np.all(new_strip.arrays['trackable'][new_strip.arrays['tracked']>0]), 'Not all tracked atoms are trackable - sort order issue?'
        #add new atoms for tracking if necessary
        if track_spacing>0:
            new_atoms_to_track=[]
            trackable_atoms = np.where(new_strip.arrays['trackable'])[0]
            trackable_atom_pos_x = new_strip[new_strip.arrays['trackable']].get_positions()[:,0]
            max_tracked_atom_pos_x = np.max(new_strip[new_strip.arrays['tracked']>0].get_positions()[:,0])
            #find distance between last tracked atom right edge of strip
            dist_to_right_edge = np.max(pos[:,0]) - max_tracked_atom_pos_x
            n_new_atoms_to_track = int(np.floor(dist_to_right_edge/track_spacing))
            for i in range(n_new_atoms_to_track):
                new_atoms_to_track.append(trackable_atoms[np.argmin(np.abs(trackable_atom_pos_x-\
                                                                           (max_tracked_atom_pos_x + (i+1)*track_spacing)))])
            new_strip.arrays['tracked'][new_atoms_to_track] = range(np.max(new_strip.arrays['tracked'])+1,
                                                                    np.max(new_strip.arrays['tracked'])+1+n_new_atoms_to_track)
        
        
        #paste in atom positions and velocities
        new_pos[pos[:,0]<(max_crop-right_hand_edge_dist)] = old_pos[((pos[:,0]>min_crop)&(pos[:,0]<right_edge_pos))]
        new_v[pos[:,0]<(max_crop-right_hand_edge_dist)] = old_v[((pos[:,0]>min_crop)&(pos[:,0]<right_edge_pos))]

        #add the length of the new segment to the total crop distance total
        added_atoms_mask = (pos[:,0]>(max_crop-right_hand_edge_dist)) & ((pos[:,0]<right_edge_pos))
        if len(pos[:,0][added_atoms_mask])>0:
            self.total_added_dist += np.max(pos[:,0][added_atoms_mask]) - np.min(pos[:,0][added_atoms_mask]) + min_x_pos_dist + max_x_pos_dist - diff
        
        old_pos_non_shifted = old_atoms_non_shifted.get_positions()
        new_pos[pos[:,0]>right_edge_pos] = old_pos_non_shifted[pos[:,0]>right_edge_pos]
        new_v[pos[:,0]>right_edge_pos] = old_v[pos[:,0]>right_edge_pos]
        new_strip.set_positions(new_pos)
        new_strip.set_velocities(new_v)
        return new_strip
    
    def get_true_height(self,strip_height):
        """
        get the true height of a strip with no strain
        """
        ny = int(strip_height/self.single_cell_height)
        if ny % 2 == 1:
            ny += 1
        true_height = ny*self.single_cell_height
        return true_height    

    def rescale_K(self,atoms,K_old,K_new,strip_height, tip_position,approximate=False):
        """
        rescale the stress intensity factor of the crack atoms
        """
        crack_atoms = atoms.copy()
        #get the strain corresponding to the old K and new K
        true_strip_height = self.get_true_height(strip_height)
        strain_old = self.K_to_strain(K_old,true_strip_height,approximate=approximate)
        strain_new = self.K_to_strain(K_new,true_strip_height,approximate=approximate)

        #rescale the y positions of the crack atoms by the ratio of the strains, taking the midpoint at 0 for atoms beyond tip position
        midpoint = (crack_atoms.positions[:,1].max() + crack_atoms.positions[:,1].min())/2
        beyond_tip_mask = crack_atoms.positions[:,0] > tip_position
        behind_tip_mask = np.logical_not(beyond_tip_mask)
        crack_atoms.positions[:,1][beyond_tip_mask] += (crack_atoms.positions[:,1][beyond_tip_mask] - midpoint)*(strain_new-strain_old)
        #rescale the x positions of the atoms by the poisson strain
        # crack_atoms.positions[:,0][beyond_tip_mask] = crack_atoms.positions[:,0][beyond_tip_mask] - self.nu*(strain_new-strain_old)*crack_atoms.positions[:,0][beyond_tip_mask]
        #for atoms behind the crack tip, change the y displacement of top and bottom halves by half the height times strain_new/strain_old
        y_disp_change = (strain_new-strain_old)*strip_height/2
        #mask for those lower than the midpoint
        lower_mask = (crack_atoms.positions[:,1] < midpoint) & behind_tip_mask
        higher_mask = (crack_atoms.positions[:,1] > midpoint) & behind_tip_mask
        crack_atoms.positions[:,1][lower_mask] = crack_atoms.positions[:,1][lower_mask] - y_disp_change
        crack_atoms.positions[:,1][higher_mask] = crack_atoms.positions[:,1][higher_mask] + y_disp_change

        return crack_atoms

    
    def check_steady_state(self,atom_1_traj,atom_2_traj,y_threshold=1):
        """
        take the y position as a function of time for 2 atoms and check to what extent they are in steady state
        atom_1_traj and atom_2_traj are both 2 dimensional arrays, with one column being simulation time, and the other
        being the y position of the atom
        """
        #take the MD trajectories of two atoms and check to what extent they are in steady state
        # first, estimate velocity by finding the times at which the atoms cross a y displacement of 1 Angstrom

        #find the times at which the atoms cross a y displacement of 1 Angstrom
        mask_traj_1 = np.abs(atom_1_traj[:,1]-atom_1_traj[0,1])>y_threshold
        mask_traj_2 = np.abs(atom_2_traj[:,1]-atom_2_traj[0,1])>y_threshold
        print('1')
        print(atom_1_traj[:,1]-atom_1_traj[0,1])
        print('2')
        print(atom_2_traj[:,1]-atom_2_traj[0,1])
        num_points = min(np.shape((atom_2_traj[mask_traj_2]))[0],1000)
        print('num points',num_points)
        masks = [mask_traj_1,mask_traj_2]
        atom_trajs = [atom_1_traj,atom_2_traj]
        break_tsteps = []
        x_pos = []
        for i,atom_traj in enumerate(atom_trajs):
            break_tsteps.append(atom_traj[:,0][masks[i]][0])
            x_pos.append(atom_traj[:,2][masks[i]][0])
        dist_between_atoms = x_pos[1]-x_pos[0]
        #compute velocity from dist_between_atoms and the time diff between times
        v = dist_between_atoms/(break_tsteps[1]-break_tsteps[0])
        #convert v from A/ps to km/s
        v_kms = v*(10**-1)
        #print velocity
        print(f'Velocity is {v_kms} km/s')
        #find sum squared overlap between trajectories for 5 ps
        diff = atom_2_traj[:,1][mask_traj_2][:num_points] - atom_1_traj[:,1][mask_traj_1][:num_points]
        #find 2 norm
        steady_state_criterion = (np.linalg.norm(diff)/num_points)*1000
        print(f'Steady state value is {steady_state_criterion}')
        return v_kms, steady_state_criterion
    
    def cut_out_tip_region(self,crack_atoms,strip_width,strip_height,strip_thickness,cut_out_width,cut_out_height,bondlength,bulk_nn,return_base_strip=False):
        #first, build a thin strip which matches the width, height and thickness supplied
        base_strip = self.build_thin_strip(strip_width,strip_height,strip_thickness,vacuum=0)
        #once base strip is built, now get the tip atoms from crack_atoms
        bond_atoms = find_tip_coordination(crack_atoms,bondlength=bondlength,bulk_nn=bulk_nn,calculate_midpoint=True)
        #now we have the bond atoms, build a mask of width cut_out_width and height cut_out_height
        #around the positions of the average of the bond atoms in base_strip
        #create new array, set bond atoms to 1 and write to file
        crack_atoms.new_array('bond_atoms',np.zeros(len(crack_atoms),dtype=int))
        crack_atoms.arrays['bond_atoms'][bond_atoms[0]] = 1
        crack_atoms.arrays['bond_atoms'][bond_atoms[1]] = 1
        ase.io.write('bond_atoms.xyz',crack_atoms)

        tip_x = (base_strip.get_positions()[bond_atoms[0],0] + base_strip.get_positions()[bond_atoms[1],0])/2
        tip_y = (base_strip.get_positions()[bond_atoms[0],1] + base_strip.get_positions()[bond_atoms[1],1])/2

        mask = (base_strip.get_positions()[:,0] > (tip_x - cut_out_width/2)) & (base_strip.get_positions()[:,0] < (tip_x + cut_out_width/2)) \
            & (base_strip.get_positions()[:,1] > (tip_y - cut_out_height/2)) & (base_strip.get_positions()[:,1] < (tip_y + cut_out_height/2))
        
        #now we have the mask, we can cut out the tip region from crack_atoms
        cut_base_strip = base_strip[mask]
        cut_out_atoms = crack_atoms[mask]

        #now we need to perform another stable sort on the cut_base_strip
        order = self.stable_sort_strip(cut_base_strip)

        #we can now use this to re-index the cut_out_atoms
        cut_out_atoms = cut_out_atoms[order]
        cut_base_strip = cut_base_strip[order]
        
        self.cut_out_mask = mask
        self.cut_out_new_order = order

        if return_base_strip:
            return cut_out_atoms, cut_base_strip
        else:
            return cut_out_atoms
        
    def draw_strip_potential_boundary(self,new_slab,partition_width,buffer_thickness,strip_width,strip_height,strip_thickness,vacuum):
        basic_slab = self.build_thin_strip(strip_width,strip_height,strip_thickness,vacuum)
        pos = basic_slab.get_positions()
        #get the atoms within +- partition_width/2 of the centre in y
        mid_point_y = (np.max(pos[:,1]) + np.min(pos[:,1]))/2
        mask = (pos[:,1] > (mid_point_y - partition_width/2)) & (pos[:,1] < (mid_point_y + partition_width/2))
        #set these atoms to be expensive, by marking the potential type as 1, and make all other atoms 2
        new_slab.arrays['potential'] = np.array([1 if x else 2 for x in mask])

        #now set the buffer atoms
        top_mask = (pos[:,1] > (mid_point_y + partition_width/2-buffer_thickness)) & (pos[:,1] < (mid_point_y + partition_width/2 + buffer_thickness))
        bot_mask = (pos[:,1] > (mid_point_y - partition_width/2 - buffer_thickness)) & (pos[:,1] < (mid_point_y - partition_width/2 + buffer_thickness))
        new_slab.arrays['buffer'] = np.array([1 if x else 0 for x in (top_mask|bot_mask)])


    def find_strip_crack_tip(self,final_crack_state,bondlength,bulk_nn,step_tolerant=False):
        tmp_bondlength = bondlength
        found_tip=False
        final_crack_state.new_array('crack_tip',np.zeros(len(final_crack_state),dtype=int))
        full_height = np.max(final_crack_state.positions[:,1])-np.min(final_crack_state.positions[:,1])
        
        check_num = 10
        for j in range(check_num):
            try:
                if not step_tolerant:
                    bond_atoms = find_tip_coordination(final_crack_state,bondlength=tmp_bondlength,bulk_nn=bulk_nn,calculate_midpoint=True)
                else:
                    bond_atoms = find_tip_non_centred(final_crack_state, bondlength=tmp_bondlength, bulk_nn=bulk_nn, nz=self.nz)
                tip_pos = (final_crack_state.get_positions()[bond_atoms,:][:,0])
                tip_pos_y = (final_crack_state.get_positions()[bond_atoms,:][:,1])
                final_crack_state.arrays['crack_tip'][bond_atoms[0]] = 1
                final_crack_state.arrays['crack_tip'][bond_atoms[1]] = 1
                # ase.io.write(f'crack_tip_{j}.xyz',final_crack_state)
                #print(tip_pos[0],tip_pos[1])
                if not step_tolerant:
                    #crack atoms must strictly be opposite
                    assert np.abs(tip_pos[0]-tip_pos[1]) < 1
                # else:
                #     #crack atoms must be within 2 bond lengths of each other, as things are a bit messier
                #     # assert np.abs(tip_pos[0]-tip_pos[1]) < 2*bondlength
                
                found_tip = True
                if not step_tolerant:
                    crack_tip_position_x = np.mean(tip_pos)
                    crack_tip_position_y = np.mean(tip_pos_y)
                else:
                    crack_tip_position_x = np.max(tip_pos)
                    crack_tip_position_y = np.mean(tip_pos_y)
                break
            except AssertionError:
                tmp_bondlength += 0.01
                #keep trying till a crack tip is found
    
        #crack tip pos is the maximum of tips_found[i]
        if not found_tip:
            # if not step_tolerant:
            raise RuntimeError('Lost crack tip!')
            # else:
            #     warnings.warn("No well-defined crack tip; things are messy! Guessing at furthest right low-coordinated atom")
            #     crack_tip_position_x = np.max(tip_pos)
            #     crack_tip_position_y = np.mean(tip_pos_y)

        

        print(f'Found crack tip at position {crack_tip_position_x}')
        
        return crack_tip_position_x, crack_tip_position_y, bond_atoms

def write_potential_and_buffer(atoms,lammps_filename):
    #get the potential and buffer array from atoms
    potential = atoms.arrays['potential']
    buffer = atoms.arrays['buffer']
    ids = np.arange(1,len(atoms)+1)

    #print the number of potential and buffer atoms of each type
    for ntype in np.unique(potential):
        print(f'Number of potential atoms of type {ntype}: {np.sum(potential==ntype)}')
        print(f'Number of potential atoms of type {ntype} including those in buffer: {np.sum((potential==ntype)|(buffer==1))}')


    #write to file
    #create a vertical array of ids, potential
    pot_arr = np.vstack((ids,potential)).T
    with open(lammps_filename, 'ab') as file:
        file.write(b'\n\n\nEvalPotential\n\n')
        np.savetxt(file, pot_arr, fmt='%i')

    #do the same with buffer
    #create a vertical array of ids, buffer
    buffer_arr = np.vstack((ids,buffer)).T
    with open(lammps_filename, 'ab') as file:
        file.write(b'\n\n\nBufferAtoms\n\n')
        np.savetxt(file, buffer_arr, fmt='%i')
    return



def set_up_simulation_lammps(lmps,tmp_file_path,atomic_mass,calc_commands,
                             sim_tstep=0.001,damping_strength_right=0.1,damping_strength_left=0.1,dump_freq=100, dump_files=True,
                             dump_name='dump.lammpstrj',thermo_freq=100,left_damp_thickness=60,
                             right_damp_thickness=60,multi_potential=False, y_fixed_length=1,
                             bond_topology=False, weak_damp_thickness_ratio=0, weak_damp_factor=1):
    """Set up the simulation by passing an active LAMMPS object a number of commands"""
    
    # ---------- Initialize Simulation --------------------- 
    lmps.command('clear') 
    lmps.command('dimension 3')
    lmps.command('boundary s s p')
    if bond_topology:
        lmps.command('atom_style full')
    else:
        lmps.command('atom_style atomic')
    lmps.command('units metal')
    lmps.command('atom_modify map yes')

    #----------Read atoms------------
    if multi_potential:
        lmps.command('fix eval_pot all property/atom i_potential ghost yes')
        lmps.command('fix eval_buffer all property/atom i_buffer ghost yes')
        lmps.command(f'read_data {tmp_file_path}/crack.lj fix eval_pot NULL EvalPotential fix eval_buffer NULL BufferAtoms')
    else:
        lmps.command(f'read_data {tmp_file_path}/crack.lj')
                 

    #---------- Determine Lowest and Highest y-coordinates ----------
    lmps.command('group crack type 1')
    lmps.command('variable ymax equal bound(crack,ymax)')
    lmps.command('variable ymin equal bound(crack,ymin)')
    lmps.command('variable xmax equal bound(crack,xmax)')
    lmps.command('variable xmin equal bound(crack,xmin)')
    #define a variable for average y position of crack atoms using ymax and ymin
    lmps.command('variable ymid equal (v_ymax+v_ymin)/2')
    #----------Define potential-------
    lmps.command(f'mass 1 {atomic_mass}')

    
    ############ set up potential ################
    #set up both potentials, ensuring that interactions between types are defined
    lmps.commands_list(calc_commands)

    #---------- Define balance fix to spread load between processors
    lmps.command(f'fix b_fix all balance 50 1.2 shift xy 10 1.05 weight time 1.1')

    #---------- Define Regions for Boundary Layers ----------
    lmps.command(f'region bottom_layer block INF INF $((v_ymin-2)) $((v_ymin+{y_fixed_length})) INF INF')
    lmps.command(f'region top_layer block INF INF $((v_ymax)-{y_fixed_length}) $(v_ymax+2) INF INF')
    lmps.command('region left_layer_fixed block $((v_xmin-2)) $((v_xmin+5)) INF INF INF INF')
    lmps.command('region right_layer_fixed block $((v_xmax-5)) $((v_xmax+2)) INF INF INF INF')
    lmps.command(f'region left_layer_thermostat_strong block $((v_xmin-2)) $((v_xmin+{left_damp_thickness/(weak_damp_thickness_ratio+1)})) INF INF INF INF')
    lmps.command(f'region right_layer_thermostat_strong block $((v_xmax-{right_damp_thickness/(weak_damp_thickness_ratio+1)})) $((v_xmax+2)) INF INF INF INF')

    lmps.command(f'region left_layer_thermostat_weak block $((v_xmin+{left_damp_thickness/(weak_damp_thickness_ratio+1)})) $((v_xmin+{left_damp_thickness})) INF INF INF INF')
    lmps.command(f'region right_layer_thermostat_weak block $((v_xmax-{right_damp_thickness})) $((v_xmax-{right_damp_thickness/(weak_damp_thickness_ratio+1)})) INF INF INF INF')

    #---------- Set groups for boundary layers ----------
    # Identify atoms within 1 unit of the top and bottom boundaries
    lmps.command('group top_atoms region top_layer')
    lmps.command('group bottom_atoms region bottom_layer')
    lmps.command('group left_atoms region left_layer_fixed')
    lmps.command('group right_atoms region right_layer_fixed')
    lmps.command('group left_thermo_weak region left_layer_thermostat_weak')
    lmps.command('group right_thermo_weak region right_layer_thermostat_weak')
    lmps.command('group left_thermo_strong region left_layer_thermostat_strong')
    lmps.command('group right_thermo_strong region right_layer_thermostat_strong')
    #lmps.command('group top_atoms subtract top_atoms_full right_atoms')
    #lmps.command('group bottom_atoms subtract bottom_atoms_full right_atoms')

    # --------- Fix the edge atoms to prevent movement -----------
    lmps.command('fix 1 top_atoms setforce 0.0 0.0 NULL')
    lmps.command('fix 2 bottom_atoms setforce 0.0 0.0 NULL')
    lmps.command('fix 3 left_atoms setforce 0.0 0.0 NULL')
    lmps.command('fix 4 right_atoms setforce 0.0 0.0 NULL')
    lmps.command('velocity left_atoms set 0.0 0.0 0.0')
    # --------- Turn right edge atoms into a rigid body to prevent any curvature -------------
    #lmps.command(f'fix 4 right_atoms rigid single force 1 on off off torque 1 off off off langevin 0.0 0.0 {damping_strength_right} 1029')

    # --------- Create groups for atoms treated with different ensembles --------------------
    # lmps.command('group nvt_atoms union left_thermo right_thermo')
    lmps.command('group nve_atoms subtract all right_atoms')
    lmps.command('group non_fixed_atoms subtract all top_atoms bottom_atoms right_atoms')

    # ---------- set timestep length -------------
    lmps.command(f'timestep {sim_tstep}')

    # ---------- Apply a thermostat to control the temperature ------------
    lmps.command('fix 5 nve_atoms nve')
    lmps.command(f'fix therm_weak_left left_thermo_weak langevin 0.0 0.0 {damping_strength_left*weak_damp_factor} 1029')
    lmps.command(f'fix therm_weak_right right_thermo_weak langevin 0.0 0.0 {damping_strength_right*weak_damp_factor} 1029')
    lmps.command(f'fix therm_strong_left left_thermo_strong langevin 0.0 0.0 {damping_strength_left} 1029')
    lmps.command(f'fix therm_strong_right right_thermo_strong langevin 0.0 0.0 {damping_strength_right} 1029')



    # Add a dump command to save .lammpstrj files every 100 timesteps during equilibration
    if dump_files:
        #lmps.command(f'dump myDump all atom {dump_freq} {tmp_file_path}/{dump_name}')
        lmps.command(f'dump myDump all custom {dump_freq} {tmp_file_path}/{dump_name} id type xs ys zs vx vy vz')
        lmps.command('dump_modify myDump append yes')

    # Specify the output frequency for thermo data
    lmps.command(f'thermo {thermo_freq}')


def set_up_eq_crack_simulation(lmps,tmp_file_path,atomic_mass,calc_commands,
                             sim_tstep=0.001,dump_freq=100, dump_files=True,
                             dump_name='dump.lammpstrj',thermo_freq=100,T=100,
                             damping_strength=0.1,rseed=1029):
    """Set up the simulation by passing an active LAMMPS object a number of commands"""
    
    # ---------- Initialize Simulation --------------------- 
    lmps.command('clear') 
    lmps.command('dimension 3')
    lmps.command('boundary s s p')
    lmps.command('atom_style atomic')
    lmps.command('units metal')


    lmps.command(f'read_data {tmp_file_path}/crack.lj')
                 

    #---------- Determine Lowest and Highest y-coordinates ----------
    lmps.command('group crack type 1')
    lmps.command('variable ymax equal bound(crack,ymax)')
    lmps.command('variable ymin equal bound(crack,ymin)')
    lmps.command('variable xmax equal bound(crack,xmax)')
    lmps.command('variable xmin equal bound(crack,xmin)')
    #define a variable for average y position of crack atoms using ymax and ymin
    lmps.command('variable ymid equal (v_ymax+v_ymin)/2')
    #----------Define potential-------
    lmps.command(f'mass 1 {atomic_mass}')

    
    ############ set up potential ################
    #set up both potentials, ensuring that interactions between types are defined
    lmps.commands_list(calc_commands)

    #---------- Define balance fix to spread load between processors
    lmps.command(f'fix b_fix all balance 50 1.2 shift xy 10 1.05 weight time 1.1')

    #---------- Define Regions for Boundary Layers ----------
    lmps.command(f'region bottom_layer block INF INF $((v_ymin-2)) $((v_ymin)) INF INF')
    lmps.command(f'region top_layer block INF INF $((v_ymax)) $(v_ymax+2) INF INF')
    lmps.command('region left_layer_fixed block $((v_xmin-2)) $((v_xmin+5)) INF INF INF INF')
    lmps.command('region right_layer_fixed block $((v_xmax-5)) $((v_xmax+2)) INF INF INF INF')

    #---------- Set groups for boundary layers ----------
    lmps.command('group top_atoms region top_layer')
    lmps.command('group bottom_atoms region bottom_layer')
    lmps.command('group left_atoms region left_layer_fixed')
    lmps.command('group right_atoms region right_layer_fixed')
    lmps.command('group non_fixed_atoms subtract all top_atoms left_atoms bottom_atoms right_atoms')

    # --------- Fix the edge atoms to prevent movement -----------
    lmps.command('fix 1 top_atoms setforce 0.0 0.0 NULL')
    lmps.command('fix 2 bottom_atoms setforce 0.0 0.0 NULL')
    # lmps.command('fix 3 left_atoms setforce 0.0 0.0 NULL')
    lmps.command('fix 4 right_atoms setforce 0.0 0.0 NULL')

    # ---------- set timestep length -------------
    lmps.command(f'timestep {sim_tstep}')

    # ---------- Apply a thermostat to control the temperature ------------
    lmps.command('fix 5 non_fixed_atoms nve')
    lmps.command(f'fix therm non_fixed_atoms langevin {T} {T} {damping_strength} {rseed}')


    # Add a dump command to save .lammpstrj files every 100 timesteps during equilibration
    if dump_files:
        #lmps.command(f'dump myDump all atom {dump_freq} {tmp_file_path}/{dump_name}')
        lmps.command(f'dump myDump all custom {dump_freq} {tmp_file_path}/{dump_name} id type xs ys zs vx vy vz')
        #lmps.command('dump_modify myDump append yes')

    # Specify the output frequency for thermo data
    lmps.command(f'thermo {thermo_freq}')
