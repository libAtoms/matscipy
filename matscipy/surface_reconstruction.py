import numpy as np
from numpy import sqrt
from ase.lattice.cubic import Diamond, FaceCenteredCubic, SimpleCubic, BodyCenteredCubic

from ase.optimize import LBFGS

from ase.constraints import FixAtoms

from matscipy.cauchy_born import CubicCauchyBorn
from ase.optimize.precon import PreconLBFGS
import ase


class SurfaceReconstruction:
    """Object for mapping and applying a surface reconstruction in simple and multi-lattices.

    The relaxation of a given free surface of a crystal at a given orientation can be mapped and
    saved. This relaxation can then be applied to atoms in a different ASE atoms object.
    This is extremely useful in the case of fracture mechanics, where surface relaxation
    can be mapped to the internal crack surfaces of an atoms object.
    """

    def __init__(self, el, a0, calc, directions,
                 surf_dir, lattice, multilattice=False):
        """Parameters
        ----------
        el : string
            Element crystal is composed of in ASE format
        a0 : float
            Lattice constant of crystal
        calc : ASE calculator object
            Calculator for surface relaxation
        directions : list
            The x, y and z lab frame directions expressed in the lattice frame.
            e.g directions=[[1,1,1], [0,-1,1], [-2,1,1]]
        surf_dir : int
            Direction which is normal to the free surface. Can be 0, 1 or 2 for
            lab x, lab y or lab z (as defined by directions).
        lattice : ASE ase.lattice.cubic function
            Callable function which can be used to generate the crystal.
        """
        # directions is a list of 3 vectors specifiying the surface of interest
        # in the orientation specified by the problem. surf_dir says which axis is free surface
        # e.g directions=[[1,1,1], [0,-1,1], [-2,1,1]]
        self.el = el
        self.a0 = a0
        self.calc = calc
        self.directions = directions
        self.surf_dir = surf_dir
        self.inter_planar_dist = a0 / np.linalg.norm(directions[surf_dir])
        # build the rotation matrix from the directions
        A = np.zeros([3, 3])
        for i, direction in enumerate(directions):
            direction = np.array(direction)
            A[:, i] = direction / np.linalg.norm(direction)
        self.A = A  # rotation matrix
        self.eval_cb = None
        self.lattice = lattice
        self.cb = None
        self.multilattice = False
        if multilattice:
            self.cb = CubicCauchyBorn(
                self.el, self.a0, self.calc, lattice=self.lattice)
            self.multilattice = True

    def map_surface(self, fmax=0.0001, layers=6, cutoff=10,
                    shift=0, switch=False, invert_dirs=[False, False, False]):
        """Map the relaxation of the crystal surface a certain number of layers deep.
        Parameters
        ----------
        fmax : float
            Force tolerance for relaxation
        layers : int
            Number of layers deep to map the free surface
        cutoff : float
            Cutoff of the potential being used - used to decide the number of
            atoms beneath the mapped layers that need to be modelled
        shift : float
            Amount to shift the bulk structure before the surface is mapped
            such that the top layer of surface is physically meaningful. One
            use case of this would be to adjust the top layer of atoms being mapped
            in diamond such that it matches the top layer of atoms seen on the cleavage
            plane.
        switch : bool
            Whether or not to switch the sublattices around in the mapped structure.
            (Essentially, the correct surface can always be found from some combination of shift and switch)
        invert_dirs : bool array
            Whether to invert the components of the mapped shifts. This is useful in the case where a surface which originally has mirror
            symmetry in a certain plane can break this symmetry during reconstruction. This is the case for the 110 silicon reconstructed surface,
            where the surface breaks mirror symmetry in the {001} type plane during reconstruction. Two possible surfaces can form, related through
            diad symmetry. To map the other surface, one has to switch sublattices and invert the mapped displacements along the <001> type direction.
        """
        # build a single cell to determine how many atomic surface
        # layers there are in a single unit cell made by ASE
        single_cell = self.lattice(
            directions=self.directions,
            size=[1, 1, 1],
            symbol=self.el,
            latticeconstant=self.a0,
            pbc=(1, 1, 1))
        # ase.io.write(
        #    f'{self.directions[self.surf_dir]}_unitcell.xyz', single_cell)

        if self.multilattice:
            # for a multilattice, get number of layers by just looking
            # at one of the sublattices
            self.cb.set_sublattices(single_cell, self.A)
            n_layer_per_cell = len(np.unique(single_cell.get_positions()[
                                   self.cb.lattice1mask][:, self.surf_dir].round(decimals=6)))
            # check if multilattice atoms are coplanar
            diff = single_cell.get_positions()[self.cb.lattice1mask][0, self.surf_dir] - \
                single_cell.get_positions(
            )[self.cb.lattice2mask][0, self.surf_dir]
            # print(diff)
            # if np.abs(diff)<0.001:
            #    self.split_cb_pair = False
            # else:
            #    self.split_cb_pair = True
        else:
            # otherwise, just look at all the layers directly
            n_layer_per_cell = len(np.unique(single_cell.get_positions()[
                                   :, self.surf_dir].round(decimals=6)))

        # print('nlayer per cell', n_layer_per_cell)
        # ase.io.write('single_surface_cell.xyz',single_cell)

        # set the number of atomic unit cells to map
        # as twice the number of layers to map + a number of layers
        # equal to the potential cutoff for fixing
        height = 2 * int(layers / n_layer_per_cell) + \
            int(np.round(cutoff / self.inter_planar_dist))
        size = [1, 1, 1]
        size[self.surf_dir] *= height
        self.layers = layers

        # build the bulk structure, and the slab structure which has vacuum in
        # surface direction
        bulk = self.lattice(
            directions=self.directions,
            size=size,
            symbol=self.el,
            latticeconstant=self.a0,
            pbc=(1, 1, 1))

        # shift the cell slightly and wrap to stop dodgy surfaces when vacuum added
        # as well as to add on any user-defined shift
        shift_vec = np.zeros([3])
        shift_vec[self.surf_dir] += 0.01 + shift
        bulk.positions += shift_vec
        bulk.wrap()

        cell = bulk.get_cell()
        # vacuum along self.surf_dir axis (surface normal)
        cell[self.surf_dir, :] *= 2
        slab = bulk.copy()
        slab.set_cell(cell, scale_atoms=False)

        # get the mask for the bottom surface and fix it
        pos_before = slab.get_positions()
        # fix bottom atoms to stop lower surface reconstruction
        mask = pos_before[:, self.surf_dir] < cutoff
        slab.set_constraint(FixAtoms(mask=mask))
        slab.calc = self.calc
        # print('fbefore',slab.get_forces())
        # ase.io.write('0_slab.xyz', slab)

        # run an optimisation to relax the surface
        opt_slab = PreconLBFGS(slab)
        opt_slab.run(fmax=fmax)
        # ase.io.write('1_slab.xyz', slab)
        pos_after = slab.get_positions()

        # measure the shift between the final position and the intial position
        pos_shift = pos_after - pos_before
        # print('POS SHIFT', pos_shift)
        # print('fafter',slab.get_forces())

        # invert measured shifts if required
        for direc in range(len(invert_dirs)):
            if invert_dirs[direc]:
                pos_shift[:, direc] *= -1

        # for a multi-lattice:
        if self.multilattice:
            # if this is a multi-lattice

            # set which atoms are in each multilattice based on the bulk structure. This is important because
            # different relaxations happen for each multilattice atom in each
            # atomic layer near the surface
            self.cb.set_sublattices(bulk, self.A)
            if switch:
                self.cb.switch_sublattices(bulk)
            # split the full lattice into the two multilattices
            total_layers = n_layer_per_cell * height
            n_atoms_per_layer = int(len(slab) / total_layers)

            # get the indices of the atoms based on how close they are to the
            # surface in order, for each sub lattice
            sorted_surf_indices_lattice1 = np.argsort(
                pos_before[self.cb.lattice1mask][:, self.surf_dir])
            sorted_surf_indices_lattice2 = np.argsort(
                pos_before[self.cb.lattice2mask][:, self.surf_dir])

            # split these into layers and save them to an array.
            # each layer holds the indices of the atoms it contains [note - these are the indices of the array
            # pos_shift[self.cb.latticemask] - not pos_shift]
            surf_layer_list_lattice1 = np.zeros(
                [total_layers, int(n_atoms_per_layer / 2)], dtype=int)
            surf_layer_list_lattice2 = np.zeros(
                [total_layers, int(n_atoms_per_layer / 2)], dtype=int)
            for i in range(total_layers):
                surf_layer_list_lattice1[i, :] = sorted_surf_indices_lattice1[i * int(
                    n_atoms_per_layer / 2):(i + 1) * int(n_atoms_per_layer / 2)]
                surf_layer_list_lattice2[i, :] = sorted_surf_indices_lattice2[i * int(
                    n_atoms_per_layer / 2):(i + 1) * int(n_atoms_per_layer / 2)]

            # get the inter-layer distance in the direction we care about.
            self.inter_surf_dist = np.abs(pos_before[self.cb.lattice1mask][surf_layer_list_lattice1[1, :]][0, self.surf_dir]
                                          - pos_before[self.cb.lattice1mask][surf_layer_list_lattice1[0, :]][0, self.surf_dir])

            # now go through the top layers that we are applying a surface relaxation to and save the shifts of
            # the lattice 1 and lattice 2 atoms in these layers in the form
            # [x,y,z] for each layer

            self.relaxation_array_lattice1 = np.zeros([layers, 3])
            self.relaxation_array_lattice2 = np.zeros([layers, 3])
            for layer in range(layers):
                pos_shifts_lattice1 = pos_shift[self.cb.lattice1mask][
                    surf_layer_list_lattice1[total_layers - layer - 1, :]]
                pos_shifts_lattice2 = pos_shift[self.cb.lattice2mask][
                    surf_layer_list_lattice2[total_layers - layer - 1, :]]
                self.relaxation_array_lattice1[layer,
                                               :] = pos_shifts_lattice1[0, :]
                self.relaxation_array_lattice2[layer,
                                               :] = pos_shifts_lattice2[0, :]

        else:
            # for a non-multilattice
            # get the total number of in the surface from the number of layers
            # per cell
            total_layers = n_layer_per_cell * height
            n_atoms_per_layer = int(len(slab) / total_layers)
            # print(n_atoms_per_layer)
            # print(len(slab))
            # print(pos_shift)
            # get the indices of the atoms sorted in height order
            sorted_surf_indices = np.argsort(pos_before[:, self.surf_dir])

            # split these into layers and save them to an array.
            # each layer holds the indices of the atoms it contains [note - these are the indices of the array
            # pos_shift[] - not pos_shift]
            surf_layer_list = np.zeros(
                [total_layers, n_atoms_per_layer], dtype=int)
            for i in range(total_layers):
                surf_layer_list[i, :] = sorted_surf_indices[i *
                                                            n_atoms_per_layer:((i + 1) * n_atoms_per_layer)]
            # print(surf_layer_list)
            self.inter_surf_dist = np.abs(
                pos_before[surf_layer_list[1, :]][0, self.surf_dir] - pos_before[surf_layer_list[0, :]][0, self.surf_dir])
            self.relaxation_array = np.zeros([layers, 3])
            for layer in range(layers):
                pos_shifts_by_layer = pos_shift[surf_layer_list[total_layers - layer - 1, :]]
                self.relaxation_array[layer, :] = pos_shifts_by_layer[0, :]
            # print(self.relaxation_array)

    def identify_layers(
            self,
            atoms,
            surface_coords,
            xlim=None,
            ylim=None,
            zlim=None,
            search_dir=-1,
            atoms_for_cb=None,
            read_from_atoms=False):
        """Function for identifying the layers adjacent to a surface, in a supplied atoms structure.
        This allows a mapped relaxation to be easily applied to a surface.
        Parameters
        ----------
        atoms : ASE atoms object
            Atoms object which contains surface to relax
        surface_coords : array_like
            Coordinates of a point that lies on the free surface (or within one layer distance above it)
        xlim : array_like
            [x lower, x upper], x range to apply relaxation
        ylim : array_like
            [y lower, y upper], y range to apply relaxation
        zlim : array_like
            [z lower, z upper], z range to apply relaxation
        search_dir : int
            -1: surface is below point provided, 1: surface is above point provided
        atoms_for_cb : ASE atoms object
            atoms object which allows the easy identification of sub-lattices in a multi-lattice
            crystal using the CubicCauchyBorn set_sublattices function. The difference between this and atoms may
            be periodic boundary conditions, for example.
        read_from_atoms : bool
            Whether or not to read in the cauchy-born sublattices from the atoms structure. This can be done if the
            atoms object contains arrays called lattice1mask and lattice2mask

        Returns
        -------
        layer_mask_set : array_like, bool
            A 2D array dimensions [nlayers,natoms] where a row i is a mask for
            'atoms' corresponding to a layer i deep from the surface. In the case of
            a multi-lattice, two masks are returned, one for each of the sublattices on each layer.
        """
        # function for identifying the layers adjacent to a surface, where the top of the surface
        # is found near coordinates surface_coords
        pos = atoms.get_positions()

        # get the masks for the limits imposed by xlim,ylim and zlim
        if xlim is not None:
            xmask = (pos[:, 0] < xlim[1]) & (pos[:, 0] > xlim[0])
        else:
            xmask = np.ones([len(atoms)], dtype=bool)

        if ylim is not None:
            ymask = (pos[:, 1] < ylim[1]) & (pos[:, 1] > ylim[0])
        else:
            ymask = np.ones([len(atoms)], dtype=bool)

        if zlim is not None:
            zmask = (pos[:, 2] < zlim[1]) & (pos[:, 2] > zlim[0])
        else:
            zmask = np.ones([len(atoms)], dtype=bool)

        lim_mask = xmask & ymask & zmask

        # now, get the mask for the provided atoms structure
        # for each individual surface layer
        layer_mask_set = np.zeros([len(atoms), self.layers], dtype=bool)
        for layer in range(self.layers):
            # this is the crucial line, essentially finds a mask for the
            # atoms that lie within a +- 1 layer distance from the starting point.
            # we then shift down or up a layer each iteration, depending on
            # search_dir
            if search_dir == -1:
                # this corresponds to searching downwards for surface layers
                layer_mask = np.logical_and((pos[:, self.surf_dir] <= surface_coords[self.surf_dir] - (
                    layer * self.inter_surf_dist)), (pos[:, self.surf_dir] > surface_coords[self.surf_dir] - ((layer + 1) * self.inter_surf_dist)))
            elif search_dir == 1:
                # this corresponds to searching upward for surface layers
                layer_mask = np.logical_and((pos[:, self.surf_dir] >= surface_coords[self.surf_dir] + (
                    layer * self.inter_surf_dist)), (pos[:, self.surf_dir] < surface_coords[self.surf_dir] + ((layer + 1) * self.inter_surf_dist)))
            else:
                print(
                    'search_dir should be -1 for a downward surface, 1 for an upward surface!')
                raise ValueError
            layer_mask = layer_mask & lim_mask

            # add the layer mask found here to the full set
            layer_mask_set[:, layer] = layer_mask

        if not self.multilattice:
            return layer_mask_set
        else:
            # for a multi-lattice, also have to account for shifts between multilattice atoms
            # in layers, so need to refine the masks further to be the lattice1
            # and lattice2 atoms in each layer
            if atoms_for_cb is None:
                atoms_for_cb = atoms
            # note that to find the sublattices, need to use a version of the
            # atoms structure with no free surface
            if self.eval_cb is None:
                self.eval_cb = self.cb
            self.eval_cb.set_sublattices(
                atoms_for_cb, self.A, read_from_atoms=read_from_atoms)
            layer_mask_set_lattice1 = np.zeros_like(layer_mask_set, dtype=bool)
            layer_mask_set_lattice2 = np.zeros_like(layer_mask_set, dtype=bool)
            for layer in range(self.layers):
                layer_mask_set_lattice1[:,
                                        layer] = layer_mask_set[:,
                                                                layer] & self.eval_cb.lattice1mask
                layer_mask_set_lattice2[:,
                                        layer] = layer_mask_set[:,
                                                                layer] & self.eval_cb.lattice2mask
            # if (self.split_cb_pair and (search_dir==1)):
                # for some surfaces, the surface formed will split a pair of atoms on different multilattices.
                # In this case, you need to flip the order of the lattice masks you return, which is essentially
                # exploiting the fact that a surface in the upward direction has diad symmetry with the surface in the
                # downward direction.
            #    print('switching lattice masks')
            #    return layer_mask_set_lattice2, layer_mask_set_lattice1
            # else:
            return layer_mask_set_lattice1, layer_mask_set_lattice2

    def apply_surface_shift(
            self,
            atoms,
            surface_coords,
            cb=None,
            xlim=None,
            ylim=None,
            zlim=None,
            search_dir=-1,
            atoms_for_cb=None,
            read_from_atoms=False):
        """Function which applies the mapped out deformation to the surface layers
        in a supplied atoms structure.
        Parameters
        ----------
        atoms : ASE atoms object
            Atoms object which contains surface to relax
        surface_coords : array_like
            Coordinates of a point that lies on the free surface (or within one layer distance above it)
        xlim : array_like
            [x lower, x upper], x range to apply relaxation
        ylim : array_like
            [y lower, y upper], y range to apply relaxation
        zlim : array_like
            [z lower, z upper], z range to apply relaxation
        search_dir : int
            -1: surface is below point provided, 1: surface is above point provided
        atoms_for_cb : ASE atoms object
            atoms object which allows the easy identification of sub-lattices in a multi-lattice
            crystal using the CubicCauchyBorn set_sublattices function. The difference between this and atoms may
            be periodic boundary conditions, for example.
        read_from_atoms : bool
            Whether or not to read in the cauchy-born sublattices from the atoms structure. This can be done if the
            atoms object contains arrays called lattice1mask and lattice2mask
        """
        # This finds the layers of atoms that need shifting and applies the
        # surface shift accordingly.
        self.eval_cb = cb
        # print(self.eval_cb)
        if self.multilattice:
            layer_mask_set_lattice1, layer_mask_set_lattice2 = self.identify_layers(
                atoms, surface_coords, xlim=xlim, ylim=ylim, zlim=zlim, search_dir=search_dir, atoms_for_cb=atoms_for_cb, read_from_atoms=read_from_atoms)
            for layer in range(self.layers):
                # print('diff', self.relaxation_array_lattice2[layer, :])
                # print('initial',atoms.positions[:, :][layer_mask_set_lattice1[:,layer]])
                lattice_1_shift = self.relaxation_array_lattice1[layer, :]
                lattice_2_shift = self.relaxation_array_lattice2[layer, :]

                # if the surface is going upward, invert surf_dir component of
                # applied shift
                lattice_1_shift[self.surf_dir] = (-search_dir) * \
                    lattice_1_shift[self.surf_dir]
                lattice_2_shift[self.surf_dir] = (-search_dir) * \
                    lattice_2_shift[self.surf_dir]
                atoms.positions[:,
                                :][layer_mask_set_lattice1[:,
                                                           layer]] += lattice_1_shift
                atoms.positions[:,
                                :][layer_mask_set_lattice2[:,
                                                           layer]] += lattice_2_shift
                # print('after',atoms.positions[:, :][layer_mask_set_lattice1[:,layer]])

        else:
            layer_mask_set = self.identify_layers(
                atoms,
                surface_coords,
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                search_dir=search_dir)
            for layer in range(self.layers):
                shift = self.relaxation_array[layer, :]
                shift[self.surf_dir] = (-search_dir) * shift[self.surf_dir]
                atoms.positions[:, :][layer_mask_set[:, layer]
                                      ] += shift

    def map_pandey_111(self, fmax=0.0001, layers=6, cutoff=10,
                       orientation=0, shift=0, switch=False):
        """Map the diamond structure 111 pandey relaxation of the crystal surface a certain number of layers deep.
        As this relaxation breaks surface symmetry, one can map it in 3 different orientations.
        Parameters
        ----------
        fmax : float
            Force tolerance for relaxation
        layers : int
            Number of layers deep to map the free surface
        cutoff : float
            Cutoff of the potential being used - used to decide the number of
            atoms beneath the mapped layers that need to be modelled
        shift : float
            Amount to shift the bulk structure before the surface is mapped
            such that the top layer of surface is physically meaningful. One
            use case of this would be to adjust the top layer of atoms being mapped
            in diamond such that it matches the top layer of atoms seen on the cleavage
            plane.
        switch : bool
            Whether or not to switch the sublattices around in the mapped structure.
            (Essentially, the correct surface can always be found from some combination of shift and switch)
        orientation : 0,1,2
            Which of the three different orientations of the Pandey reconstruction to map
        """
        # map out the pandey reconstruction on a diamond structure 111 surface
        # can then apply it later (this requires a seperate function as the unit cell of 2x1 reconstruction
        # breaks the symmetry of the surface)
        # self.split_cb_pair = True
        # 111 type surface always splits a cb pair
        ny = 1
        nx = 1
        nz = int(layers / 3 + int(cutoff / (self.a0 / np.sqrt(3))) + 1)
        self.layers = layers
        self.pandey_dirs = [[1, -1, 0], [1, 1, -2], [1, 1, 1]]
        # build the pandey frame rotation matrix
        U = np.zeros([3, 3])
        for i, direction in enumerate(self.pandey_dirs):
            direction = np.array(direction)
            U[:, i] = direction / np.linalg.norm(direction)
        self.U = U  # pandey frame rotation matrix
        a = Diamond(self.el,
                    size=[1, 1, nz],
                    latticeconstant=self.a0,
                    directions=self.pandey_dirs
                    )

        self.cb.set_sublattices(a, self.U)
        if switch:
            self.cb.switch_sublattices(a)
        sorted_z_vals = np.sort(a.get_positions()[self.cb.lattice1mask][:, 2])
        self.inter_surf_dist = sorted_z_vals[2] - sorted_z_vals[0]
        # a.translate([0,+0.001,-shift*self.inter_surf_dist]) #REMOVE SHIFT
        # FROM HERE
        a.translate([0.01, 0.01, 0])
        a.wrap()
        # print('INTER SURF DIST', self.inter_surf_dist)
        # seed the 2x1 reconstruction on the top plane
        sx, sy, sz = a.get_cell().diagonal()
        # ([sx/(12*nx), sy/(4*nx), sz/(6*nz)])
        a.translate([0, 0, sz / (6 * nz)])
        a.set_scaled_positions(a.get_scaled_positions() % 1.0)
        pos_initial_unrotated = a.get_positions()

        bulk = a.copy()

        bondlen = self.a0 * sqrt(3) / 4

        x, y, z = a.positions.T
        mask = np.abs(z - z.max()) < 0.1
        top1, top2 = np.arange(len(a))[mask]
        topA, topB = np.arange(len(a))[np.logical_and(np.abs(z - z.max()) < bondlen,
                                                      np.logical_not(mask))]
        a.set_distance(top1, top2, bondlen)
        a.set_distance(topA, topB, bondlen)
        # ase.io.write('intermediate_atoms.xyz',a)
        x, y, z = a.positions.T
        y1 = (y[top1] + y[top2]) / 2
        yA = (y[topA] + y[topB]) / 2
        y[top1] += yA - y1 + a.cell[1, 1] / 2
        y[top2] += yA - y1 + a.cell[1, 1] / 2
        # x[top1] += a.cell[0,0]/2
        # x[topB] -= a.cell[0,0]/2
        a.set_positions(np.transpose([x, y, z]))
        # a.wrap()

        # ase.io.write('0_bulk.xyz', bulk)
        # ase.io.write('1_bulk.xyz', a)
        # relax the structure to get the 2x1 111 reconstruction
        # run an optimisation to relax the surface

        # add some vacuum
        cell = a.get_cell()
        # vacuum along self.surf_dir axis (surface normal)
        cell[2, :] *= 2
        a.set_cell(cell, scale_atoms=False)
        a.calc = self.calc

        # fix bottom atoms to stop lower surface reconstruction
        # print('POS INITIAL UNROTATED',pos_initial_unrotated)
        mask = pos_initial_unrotated[:, 2] < cutoff
        # print(pos_initial_unrotated[:,2])
        a.set_constraint(FixAtoms(mask=mask))
        opt_a = PreconLBFGS(a)
        opt_a.run(fmax=fmax)
        # ase.io.write('2_bulk.xyz', a)
        pos_after_unrotated = a.get_positions()
        # print(pos_after_unrotated[:, 1] < 0)
        # print(len(pos_after_unrotated[:, 1] < 0))
        # print(pos_after_unrotated[:, 1] > a.cell[1, 1])
        # print(len(pos_after_unrotated[:, 1] > a.cell[1, 1]))
        atoms_outside_mask = (pos_after_unrotated[:, 1] < 0) | (
            pos_after_unrotated[:, 1] > a.cell[1, 1])
        pushed_out_atoms = a[atoms_outside_mask]
        wrapped = False
        if len(pushed_out_atoms) > 0:
            print('detected atoms to be wrapped')
            wrapped = True
            a.wrap()
            pos_after_unrotated = a.get_positions()
            # ase.io.write('3_bulk.xyz', a)

        pos_diff_unrotated = pos_after_unrotated - pos_initial_unrotated
        # print('pos diff unrotated',pos_diff_unrotated)
        # map out the layers in the original structure
        surface_coords = pos_initial_unrotated[np.argmax(
            pos_initial_unrotated[:, 2]), :] + [0, 0, 0.01]
        tmp = self.surf_dir
        self.surf_dir = 2
        layer_mask_set_lattice1_atom1, layer_mask_set_lattice1_atom2, layer_mask_set_lattice2_atom1, layer_mask_set_lattice2_atom2 = \
            self.identify_pandey_layers(
                bulk, surface_coords, read_from_atoms=True, frame_dirs=self.pandey_dirs)
        self.surf_dir = tmp

        # relaxation array lattice 1 atom 1
        self.ral1a1 = np.zeros([layers, 3])
        # relaxation array lattice 1 atom 2
        self.ral1a2 = np.zeros([layers, 3])
        # relaxation array lattice 2 atom 1
        self.ral2a1 = np.zeros([layers, 3])
        # relaxation array lattice 2 atom 2
        self.ral2a2 = np.zeros([layers, 3])

        # print(np.shape(pos_diff_unrotated))
        # print(np.shape(layer_mask_set_lattice1_atom1))
        # generate rotation matrix to get from pandey frame to lab frame
        # orientation rotation tensor
        OR = np.zeros([3, 3])
        OR[0, 0] = np.cos(2 * (np.pi / 3) * orientation)
        OR[0, 1] = -np.sin(2 * (np.pi / 3) * orientation)
        OR[1, 0] = np.sin(2 * (np.pi / 3) * orientation)
        OR[1, 1] = np.cos(2 * (np.pi / 3) * orientation)
        OR[2, 2] = 1
        # print('OR',OR)

        R = np.transpose(self.A) @ self.U @ np.transpose(OR)
        # print('with no rotation',np.transpose(self.A)@self.U)
        # print('with rotation',R)
        for layer in range(layers):
            # print(layer)
            # print(
            # len(pos_diff_unrotated[layer_mask_set_lattice2_atom2[:, layer]]))

            self.ral1a1[layer,
                        :] = R @ pos_diff_unrotated[layer_mask_set_lattice1_atom1[:,
                                                                                  layer]][0,
                                                                                          :]
            self.ral1a2[layer,
                        :] = R @ pos_diff_unrotated[layer_mask_set_lattice1_atom2[:,
                                                                                  layer]][0,
                                                                                          :]
            self.ral2a1[layer,
                        :] = R @ pos_diff_unrotated[layer_mask_set_lattice2_atom1[:,
                                                                                  layer]][0,
                                                                                          :]
            self.ral2a2[layer,
                        :] = R @ pos_diff_unrotated[layer_mask_set_lattice2_atom2[:,
                                                                                  layer]][0,
                                                                                          :]
        # print(self.ral1a1, self.ral1a2, self.ral2a1, self.ral2a2)

    def identify_pandey_layers(self,
                               atoms,
                               surface_coords,
                               xlim=None,
                               ylim=None,
                               zlim=None,
                               search_dir=-1,
                               atoms_for_cb=None,
                               read_from_atoms=False,
                               frame_dirs=None,
                               orientation=0):
        """Function for identifying the layers adjacent to a surface, in a supplied atoms structure.
        This allows a mapped relaxation to be easily applied to a surface. This implementation is specific
        to the 111 Pandey reconstruction, as more atoms need to be identified due to the breaking of surface
        symmetry.

        Parameters
        ----------
        atoms : ASE atoms object
            Atoms object which contains surface to relax
        surface_coords : array_like
            Coordinates of a point that lies on the free surface (or within one layer distance above it)
        xlim : array_like
            [x lower, x upper], x range to apply relaxation
        ylim : array_like
            [y lower, y upper], y range to apply relaxation
        zlim : array_like
            [z lower, z upper], z range to apply relaxation
        search_dir : int
            -1: surface is below point provided, 1: surface is above point provided
        atoms_for_cb : ASE atoms object
            atoms object which allows the easy identification of sub-lattices in a multi-lattice
            crystal using the CubicCauchyBorn set_sublattices function. The difference between this and atoms may
            be periodic boundary conditions, for example.
        read_from_atoms : bool
            Whether or not to read in the cauchy-born sublattices from the atoms structure. This can be done if the
            atoms object contains arrays called lattice1mask and lattice2mask
        frame_dirs :
            The frame in which the Pandey reconstruction was mapped out - the atoms in the supplied frame will need to
            be rotated to this frame before the reconstruction can be applied
        orientation : 0,1 or 2
            Which of the three different orientations of the Pandey reconstructions are being applied to the surface


        Returns
        -------
        layer_mask_set : array_like, bool
            A 2D array dimensions [nlayers,natoms] where a row i is a mask for
            'atoms' corresponding to a layer i deep from the surface. In the case of
            a multi-lattice, two masks are returned, one for each of the sublattices on each layer.
        """
        # this function needs to identify the 4 sets of atoms per layer
        # that reconstruct (breaking symmetry)
        # it does this by taking an atoms structure, reading the sublattices
        # identifying the layers down from a point and then in each layer
        # identifying the 4 unique types of atom
        # to do this, it takes the positions of all the atoms of one sublattice
        # rotates these position vectors from the supplied frame to the mapped reconstruction frame
        # and then goes from left to right, shifting th

        # get the layer masks

        layer_mask_set_lattice1, layer_mask_set_lattice2 = \
            self.identify_layers(atoms, surface_coords, read_from_atoms=read_from_atoms, xlim=xlim,
                                 ylim=ylim, zlim=zlim, atoms_for_cb=atoms_for_cb, search_dir=search_dir)

        # rotate the lattice structure to match the pandey directions
        # build the rotation matrix
        # this is equivalent to transforming from the coordinate system
        # given by 'dirs' (the lab frame) to the lattice frame, and then from there
        # to the frame where the pandey reconstruction is defined.

        if frame_dirs is not None:
            T = np.zeros([3, 3])
            for i, direction in enumerate(frame_dirs):
                direction = np.array(direction)
                T[:, i] = direction / np.linalg.norm(direction)
        else:
            T = self.A

        OR = np.zeros([3, 3])
        OR[0, 0] = np.cos(2 * (np.pi / 3) * orientation)
        OR[0, 1] = -np.sin(2 * (np.pi / 3) * orientation)
        OR[1, 0] = np.sin(2 * (np.pi / 3) * orientation)
        OR[1, 1] = np.cos(2 * (np.pi / 3) * orientation)
        OR[2, 2] = 1

        # OR - Orientation rotation tensor
        # T - lab frame to lattice frame
        # U - pandey map frame to lattice frame
        R = OR @ np.transpose(self.U) @ T
        # print(self.U, T, R)

        layer_mask_set_lattice1_atom1 = np.zeros_like(
            layer_mask_set_lattice1, dtype=bool)
        layer_mask_set_lattice1_atom2 = np.zeros_like(
            layer_mask_set_lattice1, dtype=bool)
        layer_mask_set_lattice2_atom1 = np.zeros_like(
            layer_mask_set_lattice1, dtype=bool)
        layer_mask_set_lattice2_atom2 = np.zeros_like(
            layer_mask_set_lattice1, dtype=bool)

        # find an exclusion zone using the position of the leftmost lattice1
        # atom in the third layer (this is where we cut the cell in map_pandey)
        layer3 = atoms[layer_mask_set_lattice1[:, 2]]
        layer3pos = layer3.get_positions()
        # rotate
        rotated_layer3 = np.zeros_like(layer3pos)
        for j in range(np.shape(layer3pos)[0]):
            rotated_layer3[j, :] = R @ layer3pos[j, :]

        # find the leftmost point of layer 3
        leftpos = np.min(rotated_layer3[:, 1])
        pos = atoms.get_positions()
        rotated_pos = np.zeros_like(pos)
        for j in range(np.shape(pos)[0]):
            rotated_pos[j, :] = R @ pos[j, :]

        mask = rotated_pos[:, 1] < leftpos - 0.01

        # print('SEARCH DIR,', search_dir)
        for i in range(np.shape(layer_mask_set_lattice1)[1]):
            # print(np.shape(layer_mask_set_lattice1))
            lattice1atoms = atoms[layer_mask_set_lattice1[:, i]]
            pos_lattice_1_atoms = lattice1atoms.get_positions()
            # rotate
            rotated_pos_lattice1 = np.zeros_like(pos_lattice_1_atoms)
            for j in range(np.shape(pos_lattice_1_atoms)[0]):
                rotated_pos_lattice1[j, :] = R @ pos_lattice_1_atoms[j, :]

            # sort by the y position of the atoms in the rotated frame
            # that is used to map the pandey reconstruction
            # print('LAYER MASK LATTICE 1',layer_mask_set_lattice1[:,i])
            atom_indices = np.where(layer_mask_set_lattice1[:, i] == True)[0]
            pos_indices = np.argsort(rotated_pos_lattice1[:, 1])
            n_unique = 0
            # if the search direction is 1 (looking at a lower surface), run
            # this backwards
            if search_dir == 1:
                k = len(pos_indices) - 1
            else:
                k = 0
            compare_pos = rotated_pos_lattice1[:, 1][pos_indices[k]]
            for j in range(len(pos_indices)):
                if search_dir == 1:
                    k = len(pos_indices) - j - 1
                else:
                    k = j
                # build masks which hold the different types of atoms in
                # the symmetry broken reconstruction (using the y axis in the rotated
                # frame to differentiate them from eachother)
                # print(k, search_dir)
                # print(-search_dir*rotated_pos_lattice1[:, 1][pos_indices[k]],
                #      (-search_dir*(compare_pos-(search_dir*0.01))))
                if (-search_dir * rotated_pos_lattice1[:, 1][pos_indices[k]]
                    ) > (-search_dir * (compare_pos - (search_dir * 0.01))):
                    # if
                    # rotated_pos_lattice1[:,1][pos_indices[k]]<(compare_pos-0.01):
                    n_unique += 1
                    compare_pos = rotated_pos_lattice1[:, 1][pos_indices[k]]
                if n_unique % 2 == 0:
                    layer_mask_set_lattice1_atom1[:,
                                                  i][atom_indices[pos_indices[k]]] = True
                else:
                    layer_mask_set_lattice1_atom2[:,
                                                  i][atom_indices[pos_indices[k]]] = True

            # repeat for lattice 2
            lattice2atoms = atoms[layer_mask_set_lattice2[:, i]]
            pos_lattice_2_atoms = lattice2atoms.get_positions()

            # rotate
            rotated_pos_lattice2 = np.zeros_like(pos_lattice_2_atoms)
            for j in range(np.shape(pos_lattice_2_atoms)[0]):
                rotated_pos_lattice2[j, :] = R @ pos_lattice_2_atoms[j, :]
            # sort by the y position of the atoms in the rotated frame
            # that is used to map the pandey reconstruction
            atom_indices = np.where(layer_mask_set_lattice2[:, i] == True)[0]
            # print(atom_indices)
            pos_indices = np.argsort(rotated_pos_lattice2[:, 1])
            n_unique = 0
            if search_dir == 1:
                k = len(pos_indices) - 1
            else:
                k = 0
            compare_pos = rotated_pos_lattice2[:, 1][pos_indices[k]]
            for j in range(len(pos_indices)):
                if search_dir == 1:
                    k = len(pos_indices) - j - 1
                else:
                    k = j
                # build masks which hold the different types of atoms in
                # the symmetry broken reconstruction (using the y axis in the rotated
                # frame to differentiate them from eachother)
                if (-search_dir * rotated_pos_lattice2[:, 1][pos_indices[k]]
                    ) > (-search_dir * (compare_pos - (search_dir * 0.01))):
                    n_unique += 1
                    compare_pos = rotated_pos_lattice2[:, 1][pos_indices[k]]
                if n_unique % 2 == 0:
                    layer_mask_set_lattice2_atom1[:,
                                                  i][atom_indices[pos_indices[k]]] = True
                else:
                    layer_mask_set_lattice2_atom2[:,
                                                  i][atom_indices[pos_indices[k]]] = True

        # print(layer_mask_set_lattice1_atom1[:,layer])
        try:
            atoms.new_array('atomno', np.zeros(len(atoms), dtype=int))
        except RuntimeError:
            pass
        atomno = atoms.arrays['atomno']
        for layer in range(self.layers):
            atomno[layer_mask_set_lattice1_atom1[:, layer]] = 1
            atomno[layer_mask_set_lattice1_atom2[:, layer]] = 2
            atomno[layer_mask_set_lattice2_atom1[:, layer]] = 3
            atomno[layer_mask_set_lattice2_atom2[:, layer]] = 4
        return layer_mask_set_lattice1_atom1, layer_mask_set_lattice1_atom2, layer_mask_set_lattice2_atom1, layer_mask_set_lattice2_atom2

    def apply_pandey_111(self,
                         atoms,
                         surface_coords,
                         cb=None,
                         xlim=None,
                         ylim=None,
                         zlim=None,
                         search_dir=-1,
                         atoms_for_cb=None,
                         read_from_atoms=False,
                         orientation=0):
        """Function which applies the mapped out Pandey deformation to the surface layers
        in a supplied atoms structure.
        Parameters
        ----------
        atoms : ASE atoms object
            Atoms object which contains surface to relax
        surface_coords : array_like
            Coordinates of a point that lies on the free surface (or within one layer distance above it)
        xlim : array_like
            [x lower, x upper], x range to apply relaxation
        ylim : array_like
            [y lower, y upper], y range to apply relaxation
        zlim : array_like
            [z lower, z upper], z range to apply relaxation
        search_dir : int
            -1: surface is below point provided, 1: surface is above point provided
        atoms_for_cb : ASE atoms object
            atoms object which allows the easy identification of sub-lattices in a multi-lattice
            crystal using the CubicCauchyBorn set_sublattices function. The difference between this and atoms may
            be periodic boundary conditions, for example.
        read_from_atoms : bool
            Whether or not to read in the cauchy-born sublattices from the atoms structure. This can be done if the
            atoms object contains arrays called lattice1mask and lattice2mask
        orientation : 0,1 or 2
            Which of the three different orientations of the Pandey reconstructions are being applied to the surface
        """
        self.eval_cb = cb

        # if this is a lower surface, we need to shift the orientation of the 111
        # that gets applied if it is not 0. This is because the lower surface has diad
        # symmetry with the upper surface, which means that 0 does not change,
        # but 1 and 2 switch.
        if search_dir == 1:
            if orientation == 1:
                orientation = 2
            elif orientation == 2:
                orientation = 1

        # apply the pandey reconstruction on a diamond structure 111 surface within some set of layers
        # get masks for atoms
        layer_mask_set_lattice1_atom1, layer_mask_set_lattice1_atom2, layer_mask_set_lattice2_atom1, layer_mask_set_lattice2_atom2 = \
            self.identify_pandey_layers(atoms, surface_coords, read_from_atoms=read_from_atoms,
                                        xlim=xlim, ylim=ylim, zlim=zlim, search_dir=search_dir,
                                        atoms_for_cb=atoms_for_cb, orientation=orientation)
        # ase.io.write(f'{self.directions[self.surf_dir]}1.xyz', atoms)

        for layer in range(self.layers):
            lattice_1_atom_1shift = self.ral1a1[layer, :]
            lattice_1_atom_2shift = self.ral1a2[layer, :]
            lattice_2_atom_1shift = self.ral2a1[layer, :]
            lattice_2_atom_2shift = self.ral2a2[layer, :]
            # print(f'atom 1, layer{layer}', lattice_1_atom_1shift)
            # print(f'atom 2, layer{layer}', lattice_1_atom_2shift)
            # print(f'atom 3, layer{layer}', lattice_2_atom_1shift)
            # print(f'atom 4, layer{layer}', lattice_2_atom_2shift)
            # print(self.surf_dir)
            # invert component based on surf_dir if necessary

            if search_dir == 1:
                R = np.array([[-1, 0, 0],
                              [0, -1, 0],
                              [0, 0, 1]])
                lattice_1_atom_1shift = R @ \
                    lattice_1_atom_1shift
                lattice_1_atom_2shift = R @ \
                    lattice_1_atom_2shift
                lattice_2_atom_1shift = R @ \
                    lattice_2_atom_1shift
                lattice_2_atom_2shift = R @ \
                    lattice_2_atom_2shift

            # apply shifts to different atom types
            atoms.positions[:,
                            :][layer_mask_set_lattice1_atom1[:,
                                                             layer]] += lattice_1_atom_1shift
            atoms.positions[:,
                            :][layer_mask_set_lattice1_atom2[:,
                                                             layer]] += lattice_1_atom_2shift
            atoms.positions[:,
                            :][layer_mask_set_lattice2_atom1[:,
                                                             layer]] += lattice_2_atom_1shift
            atoms.positions[:,
                            :][layer_mask_set_lattice2_atom2[:,
                                                             layer]] += lattice_2_atom_2shift

        atoms.wrap()

    def map_si_110_3x1(self, fmax=0.0001, layers=6, cutoff=10,
                       switch=False, orientation=0, shift=0, permute=False):
        """Map the silicon 110 3x1 relaxation of the crystal.
        ------------------------------
        fmax : float
            Force tolerance for relaxation
        layers : int
            Number of layers deep to map the free surface
        cutoff : float
            Cutoff of the potential being used - used to decide the number of
            atoms beneath the mapped layers that need to be modelled
        shift : float
            Amount to shift the bulk structure before the surface is mapped
            such that the top layer of surface is physically meaningful. One
            use case of this would be to adjust the top layer of atoms being mapped
            in diamond such that it matches the top layer of atoms seen on the cleavage
            plane.
        switch : bool
            Whether or not to switch the sublattices around in the mapped structure.
            (Essentially, the correct surface can always be found from some combination of shift and switch)
        orientation : int, 1 or 0
            Whether to flip the reconstruction by 180 degrees. This is necessary for the lower surface.
        """
        # map out the pandey reconstruction on a diamond structure 111 surface
        # can then apply it later (this requires a seperate function as the unit cell of 3x1 reconstruction
        # breaks the symmetry of the surface)
        # self.split_cb_pair = True
        # 111 type surface always splits a cb pair
        ny = 3
        nx = 1
        nz = 3 * int(layers)
        self.layers = layers
        self.si_110_3x1_dirs = [[0, 0, 1], [1, -1, 0], [1, 1, 0]]
        # build the reconstruction frame rotation matrix
        U = np.zeros([3, 3])
        for i, direction in enumerate(self.si_110_3x1_dirs):
            direction = np.array(direction)
            U[:, i] = direction / np.linalg.norm(direction)
        self.U = U  # reconstruction frame rotation matrix
        a = Diamond(self.el,
                    size=[1, ny, nz],
                    latticeconstant=self.a0,
                    directions=self.si_110_3x1_dirs
                    )

        self.cb.set_sublattices(a, self.U)
        if switch:
            self.cb.switch_sublattices(a)
        sorted_z_vals = np.sort(a.get_positions()[self.cb.lattice1mask][:, 2])
        self.inter_surf_dist = sorted_z_vals[3] - sorted_z_vals[0]
        # REMOVE SHIFT FROM HERE
        a.translate([-0.01, 0.0, -shift * self.inter_surf_dist])
        # a.translate([0.01,0.01,-1])
        a.wrap()
        # print('INTER SURF DIST', self.inter_surf_dist)
        # seed the 3x1 reconstruction on the top plane
        pos_initial_unrotated = a.get_positions()

        bulk = a.copy()
        top_atom_mask = a.positions[:, 2] > (
            np.max(a.positions[:, 2]) - self.inter_surf_dist)
        top_atom_indices = np.where(top_atom_mask)[0]
        sorted_indices = top_atom_indices[np.argsort(
            a.positions[top_atom_indices, 1])]

        if permute:
            a.positions[sorted_indices[2], 2] += -1
            a.positions[sorted_indices[3], 2] += -1
            a.positions[sorted_indices[5], 2] += -1
            a.positions[sorted_indices[0], 2] += -1
        else:
            a.positions[sorted_indices[1], 2] += -1
            a.positions[sorted_indices[2], 2] += -1
            a.positions[sorted_indices[4], 2] += -1
            a.positions[sorted_indices[5], 2] += -1

        # ase.io.write('0_bulk.xyz', bulk)
        # ase.io.write('1_bulk.xyz', a)
        # relax the structure to get the 3x1 111 reconstruction
        # run an optimisation to relax the surface

        # add some vacuum
        cell = a.get_cell()
        # vacuum along self.surf_dir axis (surface normal)
        cell[2, :] *= 2
        a.set_cell(cell, scale_atoms=False)
        a.calc = self.calc

        # fix bottom atoms to stop lower surface reconstruction
        # print('POS INITIAL UNROTATED',pos_initial_unrotated)
        mask = pos_initial_unrotated[:, 2] < cutoff
        # print(pos_initial_unrotated[:,2])
        a.set_constraint(FixAtoms(mask=mask))
        opt_a = PreconLBFGS(a)
        opt_a.run(fmax=fmax)
        # ase.io.write('2_bulk.xyz', a)
        pos_after_unrotated = a.get_positions()
        # print(pos_after_unrotated[:, 1] < 0)
        # print(len(pos_after_unrotated[:, 1] < 0))
        # print(pos_after_unrotated[:, 1] > a.cell[1, 1])
        # print(len(pos_after_unrotated[:, 1] > a.cell[1, 1]))
        atoms_outside_mask = (pos_after_unrotated[:, 1] < 0) | (
            pos_after_unrotated[:, 1] > a.cell[1, 1])
        pushed_out_atoms = a[atoms_outside_mask]
        wrapped = False
        if len(pushed_out_atoms) > 0:
            # print('detected atoms to be wrapped')
            wrapped = True
            # a.wrap()
            pos_after_unrotated = a.get_positions()
            # ase.io.write('3_bulk.xyz', a)

        pos_diff_unrotated = pos_after_unrotated - pos_initial_unrotated
        # print('pos diff unrotated',pos_diff_unrotated)
        # map out the layers in the original structure
        surface_coords = pos_initial_unrotated[np.argmax(
            pos_initial_unrotated[:, 2]), :] + [0, 0, 0.01]
        tmp = self.surf_dir
        self.surf_dir = 2
        layer_mask_set_lattice1_atom1, layer_mask_set_lattice1_atom2, layer_mask_set_lattice1_atom3, layer_mask_set_lattice2_atom1, layer_mask_set_lattice2_atom2,\
            layer_mask_set_lattice2_atom3 = self.identify_si_110_layers(
                bulk, surface_coords, read_from_atoms=True, frame_dirs=self.si_110_3x1_dirs, switch=switch)
        self.surf_dir = tmp

        # relaxation array lattice 1 atom 1
        self.ral1a1 = np.zeros([layers, 3])
        # relaxation array lattice 1 atom 2
        self.ral1a2 = np.zeros([layers, 3])
        # relaxation array lattice 1 atom 3
        self.ral1a3 = np.zeros([layers, 3])
        # relaxation array lattice 2 atom 1
        self.ral2a1 = np.zeros([layers, 3])
        # relaxation array lattice 2 atom 2
        self.ral2a2 = np.zeros([layers, 3])
        # relaxation array lattice 2 atom 3
        self.ral2a3 = np.zeros([layers, 3])

        # print(np.shape(pos_diff_unrotated))
        # print(np.shape(layer_mask_set_lattice1_atom1))
        # generate rotation matrix to get from pandey frame to lab frame
        # orientation rotation tensor
        OR = np.zeros([3, 3])
        OR[0, 0] = np.cos((np.pi) * orientation)
        OR[0, 1] = -np.sin((np.pi) * orientation)
        OR[1, 0] = np.sin((np.pi) * orientation)
        OR[1, 1] = np.cos((np.pi) * orientation)
        OR[2, 2] = 1
        # print('OR',OR)

        R = np.transpose(self.A) @ self.U @ np.transpose(OR)
        # print('with no rotation',np.transpose(self.A)@self.U)
        # print('with rotation',R)
        for layer in range(layers):
            # print(layer)
            # print(
            # len(pos_diff_unrotated[layer_mask_set_lattice2_atom2[:, layer]]))

            self.ral1a1[layer,
                        :] = R @ pos_diff_unrotated[layer_mask_set_lattice1_atom1[:,
                                                                                  layer]][0,
                                                                                          :]
            self.ral1a2[layer,
                        :] = R @ pos_diff_unrotated[layer_mask_set_lattice1_atom2[:,
                                                                                  layer]][0,
                                                                                          :]
            self.ral1a3[layer,
                        :] = R @ pos_diff_unrotated[layer_mask_set_lattice1_atom3[:,
                                                                                  layer]][0,
                                                                                          :]
            self.ral2a1[layer,
                        :] = R @ pos_diff_unrotated[layer_mask_set_lattice2_atom1[:,
                                                                                  layer]][0,
                                                                                          :]
            self.ral2a2[layer,
                        :] = R @ pos_diff_unrotated[layer_mask_set_lattice2_atom2[:,
                                                                                  layer]][0,
                                                                                          :]
            self.ral2a3[layer,
                        :] = R @ pos_diff_unrotated[layer_mask_set_lattice2_atom3[:,
                                                                                  layer]][0,
                                                                                          :]
        # print(self.ral1a1, self.ral1a2, self.ral2a1, self.ral2a2)

    def identify_si_110_layers(self,
                               atoms,
                               surface_coords,
                               xlim=None,
                               ylim=None,
                               zlim=None,
                               search_dir=-1,
                               atoms_for_cb=None,
                               read_from_atoms=False,
                               frame_dirs=None,
                               orientation=0,
                               switch=False):
        """Function for identifying the layers adjacent to a surface, in a supplied atoms structure.
        This allows a mapped relaxation to be easily applied to a surface. This implementation is specific
        to the 111 Pandey reconstruction, as more atoms need to be identified due to the breaking of surface
        symmetry.

        Parameters
        ----------
        atoms : ASE atoms object
            Atoms object which contains surface to relax
        surface_coords : array_like
            Coordinates of a point that lies on the free surface (or within one layer distance above it)
        xlim : array_like
            [x lower, x upper], x range to apply relaxation
        ylim : array_like
            [y lower, y upper], y range to apply relaxation
        zlim : array_like
            [z lower, z upper], z range to apply relaxation
        search_dir : int
            -1: surface is below point provided, 1: surface is above point provided
        atoms_for_cb : ASE atoms object
            atoms object which allows the easy identification of sub-lattices in a multi-lattice
            crystal using the CubicCauchyBorn set_sublattices function. The difference between this and atoms may
            be periodic boundary conditions, for example.
        read_from_atoms : bool
            Whether or not to read in the cauchy-born sublattices from the atoms structure. This can be done if the
            atoms object contains arrays called lattice1mask and lattice2mask
        frame_dirs :
            The frame in which the Pandey reconstruction was mapped out - the atoms in the supplied frame will need to
            be rotated to this frame before the reconstruction can be applied
        orientation : 0,1 or 2
            Which of the three different orientations of the Pandey reconstructions are being applied to the surface


        Returns
        -------
        layer_mask_set : array_like, bool
            A 2D array dimensions [nlayers,natoms] where a row i is a mask for
            'atoms' corresponding to a layer i deep from the surface. In the case of
            a multi-lattice, two masks are returned, one for each of the sublattices on each layer.
        """
        # this function needs to identify the 4 sets of atoms per layer
        # that reconstruct (breaking symmetry)
        # it does this by taking an atoms structure, reading the sublattices
        # identifying the layers down from a point and then in each layer
        # identifying the 4 unique types of atom
        # to do this, it takes the positions of all the atoms of one sublattice
        # rotates these position vectors from the supplied frame to the mapped reconstruction frame
        # and then goes from left to right, shifting th

        # get the layer masks

        layer_mask_set_lattice1, layer_mask_set_lattice2 = \
            self.identify_layers(atoms, surface_coords, read_from_atoms=read_from_atoms, xlim=xlim,
                                 ylim=ylim, zlim=zlim, atoms_for_cb=atoms_for_cb, search_dir=search_dir)

        # rotate the lattice structure to match the pandey directions
        # build the rotation matrix
        # this is equivalent to transforming from the coordinate system
        # given by 'dirs' (the lab frame) to the lattice frame, and then from there
        # to the frame where the pandey reconstruction is defined.

        if frame_dirs is not None:
            T = np.zeros([3, 3])
            for i, direction in enumerate(frame_dirs):
                direction = np.array(direction)
                T[:, i] = direction / np.linalg.norm(direction)
        else:
            T = self.A

        OR = np.zeros([3, 3])
        OR[0, 0] = np.cos((np.pi) * orientation)
        OR[0, 1] = -np.sin((np.pi) * orientation)
        OR[1, 0] = np.sin((np.pi) * orientation)
        OR[1, 1] = np.cos((np.pi) * orientation)
        OR[2, 2] = 1

        # OR - Orientation rotation tensor
        # T - lab frame to lattice frame
        # U - lattice map frame to lattice frame
        R = OR @ np.transpose(self.U) @ T
        # print(self.U, T, R)

        layer_mask_set_lattice1_atom1 = np.zeros_like(
            layer_mask_set_lattice1, dtype=bool)
        layer_mask_set_lattice1_atom2 = np.zeros_like(
            layer_mask_set_lattice1, dtype=bool)
        layer_mask_set_lattice1_atom3 = np.zeros_like(
            layer_mask_set_lattice1, dtype=bool)
        layer_mask_set_lattice2_atom1 = np.zeros_like(
            layer_mask_set_lattice1, dtype=bool)
        layer_mask_set_lattice2_atom2 = np.zeros_like(
            layer_mask_set_lattice1, dtype=bool)
        layer_mask_set_lattice2_atom3 = np.zeros_like(
            layer_mask_set_lattice1, dtype=bool)

        # find an exclusion zone using the position of the leftmost lattice1
        # atom in the first layer (this is where we cut the cell in map_pandey)
        if switch:
            layer1 = atoms[layer_mask_set_lattice2[:, 0]]
        else:
            layer1 = atoms[layer_mask_set_lattice1[:, 0]]

        layer1pos = layer1.get_positions()
        # rotate
        rotated_layer1 = np.zeros_like(layer1pos)
        for j in range(np.shape(layer1pos)[0]):
            rotated_layer1[j, :] = R @ layer1pos[j, :]

        # find the leftmost point of layer 1
        leftpos = np.min(rotated_layer1[:, 1])
        pos = atoms.get_positions()
        rotated_pos = np.zeros_like(pos)
        for j in range(np.shape(pos)[0]):
            rotated_pos[j, :] = R @ pos[j, :]

        mask = rotated_pos[:, 1] < leftpos - 0.01
        for i in range(np.shape(layer_mask_set_lattice1)[1]):
            # print(np.shape(layer_mask_set_lattice1))
            lattice1atoms = atoms[layer_mask_set_lattice1[:, i]]
            pos_lattice_1_atoms = lattice1atoms.get_positions()
            # rotate
            rotated_pos_lattice1 = np.zeros_like(pos_lattice_1_atoms)
            for j in range(np.shape(pos_lattice_1_atoms)[0]):
                rotated_pos_lattice1[j, :] = R @ pos_lattice_1_atoms[j, :]

            # sort by the y position of the atoms in the rotated frame
            # that is used to map the pandey reconstruction
            # print('LAYER MASK LATTICE 1',layer_mask_set_lattice1[:,i])
            atom_indices = np.where(layer_mask_set_lattice1[:, i] == True)[0]
            pos_indices = np.argsort(rotated_pos_lattice1[:, 1])
            n_unique = 0
            # if the search direction is 1 (looking at a lower surface), run
            # this backwards
            if search_dir == 1:
                k = len(pos_indices) - 1
            else:
                k = 0
            compare_pos = rotated_pos_lattice1[:, 1][pos_indices[k]]
            for j in range(len(pos_indices)):
                if search_dir == 1:
                    k = len(pos_indices) - j - 1
                else:
                    k = j
                # build masks which hold the different types of atoms in
                # the symmetry broken reconstruction (using the y axis in the rotated
                # frame to differentiate them from eachother)
                # print(k, search_dir)
                # print(-search_dir*rotated_pos_lattice1[:, 1][pos_indices[k]],
                #      (-search_dir*(compare_pos-(search_dir*0.01))))
                if (-search_dir * rotated_pos_lattice1[:, 1][pos_indices[k]]
                    ) > (-search_dir * (compare_pos - (search_dir * 0.01))):
                    # if
                    # rotated_pos_lattice1[:,1][pos_indices[k]]<(compare_pos-0.01):
                    n_unique += 1
                    compare_pos = rotated_pos_lattice1[:, 1][pos_indices[k]]
                if n_unique % 3 == 0:
                    layer_mask_set_lattice1_atom1[:,
                                                  i][atom_indices[pos_indices[k]]] = True
                elif n_unique % 3 == 1:
                    layer_mask_set_lattice1_atom2[:,
                                                  i][atom_indices[pos_indices[k]]] = True
                elif n_unique % 3 == 2:
                    layer_mask_set_lattice1_atom3[:,
                                                  i][atom_indices[pos_indices[k]]] = True

            # repeat for lattice 2
            lattice2atoms = atoms[layer_mask_set_lattice2[:, i]]
            pos_lattice_2_atoms = lattice2atoms.get_positions()

            # rotate
            rotated_pos_lattice2 = np.zeros_like(pos_lattice_2_atoms)
            for j in range(np.shape(pos_lattice_2_atoms)[0]):
                rotated_pos_lattice2[j, :] = R @ pos_lattice_2_atoms[j, :]
            # sort by the y position of the atoms in the rotated frame
            # that is used to map the pandey reconstruction
            atom_indices = np.where(layer_mask_set_lattice2[:, i] == True)[0]
            # print(atom_indices)
            pos_indices = np.argsort(rotated_pos_lattice2[:, 1])
            n_unique = 0
            if search_dir == 1:
                k = len(pos_indices) - 1
            else:
                k = 0
            compare_pos = rotated_pos_lattice2[:, 1][pos_indices[k]]
            for j in range(len(pos_indices)):
                if search_dir == 1:
                    k = len(pos_indices) - j - 1
                else:
                    k = j
                # build masks which hold the different types of atoms in
                # the symmetry broken reconstruction (using the y axis in the rotated
                # frame to differentiate them from eachother)
                if (-search_dir * rotated_pos_lattice2[:, 1][pos_indices[k]]
                    ) > (-search_dir * (compare_pos - (search_dir * 0.01))):
                    n_unique += 1
                    compare_pos = rotated_pos_lattice2[:, 1][pos_indices[k]]
                if n_unique % 3 == 0:
                    layer_mask_set_lattice2_atom1[:,
                                                  i][atom_indices[pos_indices[k]]] = True
                elif n_unique % 3 == 1:
                    layer_mask_set_lattice2_atom2[:,
                                                  i][atom_indices[pos_indices[k]]] = True
                elif n_unique % 3 == 2:
                    layer_mask_set_lattice2_atom3[:,
                                                  i][atom_indices[pos_indices[k]]] = True

        # print(layer_mask_set_lattice1_atom1[:,layer])
        try:
            atoms.new_array('atomno', np.zeros(len(atoms), dtype=int))
        except RuntimeError:
            pass
        atomno = atoms.arrays['atomno']
        for layer in range(self.layers):
            atomno[layer_mask_set_lattice1_atom1[:, layer]] = 1
            atomno[layer_mask_set_lattice1_atom2[:, layer]] = 2
            atomno[layer_mask_set_lattice2_atom1[:, layer]] = 3
            atomno[layer_mask_set_lattice2_atom2[:, layer]] = 4
        return layer_mask_set_lattice1_atom1, layer_mask_set_lattice1_atom2, layer_mask_set_lattice1_atom3, layer_mask_set_lattice2_atom1, layer_mask_set_lattice2_atom2, layer_mask_set_lattice2_atom3

    def apply_si_110(self,
                     atoms,
                     surface_coords,
                     cb=None,
                     xlim=None,
                     ylim=None,
                     zlim=None,
                     search_dir=-1,
                     atoms_for_cb=None,
                     read_from_atoms=False,
                     orientation=0,
                     switch=False):
        """Function which applies the mapped out Si deformation to the surface layers
        in a supplied atoms structure.
        Parameters
        ----------
        atoms : ASE atoms object
            Atoms object which contains surface to relax
        surface_coords : array_like
            Coordinates of a point that lies on the free surface (or within one layer distance above it)
        xlim : array_like
            [x lower, x upper], x range to apply relaxation
        ylim : array_like
            [y lower, y upper], y range to apply relaxation
        zlim : array_like
            [z lower, z upper], z range to apply relaxation
        search_dir : int
            -1: surface is below point provided, 1: surface is above point provided
        atoms_for_cb : ASE atoms object
            atoms object which allows the easy identification of sub-lattices in a multi-lattice
            crystal using the CubicCauchyBorn set_sublattices function. The difference between this and atoms may
            be periodic boundary conditions, for example.
        read_from_atoms : bool
            Whether or not to read in the cauchy-born sublattices from the atoms structure. This can be done if the
            atoms object contains arrays called lattice1mask and lattice2mask
        orientation : 0,1 or 2
            Which of the three different orientations of the Pandey reconstructions are being applied to the surface
        """
        self.eval_cb = cb

        # apply the reconstruction on a diamond structure 110 surface within some set of layers
        # get masks for atoms
        layer_mask_set_lattice1_atom1, layer_mask_set_lattice1_atom2, layer_mask_set_lattice1_atom3, layer_mask_set_lattice2_atom1, layer_mask_set_lattice2_atom2,\
            layer_mask_set_lattice2_atom3 = \
            self.identify_si_110_layers(atoms, surface_coords, read_from_atoms=read_from_atoms,
                                        xlim=xlim, ylim=ylim, zlim=zlim, search_dir=search_dir,
                                        atoms_for_cb=atoms_for_cb, orientation=orientation, switch=switch)
        # ase.io.write(f'{self.directions[self.surf_dir]}1.xyz', atoms)

        for layer in range(self.layers):
            lattice_1_atom_1shift = self.ral1a1[layer, :]
            lattice_1_atom_2shift = self.ral1a2[layer, :]
            lattice_1_atom_3shift = self.ral1a3[layer, :]
            lattice_2_atom_1shift = self.ral2a1[layer, :]
            lattice_2_atom_2shift = self.ral2a2[layer, :]
            lattice_2_atom_3shift = self.ral2a3[layer, :]
            # invert component based on surf_dir if necessary

            if search_dir == 1:
                # TODO does this rotation matrix make this non-general? what
                # was I thinking? this needs testing
                R = np.array([[-1, 0, 0],
                              [0, -1, 0],
                              [0, 0, 1]])
                lattice_1_atom_1shift = R @ \
                    lattice_1_atom_1shift
                lattice_1_atom_2shift = R @ \
                    lattice_1_atom_2shift
                lattice_1_atom_3shift = R @ \
                    lattice_1_atom_3shift
                lattice_2_atom_1shift = R @ \
                    lattice_2_atom_1shift
                lattice_2_atom_2shift = R @ \
                    lattice_2_atom_2shift
                lattice_2_atom_3shift = R @ \
                    lattice_2_atom_3shift

            # apply shifts to different atom types
            atoms.positions[:,
                            :][layer_mask_set_lattice1_atom1[:,
                                                             layer]] += lattice_1_atom_1shift
            atoms.positions[:,
                            :][layer_mask_set_lattice1_atom2[:,
                                                             layer]] += lattice_1_atom_2shift
            atoms.positions[:,
                            :][layer_mask_set_lattice1_atom3[:,
                                                             layer]] += lattice_1_atom_3shift
            atoms.positions[:,
                            :][layer_mask_set_lattice2_atom1[:,
                                                             layer]] += lattice_2_atom_1shift
            atoms.positions[:,
                            :][layer_mask_set_lattice2_atom2[:,
                                                             layer]] += lattice_2_atom_2shift
            atoms.positions[:,
                            :][layer_mask_set_lattice2_atom3[:,
                                                             layer]] += lattice_2_atom_3shift

        atoms.wrap()
