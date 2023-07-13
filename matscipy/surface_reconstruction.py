import numpy as np
from ase.lattice.cubic import Diamond, FaceCenteredCubic, SimpleCubic, BodyCenteredCubic

from ase.optimize import LBFGS

from ase.constraints import FixAtoms

from matscipy.cauchy_born import CubicCauchyBorn
from ase.optimize.precon import PreconLBFGS
import ase.io


class SurfaceReconstruction:
    """Object for mapping and applying a surface reconstruction in simple and multi-lattices.

    The relaxation of a given free surface of a crystal at a given orientation can be mapped and
    saved. This relaxation can then be applied to atoms in a different ASE atoms object.
    This is extremely useful in the case of fracture mechanics, where surface relaxation
    can be mapped to the internal crack surfaces of an atoms object.
    """

    def __init__(self, el, a0, calc, directions, surf_dir, lattice='diamond'):
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
        lattice : string
            Crystal lattice type.
        """
        # directions is a list of 3 vectors specifiying the surface of interest
        # in the orientation specified by the problem. surf_dir says which axis is free surface
        # e.g directions=[[1,1,1], [0,-1,1], [-2,1,1]]
        self.el = el
        self.a0 = a0
        self.calc = calc
        self.directions = directions
        self.surf_dir = surf_dir

        # build the rotation matrix from the directions
        A = np.zeros([3, 3])
        for i, direction in enumerate(directions):
            direction = np.array(direction)
            A[:, i] = direction / np.linalg.norm(direction)
        self.A = A  # rotation matrix
        self.eval_cb = None
        if lattice == 'diamond':
            self.lattice = Diamond
            self.cb = CubicCauchyBorn(
                self.el, self.a0, self.calc, lattice=Diamond)

    def map_surface(self, fmax=0.0001, layers=6):
        """Map the relaxation of the crystal surface a certain number of layers deep.
        Parameters
        ----------
        fmax : float
            Force tolerance for relaxation
        layers : int
            Number of layers deep to map the free surface
        """
        # set the number of atomic layers to model
        height = 30
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
        cell = bulk.get_cell()
        # vacuum along self.surf_dir axis (surface normal)
        cell[self.surf_dir, :] *= 2
        slab = bulk.copy()
        slab.set_cell(cell, scale_atoms=False)

        # get the mask for the bottom surface and fix it
        pos_before = slab.get_positions()
        # fix bottom atoms to stop lower surface reconstruction
        mask = pos_before[:, self.surf_dir] < 30
        slab.set_constraint(FixAtoms(mask=mask))
        ase.io.write('initial_slab.xyz', slab)
        slab.calc = self.calc

        # run an optimisation to relax the surface
        opt_slab = PreconLBFGS(slab)
        opt_slab.run(fmax=fmax)

        pos_after = slab.get_positions()

        # measure the shift between the final position and the intial position
        pos_shift = pos_after - pos_before

        # for a multi-lattice:
        if self.cb is not None:
            # if this is a multi-lattice

            # set which atoms are in each multilattice based on the bulk structure. This is important because
            # different relaxations happen for each multilattice atom in each
            # atomic layer near the surface
            self.cb.set_sublattices(bulk, self.A)
            # in a multilattice, there are 2 atomic layers per unit cell, so
            # set the height to double the atomic layers
            total_layers = 2 * height
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
            # for a non-multilattice #need to look at this ngl
            total_layers = height
            n_atoms_per_layer = int(len(slab) / height)
            sorted_surf_indices = np.argsort(pos_before[:, self.surf_dir])
            surf_layer_list = np.zeros([height, n_atoms_per_layer], dtype=int)
            for i in range(height):
                surf_layer_list[i, :] = sorted_surf_indices[i *
                                                            n_atoms_per_layer:((i + 1) * n_atoms_per_layer)]

            self.inter_surf_dist = np.abs(
                pos_before[surf_layer_list[1, :]][0, surf_dir] - pos_before[surf_layer_list[0, :]][0, surf_dir])
            self.relaxation_array = np.zeros([layers, 3])
            for layer in range(layers):
                pos_shifts_by_layer = pos_shift[surf_layer_list[layer, :]]
                self.relaxation_array[layer, :] = pos_shifts_by_layer[0, :]

    def identify_layers(
            self,
            atoms,
            surface_coords,
            xlim=None,
            ylim=None,
            zlim=None,
            search_dir=-1,
            atoms_for_cb=None):
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

        if self.eval_cb is None:
            return layer_mask_set
        else:
            # for a multi-lattice, also have to account for shifts between multilattice atoms
            # in layers, so need to refine the masks further to be the lattice1
            # and lattice2 atoms in each layer
            if atoms_for_cb is None:
                atoms_for_cb = atoms
            # note that to find the sublattices, need to use a version of the
            # atoms structure with no free surface
            if self.eval_cb.lattice1mask is None:
                self.eval_cb.set_sublattices(atoms_for_cb, self.A)
            layer_mask_set_lattice1 = np.zeros_like(layer_mask_set, dtype=bool)
            layer_mask_set_lattice2 = np.zeros_like(layer_mask_set, dtype=bool)
            for layer in range(self.layers):
                layer_mask_set_lattice1[:,
                                        layer] = layer_mask_set[:,
                                                                layer] & self.eval_cb.lattice1mask
                layer_mask_set_lattice2[:,
                                        layer] = layer_mask_set[:,
                                                                layer] & self.eval_cb.lattice2mask
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
            atoms_for_cb=None):
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
        """
        # This finds the layers of atoms that need shifting and applies the
        # surface shift accordingly.
        self.eval_cb = cb
        if self.eval_cb is not None:
            layer_mask_set_lattice1, layer_mask_set_lattice2 = self.identify_layers(
                atoms, surface_coords, xlim=xlim, ylim=ylim, zlim=zlim, search_dir=search_dir, atoms_for_cb=atoms_for_cb)
            for layer in range(self.layers):
                # print('diff',self.relaxation_array_lattice1[layer,:])
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
                                      ] += (-search_dir) * self.relaxation_array
