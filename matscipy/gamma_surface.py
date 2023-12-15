import numpy as np
from ase.build import stack
from matscipy.dislocation import FixedLineAtoms
from ase.optimize import BFGSLineSearch
from ase.constraints import UnitCellFilter
from matscipy.utils import validate_cubic_cell, complete_basis
import inspect
from matscipy.dislocation import CubicCrystalDislocation, \
                                 CubicCrystalDissociatedDislocation
from ase.units import _e


class GammaSurface():
    '''
    A class for generating gamma surface/generalised stacking fault images & plots
    '''
    def __init__(self, a, surface_direction, glide_direction=None,
                 crystalstructure=None, symbol="C"):
        '''
        Initialise by cutting and rotating the input structure.

        Parameters
        ----------
        a: float or ase.Atoms
            Lattice Constant or Starting structure to generate gamma surface
            from (Operates similarly to CubicCrystalDislocation)
            If lattice constant is provided, crystalstructure must also be set
        surface_direction: np.array of int or subclass of
            matscipy.dislocation.CubicCrystalDissociatedDislocation
            Vector direction of gamma surface, in miller index notation
            EG: np.array([0, 0, 1]), np.array([-1, 1, 0]), np.array([1, 1, 1])
            A subclass of matscipy.dislocation.CubicCrystalDissociatedDislocation 
            (EG: DiamondGlideScrew or FCCEdge110Dislocation)
        glide_direction: np.array of int or None
            Basis vector (in miller indices) to form the glide direction of 
            the stacking fault, which is oriented along the y axis of generated images
            Should be orthogonal to surface_direction
            If None, a suitable glide_direction will be found automatically
        crystalstructure: str
            Crystal Structure to use in building a base cubic cell
            Required when a is a lattice constant
            Current accepted values: "fcc", "bcc", "diamond"
        symbol:str
            Chemical symbol to feed to ase.lattice.cubic cell builders
            Required when a is a lattice constant
            
        Attributes
        ------------
        self.cut_at : ase.atoms.Atoms object
            Cut Atoms object used as base for gamma surface image generation
        self.surf_directions : dict
            Dict giving miller indices for "x", "y" and "z" directions of gamma surface plot
            -   self.surf_directions["z"] = surface_direction
            -   self.surf_directions["y"] = glide_direction, if glide_direction was specified
        self.nx, self.ny : int
            Dimensions of the (nx, ny) gamma surface grid
        self.images : list of ase.atoms.Atoms objects
            Generated gamma surface images (populated after self.generate_images is called)
        '''

        self.images = []
        self.calc = None
        self.nx = 0
        self.ny = 0
        self.x_disp = 0
        self.y_disp = 0
        self.surface_area = 0
        self.offset = 0
        self.crystalstructure = None
        self.ylims = [0, 1]
        self.Es = None

        axes = None

        # Check if surface_direction is some kind of dislocation
        disloc = False
        if inspect.isclass(surface_direction):
            if issubclass(surface_direction, CubicCrystalDislocation):
                # Passed a class
                disloc = True
                dissociated = issubclass(surface_direction, CubicCrystalDissociatedDislocation)
        elif isinstance(surface_direction, CubicCrystalDislocation):
            # Passed an instance
            disloc = True
            dissociated = isinstance(surface_direction, CubicCrystalDissociatedDislocation)
        
        if disloc:
            # surface_direction was some kind of CubicCrystalDislocation
            if dissociated:
                disloc = surface_direction.left_dislocation
            else:
                disloc = surface_direction
            # Dislocation object was found
            axes = disloc.axes.copy()
            self.offset = -disloc.unit_cell_core_position_dimensionless[1]
            crystalstructure = disloc.crystalstructure

            self.ylims = [0, disloc.glide_distance_dimensionless]

            self.surf_directions = {
                "x": axes[2, :],
                "y": axes[0, :],
                "z": axes[1, :]
            }
            ax = axes.copy()
            ax[0, :] = self.surf_directions["x"]
            ax[1, :] = self.surf_directions["y"]
            ax[2, :] = self.surf_directions["z"]
        else:
            # surface_direction is a vector for the basis
            z, y, x = complete_basis(surface_direction, glide_direction)

            # z, y, x -> x, y, z swap means chirality is wrong
            if glide_direction is None:
                y, x = x, y
            else:
                x = - x

            self.surf_directions = {
                "x": np.array(x),
                "y": np.array(y),
                "z": np.array(z)
                }
            ax = np.array([x, y, z])

        alat, self.cut_at = validate_cubic_cell(a, axes=ax, 
                                                crystalstructure=crystalstructure,
                                                symbol=symbol)
        self.offset *= alat

    def _vec_to_miller(self, vec, latex=True, brackets="["):
        '''
        Converts np.array vec to string miller notation

        vec: np.array
            array to convert to string miller notation
            e.g. (100) or (11-2)
        latex: bool
            Use LaTeX expressions to construct representation
        brackets: str
            Bracketing style. 
            Options are "(" or ")" for "()", "{" or "}" for "{}",
             or "[" or "]" for "[]"
            Other options can be supplied by passing a len(2) list of str
        '''
        # Sort out bracketing
        bracket_types = [
            [["(", ")", "()"], ["(", ")"]],
            [["{", "}", r"{}"], ["{", "}"]],
            [["[", "]", "[]"], ["[", "]"]]
        ]
        found = False
        for key, val in bracket_types:
            if brackets in key:
                lbrac = val[0]
                rbrac = val[1]
                found = True
        if not found:
            if len(brackets) ==2:
                lbrac = brackets[0]
                rbrac = brackets[1]
            else:
                lbrac = "("
                rbrac = ")"

        int_vec = vec.astype(int)
        l = []
        for item in int_vec:
            if latex and item < 0:
                l.extend(r"$\overline{" + str(-item) + "}$")
            else:
                l.extend(str(item))
        return lbrac + "".join(l) + rbrac

    def _gen_cellmove_images(self, base_struct, 
                             nx, ny, x_points, y_points, vacuum=0.0):
        # Add vacuum
        half_dist = base_struct.cell[2, 2] / 2

        atom_mask = base_struct.get_positions()[:, 2] > half_dist

        cell = base_struct.cell[:, :].copy()
        cell[2, 2] += vacuum
        pos = base_struct.get_positions()
        pos[atom_mask, 2] += vacuum
        base_struct.set_positions(pos)

        # Gen images
        images = []
        for i in range(nx):
            for j in range(ny):
                offset = x_points[i] + y_points[j]
                self.offsets.append(offset)

                new_cell = cell.copy()
                new_cell[2, :] += offset

                ats = base_struct.copy()

                ats.set_cell(new_cell, scale_atoms=False)

                images.append(ats)
        return images

    def _gen_atommove_images(self, base_struct, nx, ny, x_points, y_points, vacuum):
        slab = stack(base_struct.copy(), base_struct.copy())
        base_pos = slab.get_positions()

        top_idxs = np.arange(len(base_struct), len(slab))

        if vacuum:
            slab.set_pbc([True, True, False])

        images = []
        for i in range(nx):
            for j in range(ny):
                offset = x_points[i] + y_points[j]
                self.offsets.append(offset)
                ats = slab.copy()
                pos = base_pos.copy()
                pos[top_idxs, :] += offset
                ats.set_positions(pos)

                images.append(ats)
        return images

    def generate_images(self, nx, ny, z_reps=1, z_offset=0.0, cell_strain=0.0, vacuum=0.0,
                        path_xlims=[0, 1], path_ylims=None, cell_move=True):
        '''
        Generate gamma surface images on an (nx, ny) grid

        Parameters
        ----------
        nx: int
            Number of points in the x direction
        ny: int
            Number of points in the y direction
        z_reps: int
            Number of supercell copies in z
            (increases separation between periodic surfaces)
        z_offset: float
            Offset in the z direction (in A) to apply to all atoms
            Used to select different stacking fault planes
            sharing the same normal direction
            (e.g. glide and shuffle planes in Diamond)
        cell_strain: float
            Fractional strain to apply in z direction to cell
            only (0.1 = +10% strain; atoms aren't moved)
            Helpful for issues when atoms are extremely close in unrelaxed images.
        vacuum: float
            Additional vacuum layer (in A) to add between periodic
            images of the gamma surface
        vacuum_offset: float
            Offset (in A) applied to the position of
            the vacuum layer in the cell
            The position of the vacuum layer is given by:
            vac_pos = self.cut_at.cell[2, 2] * (1 + cell_strain) / 2 + vacuum_offset
        path_xlims: list or array of floats
            Limits (in fractional coordinates) of the stacking fault path in the x direction
        path_ylims: list or array of floats
            Limits (in fractional coordinates) of the stacking fault path in the x direction
            If not supplied, will be set to either [0, 1],
            or will be set based on the glide distance of the supplied dislocation
        cell_move: bool
            Toggles using the cell move method (True) or atom move method (False)

        Returns
        --------
        images: list of ase.atoms.Atoms
            nx*ny length list of gamma surface images
        '''

        if path_ylims is None:
            y_lims = self.ylims
        else:
            y_lims = path_ylims

        x_lims = path_xlims

        self.nx = nx
        self.ny = ny

        xs = np.linspace(x_lims[0], x_lims[1], nx)
        ys = np.linspace(y_lims[0], y_lims[1], ny)

        base_struct = self.cut_at * (1, 1, z_reps)

        # Atom offset
        offset = z_offset + self.offset
        pos = base_struct.get_positions()
        pos[:, 2] += offset
        base_struct.set_positions(pos)
        # Wrap atoms back to select a new plane at gamma surface edge
        base_struct.wrap()

        # Realign atoms back to origin
        pos = base_struct.get_positions()
        min_z = np.min(pos[:, 2])
        pos[:, 2] -= min_z
        base_struct.set_positions(pos)

        # Apply cell strain
        new_cell = base_struct.cell[:, :].copy()
        new_cell[2, 2] += cell_strain
        base_struct.set_cell(new_cell, scale_atoms=False)

        # Surface Size
        cell = base_struct.cell[:, :]
        self.x_disp = cell[0, 0] * np.array([1, 0, 0])
        self.y_disp = cell[1, 1] * np.array([0, 1, 0])

        self.surface_area = np.linalg.norm(np.cross(cell[0, :], cell[1, :]))
        self.surface_separation = np.abs(cell[2, 2])

        dx = self.x_disp
        dy = self.y_disp


        self.offsets = []

        x_points = np.zeros((nx, 3))
        y_points = np.zeros((ny, 3))


        for i in range(nx):
            x_points[i, :] = dx * xs[i]
        for j in range(ny):
            y_points[j, :] = dy * ys[j]

        if cell_move is True:
            # Generate images via cell moves
            self.images = self._gen_cellmove_images(base_struct, nx, ny, 
                                                    x_points, y_points, vacuum)
        else:
            # Generate images via atom moves
            self.images = self._gen_atommove_images(base_struct, nx, ny, 
                                                    x_points, y_points, bool(vacuum))
        return self.images

    def relax_images(self, calculator=None, ftol=1e-3, 
                     optimiser=BFGSLineSearch, constrain_atoms=True,
                     cell_relax=True, logfile=None, steps=200):
        '''
        Utility function to relax gamma surface images using calculator

        Parameters
        ----------
        calculator: ase calculator instance
            Calculator to give forces for relaxation
        ftol: float
            force tolerance for optimisation
        optimiser: ase optimiser
            optimiser method to use
        constrain_atoms: bool
            Toggle constraint that all atoms only relax in z direction
            (normal to the gamma surface)
        cell_relax: bool
            Allow z component of cell (cell[2, 2]) to relax
            (normal to the gamma surface)
        steps: int
            Maximum number of iterations allowed for each image relaxation
            Fed directly to optimiser.run()
        '''
        if calculator is not None:
            calc = calculator
        else:
            calc = self.calc

        if calc is None:
            raise RuntimeError("No calculator supplied for relaxation")

        cell_constraint_mask = np.zeros((3, 3))
        if cell_relax:
            cell_constraint_mask[2, 2] = 1  # Allow z component relaxation

        select_all = np.full_like(self.images[0], True, dtype=bool)
        for image in self.images:

            if constrain_atoms:
                image.set_constraint(FixedLineAtoms(select_all, (0, 0, 1)))

            cell_filter = UnitCellFilter(image, mask=cell_constraint_mask)
            image.calc = calc
            opt = optimiser(cell_filter, logfile=logfile)
            opt.run(ftol, steps=steps)

            if not opt.converged():
                raise RuntimeError("An image relaxation failed to converge")

    def get_energy_densities(self, calculator=None,
                             relax=False, **relax_kwargs):
        '''
        Get (self.nx, self.ny) grid of gamma surface energies from self.images

        Parameters
        ----------
        calculator : ase calculator
            Calculator to use for finding surface energies 
            (and optionally in the relaxation)
        relax : bool
            Whether to additionally relax the images 
            (through a call to self.relax_images) before finding energies
        **relax_kwargs : keyword args
            Extra kwargs to be passed to self.relax_images if relax=True

        Returns 
        -------
        Es : np.array
        (nx, ny) array of energy densities (energy per unit surface area) 
        for the gamma surface, in eV/A**2
        '''

        if calculator is not None:
            calc = calculator
        else:
            calc = self.calc

        if calc is None:
            raise RuntimeError("No calculator supplied for energy evaluation")

        cell = self.images[0].cell[:, :]
        self.surface_area = np.linalg.norm(np.cross(cell[0, :], cell[1, :]))
        self.surface_separation = np.abs(cell[2, 2])

        Es = np.zeros((self.nx, self.ny))

        if relax:
            self.relax_images(calc, **relax_kwargs)

        idx = 0

        for i in range(self.nx):
            for j in range(self.ny):
                image = self.images[idx]
                image.calc = calc
                Es[i, j] = image.get_potential_energy()
                idx += 1

        Es -= np.min(Es)
        Es /= self.surface_area

        self.Es = Es
        return Es

    def plot_energy_densities(self, Es=None, ax=None, 
                              si=True, interpolation="bicubic"):
        '''
        Produce a matplotlib plot of the gamma surface energy,
        from the data gathered in self.generate_images
        and self.get_surface_energies

        Returns a matplotlib fig and ax object
        Es: np.array
            (nx, ny) array of energy densities. If None, uses self.Es
            (if populated from self.get_energy_densities())
        ax: matplotlib axis object
            Axis to draw plot
        si: bool
            Use SI units (J/m^2) if True, else ASE units (eV/A^2)
        interpolation: str
            arg to pass to matplotlib imshow interpolation
        '''
        import matplotlib.pyplot as plt

        if Es is None:
            Es = self.Es

        if Es is None:
            # Should have been populated from self.Es
            # If not, self.get_energy_densities() should have been called
            raise RuntimeError("No energies to use im plotting! Pass Es=Es," +
                               " or call self.get_energy_densities().")

        if si:
            mul = _e * 1e20
            units = "J/m$^2$"
        else:
            mul = 1
            units = "eV/$ \mathrm{A}^2$"

        if ax is None:

            fig, my_ax = plt.subplots()

        else:
            my_ax = ax
            fig = my_ax.get_figure()

        im = my_ax.imshow(self.Es.T * mul, origin="lower",
                          extent=[0, np.linalg.norm(self.x_disp),
                                  0, np.linalg.norm(self.y_disp)],
                          interpolation=interpolation)

        fig.colorbar(im, ax=ax, label=f"Energy Density ({units})")

        my_ax.set_xlim([0, np.linalg.norm(self.x_disp)])
        my_ax.set_ylim([0, np.linalg.norm(self.y_disp)])

        my_ax.set_xlabel("Glide in " + self._vec_to_miller(self.surf_directions["x"]) + " ($ \mathrm{A}$)")
        my_ax.set_ylabel("Glide in " + self._vec_to_miller(self.surf_directions["y"]) + " ($ \mathrm{A}$)")

        my_ax.set_title(self._vec_to_miller(self.surf_directions["z"], brackets="()") +
                        " Gamma Surface\nSurface Separation = {:0.2f} A".format(self.surface_separation))

        return fig, my_ax

    def show(self, CNA_color=True, plot_energies=False, si=False, **kwargs):
        '''
        Overload of GammaSurface.show()
        Plots an animation of the stacking fault structure,
        and optionally the associated energy densities
        (requires self.get_energy_densitities() to have been called)

        CNA_color: bool
            Toggle atom colours based on Common Neighbour Analysis (Structure identification)
        plot_energies:
            Add additional plot showing energy density profile, alongside the structure
        si: bool
            Plot energy densities in SI units (J/m^2), or "ASE" units eV/A^2
        **kwargs
            extra kwargs passed to ase.visualize.plot.plot_atoms
        '''
        from ase.visualize.plot import plot_atoms
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matscipy.utils import get_structure_types

        images = [image.copy() for image in self.images]

        if CNA_color:
            for system in images:
                atom_labels, structure_names, colors = get_structure_types(system, 
                                                                           diamond_structure=True)
                atom_colors = [colors[atom_label] for atom_label in atom_labels]

                system.set_array("colors", np.array(atom_colors))

        fig, ax = plt.subplots(ncols=2, nrows=2)
        atom_ax1, atom_ax2, atom_ax3, energy_ax = ax.flatten()

        atom_axes = [atom_ax1, atom_ax2, atom_ax3]
        rotations = ["-90z, -90x", "-90x", ""]

        directions = [ # Directions corresponding to the above rotations
            ("y", "z"),
            ("x", "z"),
            ("x", "y")
        ]

        def plot_all_atom_axes(atoms, keep_lims=True):
            for i in range(3):
                curr_ax = atom_axes[i]
                rot = rotations[i]

                xdir, ydir = directions[i]

                xlim = curr_ax.get_xlim()
                ylim = curr_ax.get_ylim()
                curr_ax.clear()

                curr_ax.set_xticks([])
                curr_ax.set_yticks([])

                curr_ax.set_xlabel(self._vec_to_miller(self.surf_directions[xdir]))
                curr_ax.set_ylabel(self._vec_to_miller(self.surf_directions[ydir]))

                if CNA_color:
                    plot_atoms(atoms, ax=curr_ax, colors=atoms.get_array("colors"),
                    rotation=rot, **kwargs)
                else: # default color are jmol (same is nglview)
                    plot_atoms(atoms, ax=curr_ax, rotation=rot, **kwargs)
                # avoid changing the size of the plot during animation
                if keep_lims:
                    curr_ax.set_xlim(xlim)
                    curr_ax.set_ylim(ylim)

        if plot_energies:
            if self.Es is not None:
                Es = self.Es
            else:
                try:
                    # Assume that calculator is attached so we can get energy densities
                    Es = self.get_energy_densities()
                except:
                    raise RuntimeError("Cannot plot energy densities before get_energy_densities is called!")

            self.plot_energy_densities(ax=energy_ax, si=si)
        else:
            energy_ax.clear()
            energy_ax.axis("off")

        plt.tight_layout()

        plot_all_atom_axes(images[-1], keep_lims=False)

        def drawimage(framedata):
            framenum, atoms = framedata
            # Plot Structure

            plot_all_atom_axes(atoms, keep_lims=True)

            # Plot energies
            if plot_energies:
                # Manually clear the colorbar
                energy_ax.images[-1].colorbar.remove()
                energy_ax.clear()
                self.plot_energy_densities(ax=energy_ax, si=si)
                pos = self.offsets[framenum]
                plt.scatter(pos[0], pos[1], marker="x", color="k")

        animation = FuncAnimation(fig, drawimage, frames=enumerate(images),
                                  save_count=len(images),
                                  init_func=lambda: None,
                                  interval=200)
        return animation


class StackingFault(GammaSurface):
    '''
    Class for stacking fault-specific generation & plotting
    '''

    def generate_images(self, n, *args, **kwargs):
        '''
        Generate gamma surface images on an (n) line
        n: int
            Number of images
        '''
        return super().generate_images(1, n, *args, **kwargs)

    def plot_energy_densities(self, Es=None, ax=None, si=False):
        '''
        Produce a matplotlib plot of the stacking fault energy, 
        from the data gathered in self.generate_images and self.get_surface_energy

        Returns a matplotlib fig and ax object
        Es: np.array
            (nx, ny) array of energy densities. If None, uses self.Es 
            (if populated from self.get_energy_densities())
        ax: matplotlib axis object
            Axis to draw plot
        si: bool
            Use SI units (J/m^2) if True, else ASE units (eV/A^2)
        '''
        import matplotlib.pyplot as plt

        if Es is None:
            Es = self.Es

        if Es is None:
            # Should have been populated from self.Es
            # If not, self.get_energy_densities() should have been called
            raise RuntimeError("No energies to use im plotting! Pass Es=Es," + 
                               " or call self.get_energy_densities().")

        if si:
            mul = _e * 1e20
            units = "J/m$^2$"
        else:
            mul = 1
            units = "eV/$ \mathrm{A}^2$"

        if ax is None:

            fig, my_ax = plt.subplots()

        else:
            my_ax = ax
            fig = my_ax.get_figure()

        my_ax.plot(np.linalg.norm(self.y_disp) * np.arange(self.ny)/(self.ny-1), self.Es[0, :] * mul)
        my_ax.set_xlabel("Glide distance in "+ self._vec_to_miller(self.surf_directions["y"]) + " ($ \mathrm{A}$)")
        my_ax.set_ylabel(f"Energy Density ({units})")
        title = self._vec_to_miller(self.surf_directions["z"], brackets="()") + " Stacking Fault\n" + \
                "Surface Separation = {:0.2f}".format(self.surface_separation) + "$ \mathrm{A}$"
        my_ax.set_title(title)
        return fig, my_ax

    def show(self, CNA_color=True, plot_energies=False, si=False, **kwargs):
        '''
        Overload of GammaSurface.show()
        Plots an animation of the stacking fault structure, 
        and optionally the associated 
        energy densities (requires self.get_energy_densitities() to have been called)

        CNA_color: bool
            Toggle atom colours based on Common Neighbour Analysis (Structure identification)
        plot_energies:
            Add additional plot showing energy density profile, alongside the structure
        si: bool
            Plot energy densities in SI units (J/m^2), or "ASE" units eV/A^2
        rotation: str
            rotation to apply to the structures, 
            passed directly to ase.visualize.plot.plot_atoms
        **kwargs
            extra kwargs passed to ase.visualize.plot.plot_atoms
        '''
        from ase.visualize.plot import plot_atoms
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matscipy.utils import get_structure_types

        images = [image.copy() for image in self.images]
        nims = len(images)

        rotation = "-90z, -90x"

        if si:
            si_fac = 1e20 * _e
        else:
            si_fac = 1

        if CNA_color:
            for system in images:
                atom_labels, structure_names, colors = get_structure_types(system, 
                                                                           diamond_structure=True)
                atom_colors = [colors[atom_label] for atom_label in atom_labels]

                system.set_array("colors", np.array(atom_colors))

        if plot_energies:
            fig, ax = plt.subplots(ncols=2)
            atom_ax, energy_ax = ax

            if self.Es is not None:
                Es = self.Es
            else:
                try:
                    # Assume that calculator is attached so we can get energy densities
                    Es = self.get_energy_densities()
                except:
                    raise RuntimeError("Cannot plot energy densities before get_energy_densities is called!")

            self.plot_energy_densities(ax=energy_ax, si=si)
        else:
            fig, atom_ax = plt.subplots()

        plt.tight_layout()

        plot_atoms(images[-1], ax=atom_ax, rotation=rotation, **kwargs)

        xlim = atom_ax.get_xlim()
        ylim = atom_ax.get_ylim()

        def drawimage(framedata):
            framenum, atoms = framedata
            # Plot Structure
            atom_ax.clear()

            atom_ax.set_xticks([])
            atom_ax.set_yticks([])

            atom_ax.set_xlabel(self._vec_to_miller(self.surf_directions["y"]))
            atom_ax.set_ylabel(self._vec_to_miller(self.surf_directions["z"]))

            if CNA_color:
                plot_atoms(atoms, ax=atom_ax, colors=atoms.get_array("colors"), 
                rotation=rotation, **kwargs)
            else: # default color are jmol (same is nglview)
                plot_atoms(atoms, ax=atom_ax, rotation=rotation, **kwargs)
            # avoid changing the size of the plot during animation
            atom_ax.set_xlim(xlim) 
            atom_ax.set_ylim(ylim)

            # Plot energies
            if plot_energies:
                energy_ax.clear()
                self.plot_energy_densities(ax=energy_ax, si=si)
                plt.scatter(np.linalg.norm(self.y_disp) * framenum/(nims-1), Es[0, framenum] * si_fac, marker="x", color="k")

        animation = FuncAnimation(fig, drawimage, frames=enumerate(images),
                                  save_count=len(images),
                                  init_func=lambda: None,
                                  interval=200)
        return animation
