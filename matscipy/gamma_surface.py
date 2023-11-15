import numpy as np
from ase.build import rotate, cut
from matscipy.dislocation import FixedLineAtoms
from ase.optimize import BFGSLineSearch
import warnings
from ase.constraints import UnitCellFilter
from matscipy.utils import validate_cubic_cell
import inspect
from matscipy.dislocation import CubicCrystalDissociatedDislocation

class GammaSurface():
    '''
    A class for generating gamma surface/generalised stacking fault images & plots
    '''
    def __init__(self, a, surface_direction, y_dir=None, crystalstructure=None, symbol="C"):
        '''
        Initialise by cutting and rotating the input structure.

        Parameters
        ----------
        a: float or ase.Atoms
            Lattice Constant or Starting structure to generate gamma surface from
            (Operates similarly to CubicCrystalDislocation)
            If lattice constant is provided, crystalstructure must also be set
        surface_direction: np.array of int or subclass of matscipy.dislocation.CubicCrystalDissociatedDislocation
            Vector direction of gamma surface, in miller index notation
            EG: np.array([0, 0, 1]), np.array([-1, 1, 0]), np.array([1, 1, 1])
            A subclass of matscipy.dislocation.CubicCrystalDissociatedDislocation (EG: DiamondGlideScrew or FCCEdge110Dislocation)
        y_dir: np.array of int or None
            Basis vector (in miller indices) to form "y" axis, which should be orthogonal to surface_direction
            If None, a suitable y_dir will be found automatically
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
            Dict giving miller indeces for "x", "y" and "z" directions of gamma surface plot
            -   self.surf_directions["z"] = surface_direction
            -   self.surf_directions["y"] = y_dir, if y_dir was specified
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

        axes = None
        disloc = False
        if inspect.isclass(surface_direction):
            if issubclass(surface_direction, CubicCrystalDissociatedDislocation):
                # Passed a class which inherits from CubicCrystalDissociatedDislocation
                disloc = True
        elif isinstance(surface_direction, CubicCrystalDissociatedDislocation):
            # Passed an instance of a class which inherits from CubicCrystalDissociatedDislocation
            disloc = True
        
        if disloc:
            # Dislocation object was found
            axes = surface_direction.left_dislocation.axes.copy()
            self.offset = -surface_direction.left_dislocation.unit_cell_core_position_dimensionless[1]
            crystalstructure = surface_direction.left_dislocation.crystalstructure

            self.ylims = [0, surface_direction.left_dislocation.glide_distance_dimensionless]

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
            if y_dir is None:
                _y_dir = self._y_dir_search(surface_direction)
            else:
                _y_dir = np.array(y_dir)

                if np.abs(np.dot(surface_direction, _y_dir)) >= 1e-3:
                    # Vector basis is not orthogonal
                    msg = f"y_dir vector {_y_dir} is not orthogonal to surface_direction vector {surface_direction}; dot(surface_direction, y_dir) = {float(np.dot(surface_direction, _y_dir))}\n" + \
                        "Gamma Surface plot may not show the correct directions"
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)

            _x_dir = np.cross(_y_dir, surface_direction)
        
            self.surf_directions = {
                "x": np.array(_x_dir),
                "y": np.array(_y_dir),
                "z": np.array(surface_direction)
                }
            ax = np.array([_x_dir, _y_dir, surface_direction])
        
        alat, self.cut_at = validate_cubic_cell(a, axes=ax, crystalstructure=crystalstructure, symbol=symbol)
        self.offset *= alat

    def _y_dir_search(self, d1):
        '''
        Search for integer x, y, z components for vectors perpendicular to d1
        '''
        nmax = 5
        tol = 1e-6
        for i in range(nmax):
            for j in range(-i-1, i+1):
                for k in range(-j-1, j+1):
                    if i==j and j==k and k==0:
                        continue

                    # Try all permutations
                    test_vec = np.array([i, j, k])

                    if np.abs(np.dot(d1, test_vec)) < tol:
                        return test_vec
                    
                    test_vec = np.array([k, i, j])

                    if np.abs(np.dot(d1, test_vec)) < tol:
                        return test_vec
                    
                    test_vec = np.array([j, k, i])

                    if np.abs(np.dot(d1, test_vec)) < tol:
                        return test_vec
        
        # No nice integer vector found!
        raise RuntimeError(f"Could not automatically find an integer basis from basis vector {d1}")

    def _gen_compressed_images(self, base_struct, nx, ny, x_points, y_points, vacuum=0.0):
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

    def generate_images(self, nx, ny, z_replications=1, atom_offset=None, cell_strain=0.0, vacuum=0.0, path_xlims=[0, 1], path_ylims=None, compressed=True):
        '''
        Generate gamma surface images on an (nx, ny) grid

        Parameters
        ----------
        nx: int
            Number of points in the x direction
        ny: int
            Number of points in the y direction
        z_replications: int
            Number of supercell copies in z (increases separation between periodic surfaces)
        atom_offset: float
            Offset in the z direction (in A) to apply to all atoms
        cell_strain: float
            Fractional strain to apply in z direction to cell only (0.1 = +10% strain)
        vacuum: float
            Additional vacuum layer (in A) to add between periodic images of the gamma surface
        vacuum_offset: float
            Offset (in A) applied to the position of the vacuum layer in the cell
            The position of the vacuum layer is given by:
            vac_pos = self.cut_at.cell[2, 2] * (1 + cell_strain) / 2 + vacuum_offset
        path_xlims: list or array of floats
            Limits (in fractional coordinates) of the stacking fault path in the x direction
        path_ylims: list or array of floats
            Limits (in fractional coordinates) of the stacking fault path in the x direction
            If not supplied, will be set to either [0, 1], or will be set based on the glide distance of the supplied dislocation

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

        base_struct = self.cut_at * (1, 1, z_replications)

        # Atom offset
        if atom_offset is not None:
            offset = atom_offset
        else:
            offset = self.offset
        pos = base_struct.get_positions()
        pos[:, 2] += offset
        base_struct.set_positions(pos)
        base_struct.wrap()
        
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

        images = []

        self.offsets = []

        x_points = np.linspace(*x_lims, nx) * dx
        y_points = np.linspace(*y_lims, ny) * dy

        if compressed is True:
            self.images = self._gen_compressed_images(base_struct, nx, ny, x_points, y_points, vacuum)
        else:
            pass
        return self.images


    def relax_images(self, calculator=None, ftol=1e-3, optimiser=BFGSLineSearch, constrain_atoms=True,
                     cell_relax=True, logfile=None, **kwargs):
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
        **kwargs: Other keyword args
            Extra arguments passed to the optimiser.run() method
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

        for image in self.images:
            select_all = np.full_like(image, True, dtype=bool)

            if constrain_atoms:
                image.set_constraint(FixedLineAtoms(select_all, (0, 0, 1)))

            cell_filter = UnitCellFilter(image, mask=cell_constraint_mask)
            image.calc = calc
            opt = optimiser(cell_filter, logfile=logfile)
            opt.run(ftol, **kwargs)

    def get_surface_energies(self, calculator=None, relax=False, **relax_kwargs):
        '''
        Get (self.nx, self.ny) grid of gamma surface energies from self.images

        Parameters
        ----------
        calculator : ase calculator
            Calculator to use for finding surface energies (and optionally in the relaxation)
        relax : bool
            Whether to additionally relax the images (through a call to self.relax_images) before finding energies
        **relax_kwargs : keyword args
            Extra kwargs to be passed to self.relax_images if relax=True

        Returns 
        -------
        Es : np.array
        (nx, ny) array of energy densities (energy per unit surface area) for the gamma surface, in eV/A**2
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

    def plot_gamma_surface(self, ax=None, si=True):
        '''
        Produce a matplotlib plot of the gamma surface energy, from the data gathered in self.generate_images and self.get_surface_energioes

        Returns a matplotlib fig and ax object

        ax: matplotlib axis object
            Axis to draw plot
        si: bool
            Use SI units (J/m^2) if True, else atomic units (eV/A^2)
        '''

        from ase.units import _e
        import matplotlib.pyplot as plt

        if si:
            mul = _e * 1e20
            units = "J/m$^2$"
        else:
            mul = 1
            units = "eV/${\AA}^2$"

        if ax is None:

            fig, my_ax = plt.subplots()

        else:
            my_ax = ax
            fig = my_ax.get_figure()

        im = my_ax.imshow(self.Es.T * mul, origin="lower", extent=[0, np.linalg.norm(self.x_disp), 0, np.linalg.norm(self.y_disp)], interpolation="bicubic")
        fig.colorbar(im, ax=ax, label=f"Energy Density ({units})")

        my_ax.set_xlabel("(" + " ".join(self.surf_directions["x"].astype(str)) + ")")
        my_ax.set_ylabel("(" + " ".join(self.surf_directions["y"].astype(str)) + ")")

        my_ax.set_title("(" + " ".join(self.surf_directions["z"].astype(str)) + ") Gamma Surface\nSurface Separation = {:0.2f} A".format(self.surface_separation))

        return fig, my_ax

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
    
    def plot_gamma_surface(self, ax=None, si=False):
        '''
        Produce a matplotlib plot of the stacking fault energy, from the data gathered in self.generate_images and self.get_surface_energy

        Returns a matplotlib fig and ax object

        ax: matplotlib axis object
            Axis to draw plot
        si: bool
            Use SI units (J/m^2) if True, else atomic units (eV/A^2)
        '''
        from ase.units import _e
        import matplotlib.pyplot as plt

        if si:
            mul = _e * 1e20
            units = "J/m$^2$"
        else:
            mul = 1
            units = "eV/${\AA}^2$"
        
        if ax is None:

            fig, my_ax = plt.subplots()

        else:
            my_ax = ax
            fig = my_ax.get_figure()

        my_ax.plot(np.arange(self.ny)/(self.ny-1), self.Es[0, :] * mul)
        my_ax.set_xlabel("Position along ("+ ' '.join(self.surf_directions["y"].astype(str)) + ") path")
        my_ax.set_ylabel(f"Energy Density ({units})")
        title = "(" + ' '.join(self.surf_directions["z"].astype(str)) + ") Stacking Fault\n" + "Surface Separation = {:0.2f}".format(self.surface_separation) + "${\AA}$"
        my_ax.set_title(title)
        return fig, my_ax
