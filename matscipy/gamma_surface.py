import numpy as np
from ase.build import rotate, cut
from matscipy.dislocation import FixedLineAtoms
from ase.optimize import BFGSLineSearch
import warnings
from ase.constraints import UnitCellFilter


class GammaSurface():
    '''
    A class for generating gamma surface/generalised stacking fault images & plots
    '''
    def __init__(self, ats, surface_direction, y_dir=None):
        '''
        Initialise by cutting and rotating the input structure.

        ats: ase.Atoms
            Starting structure to generate gamma surface from
        surface_direction: np.array
            Vector direction of gamma surface, in miller index notation
            EG: np.array([0, 0, 1]), np.array([-1, 1, 0]), np.array([1, 1, 1])
        y_dir: np.array | None
            Basis vector (in miller indeces) to form "y" axis, which should be orthogonal to surface_direction
            If None, a suitable y_dir will be found automatically

        Useful attributes:
        self.cut_at
            Cut Atoms object used as base for gamma surface image generation
        self.surf_directions
            Dict giving miller indeces for "x", "y" and "z" directions of gamma surface plot
            -   self.surf_directions["z"] = surface_direction
            -   self.surf_directions["y"] = y_dir, if y_dir was specified
        self.cell_directions
            Dict giving miller indeces for "x", "y" and "z" axes of self.cut_at, relative to the initial structure
            Will differ from self.surf_directions if compress=True, and a smaller representation was found
        self.nx, self.ny
            Dimensions of the (nx, ny) gamma surface grid
        self.images
            Generated gamma surface images (populated after self.generate_images is called)
        '''

        self.images = []
        self.nx = 0
        self.ny = 0
        self.x_disp = 0
        self.y_disp = 0
        self.surface_area = 0

        def _cut_and_rotate(ats, surface_direction, y_dir):
            # Cut such that surface_direction lies in z
            eps = 1e-3
            at = cut(ats.copy(), a=surface_direction, b=y_dir, origo=[-eps]*3)
            rotate(at, at.cell[2, :].copy(), np.array([0, 0, 1]), at.cell[1, :].copy(), np.array([0, 1, 0]))
            return at

        self._base_ats = ats.copy()

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

    
        _cell_y_dir = _y_dir
        _cell_x_dir = _x_dir

        self.mapping = {
            "x" : np.array([1, 0, 0]),
            "y" : np.array([0, 1, 0])
        }
        self.cut_at = _cut_and_rotate(self._base_ats, surface_direction, _y_dir)
    
        self.surf_directions = {
            "x": np.array(_x_dir),
            "y": np.array(_y_dir),
            "z": np.array(surface_direction)
            }
        self.cell_directions = {
            "x": np.array(_cell_x_dir),
            "y": np.array(_cell_y_dir),
            "z": np.array(surface_direction)
        }

    def _y_dir_search(self, d1):
        # Search for integer x, y, z components for vectors perpendicular to d1
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

    def generate_images(self, nx, ny, z_replications=1, atom_offset=0.0, cell_strain=0.0, vacuum=0.0, vacuum_offset=0.0, path_xlims=[0.0, 1.0], path_ylims=None):
        '''
        Generate gamma surface images on an (nx, ny) grid

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
        path_xlims: list|array of floats
            Limits (in fractional coordinates) of the stacking fault path in the x direction
        path_ylims: list|array of floats
            Limits (in fractional coordinates) of the stacking fault path in the x direction
            If not supplied, will be set to path_xlims
        '''

        if path_ylims is None:
            y_lims = path_xlims
        else:
            y_lims = path_ylims
        
        x_lims = path_xlims

        self.nx = nx
        self.ny = ny

        base_struct = self.cut_at * (1, 1, z_replications)

        # Atom offset
        pos = base_struct.get_positions()
        pos[:, 2] += atom_offset
        base_struct.set_positions(pos)
        base_struct.wrap()
        
        # Apply cell strain
        new_cell = base_struct.cell[:, :].copy()
        new_cell[2, 2] += cell_strain
        base_struct.set_cell(new_cell, scale_atoms=False)

        # Add vacuum
        half_dist = new_cell[2, 2] / 2 + vacuum_offset

        atom_mask = base_struct.get_positions()[:, 2] > half_dist
        
        cell = base_struct.cell[:, :].copy()
        cell[2, 2] += vacuum
        pos = base_struct.get_positions()
        pos[atom_mask, 2] += vacuum
        base_struct.set_positions(pos)

        # Surface Size
        # TODO: This assumes cell is cuboidal - is this always true?
        self.x_disp = cell[:, :] @ self.mapping["x"]
        self.y_disp = cell[:, :] @ self.mapping["y"]

        self.surface_area = np.linalg.norm(np.cross(cell[0, :], cell[1, :]))
        self.surface_separation = np.abs(cell[2, 2])
                
        dx = self.x_disp
        dy = self.y_disp

        images = []

        self.offsets = []

        x_points = np.linspace(*x_lims, nx)
        y_points = np.linspace(*y_lims, ny)

        # Gen images
        for i in range(nx):
            for j in range(ny):
                offset = x_points[i] * dx + y_points[j] * dy
                self.offsets.append(offset)
            
                new_cell = cell.copy()
                new_cell[2, :] += offset

                ats = base_struct.copy()

                ats.set_cell(new_cell, scale_atoms=False)

                images.append(ats)
        self.images = images

    def relax_images(self, calculator, ftol=1e-3, optimiser=BFGSLineSearch, constrain_atoms=True,
                     cell_relax=True, logfile=None, **kwargs):
        '''
        Utility function to relax gamma surface images using calculator

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

        cell_constraint_mask = np.zeros((3, 3))
        if cell_relax:
            cell_constraint_mask[2, 2] = 1  # Allow z component relaxation

        for image in self.images:
            select_all = np.full_like(image, True, dtype=bool)

            if constrain_atoms:
                image.set_constraint(FixedLineAtoms(select_all, (0, 0, 1)))

            cell_filter = UnitCellFilter(image, mask=cell_constraint_mask)
            image.calc = calculator
            opt = optimiser(cell_filter, logfile=logfile)
            opt.run(ftol, **kwargs)

    def get_surface_energies(self, calculator, relax=False, **relax_kwargs):
        '''
        Get (self.nx, self.ny) grid of gamma surface energies from self.images

        calculator: ase calculator
            Calculator to use for finding surface energies (and optionally in the relaxation)
        relax: bool
            Whether to additionally relax the images (through a call to self.relax_images) before finding energies
        **relax_kwargs: keyword args
            Extra kwargs to be passed to self.relax_images if relax=True

        Returns (nx, ny) array of energy densities (energy per unit surface area) for the gamma surface, in eV/A**2
        '''

        cell = self.images[0].cell[:, :]
        self.surface_area = np.linalg.norm(np.cross(cell[0, :], cell[1, :]))
        self.surface_separation = np.abs(cell[2, 2])


        Es = np.zeros((self.nx, self.ny))

        if relax:
            self.relax_images(calculator, **relax_kwargs)
        
        idx = 0

        for i in range(self.nx):
            for j in range(self.ny):
                image = self.images[idx]
                image.calc = calculator
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
