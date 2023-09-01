import numpy as np
from ase.build import rotate, cut
from matscipy.dislocation import FixedLineAtoms
from ase.optimize import BFGSLineSearch
import warnings
import matplotlib.pyplot as plt
from ase.constraints import UnitCellFilter


class GammaSurface():
    '''
    A class for generating gamma surface/generalised stacking fault images & plots
    '''
    def __init__(self, ats, surface_direction, y_dir=None, compress=True):
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
        compress: bool
            Search for "compressed" representation of surface where number of atoms
            is minimised

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
            at = cut(ats.copy(), a=surface_direction, b=y_dir)
            rotate(at, [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0])
            rotate(at, at.cell[0, :], [1, 0, 0], at.cell[1, :], [0, 1, 0])
            return at

        self._base_ats = ats.copy()

        if y_dir is None:
            _y_dir = self._y_dir_search(surface_direction)
        else:
            _y_dir = y_dir

            if np.abs(np.dot(surface_direction, _y_dir)) >= 1E-3:
                # Vector basis is not orthogonal
                msg = f"y_dir vector {_y_dir} is not orthogonal to surface_direction vector {surface_direction}; dot(surface_direction, y_dir) = {float(np.dot(surface_direction, _y_dir))}\n" + \
                    "Gamma Surface plot may not show the correct directions"
                warnings.warn(msg, RuntimeWarning)

        _x_dir = np.cross(_y_dir, surface_direction)

        if not compress:
            _cell_y_dir = _y_dir
            _cell_x_dir = _x_dir

            self.mapping = {
                "x" : np.array([1, 0, 0]),
                "y" : np.array([0, 1, 0])
            }
            self.cut_at = _cut_and_rotate(self._base_ats, surface_direction, _y_dir)
        
        else:
            # Find linear combinations of _x_dir and _y_dir
            d4_vecs = self._find_new_axes(_y_dir, _x_dir)

            # Test each comb to find smallest representation
            at_len = np.inf
            for vec in d4_vecs:
                test_cut = _cut_and_rotate(self._base_ats, surface_direction, vec)
                if len(test_cut) < at_len:
                    at_len = len(test_cut)
                    cand_ats = test_cut.copy()
                    _cell_y_dir = vec

            _cell_x_dir = np.cross(_cell_y_dir, surface_direction)

            if y_dir is None:  # No user specified y_dir, use most compressed direction instead
                _y_dir = _cell_y_dir
                _x_dir = _cell_x_dir

            # Define mapping between cell coord system andngamma surface coord system
            self.mapping = {
                "x": np.linalg.solve(np.array([_cell_x_dir, _cell_y_dir, surface_direction]).T, _x_dir),
                "y": np.linalg.solve(np.array([_cell_x_dir, _cell_y_dir, surface_direction]).T, _y_dir)
            }
            self.cut_at = cand_ats
        
        self.surf_directions = {
            "x": _x_dir,
            "y": _y_dir,
            "z": surface_direction
            }
        self.cell_directions = {
            "x": _cell_x_dir,
            "y": _cell_y_dir,
            "z": surface_direction
        }

    def _y_dir_search(self, d1):
        # Search for integer x, y, z components for vectors perpendicular to d1
        nmax = 5
        tol = 1E-6
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

    def _find_new_axes(self, d2, d3, depth=3):
        '''
        Search for new orthogonal vectors d4, d5 via:

        d4 = a_1 * d2 + a_2 * d3
        d5 = cross(d4, d1)

        depth controls the maximum allowed magnitudes of a_1, a_2

        '''

        vecs = []

        for i in range(-depth, depth):
            if i==0:
                continue
            for j in range(-depth, depth):
                d4 = i * d2 + j * d3
                gcd = np.gcd.reduce(np.abs(d4))
                d4 = d4 / gcd  # Collapse euivalence of EG [1, 1, 1], [2, 2, 2] as basis vectors
                vecs.append(d4)

        vec = np.unique(vecs)
        return vec

    def generate_images(self, nx, ny, z_replications=1, vert_strain=0.0):
        '''
        Generate gamma surface images on an (nx, ny) grid

        nx: int
            Number of points in the x direction
        ny: int
            Number of points in the y direction
        z_replications: int
            Number of supercell copies in z (increases separation between periodic surfaces)
        vert_strain: float
            Percentage strain to apply in z direction (0.1 = +10% strain)
        '''
        self.nx = nx
        self.ny = ny

        base_struct = self.cut_at * (1, 1, z_replications)

        cell = base_struct.cell[:, :]

        self.x_disp = self.mapping["x"] @ cell[0, :] * self.mapping["x"]
        self.y_disp = self.mapping["y"] @ cell[1, :] * self.mapping["y"]

        self.surface_area = np.linalg.norm(np.cross(self.x_disp, self.y_disp))
        self.surface_separation = np.abs(cell[2, 2])
                
        dx = self.x_disp / nx
        dy = self.y_disp / ny

        images = []

        for i in range(nx):
            for j in range(ny):
                offset = i * dx + j * dy
            
                new_cell = cell.copy()
                new_cell[2, 2] *= (1.0 + vert_strain)
                new_cell[2, :] += offset

                ats = base_struct.copy()

                ats.set_cell(new_cell, scale_atoms=False)

                images.append(ats)
        self.images = images

    def relax_images(self, calculator, ftol=1E-3, optimiser=BFGSLineSearch, constrain_atoms=True,
                     cell_relax=True, **kwargs):
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
            opt = optimiser(cell_filter)
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

    def plot_gamma_surface(self, ax=None):
        '''
        Produce a matplotlib plot of the gamma surface energy, from the data gathered in self.generate_images and self.get_surface_energioes

        Returns a matplotlib fig and ax object
        '''

        if ax is None:

            fig, my_ax = plt.subplots()

        else:
            my_ax = ax
            fig = my_ax.get_figure()

        im = my_ax.imshow(self.Es.T, origin="lower", extent=[0, np.linalg.norm(self.x_disp), 0, np.linalg.norm(self.y_disp)], interpolation="bicubic")
        fig.colorbar(im, ax=ax, label="Energy Density (eV/A**2)")

        my_ax.set_xlabel("(" + " ".join(self.surf_directions["x"].astype(str)) + ")")
        my_ax.set_ylabel("(" + " ".join(self.surf_directions["y"].astype(str)) + ")")

        my_ax.set_title("(" + " ".join(self.surf_directions["z"].astype(str)) + ") Gamma Surface\nSurface Separation = {:0.2f} A".format(self.surface_separation))

        return fig, my_ax

class StackingFault(GammaSurface):
    '''
    Class for stacking fault-specific generation & plotting
    '''

    def generate_images(self, n, z_replications=1, vert_strain=0):
        '''
        Generate gamma surface images on an (n) line
        n: int
            Number of images
        z_replications: int
            Number of supercell copies in z (increases separation between periodic surfaces)
        vert_strain: float
            Percentage strain to apply in z direction (0.1 = +10% strain)
        '''
        return super().generate_images(1, n, z_replications, vert_strain)
    
    def plot_gamma_surface(self, ax=None):
        '''
        Produce a matplotlib plot of the stacking fault energy, from the data gathered in self.generate_images and self.get_surface_energy

        Returns a matplotlib fig and ax object
        '''
        if ax is None:

            fig, my_ax = plt.subplots()

        else:
            my_ax = ax
            fig = my_ax.get_figure()

        my_ax.plot(np.arange(self.ny)/self.ny, self.Es[0, :])
        my_ax.set_xlabel("Position along path")
        my_ax.set_ylabel("Energy Density (eV/A**2)")
        title = f"({' '.join(self.surf_directions["z"].astype(str))}) Stacking Fault\n" + "Surface Separation = {:0.2f} A".format(self..surface_separation)
        my_ax.set_title(title)
        return fig, my_ax
