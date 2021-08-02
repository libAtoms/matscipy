#
# Copyright 2019, 2021 Lars Pastewka (U. Freiburg)
#           2018-2021 Petr Grigorev (Warwick U.)
#           2020 James Kermode (Warwick U.)
#           2019 Arnaud Allera (U. Lyon 1)
#           2019 Wolfram G. Nöhring (U. Freiburg)
#
# matscipy - Materials science with Python at the atomic-scale
# https://github.com/libAtoms/matscipy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import numpy as np

from scipy.optimize import minimize

from ase.lattice.cubic import BodyCenteredCubic
from ase.constraints import FixAtoms, StrainFilter
from ase.optimize import FIRE
from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from ase.units import GPa  # unit conversion
from ase.lattice.cubic import SimpleCubicFactory, Diamond
from ase.io import read
from ase.geometry import get_distances

from matscipy.neighbours import neighbour_list, mic
from matscipy.elasticity import fit_elastic_constants


def make_screw_cyl(alat, C11, C12, C44,
                   cylinder_r=10, cutoff=5.5,
                   hard_core=False,
                   center=[0., 0., 0.],
                   l_extend=[0., 0., 0.],
                   symbol='W'):

    """Makes screw dislocation using atomman library

    Parameters
    ----------
    alat : float
        Lattice constant of the material.
    C11 : float
        C11 elastic constant of the material.
    C12 : float
        C12 elastic constant of the material.
    C44 : float
        C44 elastic constant of the material.
    cylinder_r : float
        radius of cylinder of unconstrained atoms around the
        dislocation  in angstrom
    cutoff : float
        Potential cutoff for marinica potentials for FS cutoff = 4.4
    hard_core : bool
        Description of parameter `hard_core`.
    center : type
        The position of the dislocation core and the center of the
                 cylinder with FixAtoms condition
    l_extend : float
        extension of the box. used for creation of initial
        dislocation position with box equivalent to the final position
    symbol : string
        Symbol of the element to pass to ase.lattice.cubic.SimpleCubicFactory
        default is "W" for tungsten

    Returns
    -------
    disloc : ase.Atoms object
        screw dislocation cylinder.
    bulk : ase.Atoms object
        bulk disk used to generate dislocation
    u : np.array
        displacement per atom.
    """
    from atomman import ElasticConstants
    from atomman.defect import Stroh

    # Create a Stroh object with junk data
    stroh = Stroh(ElasticConstants(C11=141, C12=110, C44=98),
                  np.array([0, 0, 1]))

    axes = np.array([[1, 1, -2],
                     [-1, 1, 0],
                     [1, 1, 1]])

    c = ElasticConstants(C11=C11, C12=C12, C44=C44)
    burgers = alat * np.array([1., 1., 1.])/2.

    # Solving a new problem with Stroh.solve
    stroh.solve(c, burgers, axes=axes)

    # test the solution that it does not crash
    # pos_test = uc.set_in_units(np.array([12.4, 13.5, -10.6]), 'angstrom')
    # disp = stroh.displacement(pos_test)
    # print("displacement =", uc.get_in_units(disp, 'angstrom'), 'angstrom')

    unit_cell = BodyCenteredCubic(directions=axes.tolist(),
                                  size=(1, 1, 1), symbol=symbol,
                                  pbc=(False, False, True),
                                  latticeconstant=alat)

    # make the dislocation core center of the box
    disloCenterX = alat * np.sqrt(6.)/6.0
    disloCenterY = alat * np.sqrt(2.)/6.0

    unit_cell.positions[:, 0] -= disloCenterX
    unit_cell.positions[:, 1] -= disloCenterY

    # shift to move the fixed atoms boundary condition for the
    # configuration with shifted dislocation core
    shift_x = 2.0 * center[0]
    shift_y = 2.0 * center[1]

    l_shift_x = 2.0 * l_extend[0]
    l_shift_y = 2.0 * l_extend[1]

    # size of the cubic cell as a 112 direction
    Lx = int(round((cylinder_r + 3.*cutoff + shift_x + l_shift_x)
                   / (alat * np.sqrt(6.))))

    # size of the cubic cell as a 110 direction
    Ly = int(round((cylinder_r + 3.*cutoff + shift_y + l_shift_y)
                   / (alat * np.sqrt(2.))))
    # factor 2 to make sure odd number of images is translated
    # it is important for the correct centering of the dislocation core
    bulk = unit_cell * (2*Lx, 2*Ly, 1)
    # make 0, 0, at the center
    bulk.positions[:, 0] -= Lx * alat * np.sqrt(6.)
    bulk.positions[:, 1] -= Ly * alat * np.sqrt(2.)

    # wrap
    # bulk.set_scaled_positions(bulk.get_scaled_positions())
    # apply shear here:
    # bulk.cell *= D
    # bulk.positions *= D

    x, y, z = bulk.positions.T

    radius_x_y_zero = np.sqrt(x**2 + y**2)
    mask_zero = radius_x_y_zero < cylinder_r + 2.*cutoff

    radius_x_y_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask_center = radius_x_y_center < cylinder_r + 2.*cutoff

    radius_x_y_l_shift = np.sqrt((x - l_extend[0])**2 + (y - l_extend[1])**2)
    mask_l_shift = radius_x_y_l_shift < cylinder_r + 2.*cutoff

    final_mask = mask_center | mask_zero | mask_l_shift
    # leave only atoms inside the cylinder
    bulk = bulk[final_mask]

    disloc = bulk.copy()
    # calculate and apply the displacements for atomic positions
    u = stroh.displacement(bulk.positions - center)
    u = -u if hard_core else u

    disloc.positions += u
    x, y, z = disloc.positions.T

    radius_x_y_zero = np.sqrt(x**2 + y**2)
    mask_zero = radius_x_y_zero > cylinder_r

    radius_x_y_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask_center = radius_x_y_center > cylinder_r

    radius_x_y_l_shift = np.sqrt((x - l_extend[0])**2 + (y - l_extend[1])**2)
    mask_l_shift = radius_x_y_l_shift > cylinder_r

    fix_mask = mask_center & mask_zero & mask_l_shift
    # leave only atoms inside the cylinder
    fix_atoms = FixAtoms(mask=fix_mask)
    disloc.set_constraint(fix_atoms)

    # make an "region" array to map bulk and fixed atoms
    # all atoms are "MM" by default
    region = np.full_like(disloc, "MM")
    region[fix_mask] = np.full_like(disloc[fix_mask], "fixed")
    disloc.new_array("region", region)

    # center the atoms to avoid "lost atoms" error by lammps
    center_shift = np.diagonal(bulk.cell).copy()
    center_shift[2] = 0.0  # do not shift along z direction

    disloc.positions += center_shift / 2.0
    bulk.positions += center_shift / 2.0

    return disloc, bulk, u


def make_edge_cyl(alat, C11, C12, C44,
                  cylinder_r=10, cutoff=5.5,
                  symbol='W'):
    '''
    makes edge dislocation using atomman library

    cylinder_r - radius of cylinder of unconstrained atoms around the
                 dislocation  in angstrom

    cutoff - potential cutoff for marinica potentials for FS cutoff = 4.4

    symbol : string
        Symbol of the element to pass to ase.lattuce.cubic.SimpleCubicFactory
        default is "W" for tungsten
    '''
    from atomman import ElasticConstants
    from atomman.defect import Stroh
    # Create a Stroh ojbect with junk data
    stroh = Stroh(ElasticConstants(C11=141, C12=110, C44=98),
                                   np.array([0, 0, 1]))

    axes = np.array([[1, 1, 1],
                     [1, -1, 0],
                     [1, 1, -2]])

    c = ElasticConstants(C11=C11, C12=C12, C44=C44)
    burgers = alat * np.array([1., 1., 1.])/2.

    # Solving a new problem with Stroh.solve
    # Does not work with the new version of atomman
    stroh.solve(c, burgers, axes=axes)

    unit_cell = BodyCenteredCubic(directions=axes.tolist(),
                                  size=(1, 1, 1), symbol='W',
                                  pbc=(False, False, True),
                                  latticeconstant=alat)

    bulk = unit_cell.copy()

    # shift to make the zeros of the cell betweem the atomic planes
    # and under the midplane on Y axes
    X_midplane_shift = (1.0/3.0)*alat*np.sqrt(3.0)/2.0
    Y_midplane_shift = 0.25*alat*np.sqrt(2.0)

    bulk_shift = [X_midplane_shift,
                  Y_midplane_shift,
                  0.0]

    bulk.positions += bulk_shift

    tot_r = cylinder_r + cutoff + 0.01

    Lx = int(round(tot_r/(alat*np.sqrt(3.0)/2.0)))
    Ly = int(round(tot_r/(alat*np.sqrt(2.))))

    # factor 2 to make shure odd number of images is translated
    # it is important for the correct centering of the dislocation core
    bulk = bulk * (2*Lx, 2*Ly, 1)

    center_shift = [Lx * alat * np.sqrt(3.0)/2.,
                    Ly * alat * np.sqrt(2.),
                    0.0]

    bulk.positions -= center_shift

    ED = bulk.copy()

    disp = stroh.displacement(ED.positions)

    ED.positions += disp

    x, y, z = ED.positions.T
    radius_x_y_zero = np.sqrt(x**2 + y**2)
    mask = radius_x_y_zero < tot_r

    ED = ED[mask]
    bulk = bulk[mask]

    bulk.write("before.xyz")

    ED.write("after_disp.xyz")

    x, y, z = ED.positions.T
    radius_x_y_zero = np.sqrt(x**2 + y**2)
    mask_zero = radius_x_y_zero > cylinder_r
    fix_atoms = FixAtoms(mask=mask_zero)

    ED.set_constraint(fix_atoms)

    x, y, z = bulk.positions.T
    # move lower left segment
    bulk.positions[(y < 0.0) & (x < X_midplane_shift)] -= \
        [alat * np.sqrt(3.0) / 2.0, 0.0, 0.0]
    # make the doslocation extraplane center
    bulk.positions += [(1.0/3.0)*alat*np.sqrt(3.0)/2.0, 0.0, 0.0]

    return ED, bulk


def plot_vitek(dislo, bulk,
               alat=3.16, plot_axes=None, xyscale=10):
    """Plots vitek map from ase configurations.

    Parameters
    ----------
    dislo : ase.Atoms
        Dislocation configuration.
    bulk : ase.Atoms
        Corresponding bulk configuration for calculation of displacements.
    alat : float
        Lattice parameter for calculation of neghbour list cutoff.
    plot_axes : matplotlib.Axes.axes object
        Existing axes to plot on, allows to pass existing matplotlib axes
        have full control of the graph outside the function.
        Makes possible to plot multiple differential displacement
        maps using subplots.
        Default is None, then new graph is created by plt.subplots()
        Description of parameter `plot_axes`.
    xyscale : float
        xyscale of the graph

    Returns
    -------
    None

    """
    from atomman import load
    from atomman.defect import differential_displacement

    lengthB = 0.5*np.sqrt(3.)*alat
    burgers = np.array([0.0, 0.0, lengthB])

    base_system = load("ase_Atoms", bulk)
    disl_system = load("ase_Atoms", dislo)

    neighborListCutoff = 0.95 * alat

    # plot window is +-10 angstroms from center in x,y directions, and one Burgers vector
    # thickness along z direction
    x, y, _ = bulk.positions.T

    plot_range = np.array([[x.mean() - xyscale, x.mean() + xyscale],
                          [y.mean() - xyscale, y.mean() + xyscale],
                          [-0.1, alat * 3.**(0.5) / 2.]])

    # This scales arrows such that b/2 corresponds to the
    # distance between atoms on the plot
    plot_scale = 1.885618083

    fig = differential_displacement(base_system, disl_system,
                                    burgers,
                                    cutoff=neighborListCutoff,
                                    xlim=plot_range[0],
                                    ylim=plot_range[1],
                                    zlim=plot_range[2],
                                    matplotlib_axes=plot_axes,
                                    plot_scale=plot_scale)

def show_NEB_configurations(images, bulk, xyscale=7,
                            show=True, core_positions=None):
    """Plots Vitek differential displacement maps for the list of images
       for example along the NEB path.

    Parameters
    ----------
    images : list of ase.Atoms
        List of configurations with dislocations.
    bulk : ase.Atoms
        Corresponding bulk configuration for calculation of displacements.
    xyscale : float
        xyscale of the graph
    show : bool
        Show the figure after plotting. Default is True.
    core_positions : list
        [x, y] position of dislocation core to plot

    Returns
    -------
    figure
        If the show is False else returns None

    """
    import matplotlib.pyplot as plt

    n_images = len(images)
    fig2 = plt.figure(figsize=(n_images * 4, 4))

    for i, image in enumerate(images):
        ax1 = fig2.add_subplot(1, n_images, i + 1)
        plot_vitek(image, bulk, plot_axes=ax1, xyscale=xyscale)
        if core_positions is not None:
            x, y = core_positions[i]
            ax1.scatter(x, y, marker="+", s=200, c='C1')
    if show:
        fig2.show()
        return None
    else:
        return fig2


def show_configuration(disloc, bulk, u, fixed_mask=None):
    """shows the displacement fixed atoms."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 4))

    ax1 = fig.add_subplot(131)
    ax1.set_title(r"z displacement, $\AA$")
    sc = ax1.scatter(bulk.positions[:, 0], bulk.positions[:, 1], c=u.T[2])
    ax1.axvline(0.0, color="red", linestyle="dashed")
    ax1.set_xlabel(r"x, $\AA$")
    ax1.set_ylabel(r"y, $\AA$")
    plt.colorbar(sc)

    ax2 = fig.add_subplot(132)
    ax2.set_title(r"x displacement, $\AA$")
    sc = ax2.scatter(bulk.positions[:, 0], bulk.positions[:, 1], c=u.T[0])
    ax2.set_xlabel(r"x, $\AA$")
    ax2.set_ylabel(r"y, $\AA$")
    plt.colorbar(sc, format="%.1e")

    ax3 = fig.add_subplot(133)
    ax3.set_title(r"y displacement, $\AA$")
    sc = ax3.scatter(bulk.positions[:, 0], bulk.positions[:, 1], c=u.T[1])
    plt.colorbar(sc, format="%.1e")
    ax3.set_xlabel(r"x, $\AA$")
    ax3.set_ylabel(r"y, $\AA$")

    if fixed_mask is not None:

        ax1.scatter(bulk.positions[fixed_mask, 0],
                    bulk.positions[fixed_mask, 1], c="k")

        ax2.scatter(bulk.positions[fixed_mask, 0],
                    bulk.positions[fixed_mask, 1], c="k")

        ax3.scatter(bulk.positions[fixed_mask, 0],
                    bulk.positions[fixed_mask, 1], c="k")

    plt.show()

    return None


def get_elastic_constants(pot_path=None,
                          calculator=None,
                          delta=1e-2,
                          symbol="W"):
    """
    return lattice parameter, and cubic elastic constants: C11, C12, 44
    using matscipy function
    pot_path - path to the potential

    symbol : string
        Symbol of the element to pass to ase.lattuce.cubic.SimpleCubicFactory
        default is "W" for tungsten
    """

    unit_cell = bulk(symbol)

    if (pot_path is not None) and (calculator is None):
        # create lammps calculator with the potential
        lammps = LAMMPSlib(lmpcmds=["pair_style eam/fs",
                           "pair_coeff * * %s W" % pot_path],
                           atom_types={'W': 1}, keep_alive=True)
        calculator = lammps

    unit_cell.calc = calculator

#   simple calculation to get the lattice constant and cohesive energy
#    alat0 = W.cell[0][1] - W.cell[0][0]
    sf = StrainFilter(unit_cell)  # or UnitCellFilter(W) -> to minimise wrt pos, cell
    opt = FIRE(sf)
    opt.run(fmax=1e-4)  # max force in eV/A
    alat = unit_cell.cell[0][1] - unit_cell.cell[0][0]
#    print("a0 relaxation %.4f --> %.4f" % (a0, a))
#    e_coh = W.get_potential_energy()
#    print("Cohesive energy %.4f" % e_coh)

    Cij, Cij_err = fit_elastic_constants(unit_cell,
                                         symmetry="cubic",
                                         delta=delta)

    Cij = Cij/GPa  # unit conversion to GPa

    elasticMatrix3x3 = Cij[:3, :3]
    # average of diagonal elements: C11, C22, C33
    C11 = elasticMatrix3x3.diagonal().mean()
    # make mask to extract non diagonal elements
    mask = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(mask, False)

    # average of all non diagonal elements from 1 to 3
    C12 = elasticMatrix3x3[mask].mean()

    # average of diagonal elements from 4 till 6: C44, C55, C66,
    C44 = Cij[3:, 3:].diagonal().mean()

    # A = 2.*C44/(C11 - C12)

    if (pot_path is not None) and (calculator is None):
        lammps.lmp.close()

    return alat, C11, C12, C44


def make_barrier_configurations(elastic_param=None,
                                pot_path=None, calculator=None,
                                cylinder_r=10, hard_core=False, **kwargs):
    """Creates the initial and final configurations for the NEB calculation
        The positions in FixedAtoms constrained region are average between
        final and initial configurations

    Parameters
    ----------
    pot_path : string
        Path to the potential file.
    calculator : type
        Description of parameter `calculator`.
    cylinder_r : float
        Radius of cylinder of unconstrained atoms around the
                    dislocation  in angstrom.
    hard_core : bool
        Type of the core hard or soft.
        If hard is chosen the displacement field is reversed.
    **kwargs :
        Keyword arguments to pass to make_screw_cyl() function.

    Returns
    -------
    disloc_ini : ase.Atoms
        Initial dislocation configuration.
    disloc_fin : ase.Atoms
        Final dislocation configuration.
    bulk : ase.Atoms
        Perfect bulk configuration for Vitek displacement maps

    """

    if pot_path is not None:
        alat, C11, C12, C44 = get_elastic_constants(pot_path=pot_path)
        # get the cutoff from the potential file
        with open(pot_path) as potfile:
            for i, tmp_str in enumerate(potfile):
                if i == 4:  # read the last number in the fifth line
                    cutoff = float(tmp_str.split()[-1])
                    break

    elif calculator is not None:
        alat, C11, C12, C44 = get_elastic_constants(calculator=calculator)
        cutoff = 5.0  # the value for training data for GAP from paper

    elif elastic_param is not None:
        alat, C11, C12, C44 = elastic_param
        cutoff = 5.5

    cent_x = np.sqrt(6.0) * alat / 3.0
    center = [cent_x, 0.0, 0.0]

    disloc_ini, bulk_ini, __ = make_screw_cyl(alat, C11, C12, C44,
                                              cylinder_r=cylinder_r,
                                              cutoff=cutoff,
                                              hard_core=hard_core,
                                              l_extend=center, **kwargs)

    disloc_fin, __, __ = make_screw_cyl(alat, C11, C12, C44,
                                        cylinder_r=cylinder_r,
                                        cutoff=cutoff,
                                        hard_core=hard_core,
                                        center=center, **kwargs)

    # get the fixed atoms constrain
    FixAtoms = disloc_ini.constraints[0]
    # get the indices of fixed atoms
    fixed_atoms_indices = FixAtoms.get_indices()

    # make the average position of fixed atoms
    # between initial and the last position
    ini_fix_pos = disloc_ini.get_positions()[fixed_atoms_indices]
    fin_fix_pos = disloc_fin.get_positions()[fixed_atoms_indices]

    new_av_pos = (ini_fix_pos + fin_fix_pos)/2.0

    positions = disloc_ini.get_positions()
    positions[fixed_atoms_indices] = new_av_pos
    disloc_ini.set_positions(positions, apply_constraint=False)

    positions = disloc_fin.get_positions()
    positions[fixed_atoms_indices] = new_av_pos
    disloc_fin.set_positions(positions, apply_constraint=False)

    return disloc_ini, disloc_fin, bulk_ini

def make_screw_cyl_kink(alat, C11, C12, C44,
                        cylinder_r=40, kink_length=26, kind="double", **kwargs):
    """Function to create kink configuration based on make_screw_cyl() function.
        Double kink configuration is in agreement with quadrupoles in terms of formation energy.
        Single kink configurations provide correct and stable structure, but formation energy is not accessible?

    Parameters
    ----------
    alat : float
        Lattice constant of the material.
    C11 : float
        C11 elastic constant of the material.
    C12 : float
        C12 elastic constant of the material.
    C44 : float
        C44 elastic constant of the material.
    cylinder_r : float
        radius of cylinder of unconstrained atoms around the
        dislocation  in angstrom
    kink_length : int
        Length of the cell per kink along b in unit of b, must be even.
    kind : string
        kind of the kink: right, left or double
    **kwargs :
        Keyword arguments to pass to make_screw_cyl() function.

    Returns
    -------
    kink : ase.atoms
        kink configuration
    reference_straight_disloc : ase.atoms
        reference straight dislocation configuration
    large_bulk : ase.atoms
        large bulk cell corresponding to the kink configuration
    """
    b = np.sqrt(3.0) * alat / 2.0
    cent_x = np.sqrt(6.0) * alat / 3.0

    disloc_ini, disloc_fin, bulk_ini = make_barrier_configurations((alat, C11, C12, C44),
                                                                    cylinder_r=cylinder_r, **kwargs)

    if kind == "double":

        large_bulk = bulk_ini * [1, 1, 2 * kink_length]
        reference_straight_disloc = disloc_ini * [1, 1, 2 * kink_length]

        if kink_length % 2:
            print("WARNING: length is not even!")

        kink = disloc_ini * [1, 1, kink_length // 2]
        middle_kink = disloc_fin * [1, 1, kink_length]

        middle_kink.positions += np.array((0.0, 0.0, kink.get_cell()[2][2]))

        kink.constraints[0].index = np.append(kink.constraints[0].index,
                                              middle_kink.constraints[0].get_indices() + len(kink))
        kink.extend(middle_kink)
        kink.cell[2][2] += middle_kink.cell[2][2]

        upper_kink = disloc_ini * [1, 1, kink_length // 2]
        upper_kink.positions += np.array((0.0, 0.0, kink.get_cell()[2][2]))

        kink.constraints[0].index = np.append(kink.constraints[0].index,
                                              upper_kink.constraints[0].get_indices() + len(kink))
        kink.extend(upper_kink)
        kink.cell[2][2] += upper_kink.cell[2][2]

    elif kind == "right":

        large_bulk = bulk_ini * [1, 1, kink_length]
        reference_straight_disloc = disloc_ini * [1, 1, kink_length]

        kink = disloc_ini * [1, 1, kink_length // 2]
        upper_disloc = disloc_fin * [1, 1, kink_length // 2]

        upper_disloc.positions += np.array((0.0, 0.0, kink.cell[2][2]))

        kink.extend(upper_disloc)
        kink.constraints[0].index = np.append(kink.constraints[0].index,
                                              upper_disloc.constraints[0].get_indices() + len(kink))
        kink.cell[2][2] += upper_disloc.cell[2][2]

        # we have to adjust the cell to make the kink vector periodic
        # here we remove two atomic rows. it is nicely explained in the paper
        _, _, z = large_bulk.positions.T
        right_kink_mask = z < large_bulk.cell[2][2] - 2.0 * b / 3 - 0.01

        kink = kink[right_kink_mask]

        cell = kink.cell.copy()

        # right kink is created when the kink vector is in positive x direction
        # assuming (x, y, z) is right group of vectors
        cell[2][0] += cent_x
        cell[2][2] -= 2.0 * b / 3.0
        kink.set_cell(cell, scale_atoms=False)

        # make sure all the atoms are removed and cell is modified for the bulk as well.
        large_bulk.cell[2][0] += cent_x
        large_bulk.cell[2][2] -= 2.0 * b / 3.0
        large_bulk = large_bulk[right_kink_mask]
        for constraint in kink.constraints:
            large_bulk.set_constraint(constraint)

    elif kind == "left":

        large_bulk = bulk_ini * [1, 1, kink_length]
        reference_straight_disloc = disloc_ini * [1, 1, kink_length]

        kink = disloc_fin * [1, 1, kink_length // 2]

        upper_disloc = disloc_ini * [1, 1, kink_length // 2]
        upper_disloc.positions += np.array((0.0, 0.0, kink.cell[2][2]))

        kink.extend(upper_disloc)
        kink.constraints[0].index = np.append(kink.constraints[0].index,
                                              upper_disloc.constraints[0].get_indices() + len(kink))
        kink.cell[2][2] += upper_disloc.cell[2][2]

        # we have to adjust the cell to make the kink vector periodic
        # here we remove one atomic row. it is nicely explained in the paper
        _, _, z = large_bulk.positions.T
        left_kink_mask = z < large_bulk.cell[2][2] - 1.0 * b / 3 - 0.01

        kink = kink[left_kink_mask]

        cell = kink.cell.copy()

        # left kink is created when the kink vector is in negative x direction
        # assuming (x, y, z) is right group of vectors
        cell[2][0] -= cent_x
        cell[2][2] -= 1.0 * b / 3.0
        kink.set_cell(cell, scale_atoms=False)

        # make sure all the atoms are removed and cell is modified for the bulk as well.
        large_bulk.cell[2][0] -= cent_x
        large_bulk.cell[2][2] -= 1.0 * b / 3.0
        large_bulk = large_bulk[left_kink_mask]
        for constraint in kink.constraints:
            large_bulk.set_constraint(constraint)

    else:
        raise ValueError('Kind must be "right", "left" or "double"')

    return kink, reference_straight_disloc, large_bulk

def slice_long_dislo(kink, kink_bulk, b):
    """Function to slice a long dislocation configuration to perform dislocation structure and core position analysis

    Parameters
    ----------
    kink : ase.Atoms
        kink configuration to slice
    kink_bulk : ase.Atoms
        corresponding bulk configuration to perform mapping for slicing
    b : float
        burgers vector b should be along z direction


    Returns
    -------
    sliced_kink : list of [sliced_bulk, sliced_kink]
        sliced configurations 1 b length each
    disloc_z_positions : float
        positions of each sliced configuration (center along z)
    """

    if not len(kink) == len(kink_bulk):
        raise ValueError('"kink" and "kink_bulk" must be same size')

    n_slices = int(np.round(kink.cell[2][2] / b * 3))
    atom_z_positions = kink_bulk.positions.T[2]

    kink_z_length = kink_bulk.cell[2][2]

    sliced_kink = []
    disloc_z_positions = []

    for slice_id in range(n_slices):

        shift = slice_id * b / 3.0

        upper_bound = 5.0 * b / 6.0 + shift
        lower_bound = -b / 6.0 + shift

        if upper_bound < kink_z_length:

            mask = np.logical_and(atom_z_positions < upper_bound,
                                  atom_z_positions > lower_bound)

            bulk_slice = kink_bulk.copy()[mask]
            kink_slice = kink.copy()[mask]

        else:  # take into account PBC at the end of the box

            upper_mask = atom_z_positions < (upper_bound - kink_z_length)

            mask = np.logical_or(upper_mask,
                                 atom_z_positions > lower_bound)

            bulk_slice = kink_bulk.copy()[mask]
            kink_slice = kink.copy()[mask]

            # move the bottom atoms on top of the box
            kink_slice.positions[upper_mask[mask]] += np.array(kink_bulk.cell[2])

            bulk_slice.positions[upper_mask[mask]] += np.array((kink_bulk.cell[2]))

        # print(kink_bulk[mask].positions.T[2].max())
        # print(kink_bulk[mask].positions.T[2].min())

        bulk_slice.positions -= np.array((0.0, 0.0, shift))
        kink_slice.positions -= np.array((0.0, 0.0, shift))

        bulk_slice.cell = kink_bulk.cell
        bulk_slice.cell[2][2] = b
        bulk_slice.cell[2][0] = 0

        kink_slice.cell = kink_bulk.cell
        kink_slice.cell[2][2] = b
        kink_slice.cell[2][0] = 0

        sliced_kink.append([bulk_slice, kink_slice])
        disloc_z_positions.append(b / 3.0 + shift)

    disloc_z_positions = np.array(disloc_z_positions)

    return sliced_kink, disloc_z_positions


def compare_configurations(dislo, bulk, dislo_ref, bulk_ref,
                           alat, cylinder_r=None, print_info=True, remap=True,
                           bulk_neighbours=None, origin=(0., 0.)):
    """Compares two dislocation configurations based on the gradient of
       the displacements along the bonds.

    Parameters
    ----------
    dislo : ase.Atoms
        Dislocation configuration.
    bulk : ase.Atoms
        Corresponding bulk configuration for calculation of displacements.
    dislo_ref : ase.Atoms
        Reference dislocation configuration.
    bulk_ref : ase.Atoms
        Corresponding reference bulk configuration
        for calculation of displacements.
    alat : float
        Lattice parameter for calculation of neghbour list cutoff.
    cylinder_r : float or None
        Radius of region of comparison around the dislocation coreself.
        If None makes global comparison based on the radius of
        `dislo` configuration, else compares the regions with `cylinder_r`
        around the dislocation core position.
    print_info : bool
        Flag to switch print statement about the type of the comparison
    remap: bool
        Flag to swtich off remapping of atoms between deformed and reference
        configurations. Only set this to true if atom order is the same!
    bulk_neighbours:
        Optionally pass in bulk neighbours as a tuple (bulk_i, bulk_j)
    origin: tuple
        Optionally pass in coordinate origin (x0, y0)

    Returns
    -------
    float
        The Du norm of the differences per atom.

    """

    x0, y0 = origin
    x, y, __ = bulk.get_positions().T
    x -= x0
    y -= y0
    radius = np.sqrt(x ** 2 + y ** 2)

    if cylinder_r is None:
        cutoff_radius = radius.max() - 10.

        if print_info:
            print("Making a global comparison with radius %.2f" % cutoff_radius)

    else:
        cutoff_radius = cylinder_r
        if print_info:
            print("Making a local comparison with radius %.2f" % cutoff_radius)

    cutoff_mask = radius < cutoff_radius
    second_NN_distance = alat
    if bulk_neighbours is None:
        bulk_i, bulk_j = neighbour_list('ij', bulk_ref, second_NN_distance)
    else:
        bulk_i, bulk_j = bulk_neighbours

    I_core, J_core = np.array([(i, j) for i, j in zip(bulk_i, bulk_j) if cutoff_mask[i]]).T

    if remap:
        mapping = {}
        for i in range(len(bulk)):
            mapping[i] = np.linalg.norm(bulk_ref.positions -
                                        bulk.positions[i], axis=1).argmin()
    else:
        mapping = dict(zip(list(range(len(bulk))),
                           list(range(len(bulk)))))

    u_ref = dislo_ref.positions - bulk_ref.positions

    u = dislo.positions - bulk.positions
    u_extended = np.zeros(u_ref.shape)
    u_extended[list(mapping.values()), :] = u

    du = u_extended - u_ref

    Du = np.linalg.norm(np.linalg.norm(mic(du[J_core, :] - du[I_core, :],
                                           bulk.cell), axis=1))
    return Du


def cost_function(pos, dislo, bulk, cylinder_r, elastic_param,
                  hard_core=False, print_info=True, remap=True,
                  bulk_neighbours=None, origin=(0, 0)):
    """Cost function for fitting analytical displacement field
       and detecting dislocation core position. Uses `compare_configurations`
       function for the minimisation of the core position.

    Parameters
    ----------
    pos : list of float
        Positions of the core to build the analytical solution [x, y].
    dislo : ase.Atoms
        Dislocation configuration.
    bulk : ase.Atoms
        Corresponding bulk configuration for calculation of displacements.
    cylinder_r : float or None
        Radius of region of comparison around the dislocation coreself.
        If None makes global comparison based on the radius of
        `dislo` configuration, else compares the regions with `cylinder_r`
        around the dislocation core position.
    elastic_param : list of float
        List containing alat, C11, C12, C44
    hard_core : bool
        type of the core True for hard
    print_info : bool
        Flag to switch print statement about the type of the comparison
    bulk_neighbours: tuple or None
        Optionally pass in neighbour list for bulk reference config to save
        computing it each time.
    origin: tuple
        Optionally pass in coordinate origin (x0, y0)

    Returns
    -------
    float
        Error for optimisation (result from `compare_configurations` function)

    """
    from atomman import ElasticConstants
    from atomman.defect import Stroh

    # Create a Stroh ojbect with junk data
    stroh = Stroh(ElasticConstants(C11=141, C12=110, C44=98),
                  np.array([0, 0, 1]))

    axes = np.array([[1, 1, -2],
                    [-1, 1, 0],
                    [1, 1, 1]])

    alat, C11, C12, C44 = elastic_param

    c = ElasticConstants(C11=C11, C12=C12, C44=C44)
    burgers = alat * np.array([1., 1., 1.])/2.

    # Solving a new problem with Stroh.solve
    stroh.solve(c, burgers, axes=axes)

    x0, y0 = origin
    center = (pos[0], pos[1], 0.0)
    u = stroh.displacement(bulk.positions - center)
    u = -u if hard_core else u

    dislo_guess = bulk.copy()
    dislo_guess.positions += u

    err = compare_configurations(dislo, bulk,
                                 dislo_guess, bulk,
                                 alat, cylinder_r=cylinder_r,
                                 print_info=print_info, remap=remap,
                                 bulk_neighbours=bulk_neighbours,
                                 origin=origin)

    return err


def fit_core_position(dislo_image, bulk, elastic_param, hard_core=False,
                      core_radius=10, current_pos=None, bulk_neighbours=None,
                      origin=(0, 0)):
    """
    Use `cost_function()` to fit atomic positions to Stroh solution with

    `scipy.optimize.minimize` is used to perform the fit using Powell's method.

    Parameters
    ----------

    dislo_image: ase.atoms.Atoms
    bulk: ase.atoms.Atoms
    elastic_param: array-like
        [alat, C11, C12, C44]
    hard_core: bool
    core_radius: float
    current_pos: array-like
        array [core_x, core_y] containing initial guess for core position
    bulk_neighbours: tuple
        cache of bulk neigbbours to speed up calcualtion. Should be a
        tuple (bulk_I, bulk_J) as returned by
        `matscipy.neigbbours.neighbour_list('ij', bulk, alat)`.
    origin: tuple
        Optionally pass in coordinate origin (x0, y0)

    Returns
    -------

    core_pos - array [core_x, core_y]

    """
    if current_pos is None:
        current_pos = origin
    res = minimize(cost_function, current_pos, args=(
                    dislo_image, bulk, core_radius, elastic_param,
                    hard_core, False, False, bulk_neighbours, origin),
                   method='Powell', options={'xtol': 1e-2, 'ftol': 1e-2})
    return res.x


def fit_core_position_images(images, bulk, elastic_param,
                             bulk_neighbours=None,
                             origin=(0, 0)):
    """
    Call fit_core_position() for a list of Atoms objects, e.g. NEB images

    Parameters
    ----------
    images: list
        list of Atoms object for dislocation configurations
    bulk: ase.atoms.Atoms
        bulk reference configuration
    elastic_param: list
        as for `fit_core_position()`.
    bulk_neighbours:
        as for `fit_core_position()`.
    origin: tuple
        Optionally pass in coordinate origin (x0, y0)

    Returns
    -------
    core_positions: array of shape `(len(images), 2)`
    """
    core_positions = []
    core_position = images[0].info.get('core_position', origin)
    for dislo in images:
        dislo_tmp = dislo.copy()
        core_position = fit_core_position(dislo_tmp, bulk, elastic_param,
                                          current_pos=dislo.info.get(
                                              'core_position', core_position),
                                          bulk_neighbours=bulk_neighbours,
                                          origin=origin)
        dislo.info['core_position'] = core_position
        core_positions.append(core_position)

    return np.array(core_positions)


def screw_cyl_tetrahedral(alat, C11, C12, C44,
                          scan_r=15,
                          symbol="W",
                          imp_symbol='H',
                          hard_core=False,
                          center=(0., 0., 0.)):
    """Generates a set of tetrahedral positions with `scan_r` radius.
       Applies the screw dislocation displacement for creating an initial guess
       for the H positions at dislocation core.

    Parameters
    ----------
    alat : float
        Lattice constant of the material.
    C11 : float
        C11 elastic constant of the material.
    C12 : float
        C12 elastic constant of the material.
    C44 : float
        C44 elastic constant of the material.
    scan_r : float
        Radius of the region to create tetrahedral positions.
    symbol : string
        Symbol of the element to pass to ase.lattuce.cubic.SimpleCubicFactory
        default is "W" for tungsten
    imp_symbol : string
        Symbol of the elemnt to pass creat Atoms object
        default is "H" for hydrogen
    hard_core : float
        Type of the dislocatino core if True then -u
        (sign of displacement is flipped) is applied.
        Default is False i.e. soft core is created.


    center : tuple of floats
        Coordinates of dislocation core (center) (x, y, z).
        Default is (0., 0., 0.)

    Returns
    -------
    ase.Atoms object
        Atoms object with predicted tetrahedral
        positions around dislocation core.

    """
    from atomman import ElasticConstants
    from atomman.defect import Stroh

    axes = np.array([[1, 1, -2],
                     [-1, 1, 0],
                     [1, 1, 1]])

    unit_cell = BodyCenteredCubic(directions=axes.tolist(),
                                  size=(1, 1, 1), symbol=symbol,
                                  pbc=(False, False, True),
                                  latticeconstant=alat)

    BCCTetras = BodyCenteredCubicTetrahedralFactory()

    impurities = BCCTetras(directions=axes.tolist(),
                           size=(1, 1, 1),
                           symbol=imp_symbol,
                           pbc=(False, False, True),
                           latticeconstant=alat)

    impurities = impurities[impurities.positions.T[2] < alat*1.2]
    impurities.set_cell(unit_cell.get_cell())
    impurities.wrap()

    disloCenterY = alat * np.sqrt(2.)/6.0
    disloCenterX = alat * np.sqrt(6.)/6.0

    impurities.positions[:, 0] -= disloCenterX
    impurities.positions[:, 1] -= disloCenterY

    # size of the cubic cell as a 112 direction
    Lx = int(round((scan_r)/(alat * np.sqrt(6.))))

    # size of the cubic cell as a 110 direction
    Ly = int(round((scan_r) / (alat * np.sqrt(2.))))
    # factor 2 to ,ake shure odd number of images is translated
    # it is important for the correct centering of the dislocation core
    bulk_tetra = impurities * (2*(Lx + 1), 2*(Ly + 1), 1)
    # make 0, 0, at the center

    # make 0, 0, at the center
    bulk_tetra.positions[:, 0] -= (Lx + 1) * alat * np.sqrt(6.) - center[0]
    bulk_tetra.positions[:, 1] -= (Ly + 1) * alat * np.sqrt(2.) - center[1]

    x, y, z = bulk_tetra.positions.T

    radius_x_y_zero = np.sqrt(x**2 + y**2)
    mask_zero = radius_x_y_zero < scan_r

    radius_x_y_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask_center = radius_x_y_center < scan_r

    final_mask = mask_center | mask_zero
    # leave only atoms inside the cylinder
    bulk_tetra = bulk_tetra[final_mask]

    # Create a Stroh object with junk data
    stroh = Stroh(ElasticConstants(C11=141, C12=110, C44=98),
                  np.array([0, 0, 1]))

    c = ElasticConstants(C11=C11, C12=C12, C44=C44)
    burgers = alat * np.array([1., 1., 1.])/2.

    # Solving a new problem with Stroh.solve
    stroh.solve(c, burgers, axes=axes)

    dislo_tetra = bulk_tetra.copy()

    impurities_u = stroh.displacement(bulk_tetra.positions - center)
    impurities_u = -impurities_u if hard_core else impurities_u

    dislo_tetra.positions += impurities_u

    return dislo_tetra


def screw_cyl_octahedral(alat, C11, C12, C44,
                         scan_r=15,
                         symbol="W",
                         imp_symbol='H',
                         hard_core=False,
                         center=(0., 0., 0.)):
    """Generates a set of octahedral positions with `scan_r` radius.
       Applies the screw dislocation displacement for creating an initial guess
       for the H positions at dislocation core.

    Parameters
    ----------
    alat : float
        Lattice constant of the material.
    C11 : float
        C11 elastic constant of the material.
    C12 : float
        C12 elastic constant of the material.
    C44 : float
        C44 elastic constant of the material.
    symbol : string
        Symbol of the element to pass to ase.lattuce.cubic.SimpleCubicFactory
        default is "W" for tungsten
    imp_symbol : string
        Symbol of the elemnt to pass creat Atoms object
        default is "H" for hydrogen
    symbol : string
        Symbol of the elemnt to pass creat Atoms object
    hard_core : float
        Type of the dislocatino core if True then -u
        (sign of displacement is flipped) is applied.
        Default is False i.e. soft core is created.
    center : tuple of floats
        Coordinates of dislocation core (center) (x, y, z).
        Default is (0., 0., 0.)

    Returns
    -------
    ase.Atoms object
        Atoms object with predicted tetrahedral
        positions around dislocation core.

    """
    # TODO: Make one function for impurities and pass factory to it:
    # TODO: i.e. octahedral or terahedral

    from atomman import ElasticConstants
    from atomman.defect import Stroh

    axes = np.array([[1, 1, -2],
                     [-1, 1, 0],
                     [1, 1, 1]])

    unit_cell = BodyCenteredCubic(directions=axes.tolist(),
                                  size=(1, 1, 1), symbol=symbol,
                                  pbc=(False, False, True),
                                  latticeconstant=alat)

    BCCOctas = BodyCenteredCubicOctahedralFactory()

    impurities = BCCOctas(directions=axes.tolist(),
                          size=(1, 1, 1), symbol=imp_symbol,
                          pbc=(False, False, True),
                          latticeconstant=alat)

    impurities = impurities[impurities.positions.T[2] < alat*1.2]
    impurities.set_cell(unit_cell.get_cell())
    impurities.wrap()

    disloCenterY = alat * np.sqrt(2.)/6.0
    disloCenterX = alat * np.sqrt(6.)/6.0

    impurities.positions[:, 0] -= disloCenterX
    impurities.positions[:, 1] -= disloCenterY

    L = int(round(2.0*scan_r/(alat*np.sqrt(2.)))) + 1

    bulk_octa = impurities * (L, L, 1)

    # make 0, 0, at the center
    bulk_octa.positions[:, 0] -= L * alat * np.sqrt(6.)/2. - center[0]
    bulk_octa.positions[:, 1] -= L * alat * np.sqrt(2.)/2. - center[1]

    x, y, z = bulk_octa.positions.T

    radius_x_y_zero = np.sqrt(x**2 + y**2)
    mask_zero = radius_x_y_zero < scan_r

    radius_x_y_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask_center = radius_x_y_center < scan_r

    final_mask = mask_center | mask_zero
    # leave only atoms inside the cylinder
    bulk_octa = bulk_octa[final_mask]

    # Create a Stroh object with junk data
    stroh = Stroh(ElasticConstants(C11=141, C12=110, C44=98),
                                   np.array([0, 0, 1]))

    c = ElasticConstants(C11=C11, C12=C12, C44=C44)
    burgers = alat * np.array([1., 1., 1.])/2.

    # Solving a new problem with Stroh.solve
    stroh.solve(c, burgers, axes=axes)

    dislo_octa = bulk_octa.copy()

    impurities_u = stroh.displacement(bulk_octa.positions - center)
    impurities_u = -impurities_u if hard_core else impurities_u

    dislo_octa.positions += impurities_u

    return dislo_octa


class BodyCenteredCubicTetrahedralFactory(SimpleCubicFactory):
    """A factory for creating tetrahedral lattices in bcc structure"""
    xtal_name = "bcc_tetrahedral"

    bravais_basis = [[0.0, 0.5, 0.25],
                     [0.0, 0.5, 0.75],
                     [0.0, 0.25, 0.5],
                     [0.0, 0.75, 0.5],
                     [0.5, 0.0, 0.75],
                     [0.25, 0.0, 0.5],
                     [0.75, 0.0, 0.5],
                     [0.5, 0.0, 0.25],
                     [0.5, 0.25, 0.0],
                     [0.5, 0.75, 0.0],
                     [0.25, 0.5, 0.0],
                     [0.75, 0.5, 0.0]]


class BodyCenteredCubicOctahedralFactory(SimpleCubicFactory):
    """A factory for creating octahedral lattices in bcc structure"""
    xtal_name = "bcc_octahedral"

    bravais_basis = [[0.5, 0.5, 0.0],
                     [0.0, 0.0, 0.5],
                     [0.5, 0.0, 0.0],
                     [0.5, 0.0, 0.5],
                     [0.0, 0.5, 0.0],
                     [0.0, 0.5, 0.5]]


def dipole_displacement_angle(W_bulk, dislo_coord_left, dislo_coord_right,
                              shift=0.0, mode=1.0):
    """
        Generates a simple displacement field for two dislocations in a dipole
        configuration uding simple Voltera solution as u = b/2 * angle
    """


    burgers = W_bulk.cell[2][2]

    shifted_positions = W_bulk.positions + shift - dislo_coord_left
    x, y, __ = shifted_positions.T
    displacement_left = np.arctan2(y, x) * burgers / (2.0 * np.pi)

    shifted_positions = W_bulk.positions + shift - dislo_coord_right
    x, y, __ = shifted_positions.T
    displacement_right = np.arctan2(y, mode*x) * burgers / (2.0 * np.pi)

    # make two easy core configurations

    u_dipole = np.zeros_like(W_bulk.positions)
    u_dipole.T[2] = displacement_left - mode*displacement_right

    return u_dipole


def get_u_img(W_bulk,
              dislo_coord_left,
              dislo_coord_right,
              n_img=10, n1_shift=0, n2_shift=0):
    """
        Function for getting displacemnt filed for images of quadrupole cells
        used by `make_screw_quadrupole`
    """

    u_img = np.zeros_like(W_bulk.positions)

    C1_quadrupole, C2_quadrupole, __ = W_bulk.get_cell()

    for n1 in range(-n_img, n_img + 1):
        for n2 in range(-n_img, n_img + 1):

            shift = n1 * C1_quadrupole + n2 * C2_quadrupole

            if n1 != n1_shift or n2 != n2_shift:

                u_img += dipole_displacement_angle(W_bulk,
                                                   dislo_coord_left + shift,
                                                   dislo_coord_right + shift,
                                                   shift=n2_shift*C2_quadrupole + n1_shift*C1_quadrupole)

    return u_img


def make_screw_quadrupole(alat,
                          left_shift=0,
                          right_shift=0,
                          n1u=5,
                          symbol="W"):
    r"""Generates a screw dislocation dipole configuration
       for effective quadrupole arrangement. Works for BCC systems.

    Parameters
    ----------
    alat : float
        Lattice parameter of the system in Angstrom.
    left_shift : float, optional
        Shift of the left dislocation core in number of dsitances to next
        equivalent disocation core positions needed for creation for final
        configuration for NEB. Default is 0.
    right_shift : float, optional
        shift of the right dislocation core in number of dsitances to next
        equivalent disocation core positions needed for creation for final
        configuration for NEB. Default is 0.
    n1u : int, odd number
        odd number! length of the cell a doubled distance between core along x.
        Main parameter to calculate cell vectors
    symbol : string
        Symbol of the element to pass to ase.lattuce.cubic.SimpleCubicFactory
        default is "W" for tungsten

    Returns
    -------
    disloc_quadrupole : ase.Atoms
        Resulting quadrupole configuration.
    W_bulk : ase.Atoms
        Perfect system.
    dislo_coord_left : list of float
        Coodrinates of left dislocation core [x, y]
    dislo_coord_right : list of float
        Coodrinates of right dislocation core [x, y]

    Notes
    -----

    Calculation of cell vectors
    +++++++++++++++++++++++++++
    From [1]_ we take:

    - Unit vectors for the cell are:

    .. math::

        u = \frac{1}{3}[1 \bar{2} 1];

    .. math::

        v = \frac{1}{3}[2 \bar{1} \bar{1}];

    .. math::

        z = b = \frac{1}{2}[1 1 1];

    - Cell vectors are:

    .. math::

        C_1 = n^u_1 u + n^v_1 v + C^z_1 z;

    .. math::

        C_2 = n^u_2 u + n^v_2 v + C^z_2 z;

    .. math::

        C_3 = z

    - For quadrupole arrangement n1u needs to be odd number,
      for 135 atoms cell we take n1u=5

    - To have quadrupole as as close as possible to a square one has to take:
    .. math::

        2 n^u_2 + n^v_2 = n^u_1

    .. math::

        n^v_2 \approx \frac{n^u_1}{\sqrt{3}}

    - for n1u = 5:

    .. math::

        n^v_2 \approx \frac{n^u_1}{\sqrt{3}} = 2.89 \approx 3.0

    .. math::

        n^u_2 = \frac{1}{2} (n^u_1 - n^v_2) \approx \frac{1}{2} (5-3)=1

    - Following [2]_ cell geometry is optimized by ading tilt compomemts
    Cz1 and Cz2 for our case of n1u = 3n - 1:

    Easy core

    .. math::

        C^z_1 = + \frac{1}{3}

    .. math::

        C^z_2 = + \frac{1}{6}

    Hard core

    .. math::

        C^z_1 = + \frac{1}{3}

    .. math::

        C^z_2 = + \frac{1}{6}

    may be typo in the paper check the original!

    References:
    +++++++++++

    .. [1] Ventelon, L. & Willaime, F. J 'Core structure and Peierls potential
       of screw dislocations in alpha-Fe from first principles: cluster versus
       dipole approaches' Computer-Aided Mater Des (2007) 14(Suppl 1): 85.
       https://doi.org/10.1007/s10820-007-9064-y

    .. [2] Cai W. (2005) Modeling Dislocations Using a Periodic Cell.
       In: Yip S. (eds) Handbook of Materials Modeling. Springer, Dordrecht
       https://link.springer.com/chapter/10.1007/978-1-4020-3286-8_42

    """

    unit_cell = BodyCenteredCubic(directions=[[1, -2, 1],
                                      [2, -1, -1],
                                      [1, 1, 1]],
                                  symbol=symbol,
                                  pbc=(True, True, True),
                                  latticeconstant=alat, debug=0)

    unit_cell_u, unit_cell_v, unit_cell_z = unit_cell.get_cell()

    # calculate the cell vectors according to the Ventelon paper
    # the real configrution depends on rounding check it here

    n2v = int(np.rint(n1u/np.sqrt(3.0)))
    # when the n1u - n2v difference is odd it is impossible to have
    # perfect arrangemt of translation along x with C2 equal to 0.5*n1u
    # choice of rounding between np.ceil() and np.trunc() makes a different
    # configuration but of same quality of arrangement of quadrupoles (test)
    n2u = np.ceil((n1u - n2v) / 2.)
    n1v = 0

    print("Not rounded values of C2 componets: ")
    print("n2u: %.2f,  n2v: %.2f" % ((n1u - n2v) / 2., n1u/np.sqrt(3.0)))
    print("Calculated cell vectors from n1u = %i" % n1u)
    print("n1v = %i" % n1v)
    print("n2u = %i" % n2u)
    print("n2v = %i" % n2v)

    bulk = unit_cell.copy()*[n1u, n2v, 1]

    # add another periodic shift in x direction to c2 vector
    # for proper periodicity (n^u_2=1) of the effective quadrupole arrangement

    bulk.cell[1] += n2u * unit_cell_u

    C1_quadrupole, C2_quadrupole, C3_quadrupole = bulk.get_cell()

    # calculation of dislocation cores positions
    # distance between centers of triangles along x
    # move to odd/even number -> get to upward/downward triangle
    x_core_dist = alat * np.sqrt(6.)/6.0

    # distance between centers of triangles along y
    y_core_dist = alat * np.sqrt(2.)/6.0

    # separation of the cores in a 1ux1v cell
    nx_left = 2
    nx_right = 5

    if n2v % 2 == 0:  # check if the number of cells in y direction is even
        # Even: then introduce cores between two equal halves of the cell
        ny_left = -2
        ny_right = -1

    else:  # Odd: introduce cores between two equal halves of the cell
        ny_left = 4
        ny_right = 5

    nx_left += 2.0 * left_shift
    nx_right += 2.0 * right_shift

    dislo_coord_left = np.array([nx_left * x_core_dist,
                                 ny_left * y_core_dist,
                                0.0])

    dislo_coord_right = np.array([nx_right * x_core_dist,
                                  ny_right * y_core_dist,
                                  0.0])

    # calculation of the shifts of the initial cores coordinates for the final
    # quadrupole arrangements

    # different x centering preferences for odd and even values
    if n2v % 2 == 0:  # check if the number of cells in y direction is even
        # Even:
        dislo_coord_left += (n2u - 1) * unit_cell_u
        dislo_coord_right += (n2u - 1 + np.trunc(n1u/2.0)) * unit_cell_u

    else:  # Odd:
        dislo_coord_left += n2u * unit_cell_u
        dislo_coord_right += (n2u + np.trunc(n1u/2.0)) * unit_cell_u

    dislo_coord_left += np.trunc(n2v/2.0) * unit_cell_v
    dislo_coord_right += np.trunc(n2v/2.0) * unit_cell_v

    u_quadrupole = dipole_displacement_angle(bulk,
                                             dislo_coord_left,
                                             dislo_coord_right)

    # get the image contribution from the dipoles around
    # (default value of N images to scan n_img=10)
    u_img = get_u_img(bulk,
                      dislo_coord_left,
                      dislo_coord_right)

    u_sum = u_quadrupole + u_img

    # calculate the field of neghbouring cell to estimate
    # linear u_err along C1 and C2 (see. Cai paper)

    # u_err along C2

    n1_shift = 0
    n2_shift = 1

    shift = n1_shift*C1_quadrupole + n2_shift*C2_quadrupole

    u_quadrupole_shifted = dipole_displacement_angle(bulk,
                                                     dislo_coord_left + shift,
                                                     dislo_coord_right + shift,
                                                     shift=shift)

    u_img_shifted = get_u_img(bulk,
                              dislo_coord_left,
                              dislo_coord_right,
                              n1_shift=n1_shift, n2_shift=n2_shift)

    u_sum_shifted = u_quadrupole_shifted + u_img_shifted

    delta_u = u_sum - u_sum_shifted

    delta_u_C2 = delta_u.T[2].mean()
    print("delta u c2: %.2f " % delta_u_C2)

    # u_err along C1

    n1_shift = 1
    n2_shift = 0

    shift = n1_shift*C1_quadrupole + n2_shift*C2_quadrupole

    u_quadrupole_shifted = dipole_displacement_angle(bulk,
                                                     dislo_coord_left + shift,
                                                     dislo_coord_right + shift,
                                                     shift=shift)

    u_img_shifted = get_u_img(bulk,
                              dislo_coord_left,
                              dislo_coord_right,
                              n1_shift=n1_shift, n2_shift=n2_shift)

    u_sum_shifted = u_quadrupole_shifted + u_img_shifted

    delta_u = u_sum - u_sum_shifted

    delta_u_C1 = delta_u.T[2].mean()
    print("delta u c1: %.3f" % delta_u_C1)

    x_scaled, y_scaled, __ = bulk.get_scaled_positions(wrap=False).T

    u_err_C2 = (y_scaled - 0.5)*delta_u_C2

    u_err_C1 = delta_u_C1*(x_scaled - 0.5)

    u_err = u_err_C1 + u_err_C2

    # Calculate the u_tilt to accomodate the stress (see. Cai paper)

    burgers = bulk.cell[2][2]

    u_tilt = 0.5 * burgers * (y_scaled - 0.5)

    final_u = u_sum
    final_u.T[2] += u_err - u_tilt

    disloc_quadrupole = bulk.copy()
    disloc_quadrupole.positions += final_u
    # tilt the cell according to the u_tilt
    disloc_quadrupole.cell[1][2] -= burgers/2.0
    bulk.cell[1][2] -= burgers/2.0

    return disloc_quadrupole, bulk, dislo_coord_left, dislo_coord_right


def make_screw_quadrupole_kink(alat, kind="double", n1u=5, kink_length=20, symbol="W"):
    """Generates kink configuration using make_screw_quadrupole() function
       works for BCC structure.
       The method is based on paper https://doi.org/10.1016/j.jnucmat.2008.12.053

    Parameters
    ----------
    alat : float
        Lattice parameter of the system in Angstrom.
    kind : string
        kind of the kink: right, left or double
    n1u : int
        Number of lattice vectors for the quadrupole cell (make_screw_quadrupole() function)
    kink_length : int
        Length of the cell per kink along b in unit of b, must be even.
    symbol : string
        Symbol of the element to pass to ase.lattuce.cubic.SimpleCubicFactory
        default is "W" for tungsten

    Returns
    -------

    kink : ase.atoms
        kink configuration
    reference_straight_disloc : ase.atoms
        reference straight dislocation configuration
    large_bulk : ase.atoms
        large bulk cell corresponding to the kink configuration

    """

    b = np.sqrt(3.0) * alat / 2.0
    cent_x = np.sqrt(6.0) * alat / 3.0

    ini_disloc_quadrupole, W_bulk, _, _ = make_screw_quadrupole(alat, n1u=n1u,
                                                                   left_shift=0.0,
                                                                   right_shift=0.0,
                                                                   symbol=symbol)

    fin_disloc_quadrupole, W_bulk, _, _ = make_screw_quadrupole(alat, n1u=n1u,
                                                                   left_shift=1.0,
                                                                   right_shift=1.0,
                                                                   symbol=symbol)

    reference_straight_disloc = ini_disloc_quadrupole * [1, 1, kink_length]
    large_bulk = W_bulk * [1, 1, kink_length]
    __, __, z = large_bulk.positions.T

    if kind == "left":

        # we have to adjust the cell to make the kink vector periodic
        # here we remove one atomic row . it is nicely explained in the paper
        left_kink_mask = z < large_bulk.get_cell()[2][2] - 1.0 * b / 3.0 - 0.01
        large_bulk.cell[2][0] -= cent_x
        large_bulk.cell[2][2] -= 1.0 * b / 3.0
        large_bulk = large_bulk[left_kink_mask]

        kink = fin_disloc_quadrupole * [1, 1, kink_length // 2]
        upper_kink = ini_disloc_quadrupole * [1, 1, kink_length // 2]
        upper_kink.positions += np.array((0.0, 0.0, kink.cell[2][2]))
        kink.cell[2][2] += upper_kink.cell[2][2]
        kink.extend(upper_kink)

        # left kink is created the kink vector is in negative x direction assuming (x, y, z) is right group of vectors
        kink = kink[left_kink_mask]
        kink.cell[2][0] -= cent_x
        kink.cell[2][2] -= 1.0 * b / 3.0

    elif kind == "right":

        # we have to adjust the cell to make the kink vector periodic
        # here we remove two atomic rows . it is nicely explained in the paper
        right_kink_mask = z < large_bulk.cell[2][2] - 2.0 * b / 3.0 - 0.01
        large_bulk.cell[2][0] += cent_x
        large_bulk.cell[2][2] -= 2.0 * b / 3.0
        large_bulk = large_bulk[right_kink_mask]

        kink = ini_disloc_quadrupole * [1, 1, kink_length // 2]
        upper_kink = fin_disloc_quadrupole * [1, 1, kink_length // 2]
        upper_kink.positions += np.array((0.0, 0.0, kink.cell[2][2]))
        kink.cell[2][2] += upper_kink.cell[2][2]
        kink.extend(upper_kink)

        kink = kink[right_kink_mask]
        # right kink is created when the kink vector is in positive x direction
        # assuming (x, y, z) is right group of vectors
        kink.cell[2][0] += cent_x
        kink.cell[2][2] -= 2.0 * b / 3.0

    elif kind == "double":

        # for the double kink it is kink length per kink
        kink = ini_disloc_quadrupole * [1, 1, kink_length // 2]
        middle_kink = fin_disloc_quadrupole * [1, 1, kink_length]
        middle_kink.positions += np.array((0.0, 0.0, kink.get_cell()[2][2]))

        kink.extend(middle_kink)
        kink.cell[2][2] += middle_kink.cell[2][2]

        upper_kink = ini_disloc_quadrupole * [1, 1, kink_length // 2]
        upper_kink.positions += np.array((0.0, 0.0, kink.get_cell()[2][2]))
        kink.extend(upper_kink)

        kink.cell[2][2] += upper_kink.cell[2][2]

        large_bulk = W_bulk * [1, 1, 2 * kink_length]  # double kink is double length

    else:
        raise ValueError('Kind must be "right", "left" or "double"')

    return kink, reference_straight_disloc, large_bulk


def make_edge_cyl_001_100(a0, C11, C12, C44,
                          cylinder_r,
                          cutoff=5.5,
                          tol=1e-6,
                          symbol="W"):
    """Function to produce consistent edge dislocation configuration.

    Parameters
    ----------
    alat : float
        Lattice constant of the material.
    C11 : float
        C11 elastic constant of the material.
    C12 : float
        C12 elastic constant of the material.
    C44 : float
        C44 elastic constant of the material.
    cylinder_r : float
        Radius of cylinder of unconstrained atoms around the
        dislocation  in angstrom.
    cutoff : float
        Potential cutoff for determenition of size of
        fixed atoms region (2*cutoff)
    tol : float
        Tolerance for generation of self consistent solution.
    symbol : string
        Symbol of the element to pass to ase.lattuce.cubic.SimpleCubicFactory
        default is "W" for tungsten

    Returns
    -------
    bulk : ase.Atoms object
        Bulk configuration.
    disloc : ase.Atoms object
        Dislocation configuration.
    disp : np.array
        Corresponding displacement.
    """
    from atomman import ElasticConstants
    from atomman.defect import Stroh
    # Create a Stroh object with junk data
    stroh = Stroh(ElasticConstants(C11=141, C12=110, C44=98),
                  np.array([0, 0, 1]))

    axes = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])

    c = ElasticConstants(C11=C11, C12=C12, C44=C44)
    burgers = a0 * np.array([1., 0., 0.])

    # Solving a new problem with Stroh.solve
    stroh.solve(c, burgers, axes=axes)

    unit_cell = BodyCenteredCubic(directions=axes.tolist(),
                                  size=(1, 1, 1), symbol=symbol,
                                  pbc=(False, False, True),
                                  latticeconstant=a0)

    bulk = unit_cell.copy()

    # shift to make the zeros of the cell betweem the atomic planes
    # and under the midplane on Y axes
    X_midplane_shift = -0.25*a0
    Y_midplane_shift = -0.25*a0

    bulk_shift = [X_midplane_shift,
                  Y_midplane_shift,
                  0.0]

    bulk.positions += bulk_shift

    tot_r = cylinder_r + 2*cutoff + 0.01

    Lx = int(round(tot_r/a0))
    Ly = int(round(tot_r/a0))

    # factor 2 to make sure odd number of images is translated
    # it is important for the correct centering of the dislocation core
    bulk = bulk * (2*Lx, 2*Ly, 1)

    center_shift = [Lx * a0, Ly * a0, 0.0]

    bulk.positions -= center_shift
    # bulk.write("before.xyz")

    disp1 = stroh.displacement(bulk.positions)
    disloc = bulk.copy()
    res = np.inf
    i = 0
    while res > tol:
        disloc.positions = bulk.positions + disp1
        disp2 = stroh.displacement(disloc.positions)
        res = np.abs(disp1 - disp2).max()
        disp1 = disp2
        print('disloc SCF', i, '|d1-d2|_inf =', res)
        i += 1
        if i > 10:
            raise RuntimeError('Self-consistency did \
                                not converge in 10 cycles')
    disp = disp2

    x, y, z = disloc.positions.T
    radius_x_y_zero = np.sqrt(x**2 + y**2)
    mask = radius_x_y_zero < tot_r
    disloc = disloc[mask]
    bulk = bulk[mask]
    # disloc.write("after_disp.xyz")

    x, y, z = disloc.positions.T
    radius_x_y_zero = np.sqrt(x**2 + y**2)
    mask_zero = radius_x_y_zero > cylinder_r
    fix_atoms = FixAtoms(mask=mask_zero)
    disloc.set_constraint(fix_atoms)

    return bulk, disloc, disp


def read_dislo_QMMM(filename=None, image=None):

    """
    Reads extended xyz file with QMMM configuration
    Uses "region" for mapping of QM, MM and fixed atoms
    Sets ase.constraints.FixAtoms constraint on fixed atoms

    Parameters
    ----------
    filename : path to xyz file
    image : image with "region" array to set up constraint and extract qm_mask

    Returns
    -------
    dislo_QMMM : Output ase.Atoms object
        Includes "region" array and FixAtoms constraint
    qm_mask : array mask for QM atoms mapping
    """

    if filename is not None:
        dislo_QMMM = read(filename)

    elif image is not None:
        dislo_QMMM = image

    else:
        raise RuntimeError("Please provide either path or image")

    region = dislo_QMMM.get_array("region")
    Nat = len(dislo_QMMM)

    print("Total number of atoms in read configuration: {0:7}".format(Nat))

    for region_type in np.unique(region):
        print("{0:52d} {1}".format(np.count_nonzero(region == region_type),
                                   region_type))

    if len(dislo_QMMM.constraints) == 0:

        print("Adding fixed atoms constraint")

        fix_mask = region == "fixed"
        fix_atoms = FixAtoms(mask=fix_mask)
        dislo_QMMM.set_constraint(fix_atoms)

    else:
        print("Constraints list is not zero")

    qm_mask = region == "QM"

    qm_atoms = dislo_QMMM[qm_mask]
    qm_atoms_types = np.array(qm_atoms.get_chemical_symbols())

    print("QM region atoms: {0:3d}".format(np.count_nonzero(qm_mask)))

    for qm_atom_type in np.unique(qm_atoms_types):

        print("{0:20d} {1}".format(np.count_nonzero(qm_atoms_types == qm_atom_type),
                                   qm_atom_type))

    return dislo_QMMM, qm_mask


def plot_bulk(atoms, n_planes=3, ax=None, ms=200):
    """
    Plots x, y coordinates of atoms colored according
    to non-equivalent planes in z plane
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    x, y, z = atoms.positions.T

    zlim = atoms.cell[2, 2] / n_planes

    bins = np.linspace(zlim, atoms.cell[2, 2], num=n_planes)
    bins -= atoms.cell[2, 2] / (2.0 * n_planes)

    plane_ids = np.digitize(z, bins=bins)

    for plane_id in np.unique(plane_ids):
        mask = plane_id == plane_ids
        ax.scatter(x[mask], y[mask], s=ms, edgecolor="k")


def ovito_dxa_straight_dislo_info(disloc, structure="BCC", replicate_z=3):
    """
    A function to extract information from ovito dxa analysis.
    Current version works for 1b thick configurations
    containing straight dislocations.

    Parameters
    ----------
    disloc: ase.Atoms
        Atoms object containing the atomic configuration to analyse
    replicate_z: int
        Specifies number of times to replicate the configuration
        along the dislocation line.
        Ovito dxa analysis needs at least 3b thick cell to work.

    Returns
    -------
    Results: np.array(position, b, line, angle)

    """
    from ovito.io.ase import ase_to_ovito
    from ovito.modifiers import ReplicateModifier, DislocationAnalysisModifier
    from ovito.pipeline import StaticSource, Pipeline

    dxa_disloc = disloc.copy()
    if 'fix_mask' in dxa_disloc.arrays:
        del dxa_disloc.arrays['fix_mask']

    input_crystal_structures = {"BCC": DislocationAnalysisModifier.Lattice.BCC,
                                "FCC": DislocationAnalysisModifier.Lattice.FCC,
                                "Diamond": DislocationAnalysisModifier.Lattice.CubicDiamond}

    data = ase_to_ovito(dxa_disloc)
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(ReplicateModifier(num_z=replicate_z))
    dxa = DislocationAnalysisModifier(
          input_crystal_structure=input_crystal_structures[structure])
    pipeline.modifiers.append(dxa)

    data = pipeline.compute()
    results = []
    for segment in data.dislocations.segments:

        #  insure that this is a straight dislocation in a 1b thick cell
        length = segment.length / replicate_z
        try:
            np.testing.assert_almost_equal(length, dxa_disloc.cell[2, 2],
                                           decimal=2)
        except AssertionError as error:
            print("Dislocation might not be straight:")
            print(error)

        b = segment.true_burgers_vector

        b_hat = np.array(segment.spatial_burgers_vector)
        b_hat /= np.linalg.norm(b_hat)

        lines = np.diff(segment.points, axis=0)
        angles = []
        positions = []
        for point in segment.points:
            positions.append(point[:2])

        for line in lines:
            t_hat = line / np.linalg.norm(line)
            dot = np.abs(np.dot(t_hat, b_hat))
            angle = np.degrees(np.arccos(dot))
            angles.append(angle)

        position = np.array(positions).mean(axis=0)
        line = np.array(lines).mean(axis=0)
        angle = np.array(angles).mean()
        results.append([position, b, line, angle])

    return results


def get_centering_mask(atoms, radius,
                       core_position=[0., 0., 0.],
                       extension=[0., 0., 0.],):

    center = np.diag(atoms.cell) / 2

    r = np.sqrt(((atoms.positions[:, [0, 1]]
                  - center[[0, 1]]) ** 2).sum(axis=1))
    mask = r < radius

    core_position = np.array(core_position)
    shifted_center = center + core_position
    shifted_r = np.sqrt(((atoms.positions[:, [0, 1]] -
                          shifted_center[[0, 1]]) ** 2).sum(axis=1))
    shifted_mask = shifted_r < radius

    extension = np.array(extension)
    extended_center = center + extension
    extended_r = np.sqrt(((atoms.positions[:, [0, 1]] -
                           extended_center[[0, 1]]) ** 2).sum(axis=1))
    extended_mask = extended_r < radius

    final_mask = mask | shifted_mask | extended_mask

    return final_mask


def check_duplicates(atoms, distance=0.1):
    """
    Returns a mask of atoms that have at least one other atom closer than distance
    """
    mask = atoms.get_all_distances() < distance
    duplicates = np.full_like(atoms, False)
    for i, row in enumerate(mask):
        if any(row[i+1:]):
            duplicates[i] = True
    # print(f"found {duplicates.sum()} duplicates")
    return duplicates.astype(np.bool)


class CubicCrystalDislocation:
    def __init__(self, unit_cell, alat, C11, C12, C44, axes, burgers,
                 unit_cell_core_position=None,
                 parity=None, glide_distance=None, n_planes=None,
                 self_consistent=None):
        """
        This class represents a dislocation in a cubic crystal

        The dislocation is defined by the crystal unit cell,
        elastic constants C11, C12 and C44, crystal axes,
        burgers vector and optional shift and parity vectors.

        Parameters
        ----------
        unit_cell : unit cell to build the dislocation configuration
        alat : lattice constant
        C11 : elastic constants
        C12
        C44
        axes : cell axes (b is normally along z direction)
        burgers : burgers vector of the dislocation
        unit_cell_core_position : dislocation core position in the unit cell
                                  used to shift atomic positions to
                                  make the dislocation core the center
                                  of the cell
        parity
        glide_distance : distance to the next equivalent
                         core position in the glide direction
        n_planes : int
            number of non equivalent planes in z direction
        self_consistent : float
            default value for the displacement calculation
        """

        self.unit_cell = unit_cell.copy()
        self.alat = alat
        self.C11 = C11
        self.C12 = C12
        self.C44 = C44

        self.axes = axes
        self.burgers = burgers
        if unit_cell_core_position is None:
            unit_cell_core_position = np.zeroes(3)
        self.unit_cell_core_position = unit_cell_core_position
        if parity is None:
            parity = np.zeros(2, dtype=int)
        self.parity = parity
        if glide_distance is None:
            glide_distance = 0.0
        self.glide_distance = glide_distance
        if n_planes is None:
            n_planes = 3
        self.n_planes = n_planes
        if self_consistent is None:
            self_consistent = True
        self.self_consistent = self_consistent

        self.stroh = None

    def init_stroh(self):

        from atomman import ElasticConstants
        from atomman.defect import Stroh
        c = ElasticConstants(C11=self.C11, C12=self.C12, C44=self.C44)
        self.stroh = Stroh(c, burgers=self.burgers, axes=self.axes)


    def set_burgers(self, burgers):
        self.burgers = burgers
        if self.stroh is None:
            self.init_stroh()


    def plot_unit_cell(self, ms=250, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        plot_bulk(self.unit_cell, self.n_planes, ax=ax, ms=ms)

        x_core, y_core, _ = self.unit_cell_core_position
        ax.scatter(x_core, y_core, marker="x", s=ms, c="red")
        ax.scatter(x_core + self.glide_distance, y_core, marker="x", s=ms,
                   c="blue")
        ax.set_aspect('equal')

        x0, y0, _ = np.diag(self.unit_cell.cell)

        ax.plot([0.0, 0.0, x0, x0, 0.0],
                [0.0, y0, y0, 0.0, 0.0], color="black", zorder=0)

        bulk_atoms = ax.scatter([], [], color="w", edgecolor="k",
                             label="lattice atoms")
        core1 = ax.scatter([], [], marker="x", label="initial core position",
                           c="r")
        core2 = ax.scatter([], [], marker="x", label="glide core position",
                           c="b")

        ax.legend(handles=[bulk_atoms, core1, core2], fontsize=12)
        ax.set_xlabel(r"$\AA$")
        ax.set_ylabel(r"$\AA$")


    def displacements(self, bulk_positions, center, self_consistent=True, 
                      tol=1e-6, max_iter=100, verbose=True):
        if self.stroh is None:
            self.init_stroh()

        disp1 = np.real(self.stroh.displacement(bulk_positions - center))
        if not self_consistent:
            return disp1

        res = np.inf
        i = 0
        while res > tol:
            disloc_positions = bulk_positions + disp1
            disp2 = np.real(self.stroh.displacement(disloc_positions - center))
            res = np.abs(disp1 - disp2).max()
            disp1 = disp2
            if verbose:
                print('disloc SCF', i, '|d1-d2|_inf =', res)
            i += 1
            if i > max_iter:
                raise RuntimeError(f'Self-consistency did not converge in {max_iter} cycles')
        return disp2

    def build_cylinder(self, radius,
                       core_position=np.array([0., 0., 0.]),
                       extension=np.array([0., 0., 0.]),
                       fix_width=10.0, self_consistent=None):

        if self_consistent is None:
            self_consistent = self.self_consistent

        extent = np.array([2 * (radius + fix_width),
                           2 * (radius + fix_width), 1.])
        repeat = np.ceil(extent / np.diag(self.unit_cell.cell)).astype(int)

        # if the extension and core position is
        # within the unit cell, do not add extra unit cells
        repeat_extension = np.floor(2.0 * extension /
                                    np.diag(self.unit_cell.cell)).astype(int)
        repeat_core_position = np.floor(2.0 * core_position /
                                    np.diag(self.unit_cell.cell)).astype(int)

        extra_repeat = np.stack((repeat_core_position,
                                 repeat_extension)).max(axis=0)

        repeat += extra_repeat

        repeat[2] = 1  # exactly one cell in the periodic direction

        # ensure correct parity in x and y directions
        if repeat[0] % 2 != self.parity[0]:
            repeat[0] += 1
        if repeat[1] % 2 != self.parity[1]:
            repeat[1] += 1

        bulk = self.unit_cell * repeat
        # in order to get center from an atom to the desired position
        # we have to move the atoms in the opposite direction
        bulk.positions -= self.unit_cell_core_position

        center = np.diag(bulk.cell) / 2
        shifted_center = center + core_position

        cylinder_mask = get_centering_mask(bulk, radius, core_position, extension)

        # add square borders for the case of large extension or core position
        x, y, _ = bulk.positions.T
        x_mask = x - center[0] < extension[0] + core_position[0]
        x_mask = x_mask * (x - center[0] > 0)
        y_mask = np.abs(y - center[1]) < radius
        square_mask = y_mask & x_mask

        final_mask = square_mask | cylinder_mask

        bulk = bulk[final_mask]

        # disloc is a copy of bulk with displacements applied
        disloc = bulk.copy()

        disloc.positions += self.displacements(bulk.positions, shifted_center,
                                               self_consistent=self_consistent)

        r = np.sqrt(((bulk.positions[:, [0, 1]]
                      - center[[0, 1]])**2).sum(axis=1))

        fix_mask = r > radius - fix_width

        shifted_r = np.sqrt(((bulk.positions[:, [0, 1]] -
                              shifted_center[[0, 1]]) ** 2).sum(axis=1))

        shifted_fix_max = shifted_r > radius - fix_width
        extension = np.array(extension)
        extended_center = center + extension
        extended_r = np.sqrt(((bulk.positions[:, [0, 1]] -
                               extended_center[[0, 1]]) ** 2).sum(axis=1))
        extended_fix_max = extended_r > radius - fix_width
        final_fix_mask = fix_mask & shifted_fix_max & extended_fix_max

        x, y, _ = bulk.positions.T
        x_mask = x - center[0] < extension[0] + core_position[0]
        x_mask = x_mask * (x - center[0] > 0)
        y_mask = np.abs(y - center[1]) > radius - fix_width

        # change mask only between the centers of the cylinders
        final_fix_mask[x_mask] = y_mask[x_mask]

        disloc.set_array('fix_mask', final_fix_mask)
        disloc.set_constraint(FixAtoms(mask=final_fix_mask))

        # adding vacuum and centering breaks consistency
        # of displacement =  dislo.positions - bulk.positions
        # which is essential for plot_vitek and other tools
        # I could not find a way to add vacuum to both disloc and bulk
        # without spoiling the displacement
        # disloc.center(vacuum=2 * fix_width, axis=(0, 1))

        return bulk, disloc

    def build_glide_configurations(self, radius,
                                   average_positions=False, **kwargs):

        final_core_position = np.array([self.glide_distance, 0.0, 0.0])

        bulk_ini, disloc_ini = self.build_cylinder(radius,
                                                   extension=final_core_position,
                                                   **kwargs)

        _, disloc_fin = self.build_cylinder(radius,
                                            core_position=final_core_position,
                                            **kwargs)
        if average_positions:
            # get the fixed atoms constrain
            FixAtoms = disloc_ini.constraints[0]
            # get the indices of fixed atoms
            fixed_atoms_indices = FixAtoms.get_indices()

            # make the average position of fixed atoms
            # between initial and the last position
            ini_fix_pos = disloc_ini.get_positions()[fixed_atoms_indices]
            fin_fix_pos = disloc_fin.get_positions()[fixed_atoms_indices]

            new_av_pos = (ini_fix_pos + fin_fix_pos) / 2.0

            positions = disloc_ini.get_positions()
            positions[fixed_atoms_indices] = new_av_pos
            disloc_ini.set_positions(positions, apply_constraint=False)

            positions = disloc_fin.get_positions()
            positions[fixed_atoms_indices] = new_av_pos
            disloc_fin.set_positions(positions, apply_constraint=False)

        averaged_cell = (disloc_ini.cell + disloc_fin.cell) / 2.0
        disloc_ini.set_cell(averaged_cell)
        disloc_fin.set_cell(averaged_cell)

        return bulk_ini, disloc_ini, disloc_fin

    def build_impurity_cylinder(self, disloc, impurity, radius,
                                imp_symbol="H",
                                core_position=np.array([0., 0., 0.]),
                                extension=np.array([0., 0., 0.]),
                                self_consistent=False,
                                extra_bulk_at_core=False,
                                core_radius=0.5,
                                shift=np.array([0.0, 0.0, 0.0])):

        extent = np.array([2 * radius + np.linalg.norm(self.burgers),
                           2 * radius + np.linalg.norm(self.burgers), 1.])
        repeat = np.ceil(extent / np.diag(self.unit_cell.cell)).astype(int)

        # if the extension and core position is
        # within the unit cell, do not add extra unit cells
        repeat_extension = np.floor(extension /
                                    np.diag(self.unit_cell.cell)).astype(int)
        repeat_core_position = np.floor(core_position /
                                    np.diag(self.unit_cell.cell)).astype(int)

        extra_repeat = np.stack((repeat_core_position,
                                 repeat_extension)).max(axis=0)

        repeat += extra_repeat

        repeat[2] = 1  # exactly one cell in the periodic direction

        # ensure correct parity in x and y directions
        if repeat[0] % 2 != self.parity[0]:
            repeat[0] += 1
        if repeat[1] % 2 != self.parity[1]:
            repeat[1] += 1

        impurities_unit_cell = impurity(directions=self.axes.tolist(),
                                        size=(1, 1, 1),
                                        symbol=imp_symbol,
                                        pbc=(False, False, True),
                                        latticeconstant=self.alat)

        impurities_unit_cell.cell = self.unit_cell.cell
        impurities_unit_cell.wrap(pbc=True)
        duplicates = check_duplicates(impurities_unit_cell)
        impurities_unit_cell = impurities_unit_cell[np.logical_not(duplicates)]

        impurities_bulk = impurities_unit_cell * repeat
        # in order to get center from an atom to the desired position
        # we have to move the atoms in the opposite direction
        impurities_bulk.positions -= self.unit_cell_core_position

        # build a bulk impurities cylinder
        mask = get_centering_mask(impurities_bulk,
                                  radius + np.linalg.norm(self.burgers),
                                  core_position, extension)

        impurities_bulk = impurities_bulk[mask]

        center = np.diag(impurities_bulk.cell) / 2
        shifted_center = center + core_position

        # use stroh displacement for impurities
        # disloc is a copy of bulk with displacements applied
        impurities_disloc = impurities_bulk.copy()

        core_mask = get_centering_mask(impurities_bulk,
                                       core_radius,
                                       core_position, extension)

        print(f"Ignoring {core_mask.sum()} core impurities")
        non_core_mask = np.logical_not(core_mask)

        displacemets = self.displacements(impurities_bulk.positions[non_core_mask],
                                                          shifted_center,
                                                          self_consistent=self_consistent)

        impurities_disloc.positions[non_core_mask] += displacemets

        if extra_bulk_at_core:  # add extra bulk positions at dislocation core
            bulk_mask = get_centering_mask(impurities_bulk,
                                           1.1 * self.unit_cell_core_position[1],
                                           core_position + shift, extension)

            print(f"Adding {bulk_mask.sum()} extra atoms")
            impurities_disloc.extend(impurities_bulk[bulk_mask])

        mask = get_centering_mask(impurities_disloc,
                                  radius,
                                  core_position,
                                  extension)

        impurities_disloc = impurities_disloc[mask]

        disloc_center = np.diag(disloc.cell) / 2.
        delta = disloc_center - center
        delta[2] = 0.0
        impurities_disloc.positions += delta
        impurities_disloc.cell = disloc.cell

        return impurities_disloc

class BCCScrew111Dislocation(CubicCrystalDislocation):
    def __init__(self, alat, C11, C12, C44, symbol='W'):
        axes = np.array([[1, 1, -2],
                         [-1, 1, 0],
                         [1, 1, 1]])
        burgers = alat * np.array([1, 1, 1]) / 2.0
        unit_cell_core_position = alat * np.array([np.sqrt(6.)/6.0,
                                                   np.sqrt(2.)/6.0, 0])
        parity = [0, 0]
        unit_cell = BodyCenteredCubic(directions=axes.tolist(),
                                      size=(1, 1, 1), symbol=symbol,
                                      pbc=True,
                                      latticeconstant=alat)
        glide_distance = alat * np.linalg.norm(axes[0]) / 3.0
        super().__init__(unit_cell, alat, C11, C12, C44,
                         axes, burgers, unit_cell_core_position, parity,
                         glide_distance)

        
class BCCEdge111Dislocation(CubicCrystalDislocation):
    def __init__(self, alat, C11, C12, C44, symbol='W'):
        axes = np.array([[1, 1, 1],
                         [1, -1, 0],
                         [1, 1, -2]])
        burgers = alat * np.array([1, 1, 1]) / 2.0
        unit_cell_core_position = alat * np.array([(1.0/3.0) * np.sqrt(3.0)/2.0,
                                  0.25 * np.sqrt(2.0), 0])
        parity = [0, 0]
        unit_cell = BodyCenteredCubic(directions=axes.tolist(),
                                      size=(1, 1, 1), symbol=symbol,
                                      pbc=True,
                                      latticeconstant=alat)
        glide_distance = np.linalg.norm(burgers) / 3.0
        n_planes = 6
        super().__init__(unit_cell, alat, C11, C12, C44,
                         axes, burgers, unit_cell_core_position, parity,
                         glide_distance, n_planes=n_planes)


class BCCMixed111Dislocation(CubicCrystalDislocation):
    def __init__(self, alat, C11, C12, C44, symbol='W'):
        axes = np.array([[1, -1, -2],
                         [1, 1, 0],
                         [1, -1, 1]])
        burgers = alat * np.array([1, -1, -1]) / 2.0
        parity = [0, 0]
        unit_cell = BodyCenteredCubic(directions=axes.tolist(),
                                      size=(1, 1, 1), symbol=symbol,
                                      pbc=True,
                                      latticeconstant=alat)

        # middle of the right edge of the first upward triangle
        core_position = (unit_cell.positions[1] +
                         unit_cell.positions[2]) / 2.0

        unit_cell_core_position = np.array([core_position[0],
                                            core_position[1], 0])

        glide_distance = alat * np.linalg.norm(axes[0]) / 3.0
        super().__init__(unit_cell, alat, C11, C12, C44,
                         axes, burgers, unit_cell_core_position, parity,
                         glide_distance)


class BCCEdge100Dislocation(CubicCrystalDislocation):
    def __init__(self, alat, C11, C12, C44, symbol='W'):
        axes = np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]])
        burgers = alat * np.array([1, 0, 0])
        unit_cell_core_position = alat * np.array([0.25,
                                                   0.25, 0])
        parity = [0, 0]
        unit_cell = BodyCenteredCubic(directions=axes.tolist(),
                                      size=(1, 1, 1), symbol=symbol,
                                      pbc=True,
                                      latticeconstant=alat)
        glide_distance = alat
        n_planes = 2
        super().__init__(unit_cell, alat, C11, C12, C44,
                         axes, burgers, unit_cell_core_position, parity,
                         glide_distance, n_planes=n_planes)


class BCCEdge100110Dislocation(CubicCrystalDislocation):
    def __init__(self, alat, C11, C12, C44, symbol='W'):
        axes = np.array([[1, 0, 0],
                         [0, 1, 1],
                         [0, -1, 1]])
        burgers = alat * np.array([1, 0, 0])
        unit_cell_core_position = alat * np.array([0.5,
                                                   np.sqrt(2.) / 4.0, 0])
        parity = [0, 0]
        unit_cell = BodyCenteredCubic(directions=axes.tolist(),
                                      size=(1, 1, 1), symbol=symbol,
                                      pbc=True,
                                      latticeconstant=alat)
        glide_distance = 0.5 * alat
        n_planes = 2
        super().__init__(unit_cell, alat, C11, C12, C44,
                         axes, burgers, unit_cell_core_position, parity,
                         glide_distance, n_planes=n_planes)


class DiamondGlide30degreePartial(CubicCrystalDislocation):
    def __init__(self, alat, C11, C12, C44, symbol='C'):
        axes = np.array([[1, 1, -2],
                         [1, 1, 1],
                         [1, -1, 0]])

        burgers = alat * np.array([1, -2, 1.]) / 6.

        disloCenterX = 0.5 * (alat * np.linalg.norm(axes[0])) / 6.0
        # 1/4 + 1/2 * (1/3 - 1/4) - to be in the middle of the glide set
        disloCenterY = 7.0 * (alat * np.linalg.norm(axes[1])) / 24.0

        unit_cell_core_position = np.array([disloCenterX,
                                            disloCenterY, 0])

        parity = [0, 0]

        unit_cell = Diamond(symbol, directions=axes.tolist(),
                            pbc=(False, False, True),
                            latticeconstant=alat)

        glide_distance = alat * np.linalg.norm(axes[0]) / 4.0

        n_planes = 2
        # There is very small distance between
        # atomic planes in glide configuration.
        # Due to significant anisotropy application of the self consistent
        # displacement field leads to deformation of the atomic planes.
        # This leads to the cut plane crossing one of the atomic planes and
        # thus breaking the stacking fault.
        self_consistent = False
        super().__init__(unit_cell, alat, C11, C12, C44,
                         axes, burgers, unit_cell_core_position, parity,
                         glide_distance, n_planes=n_planes,
                         self_consistent=self_consistent)


class DiamondGlide90degreePartial(CubicCrystalDislocation):
    def __init__(self, alat, C11, C12, C44, symbol='C'):
        axes = np.array([[1, 1, -2],
                         [1, 1, 1],
                         [1, -1, 0]])

        burgers = alat * np.array([1., 1., -2.]) / 6.

        disloCenterX = 0.5 * (alat * np.linalg.norm(axes[0])) / 6.0
        # 1/4 + 1/2 * (1/3 - 1/4) - to be in the middle of the glide set
        disloCenterY = 7.0 * (alat * np.linalg.norm(axes[1])) / 24.0

        unit_cell_core_position = np.array([disloCenterX,
                                            disloCenterY, 0])

        parity = [0, 0]

        unit_cell = Diamond(symbol, directions=axes.tolist(),
                            pbc=(False, False, True),
                            latticeconstant=alat)

        glide_distance = alat * np.linalg.norm(axes[0]) / 4.0

        n_planes = 2
        # There is very small distance between
        # atomic planes in glide configuration.
        # Due to significant anisotropy application of the self consistent
        # displacement field leads to deformation of the atomic planes.
        # This leads to the cut plane crossing one of the atomic planes and
        # thus breaking the stacking fault.
        self_consistent = False
        super().__init__(unit_cell, alat, C11, C12, C44,
                         axes, burgers, unit_cell_core_position, parity,
                         glide_distance, n_planes=n_planes,
                         self_consistent=self_consistent)


class CubicCrystalDissociatedDislocation(CubicCrystalDislocation):
    """
        This class represents a dissociated dislocation in a cubic crystal
        with b = b_left + b_right.
        'left_dislocation' and 'right_dislocations' are expected
        to be instances of classes derived from CubicCrystalDislocation class.
    """
    def __init__(self, left_dislocation, right_dislocation,
                 *args, **kwargs):

        self.left_dislocation = left_dislocation
        self.right_dislocation = right_dislocation

        super().__init__(*args, **kwargs)


    def build_cylinder(self, radius, partial_distance=0,
                       core_position=np.array([0., 0., 0.]),
                       extension=np.array([0., 0., 0.]),
                       fix_width=10.0, self_consistent=None):
        """
        Overloaded function to make dissociated dislocations.
        Partial distance is provided as an integer to define number
        of glide distances between two partials.

        Parameters
        ----------
        radius: float
            radius of the cell
        partial_distance: int
            distance between partials (SF length) in number of glide distances.
            Default is 0 -> non dissociated dislocation
            with b = b_left + b_right is produced
        """

        if self_consistent is None:
            self_consistent = self.self_consistent

        partial_distance_Angstrom = np.array(
            [self.glide_distance * partial_distance, 0.0, 0.0])

        bulk, disloc = self.left_dislocation.build_cylinder(radius,
                                                  extension=extension + partial_distance_Angstrom,
                                                  core_position=core_position,
                                                  fix_width=fix_width,
                                                  self_consistent=self_consistent)

        _, disloc_right = self.right_dislocation.build_cylinder(radius,
                                                      core_position=core_position + partial_distance_Angstrom,
                                                      extension=extension,
                                                      fix_width=fix_width,
                                                      self_consistent=self_consistent)

        u_right = disloc_right.positions - bulk.positions
        disloc.positions += u_right

        return bulk, disloc


class DiamondGlideScrew(CubicCrystalDissociatedDislocation):
    def __init__(self, alat, C11, C12, C44, symbol='C'):

        axes = np.array([[1, 1, -2],
                        [1, 1, 1],
                        [1, -1, 0]])

        # aiming for the resulting burgers vector
        burgers = alat * np.array([1, -1, 0]) / 2.

        disloCenterX = 0.5 * (alat * np.linalg.norm(axes[0])) / 6.0
        # 1/4 + 1/2 (1/3 - 1/4) - to be in the middle of the glide set
        disloCenterY = 7.0 * (alat * np.linalg.norm(axes[1])) / 24.0

        unit_cell_core_position = np.array([disloCenterX,
                                            disloCenterY, 0])

        parity = [0, 0]

        unit_cell = Diamond(symbol, directions=axes.tolist(),
                            pbc=(False, False, True),
                            latticeconstant=alat)

        glide_distance = alat * np.linalg.norm(axes[0]) / 4.0

        n_planes = 2

        # 30 degree
        burgers_left = alat * np.array([2., -1., -1.]) / 6.
        left30 = DiamondGlide30degreePartial(alat, C11, C12, C44)
        left30.set_burgers(burgers_left)
        # another 30 degree
        burgers_right = alat * np.array([1, -2, 1.]) / 6.
        right30 = DiamondGlide30degreePartial(alat, C11, C12, C44)
        self_consistent = False
        super().__init__(left30, right30, unit_cell, alat, C11, C12, C44,
                         axes, burgers, unit_cell_core_position, parity,
                         glide_distance, n_planes=n_planes,
                         self_consistent=self_consistent)


class DiamondGlide60Degree(CubicCrystalDissociatedDislocation):
    def __init__(self, alat, C11, C12, C44, symbol='C'):
        axes = np.array([[1, 1, -2],
                         [1, 1, 1],
                         [1, -1, 0]])

        # aiming for the resulting burgers vector
        burgers = alat * np.array([1, 0, -1]) / 2.

        disloCenterX = 0.5 * (alat * np.linalg.norm(axes[0])) / 6.0
        # 1/4 + 1/2 (1/3 - 1/4) - to be in the middle of the glide set
        disloCenterY = 7.0 * (alat * np.linalg.norm(axes[1])) / 24.0

        unit_cell_core_position = np.array([disloCenterX,
                                            disloCenterY, 0])

        parity = [0, 0]

        unit_cell = Diamond(symbol, directions=axes.tolist(),
                            pbc=(False, False, True),
                            latticeconstant=alat)

        glide_distance = alat * np.linalg.norm(axes[0]) / 4.0

        n_planes = 2

        # 30 degree
        burgers_left = alat * np.array([2., -1., -1.]) / 6.
        left30 = DiamondGlide30degreePartial(alat, C11, C12, C44)
        left30.set_burgers(burgers_left)
        # 90 degree
        burgers_right = alat * np.array([1, 1, -2.]) / 6.
        right90 = DiamondGlide90degreePartial(alat, C11, C12, C44)
        self_consistent = False
        super().__init__(left30, right90, unit_cell, alat, C11, C12, C44,
                         axes, burgers, unit_cell_core_position, parity,
                         glide_distance, n_planes=n_planes,
                         self_consistent=self_consistent)


class FixedLineAtoms:
    """Constrain atoms to move along a given direction only."""
    def __init__(self, a, direction):
        self.a = a
        self.dir = direction / np.sqrt(np.dot(direction, direction))

    def adjust_positions(self, atoms, newpositions):
        steps = newpositions[self.a] - atoms.positions[self.a]
        newpositions[self.a] = (atoms.positions[self.a] +
                                np.einsum("ij,j,k", steps, self.dir, self.dir))

    def adjust_forces(self, atoms, forces):
        forces[self.a] = np.einsum("ij,j,k", forces[self.a], self.dir, self.dir)


def gamma_line(unit_cell, calc=None, shift_dir=0, surface=2,
               size=[2, 2, 2], factor=15, n_dots=11,
               relax=True, fmax=1.0e-2, return_images=False):
    """
    This function performs a calculation of a cross-sections in 'shift_dir`
    of the generalized stacking fault (GSF) gamma
    surface with `surface` orientation.
    *A gamma surface is defined as the energy variation when the
    crystal is cut along a particular plane and then one of the
    resulting parts is displaced along a particular direction. This
    quantity is related to the energy landscape of dislocations and
    provides data out of equilibrium, preserving the crystal state.*
    For examples for the case of W and more details see section 4.2
    and figure 2 in [J. Phys.: Condens. Matter 25 (2013) 395502 (15pp)]\
                    (http://iopscience.iop.org/0953-8984/25/39/395502)

    Parameters
    ----------
    unit_cell: ase.Atoms
        Unit cell to construct gamma surface from.
        Should have a ase.calculator attached as calc
        in order to perform relaxation.
    calc: ase.calculator
        if unit_cell.calc is None set unit_cell.calc to calc
    shift_dir: int
        index of unit_cell axes to shift atoms
    surface: int
        index of unit_cell axes to be the surface normal direction
    size: list of ints
        start size of the cell
    factor: int
        factor to increase the size of the cell along
        the surface normal direction
    n_dots: int
        number of images along the gamma line
    relax: bool
        flag to perform relaxation
    fmax: float
        maximum force value for relaxation
    return_images: bool
        flag to control if the atomic configurations are returned
        together with the results

    Returns
    -------
    deltas: np.array
        shift distance of every image in Angstroms
    totens: np.array
        gamma surface energy in eV / Angstroms^2
    images: list of ase.Atoms
            images along the gamma surface. Returned if return_images is True
    """

    from ase.optimize import LBFGSLineSearch

    if unit_cell.calc is None:
        if calc is None:
            raise RuntimeError("Please set atoms calculator or provide calc")
        else:
            unit_cell.calc = calc

    size = np.array(size)
    directions = np.array([0, 1, 2])

    period = unit_cell.cell[shift_dir, shift_dir]
    size[surface] *= factor
    slab = unit_cell * size.tolist()

    top_mask = slab.positions.T[surface] > slab.cell[surface, surface] / 2.0

    surface_direction = directions == surface
    slab.pbc = (~surface_direction).tolist()
    slab.center(axis=surface, vacuum=10)

    images = []
    totens = []
    deltas = []

    for delta in np.linspace(0.0, period, num=n_dots):

        image = slab.copy()
        image.positions[:, shift_dir][top_mask] += delta

        select_all = np.full_like(image, True, dtype=bool)
        image.set_constraint(
            FixedLineAtoms(select_all, surface_direction.astype(int)))
        image.calc = unit_cell.calc

        if image.get_forces().max() < fmax:
            raise RuntimeError(
                "Initial max force is smaller than fmax! Check surface direction")

        if relax:
            opt = LBFGSLineSearch(image)
            opt.run(fmax=fmax)
            images.append(image)

        deltas.append(delta)
        totens.append(image.get_potential_energy())

    totens = np.array(totens)
    totens -= totens[0]

    surface_area_dirs = directions[~(directions == surface)]
    surface_area = (slab.cell.lengths()[surface_area_dirs[0]] *
                    slab.cell.lengths()[surface_area_dirs[1]])

    totens /= surface_area  # results in eV/A^2

    if return_images:
        return np.array(deltas), totens, images
    else:
        return np.array(deltas), totens
