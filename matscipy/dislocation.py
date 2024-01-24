#
# Copyright 2019, 2021 Lars Pastewka (U. Freiburg)
#           2018-2023 Petr Grigorev (Warwick U.)
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

"""Tools for studying structure and movement of dislocations."""

import numpy as np

from abc import ABCMeta
import warnings

from scipy.optimize import minimize

from ase.lattice.cubic import (BodyCenteredCubic, FaceCenteredCubic,
                               Diamond, SimpleCubicFactory)
from ase.constraints import FixAtoms, StrainFilter
from ase.optimize import FIRE
from ase.optimize.precon import PreconLBFGS
from ase.build import bulk, stack
from ase.calculators.lammpslib import LAMMPSlib
from ase.units import GPa  # unit conversion
from ase.io import read

from matscipy.neighbours import neighbour_list, mic
from matscipy.elasticity import fit_elastic_constants
from matscipy.elasticity import Voigt_6x6_to_full_3x3x3x3
from matscipy.elasticity import cubic_to_Voigt_6x6, coalesce_elastic_constants
from matscipy.utils import validate_cubic_cell, points_in_polygon2D


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
        Potential cutoff for Marinica potentials for FS cutoff = 4.4
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

    cutoff - potential cutoff for Marinica potentials for FS cutoff = 4.4

    symbol : string
        Symbol of the element to pass to ase.lattice.cubic.SimpleCubicFactory
        default is "W" for tungsten
    '''
    from atomman import ElasticConstants
    from atomman.defect import Stroh
    # Create a Stroh object with junk data
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

    # shift to make the zeros of the cell between the atomic planes
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

    # factor 2 to make sure odd number of images is translated
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
    bulk.positions[(y < 0.0) & (x < X_midplane_shift)] -= [alat * np.sqrt(3.0) / 2.0, 0.0, 0.0]
    # make the dislocation extra half plane center
    bulk.positions += [(1.0/3.0)*alat*np.sqrt(3.0)/2.0, 0.0, 0.0]

    return ED, bulk


def plot_vitek(dislo, bulk,
               alat=3.16, plot_axes=None, xyscale=10):
    """
    Plots vitek map from ase configurations.

    Parameters
    ----------
    dislo : ase.Atoms
        Dislocation configuration.
    bulk : ase.Atoms
        Corresponding bulk configuration for calculation of displacements.
    alat : float
        Lattice parameter for calculation of neighbour list cutoff.
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

    # plot window is +-10 angstroms from center in x,y directions,
    # and one Burgers vector thickness along z direction
    x, y, _ = bulk.positions.T

    plot_range = np.array([[x.mean() - xyscale, x.mean() + xyscale],
                          [y.mean() - xyscale, y.mean() + xyscale],
                          [-0.1, alat * 3.**(0.5) / 2.]])

    # This scales arrows such that b/2 corresponds to the
    # distance between atoms on the plot
    plot_scale = 1.885618083

    _ = differential_displacement(base_system, disl_system,
                                  burgers,
                                  cutoff=neighborListCutoff,
                                  xlim=plot_range[0],
                                  ylim=plot_range[1],
                                  zlim=plot_range[2],
                                  matplotlib_axes=plot_axes,
                                  plot_scale=plot_scale)


def show_NEB_configurations(images, bulk, xyscale=7,
                            show=True, core_positions=None):
    """
    Plots Vitek differential displacement maps for the list of images
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
                          symbol="W",
                          verbose=True,
                          fmax=1e-4,
                          smax=1e-3):
    """
    return lattice parameter, and cubic elastic constants: C11, C12, 44
    using matscipy function
    pot_path - path to the potential

    symbol : string
        Symbol of the element to pass to ase.lattice.cubic.SimpleCubicFactory
        default is "W" for tungsten
    """

    unit_cell = bulk(symbol, cubic=True)

    if (pot_path is not None) and (calculator is None):
        # create lammps calculator with the potential
        lammps = LAMMPSlib(lmpcmds=["pair_style eam/fs",
                           "pair_coeff * * %s W" % pot_path],
                           atom_types={'W': 1}, keep_alive=True)
        calculator = lammps

    unit_cell.calc = calculator

    #   simple calculation to get the lattice constant and cohesive energy
    #    alat0 = W.cell[0][1] - W.cell[0][0]
    sf = StrainFilter(unit_cell)
    # or UnitCellFilter(W)
    # -> to minimise wrt pos, cell
    opt = PreconLBFGS(sf, precon=None, logfile="-" if verbose else None)
    opt.run(fmax=fmax, smax=smax)  # max force in eV/A
    
    alat = unit_cell.cell.lengths()[0]
    #    print("a0 relaxation %.4f --> %.4f" % (a0, a))
    #    e_coh = W.get_potential_energy()
    #    print("Cohesive energy %.4f" % e_coh)

    Cij, Cij_err = fit_elastic_constants(unit_cell,
                                         symmetry="cubic",
                                         delta=delta,
                                         verbose=verbose)

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


def make_screw_cyl_kink(alat, C11, C12, C44, cylinder_r=40,
                        kink_length=26, kind="double", **kwargs):
    """
    Function to create kink configuration based on make_screw_cyl() function.
    Double kink configuration is in agreement with
    quadrupoles in terms of formation energy.
    Single kink configurations provide correct and stable structure,
    but formation energy is not accessible?

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

    (disloc_ini,
     disloc_fin,
     bulk_ini) = make_barrier_configurations((alat, C11, C12, C44),
                                             cylinder_r=cylinder_r,
                                             **kwargs)

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

        # make sure all the atoms are removed and cell is modified
        # for the bulk as well.
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

        # make sure all the atoms are removed and cell is modified
        # for the bulk as well.
        large_bulk.cell[2][0] -= cent_x
        large_bulk.cell[2][2] -= 1.0 * b / 3.0
        large_bulk = large_bulk[left_kink_mask]
        for constraint in kink.constraints:
            large_bulk.set_constraint(constraint)

    else:
        raise ValueError('Kind must be "right", "left" or "double"')

    return kink, reference_straight_disloc, large_bulk


def slice_long_dislo(kink, kink_bulk, b):
    """Function to slice a long dislocation configuration to perform
       dislocation structure and core position analysis

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
                                                   shift=n2_shift * C2_quadrupole + n1_shift * C1_quadrupole)

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


def make_screw_quadrupole_kink(alat, kind="double",
                               n1u=5, kink_length=20, symbol="W"):
    """Generates kink configuration using make_screw_quadrupole() function
       works for BCC structure.
       The method is based on paper
       https://doi.org/10.1016/j.jnucmat.2008.12.053

    Parameters
    ----------
    alat : float
        Lattice parameter of the system in Angstrom.
    kind : string
        kind of the kink: right, left or double
    n1u : int
        Number of lattice vectors for the quadrupole cell
        (make_screw_quadrupole() function)
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

    (ini_disloc_quadrupole,
     W_bulk, _, _) = make_screw_quadrupole(alat, n1u=n1u,
                                           left_shift=0.0,
                                           right_shift=0.0,
                                           symbol=symbol)

    (fin_disloc_quadrupole,
     W_bulk, _, _) = make_screw_quadrupole(alat, n1u=n1u,
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

        # left kink is created the kink vector is in negative x direction
        # assuming (x, y, z) is right group of vectors
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

        # double kink is double length
        large_bulk = W_bulk * [1, 1, 2 * kink_length]

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
            raise RuntimeError('Self-consistency did ' +
                               'not converge in 10 cycles')
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

    input_crystal_structures = {"bcc": DislocationAnalysisModifier.Lattice.BCC,
                                "fcc": DislocationAnalysisModifier.Lattice.FCC,
                                "diamond": DislocationAnalysisModifier.Lattice.CubicDiamond}

    data = ase_to_ovito(dxa_disloc)
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(ReplicateModifier(num_z=replicate_z))
    dxa = DislocationAnalysisModifier(
          input_crystal_structure=input_crystal_structures[structure.lower()])
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
    Returns a mask of atoms that have at least
    one other atom closer than distance
    """
    mask = atoms.get_all_distances() < distance
    duplicates = np.full_like(atoms, False)
    for i, row in enumerate(mask):
        if any(row[i+1:]):
            duplicates[i] = True
    # print(f"found {duplicates.sum()} duplicates")
    return duplicates.astype(np.bool)

class AnisotropicDislocation:
    """
    Displacement and displacement gradient field of straight dislocation 
    in anisotropic elastic media. Ref: pp. 467 in J.P. Hirth and J. Lothe, 
    Theory of Dislocations, 2nd ed. Similar to class `CubicCrystalDislocation`.
    """

    def __init__(self, axes, slip_plane, disloc_line, burgers,
                 C11=None, C12=None, C44=None, C=None):
        """
        Setup a dislocation in a cubic crystal. 
        C11, C12 and C44 are the elastic constants in the Cartesian geometry.
        The arg 'axes' is a 3x3 array containing axes of frame of reference (eg.crack system).
        The dislocation parameters required are the slip plane normal ('slip_plane'), the 
        dislocation line direction ('disloc_line') and the Burgers vector ('burgers').
        """

        # normalize axes of the cell (for NCFlex: crack system coordinates) 
        A = np.array([ np.array(v)/np.linalg.norm(v) for v in axes ])

        # normalize axis of dislocation coordinates
        n = np.array(slip_plane) ; n =  n / np.linalg.norm(n) 
        xi = np.array(disloc_line) ; xi = xi / np.linalg.norm(xi) 
        m = np.cross(n, xi) # dislocation glide direction
        m = m / np.linalg.norm(m)    
        
        # Rotate vectors from dislocation system to crack coordinates
        m = np.einsum('ij,j', A, m)
        n = np.einsum('ij,j', A, n)
        xi = np.einsum('ij,j', A, xi)
        b = np.einsum('ij,j', A, burgers)

        
        
        # define elastic constant in Voigt notation
        Cijkl = coalesce_elastic_constants(C11, C12, C44, C, convention="Cijkl")

        # rotate elastic matrix
        cijkl = np.einsum('ig,jh,ghmn,km,ln', \
            A, A, Cijkl, A, A)
    
        # solve the Stroh sextic formalism: the same as Stroh.py from atomman
        # this part can also be replaced by calling Stroh.py from atomman
        mm = np.einsum('i,ijkl,l', m, cijkl, m)
        mn = np.einsum('i,ijkl,l', m, cijkl, n)
        nm = np.einsum('i,ijkl,l', n, cijkl, m)
        nn = np.einsum('i,ijkl,l', n, cijkl, n)

        # TODO: Replace with np.linalg.solve as it is more stable than np.linalg.inv
        nninv = np.linalg.inv(nn)
        mn_nninv = np.dot(mn, nninv)      
        #mn_nninv_2 = np.linalg.solve(nn.T, mn.T).T # works, consistent with mn_nninv
        #nninv_2 = np.linalg.solve(mn, mn_nninv_2) # TODO: Resolve LinAlgError: Singular matrix

        N = np.zeros((6,6), dtype=float)
        N[0:3,0:3] = -np.dot(nninv, nm)
        N[0:3,3:6] = -nninv
        N[3:6,0:3] = -(np.dot(mn_nninv, nm) - mm)
        N[3:6,3:6] = -mn_nninv

        # slove the eigenvalue problem
        Np, Nv = np.linalg.eig(N)

        # The eigenvector Nv contains the vectors A and L.
        # Normalize A and L, such that 2*A*L=1 
        for i in range(0,6):
            norm = 2.0 * np.dot(Nv[0:3, i], Nv[3:6, i])
            Nv[0:3, i] /= np.sqrt(norm)
            Nv[3:6, i] /= np.sqrt(norm)

        # Store key variables
        self.m = m
        self.n = n
        self.xi = xi
        self.b = b
        self.Np = Np
        self.Nv = Nv
    

    def displacement(self, bulk_positions, center=np.array([0.0,0.0,0.0])):

        """
        Displacement field of a straight dislocation. Currently only for 2D, can be extended.
        Parameters
        ----------
        bulk_positions : array
            Positions of atoms in the bulk cell (with axes == self.axes)
        center : 3x1 array
            Position of the dislocation core within the cell
        Returns
        -------
        disp : array
            Stroh displacements along the crack running direction (axes[0]) and crack plane normal (axes[1]).
        """

        # Get atomic positions wrt dislocation core in Cartesian coordinates
        coordinates = [ vec-center for vec in bulk_positions]

        # calculation
        signs = np.sign(np.imag(self.Np))
        signs[np.where(signs==0.0)] = 1.0
        A = self.Nv[0:3,:]
        L = self.Nv[3:6,:]
        D = np.einsum('i,ij', self.b, L)
        constant_factor = signs * A * D

        eta = (np.expand_dims(np.einsum('i,ji', self.m, coordinates), axis=1)
            + np.outer(np.einsum('i,ji', self.n, coordinates), self.Np))

        # get the displacements
        disp = ((1.0/(2.0 * np.pi * 1.0j))
            * np.einsum('ij,kj', np.log(eta), constant_factor))

        return np.real(disp) 


    def deformation_gradient(self, bulk_positions, center=np.array([0.0,0.0,0.0]), return_2D=False):
        """
        3D displacement gradient tensor of the dislocation. 

        Parameters
        ----------
        bulk_positions : array
            Positions of atoms in the bulk cell (with axes == self.axes)
        center : 3x1 array
            Position of the dislocation core within the cell

        Returns
        -------
        du_dx, du_dy, du/dz, dv_dx, dv_dy, dv/dz, dw/dx, dw/dy, dw/dz : 1D arrays
            Displacement gradients:
            dx, dy, dz: changes in dislocation core position along the three axes of self.axes
            du, dv, dw: changes in displacements of atoms along the three axes of self.axes, in response to changes in dislocation core position
        """

        # Get atomic positions wrt dislocation core in Cartesian coordinates
        coordinates = [ vec-center for vec in bulk_positions]

        signs = np.sign(np.imag(self.Np))
        signs[np.where(signs==0.0)] = 1.0
        
        A = self.Nv[0:3,:]
        L = self.Nv[3:6,:]
        D = signs * np.einsum('i,ij', self.b, L)

        eta = (np.expand_dims(np.einsum('i,ji', self.m, coordinates), axis=1)
               + np.outer(np.einsum('i,ji', self.n, coordinates), self.Np))
        
        pref = (1.0/(2.0 * np.pi * 1.0j))

        # Compute the displacement gradient components
        nat = len(coordinates)
        grad3D_T = np.zeros((3,3,nat))
        for i in range(3):
            for j in range(3):
                grad3D_T[i,j,:] = np.real(( pref * np.einsum('ij,j', 1/eta, (self.m[j] + self.Np * self.n[j]) * A[i,:] * D) ))

        # Add unity matrix along the diagonal block, to turn this into the deformation gradient tensor.
        for i in range(3):
            grad3D_T[i,i,:] += np.ones(nat)
        
        # Transpose to get the correct shape
        # Form: np.transpose([[du_dx, du_dy, du_dz], [dv_dx, dv_dy, dv_dz], [dw_dx, dw_dy, dw_dz]])
        grad3D = np.transpose(grad3D_T)
        
        if return_2D:
            # Form: np.transpose([[du_dx, du_dy], [dv_dx, dv_dy]])
            grad2D = grad3D[:, 0:2, 0:2]
            return grad2D

        return grad3D


class CubicCrystalDislocation(metaclass=ABCMeta):
    '''
    Abstract base class for modelling a single dislocation
    '''

    # Mandatory Attributes of CubicCrystalDislocation with no defaults
    # These should be set by the child dislocation class, or by CubicCrystalDislocation.__init__
    # (see https://stackoverflow.com/questions/472000/usage-of-slots for more details on __slots__)
    __slots__ = ("burgers_dimensionless", "unit_cell_core_position_dimensionless",
                 "glide_distance_dimensionless", "crystalstructure", "axes",
                 "C", "alat", "unit_cell", "name")

    # Attributes with defaults
    # These may be overridden by child classes
    parity = np.zeros(2)
    n_planes = 3
    self_consistent = True
    pbc = [True, True, True]
    stroh = None
    ADstroh = None

    # Available solvers for atomic displacements
    avail_methods = [
        "atomman", "adsl"
        ]
    
    # Attempt to load the atomman module once, 
    try:
        import atomman as atm
        atomman = atm
    except ImportError:
        atomman = None

    def __init__(self, a, C11=None, C12=None, C44=None, C=None, symbol="W"):
        """
        This class represents a dislocation in a cubic crystal

        The dislocation is defined by the crystal unit cell,
        elastic constants, crystal axes,
        burgers vector and optional shift and parity vectors.

        The crystal used as the base for constructing dislocations can be defined 
        by a lattice constant and a chemical symbol. For multi-species systems, an ase 
        Atoms object defining the cubic unit cell can instead be used.

        Elastic constants can be defined either through defining C11, C12 and C44, or by passing 
        the full 6x6 elasticity matrix

        Parameters
        ----------
        a : lattice constant OR cubic ase.atoms.Atoms object
            Lattice constant passed to ase.lattice.cubic constructor or
            Atoms object used by ase.build.cut to get unit cell in correct orientation
        C11, C12, C44 : float
            Elastic Constants (in GPa) for cubic symmetry
        C : np.array
            Full 6x6 elasticity matrix (in GPa)
        symbol : str
            Chemical symbol used to construct unit cell (if "a" is a lattice constant)


        Attributes
        ------------------------------
        alat : float
            lattice parameter (in A)
        unit_cell : ase.atoms.Atoms object
            bulk cell used to generate dislocations
        axes : np.array of float
            unit cell axes (b is normally along z direction)
        burgers_dimensionless : np.array of float
            burgers vector of the dislocation class, in a dimensionless alat=1 system
        burgers : np.array of float
            burgers vector of the atomistic dislocation
        unit_cell_core_position : np.array of float
            dislocation core position in the unit cell used to shift atomic positions to
            make the dislocation core the center of the cell
        parity : np.array
        glide_distance : float
            distance (in A) to the next equivalent core position in the glide direction
        n_planes : int
            number of non equivalent planes in z direction
        self_consistent : float
            default value for the displacement calculation
        crystalstructure : str
            Name of basic structure defining atomic geometry ("fcc", "bcc", "diamond")
        """

        # Create copies of immutable class attributes
        # Prevents side effects when E.G. changing burgers vector of an instance
        # Also makes changing cls.var and instance.var work as expected for these variables
        self.burgers_dimensionless = self.burgers_dimensionless.copy()

        # Skip these for dissociated dislocations, as these operate via classmethod properties
        if not issubclass(self.__class__, CubicCrystalDissociatedDislocation):
            self.axes = self.axes.copy()
            self.unit_cell_core_position_dimensionless = self.unit_cell_core_position_dimensionless.copy()
            self.parity = self.parity.copy()
        
            self.alat, self.unit_cell = validate_cubic_cell(a, symbol, self.axes, self.crystalstructure, self.pbc)

            # Empty dict for solver methods (e.g. AnisotropicDislocation.displacements)
            self.solvers = {
                m : None for m in self.avail_methods
            }

        # Sort out elasticity matrix into 6x6 convention (as we know the system is cubic)
        self.C = coalesce_elastic_constants(C11, C12, C44, C, convention="Cij")

    def init_solver(self, method="atomman"):
        '''
        Run the correct solver initialiser

        Solver init should take only the self arg, and set self.solver[method] to point to the function 
        to calculate displacements

        e.g. self.solvers["atomman"] = self.stroh.displacement
        '''
        solver_initters = {
            "atomman" : self.init_atomman,
            "adsl" : self.init_adsl
        }

        # Execute init routine
        solver_initters[method]()
    
    def get_solvers(self, method="atomman"):
        '''
        Get list of dislocation displacement solvers

        As CubicCrystalDislocation models a single core, there is only one solver,
        therefore solvers is a len(1) list

        Parameters
        ----------
        use_atomman: bool
            Use the Stroh solver included in atomman (requires atomman package) to
            solve for displacements, or use the AnisotropicDislocation class

        Returns
        -------
        solvers: list
            List of functions f(positions) = displacements
        '''
        has_atomman = False if self.atomman is None else True

        if type(method) != str:
            if has_atomman:
                warnings.warn("non-string method arg, falling back to method='atomman'")
                method = "atomman"
            else:
                warnings.warn("non-string method arg, falling back to method='adsl'")
                method = "adsl"

        if method.lower() not in self.avail_methods:
            # Passed an invalid method
            if has_atomman:
                warnings.warn(f"method='{method}' not one of {self.avail_methods}, falling back to method='atomman'")
                method = "atomman"
            else:
                warnings.warn(f"method='{method}' not one of {self.avail_methods}, falling back to method='adsl'")
                method = "adsl"

        if method.lower() == "atomman" and not has_atomman:
            # Atomman import failed
            warnings.warn("Import of atomman failed, falling back to method='adsl'")
            method="adsl"

        if self.solvers[method] is None:
            # Method needs initialising
            self.init_solver(method)
        
        return [self.solvers[method]]

    def init_atomman(self):
        '''
        Init atomman stroh solution solver (requires atomman)
        '''
        c = self.atomman.ElasticConstants(Cij=self.C)
        self.stroh = self.atomman.defect.Stroh(c, burgers=self.burgers, axes=self.axes)

        self.solvers["atomman"] = self.stroh.displacement

    def init_adsl(self):
        '''
        Init adsl (Anisotropic DiSLocation) solver
        '''
        # Extract consistent parameters
        axes = self.axes
        slip_plane = axes[1].copy() 
        disloc_line = axes[2].copy() 

        # Setup AnistoropicDislocation object
        self.ADstroh = AnisotropicDislocation(axes, slip_plane, disloc_line, 
                                           self.burgers, C=self.C)
        
        self.solvers["adsl"] = self.ADstroh.displacement

    # @property & @var.setter decorators used to ensure var and var_dimensionless don't get out of sync
    @property
    def burgers(self):
        return self.burgers_dimensionless * self.alat
    
    @burgers.setter
    def burgers(self, burgers):
        self.burgers_dimensionless = burgers / self.alat

    def set_burgers(self, burgers):
        self.burgers = burgers

    def invert_burgers(self):
        '''
        Modify dislocation to produce same dislocation with opposite burgers vector
        '''
        self.burgers_dimensionless *= -1

    @property
    def unit_cell_core_position(self):
        return self.unit_cell_core_position_dimensionless * self.alat
    
    @unit_cell_core_position.setter
    def unit_cell_core_position(self, position):
        self.unit_cell_core_position_dimensionless = position / self.alat

    @property
    def glide_distance(self):
        return self.glide_distance_dimensionless * self.alat
    
    @glide_distance.setter
    def glide_distance(self, distance):
        self.glide_distance_dimensionless = distance / self.alat

    def _build_supercell(self, targetx, targety):
        '''
        Build supercell in 2D from self.unit_cell, with target dimensions (targetx, targety), specified in Angstrom.
        Supercell is constructed to have cell lengths >= the target value (i.e. cell[0, 0] >= targetx)

        Parameters
        ----------
        targetx: float
            Target length (in Ang) of the cell in the "x" direction (i.e. cell[0, 0])
        targety: float
            Target length (in Ang) of the cell in the "y" direction (i.e. cell[1, 0])

        
        Returns a supercell of self.unit_cell with cell[0, 0] >= targetx and cell[1, 1] >= targety
        '''

        base_cell = self.unit_cell

        xreps = np.ceil(targetx / (base_cell.cell[0, 0] - 1e-2)).astype(int)
        yreps = np.ceil(targety / (base_cell.cell[1, 1] - 1e-2)).astype(int)

        sup = base_cell.copy() * (xreps, yreps, 1)
        return sup

    def _build_bulk_cyl(self, radius, core_positions, fix_rad, extension,
                        self_consistent, method, verbose):
        '''
        Build bulk cylinder config from args supplied by self.build_cylinder

        Parameters
        ----------
        radius: float
            Radius for a radial mask for cyl config
        core_positions: np.array
            Positions of all dislocation cores
            Should be of shape (ncores, 3)
            Can add fictitious core positions in order to extend cyl as needed
            e.g. for glide configs, etc
        fix_rad: float
            Radius for fixed atoms constraints
        extension: np.array
            (N, 3) array of vector extensions from each core.
            Will add a fictitious core at position 
            core_positions[i, :] + extension[i, :], for the 
            purposes of adding extra atoms

        Returns
        -------
        cyl: ase.Atoms
            Bulk cyl, cut down based on radius, complete
            with FixAtoms constraints
        '''
        from matscipy.utils import radial_mask_from_polygon2D


        if self_consistent is None:
            self_consistent = self.self_consistent

        if len(core_positions.shape) == 1:
            core_positions = core_positions[np.newaxis, :]

        if len(extension.shape) == 1:
            extension = extension[np.newaxis, :]

        if np.all(extension.shape != core_positions.shape):            
            if extension.shape[0] < core_positions.shape[0] and \
                extension.shape[1] == core_positions.shape[1]:
                # Too few extensions passed, assume that the extensions are for the first N cores
                # Pad extension with zeros
                extension = np.vstack([extension, np.zeros((
                    core_positions.shape[0] - extension.shape[0], core_positions.shape[1]
                ))])
            else:
                raise ValueError(f"{extension.shape=} does not match {core_positions.shape=}")

        # Only care about non-zero extensions
        exts = extension + core_positions

        xmax = np.max([np.max(core_positions[:, 0]), np.max(exts[:, 0])])
        xmin = np.min([np.min(core_positions[:, 0]), np.min(exts[:, 0])])

        ymax = np.max([np.max(core_positions[:, 1]), np.max(exts[:, 1])])
        ymin = np.min([np.min(core_positions[:, 1]), np.min(exts[:, 1])])

        xlen = xmax - xmin + 2*radius
        ylen = ymax - ymin + 2*radius

        sup = self._build_supercell(xlen, ylen)

        center = np.diag(sup.cell) / 2

        # Shift everything so that the dislocations are reasonably central

        shift = center - 0.5 * (np.array([xmax, ymax, 0]) + np.array([xmin, ymin, 0]))

        pos = sup.get_positions()
        pos += shift
        sup.set_positions(pos)
        sup.wrap()

        new_core_positions = core_positions.copy()

        new_core_positions += shift

        # Mask out extensions that are all zeros
        nonzero_exts = np.sum(np.abs(extension), axis=-1) > 0.0

        exts = exts[nonzero_exts, :]

        mask_positions = np.vstack([new_core_positions, exts + shift])

        idxs = np.argsort(mask_positions, axis=0)
        mask_positions = np.take_along_axis(mask_positions, idxs, axis=0)
        
        mask = radial_mask_from_polygon2D(sup.get_positions(), mask_positions, radius, inner=True)

        cyl = sup[mask]

        # disloc is a copy of bulk with displacements applied
        disloc = cyl.copy()

        disloc.positions += self.displacements(cyl.positions, new_core_positions,
                                               self_consistent=self_consistent, method=method,
                                               verbose=verbose)

        if fix_rad:
            fix_mask = ~radial_mask_from_polygon2D(cyl.get_positions(), mask_positions, radius - fix_rad, inner=True)

            fix_atoms = FixAtoms(mask=fix_mask)
            disloc.set_constraint(fix_atoms)
            disloc.arrays["fix_mask"] = fix_mask

        return cyl, disloc, new_core_positions

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

    def self_consistent_displacements(self, solvers, bulk_positions, core_positions, 
                                      tol=1e-6, max_iter=100, verbose=True, mixing=0.8):
        '''
        Compute dislocation displacements self-consistently, with max_iter capping the number of iterations
        Each dislocation core uses a separate solver, which computes the displacements associated with positions 
        relative to that core (i.e. that the core is treated as lying at [0, 0, 0])

        Parameters
        ----------
        solvers: list
            List of functions, solvers[i] computes the displacement field for the dislocation
            defined by core_positions[i, :]
        bulk_positions: np.array
            (N, 3) array of positions of atoms in the bulk configuration
        core_positions: np.array
            Positions of each dislocation core (shape = (ncores, 3))
        tol: float
            Displacement tolerance for the convergence of the self-consistent loop
        max_iter: int
            Maximum number of iterations of the self-consistent cycle to run before throwing 
            an exception
        verbose: bool
            Enable/Disable printing progress of the self-consistent cycle each iteration

        Returns
        -------
        displacements: np.array
        shape (N, 3) array of displacements to apply to bulk_positions to form
        the dislocation structure

        Raises a RuntimeError if max_iter is reached without convergence
        '''
        if len(core_positions.shape) == 1:
            core_positions = core_positions[np.newaxis, :]

        ncores = core_positions.shape[0]

        disp1 = np.zeros_like(bulk_positions)
        for i in range(ncores):
            disp1 += solvers[i](bulk_positions - core_positions[i, :]).real
        
        if max_iter == 0:
            return disp1
        
        # Self-consistent calculation
        res = np.inf
        for i in range(max_iter):
            disloc_positions = bulk_positions + disp1
            
            disp2 = np.zeros_like(bulk_positions)
            for j in range(ncores):
                disp2 += solvers[j](disloc_positions - core_positions[j, :]).real

            # Add mixing of previous iteration
            disp2 = mixing * disp2 + (1-mixing) * disp1

            res = np.abs(disp1 - disp2).max()
            disp1 = disp2
            if verbose:
                print('disloc SCF', i, '|d1-d2|_inf =', res)
            if res < tol:
                return disp2
            
        raise RuntimeError('Self-consistency ' +
                            f'did not converge in {max_iter} cycles')

    def displacements(self, bulk_positions, core_positions, method="atomman",
                      self_consistent=True, tol=1e-6, max_iter=100, verbose=True, mixing=0.5):
        '''
        Compute dislocation displacements self-consistently, with max_iter capping the number of iterations
        Each dislocation core uses a separate solver, which computes the displacements associated with positions 
        relative to that core (i.e. that the core is treated as lying at [0, 0, 0])

        Parameters
        ----------
        bulk_positions: np.array
            (N, 3) array of positions of atoms in the bulk configuration
        core_positions: np.array
            Positions of each dislocation core (shape = (ncores, 3))
        use_atomman: bool
            Use the Stroh solver included in atomman (requires atomman package) to
            solve for displacements, or use the AnisotropicDislocation class
        self_consistent: bool
            Whether to detemine the dislocation displacements in a self-consistent manner
            (self_consistent=False is equivalent to max_iter=0)
        tol: float
            Displacement tolerance for the convergence of the self-consistent loop
        max_iter: int
            Maximum number of iterations of the self-consistent cycle to run before throwing 
            an exception
        verbose: bool
            Enable/Disable printing progress of the self-consistent cycle each iteration

        Returns
        -------
        displacements: np.array
        shape (N, 3) array of displacements to apply to bulk_positions to form
        the dislocation structure

        Raises a RuntimeError if max_iter is reached without convergence
        '''
        
        solvers = self.get_solvers(method)

        if not self_consistent:
            max_iter = 0

        disp = self.self_consistent_displacements(solvers, bulk_positions, core_positions, 
                                                  tol, max_iter, verbose, mixing)

        return disp
    

    def build_cylinder(self, radius,
                       core_position=np.array([0., 0., 0.]),
                       extension=np.array([0, 0, 0]),
                       fix_width=10.0, self_consistent=None,
                       method="atomman",
                       verbose=True):
        '''
        Build dislocation cylinder for single dislocation system

        Parameters
        ----------
        radius: float
            Radius of bulk surrounding the dislocation core
        core_position: np.array
            Vector offset of the core position from default site
        extension: np.array
            Add extra bulk to the system, to also surround a fictitious core
            that is at position core_position + extension
        fix_width: float
            Defines a region to apply the FixAtoms ase constraint to
            Fixed region is given by (radius - fix_width) <= r <= radius,
            where the position r is measured from the dislocation core
            (or from the glide path, if extra_glide_widths is given)
        self_consistent: bool
            Controls whether the displacement field used to construct the dislocation is converged 
            self-consistently. If None (default), the value of self.self_consistent is used
        use_atomman: bool
            Use the Stroh solver included in atomman (requires atomman package) to
            solve for displacements, or use the AnisotropicDislocation class
        '''

        core_positions = np.array([
            core_position + self.unit_cell_core_position
        ])

        bulk, disloc, core_positions = self._build_bulk_cyl(radius, core_positions, fix_width,
                                                            extension, self_consistent, method, verbose)

        disloc.info["core_positions"] = [list(core_positions[0, :])]
        disloc.info["burgers_vectors"] = [list(self.burgers)]
        disloc.info["dislocation_types"] = [self.name]
        disloc.info["dislocation_classes"] = [str(self.__class__)]

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
                                            extension=-final_core_position,
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
    
    @staticmethod
    def view_cyl(system, scale=0.5, CNA_color=True, add_bonds=False,
                     line_color=[0, 1, 0], disloc_names=None, hide_arrows=False):
        '''
        NGLview-based visualisation tool for structures generated from the dislocation class

        Parameters
        ----------
        system: ase.Atoms
            Dislocation system to view
        scale: float
            Adjust radiusScale of add_ball_and_stick (add_bonds=True) or add_spacefill (add_bonds=False)
        CNA_color: bool
            Turn on atomic/bond colours based on Common Neighbour Analysis structure identification
        add_bonds: bool
            Add atomic bonds
        line_color: list or list of lists
            [R, G, B] values for the colour of dislocation lines
            if a list of lists, line_color[i] defines the colour of the ith dislocation (see structure.info["dislocation_types"])
        disloc_names: list of str
            Manual override for the automatic naming of the dislocation lines 
            (see structure.info["dislocation_types"] for defaults)
        hide_arrows: bool
            Hide arrows for placed on the dislocation lines
        '''

        def _plot_disloc_line(view, disloc_pos, disloc_name, z_length, color=[0, 1, 0]):
            '''Add dislocation line to the view as a cylinder and two cones.
            The cylinder is hollow by default so the second cylinder is needed to close it.
            In case partial distance is provided, two dislocation lines are added and are shifter accordingly.
            
            Parameters
            ----------
            view: NGLview
                NGLview plot of the atomic structure
            disloc_pos: list or array
                Position of the dislocation core
            color: list
                [R, G, B] colour of the dislocation line
            '''

            view.shape.add_cylinder((disloc_pos[0], disloc_pos[1], -2.0), 
                                    (disloc_pos[0], disloc_pos[1], z_length - 0.5),
                                    color,
                                    [0.3],
                                    disloc_name)
            
            view.shape.add_cone((disloc_pos[0], disloc_pos[1], -2.0), 
                                (disloc_pos[0], disloc_pos[1], 0.5),
                                color,
                                [0.3],
                                disloc_name)
            
            view.shape.add_cone((disloc_pos[0], disloc_pos[1], z_length - 0.5), 
                                (disloc_pos[0], disloc_pos[1], z_length + 1.0),
                                color,
                                [0.55],
                                disloc_name)
    
        from nglview import show_ase, ASEStructure
        from matscipy.utils import get_structure_types
        # custom tooltip for nglview to avoid showing molecule and residue names
        # that are not relevant for dislocation structures
        tooltip_js = """
                        this.stage.mouseControls.add('hoverPick', (stage, pickingProxy) => {
                            let tooltip = this.stage.tooltip;
                            if(pickingProxy && pickingProxy.atom && !pickingProxy.bond){
                                let atom = pickingProxy.atom;
                                if (atom.structure.name.length > 0){
                                    tooltip.innerText = atom.atomname + " atom: " + atom.structure.name;
                                } else {
                                    tooltip.innerText = atom.atomname + " atom";
                                }
                            } else if (pickingProxy && pickingProxy.bond){
                                let bond = pickingProxy.bond;
                                if (bond.structure.name.length > 0){
                                tooltip.innerText = bond.atom1.atomname + "-" + bond.atom2.atomname + " bond: " + bond.structure.name;
                                } else {
                                    tooltip.innerText = bond.atom1.atomname + "-" + bond.atom2.atomname + " bond";
                                }
                            } else if (pickingProxy && pickingProxy.unitcell){
                                tooltip.innerText = "Unit cell";
                            }
                        });
                        """
        
        
        # Check system contains all keys in structure.info we need to plot the dislocations
        for expected_key in ["core_positions", "burgers_vectors", "dislocation_types", "dislocation_classes"]:
            
            if not expected_key in system.info.keys():
                raise RuntimeError(f"{expected_key} not found in system.info, regenerate system from self.build_cylinder()")
        
        # Validate line_color arg
        if type(line_color) in [list, np.array, tuple]:
            if type(line_color[0]) in [list, np.array, tuple]:
                # Given an RGB value per dislocation
                colours = line_color
            else:
                # Assume we're given a single RBG value for all dislocs
                colours = [line_color] * len(system.info["dislocation_types"])
                
        # Check if system contains a diamond dislocation (e.g. DiamondGlideScrew, Diamond90degreePartial)
        diamond_structure = "Diamond" in system.info["dislocation_classes"][0]

        atom_labels, structure_names, colors = get_structure_types(system, 
                                                                diamond_structure=diamond_structure)
        if disloc_names is None:
            disloc_names = system.info["dislocation_types"]

        disloc_positions = system.info["core_positions"]

        if len(disloc_names) != len(disloc_positions):
            raise RuntimeError("Number of dislocation positions is not equal to number of dislocation types/names")

        view = show_ase(system)
        view.hide([0])
        
        if add_bonds: # add bonds between all atoms to have bonds between structures
            component = view.add_component(ASEStructure(system), default_representation=False, name='between structures')
            component.add_ball_and_stick(cylinderOnly=True, radiusType='covalent', radiusScale=scale, aspectRatio=0.1)
        
        # Add structure and struct colours
        for structure_type in np.unique(atom_labels):
            # every structure type is a different component
            mask = atom_labels == structure_type
            component = view.add_component(ASEStructure(system[mask]), 
                                        default_representation=False, name=str(structure_names[structure_type]))
            if CNA_color:
                if add_bonds:
                    component.add_ball_and_stick(color=colors[structure_type], radiusType='covalent', radiusScale=scale)
                else:
                    component.add_spacefill(color=colors[structure_type], radiusType='covalent', radiusScale=scale)
            else:
                if add_bonds:
                    component.add_ball_and_stick(radiusType='covalent', radiusScale=scale)
                else:
                    component.add_spacefill(radiusType='covalent', radiusScale=scale)
                        
        component.add_unitcell()

        # Plot dislocation lines
        z_length = system.cell[2, 2]

        if not hide_arrows:
            for i in range(len(disloc_positions)):
                _plot_disloc_line(view, disloc_positions[i], disloc_names[i] + " dislocation line", z_length, colours[i])

        view.camera = 'orthographic'
        view.parameters = {"clipDist": 0}

        #view._remote_call("setSize", target="Widget", args=["400px", "300px"])
        view.layout.width = '100%'
        view.layout.height = '300px'
        view.center()

        view._js(tooltip_js)
        return view


class CubicCrystalDissociatedDislocation(CubicCrystalDislocation, metaclass=ABCMeta):
    '''
    Abstract base class for modelling dissociated dislocation systems
    '''
    # Inherits all slots from CubicCrystalDislocation as well
    __slots__ = ("left_dislocation", "right_dislocation")

    # Space for overriding the burgers vectors from cls.left_dislocation.burgers_dimensionless
    new_left_burgers = None
    new_right_burgers = None
    def __init__(self, a, C11=None, C12=None, C44=None, C=None, symbol="W"):
        """
        This class represents a dissociated dislocation in a cubic crystal
        with burgers vector b = b_left + b_right.

        Args:
            identical to CubicCrystalDislocation

        Raises:
            ValueError: If resulting burgers vector
                        burgers is not a sum of burgers vectors of
                        left and right dislocations.
            ValueError: If one of the properties of
                        left and righ dislocations are not the same.
        """

        if not (isinstance(self.left_dislocation, CubicCrystalDislocation) or \
                isinstance(self.left_dislocation, CubicCrystalDissociatedDislocation)):
            self.left_dislocation = self.left_dislocation(a, C11, C12, C44, C, symbol)
        if not (isinstance(self.right_dislocation, CubicCrystalDislocation) or \
                isinstance(self.right_dislocation, CubicCrystalDissociatedDislocation)):
            self.right_dislocation = self.right_dislocation(a, C11, C12, C44, C, symbol)

        # Change disloc burgers vectors, if requested
        if self.new_left_burgers is not None:
            self.left_dislocation.burgers_dimensionless = self.new_left_burgers.copy()
        if self.new_right_burgers is not None:
            self.right_dislocation.burgers_dimensionless = self.new_right_burgers.copy()
        
        # Set self.attrs based on left disloc attrs
        left_dislocation = self.left_dislocation
        right_dislocation = self.right_dislocation

        super().__init__(a, C11, C12, C44, C, symbol)

        # Validation of disloc inputs
        try:
            np.testing.assert_almost_equal(left_dislocation.burgers +
                                           right_dislocation.burgers,
                                           self.burgers)
        except AssertionError as error:
            print(error)
            raise ValueError("Burgers vectors of left and right dislocations " +
                             "do not add up to the desired vector")

        # checking that parameters of
        # left and right dislocations are the same
        try:
            assert left_dislocation.alat == right_dislocation.alat
            assert np.allclose(left_dislocation.C, right_dislocation.C)

            np.testing.assert_equal(left_dislocation.unit_cell.get_chemical_symbols(),
                                    right_dislocation.unit_cell.get_chemical_symbols())
            np.testing.assert_equal(left_dislocation.unit_cell.cell.cellpar(),
                                    right_dislocation.unit_cell.cell.cellpar())
            np.testing.assert_equal(left_dislocation.unit_cell.positions,
                                    right_dislocation.unit_cell.positions)

            np.testing.assert_equal(left_dislocation.axes,
                                    right_dislocation.axes)

            np.testing.assert_equal(left_dislocation.unit_cell_core_position,
                                    right_dislocation.unit_cell_core_position)

            np.testing.assert_equal(left_dislocation.parity,
                                    right_dislocation.parity)

            np.testing.assert_equal(left_dislocation.glide_distance,
                                    right_dislocation.glide_distance)

            assert left_dislocation.n_planes == right_dislocation.n_planes
            assert left_dislocation.self_consistent == right_dislocation.self_consistent

        except AssertionError as error:
            print("Parameters of left and right partials are not the same!")
            print(error)
            raise ValueError("Parameters of left and right" +
                             "partials must be the same")

    # classmethod properties that get props from left dislocation
    # Used so e.g. cls.crystalstructure is setup prior to __init__
    # as is the case with CubicCrystalDislocation

    @classmethod
    @property
    def crystalstructure(cls):
        return cls.left_dislocation.crystalstructure
    
    @classmethod
    @property
    def axes(cls):
        return cls.left_dislocation.axes
    
    @classmethod
    @property
    def unit_cell_core_position_dimensionless(cls):
        return cls.left_dislocation.unit_cell_core_position_dimensionless
    
    @classmethod
    @property
    def parity(cls):
        return cls.left_dislocation.parity
    
    
    @classmethod
    @property
    def n_planes(cls):
        return cls.left_dislocation.n_planes
    
    @classmethod
    @property
    def self_consistent(cls):
        return cls.left_dislocation.self_consistent
    
    @classmethod
    @property
    def glide_distance_dimensionless(cls):
        return cls.left_dislocation.glide_distance_dimensionless
    
    @property
    def alat(self):
        return self.left_dislocation.alat
    
    @property
    def unit_cell(self):
        return self.left_dislocation.unit_cell

    def invert_burgers(self):
        '''
        Modify dislocation to produce same dislocation with opposite burgers vector
        '''
        self.burgers_dimensionless *= -1

        self.left_dislocation.invert_burgers()
        self.right_dislocation.invert_burgers()

        # Flip dislocations
        self.left_dislocation, self.right_dislocation = self.right_dislocation, self.left_dislocation

    def get_solvers(self, method="atomman"):
        '''
        Overload of CubicCrystalDislocation.get_solvers
        Get list of dislocation displacement solvers

        Parameters
        ----------
        use_atomman: bool
            Use the Stroh solver included in atomman (requires atomman package) to
            solve for displacements, or use the AnisotropicDislocation class

        Returns
        -------
        solvers: list
            List of functions f(positions) = displacements
        '''
        solvers = self.left_dislocation.get_solvers(method)

        solvers.extend(
            self.right_dislocation.get_solvers(method)
        )

        return solvers

    def build_cylinder(self, radius, partial_distance=0,
                       core_position=np.array([0., 0., 0.]),
                       extension=np.array([[0., 0., 0.],
                                          [0., 0., 0.]]),
                       fix_width=10.0, self_consistent=None,
                       method="atomman",
                       verbose=True):
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
        use_atomman: bool
            Use the Stroh solver included in atomman (requires atomman package) to
            solve for displacements, or use the AnisotropicDislocation class
        extension: np.array
            Shape (2, 3) array giving additional extension vectors from each dislocation core.
            Used to add extra bulk, e.g. to set up glide configurations.
        """
        
        partial_distance_Angstrom = np.array(
            [self.glide_distance * partial_distance, 0.0, 0.0])
            

        core_positions = np.array([
            core_position + self.unit_cell_core_position,
            core_position + self.unit_cell_core_position + partial_distance_Angstrom
        ])

        bulk, disloc, core_positions = self._build_bulk_cyl(radius, core_positions, fix_width, extension,
                                                            self_consistent, method, verbose)

        if partial_distance > 0:
            # Specify left & right dislocation separately
            disloc.info["core_positions"] = [list(core_positions[0, :]), list(core_positions[1, :])]
            disloc.info["burgers_vectors"] = [list(self.left_dislocation.burgers), list(self.right_dislocation.burgers)]
            disloc.info["dislocation_types"] = [self.left_dislocation.name, self.right_dislocation.name]
            disloc.info["dislocation_classes"] = [str(self.left_dislocation.__class__), str(self.right_dislocation.__class__)]
        else:
            # Perfect, non-dissociated dislocation, only show values for single dislocation
            disloc.info["core_positions"] = [list(core_positions[0, :])]
            disloc.info["burgers_vectors"] = [list(self.burgers)]
            disloc.info["dislocation_types"] = [self.name]
            disloc.info["dislocation_classes"] = [str(self.__class__)]

        return bulk, disloc

class CubicCrystalDislocationQuadrupole(CubicCrystalDissociatedDislocation):
    burgers_dimensionless = np.zeros(3)
    def __init__(self, disloc_class, *args, **kwargs):
        '''
        Initialise dislocation quadrupole class

        Arguments
        ---------
        disloc_class: Subclass of CubicCrystalDislocation
            Dislocation class to create (e.g. DiamondGlide90DegreePartial)

        *args, **kwargs
            Parameters fed to CubicCrystalDislocation.__init__()
        '''

        if isinstance(disloc_class, CubicCrystalDislocation):
            disloc_cls = disloc_class.__class__
        else:
            disloc_cls = disloc_class

        self.left_dislocation = disloc_cls(*args, **kwargs)
        self.right_dislocation = disloc_cls(*args, **kwargs)

        self.left_dislocation.invert_burgers()
        
        super().__init__(*args, **kwargs)

        self.glides_per_unit_cell = np.floor((self.unit_cell.cell[0, 0] + self.unit_cell.cell[1, 0]) 
                                              / (self.glide_distance - 1e-2)).astype(int)

    # Overload all of the classmethod properties from CubicCrystalDissociatedDislocation, as these cannot be
    # known at class level (don't know which disloc to make a quadrupole of, so can't know what axes should be)
        
    @property
    def crystalstructure(self):
        return self.left_dislocation.crystalstructure
    
    @property
    def axes(self):
        return self.left_dislocation.axes
    
    @property
    def unit_cell_core_position_dimensionless(self):
        return self.left_dislocation.unit_cell_core_position_dimensionless
    
    @property
    def parity(self):
        return self.left_dislocation.parity

    @property
    def n_planes(self):
        return self.left_dislocation.n_planes
    
    @property
    def self_consistent(self):
        return self.left_dislocation.self_consistent
    
    @property
    def glide_distance_dimensionless(self):
        return self.left_dislocation.glide_distance_dimensionless


    def periodic_displacements(self, positions, v1, v2, core_positions, disp_tol=1e-3, max_neighs=60, 
                               verbose="periodic", **kwargs):
        '''
        Calculate the stroh displacements for the periodic structure defined by 2D lattice vectors v1 & v2
        given the core positions.

        positions: np.array
            Atomic positions to calculate displacements at
        v1, v2: np.array
            Lattice vectors defining periodicity in XY plane (dislocation line is in Z)
        core_positions: np.array
            positions of the all dislocation cores (including all partial cores)
        disp_tol: float
            Displacement tolerance controlling the termination of displacement convergence
        max_neighs: int
            Maximum number of nth-nearest neighbour cells to consider
            max_neighs=0 is just the current cell, max_neighs=1 also includes nearest-neighbour cells, ...
        verbose: bool, str or None
            Controls verbosity of the displacement solving
            False, None, or "off" : No printing
            "periodic" (default) : Prints status of the periodic displacement convergence
            True : Also prints status of self-consistent solutions for each dislocation
                    (i.e. verbose=True for CubicCrystalDislocation.displacements())
        **kwargs
            Other keyword arguments fed to self.displacements() (e.g. method="adsl")
        returns
        -------
        disps: np.array
            displacements
        '''

        ncores = core_positions.shape[0]

        if type(verbose) == str:
            if verbose.lower() == "off":
                periodic_verbose = False
                disloc_verbose = False
            elif verbose.lower() == "periodic":
                periodic_verbose = True
                disloc_verbose = False
            else:
                warnings.warn(f"Unrecognised string verbosity '{verbose}'")
                periodic_verbose = True
                disloc_verbose = False
        elif type(verbose) == bool:
            if verbose:
                periodic_verbose = True
                disloc_verbose = True
            else:
                periodic_verbose = False
                disloc_verbose = False
        elif verbose is None:
            periodic_verbose = False
            disloc_verbose = False
        else:
            warnings.warn(f"verbose argument '{verbose}' of type {type(verbose)} not understood.")
            periodic_verbose = True
            disloc_verbose = False

        if "self_consistent" in kwargs:
            # Disable disloc verbosity if no self-consistent solve of disloc displacements
            # CubicCrystalDislocation.displacements doesn't print unless SCF is turned on 
            disloc_verbose = disloc_verbose * bool(kwargs["self_consistent"])
            self_consistent = bool(kwargs[self_consistent])
        else:
            kwargs["self_consistent"] = self.self_consistent

        displacements = np.zeros_like(positions)
        disp_update = np.zeros_like(positions)
        
        for neigh_idx in range(max_neighs):
            disp_update = np.zeros_like(positions)
            for ix in range(-neigh_idx, neigh_idx+1):
                for iy in range(-neigh_idx, neigh_idx+1):
                    if ix < neigh_idx and ix > -neigh_idx and iy < neigh_idx and iy > -neigh_idx:
                        # Looping over cells covered by previous iteration, skip
                        continue

                    # In a new cell
                    if disloc_verbose:
                        print(f"Displacements for cell {ix} {iy}")

                    core_pos = core_positions.copy()
                    for i in range(ncores):
                        core_pos[i, :] += ix * v1 + iy * v2
                    disp_update += self.displacements(
                        positions, core_pos,
                        verbose=disloc_verbose, **kwargs)

            displacements += disp_update
            max_disp_change = np.max(np.linalg.norm(disp_update, axis=-1))

            if periodic_verbose:
                print(f"Periodic displacement iteration {neigh_idx} -> max(|dr|) = {np.round(max_disp_change, 4)}")

            if max_disp_change < disp_tol and ix > 0:
                
                if periodic_verbose:
                    print(f"Periodic displacements converged to r_tol={disp_tol} after {neigh_idx} iterations")

                return displacements
        warnings.warn("Periodic quadrupole displacments did not converge!")
        return displacements


    def build_quadrupole(self, glide_separation=4, partial_distance=0, extension=0, verbose='periodic', 
                         left_offset=None, right_offset=None, **kwargs):
        '''
        Build periodic quadrupole cell

        Parameters
        ----------
        glide_separation: int
            Number of glide distances separating the two dislocation cores
        partial_distance: int or list of int
            Distance between dissociated partial dislocations
            If a list, specify a partial distance for each dislocation
            If an int, specify a partial distance for both dislocations
        extension: int
            Argument to modify the minimum length of the cell in the glide (x) direction.
            Cell will always be at least 
            (2 * glide_separation) + partial_distance + extension glide vectors long
            Useful for setting up images for glide barrier calculations, as 
            start and end structures must be the same size for the NEB method
        verbose: bool, str, or None
            Verbosity value to be fed to CubicCrystalDislocationQuadrupole.periodic_displacements
        left_offset, right_offset: np.array
            Translational offset (in Angstrom) for the left and right dislocation cores 
            (i.e. for the +b and -b cores)
        
        '''

        if left_offset is None:
            left_offset = np.zeros(3)
        if right_offset is None:
            right_offset = np.zeros(3)

        if glide_separation < 1:
            raise RuntimeError("glide_separation should be >= 1")
        elif glide_separation < 3:
            msg = "glide_separation is very small. Resulting structure may be very unstable."
            warnings.warn(msg)

        assert extension >= 0
        assert partial_distance >= 0

        core_separation = glide_separation * self.glide_distance
        core_vec = np.array([core_separation, 0, 0])

        partial_vec = np.array([partial_distance * self.glide_distance, 0, 0])
        # Additional space to add to the cell
        extra_extension = np.array([extension + partial_distance, 0, 0]) * self.glide_distance
        
        # Replicate the unit cell enough times to fill the target cell
        cell = self.unit_cell.cell[:, :].copy()

        # Integer number of glide distances which fit in a single unit cell
        glides_per_unit_cell_x = self.glides_per_unit_cell
        glides_per_unit_cell_y = np.floor(np.linalg.norm(cell[1, :]) / (self.glide_distance - 1e-2)).astype(int)

        # Number of unit cells in x (glide) direction
        xreps = np.ceil((2*glide_separation + extension + 2*partial_distance) / glides_per_unit_cell_x).astype(int)


        # Number of unit cells in y direction
        yreps = np.ceil(glide_separation / glides_per_unit_cell_y).astype(int)

        quad_bulk = self.unit_cell.copy() * (xreps, yreps, 1)
        cell = quad_bulk.cell[:, :].copy()


        # New cell based on old cell vectors
        # Rhomboid shape enclosing both cores, + any stacking fault
        new_cell = np.array([
            cell[1, :] + 0.5 * cell[0, :],
            -cell[1, :] + 0.5 *  cell[0, :],
            cell[2, :]
        ])

        if self.crystalstructure == "bcc" and np.floor(glide_separation).astype(int) % 3 == 1:
            new_cell[0, 0] -= self.glide_distance * 3/2
            new_cell[1, 0] += self.glide_distance * 3/2
            
        quad_bulk.set_cell(new_cell)
        quad_bulk.wrap()
        pos = quad_bulk.get_positions()

        quad_disloc = quad_bulk.copy()

        # Get the core positions, translate everything so the cores are central in the cell
        lens = np.sum(new_cell, axis=0)
        core_pos_1 = lens/2 - 0.5 * core_vec - 0.5 * partial_vec
        core_pos_2 = core_pos_1 + core_vec
        pos += core_pos_1 - self.left_dislocation.unit_cell_core_position

        core_pos_1 += left_offset
        core_pos_2 += right_offset

        core_positions = np.array([
            core_pos_1,
            core_pos_2
        ])

        if partial_distance > 0.0:
            partial_core_pos = core_positions.copy()
            partial_core_pos[:, 0] += partial_vec[0]
            old_core_positions = core_positions
            
            core_positions = np.array([
                old_core_positions[0, :],
                partial_core_pos[0, :],
                old_core_positions[1, :],
                partial_core_pos[1, :]
            ])

        quad_bulk.set_positions(pos)
        quad_bulk.wrap()
        pos = quad_bulk.get_positions()

        # Apply disloc displacements to quad_disloc
        disps = self.periodic_displacements(pos, new_cell[0, :], new_cell[1, :], core_positions, 
                                            verbose=verbose, **kwargs)

        pos += disps
        quad_disloc.set_positions(pos)
        quad_disloc.wrap()

        if partial_distance > 0 and isinstance(self.right_dislocation, CubicCrystalDissociatedDislocation):
            # Dissociated dislocation quadrupole
            # 4 dislocations in total
            # "Left" dissociated dislocation
            pos1 = core_pos_1
            pos2 = core_pos_1 + np.array([partial_distance * self.glide_distance, 0, 0])
            # "Right" dissociated dislocation
            pos3 = core_pos_2
            pos4 = core_pos_2 + np.array([partial_distance * self.glide_distance, 0, 0])
                                         
            quad_disloc.info["core_positions"] = [list(pos1), list(pos2), list(pos3), list(pos4)]

            # Quadrupole is a "dissociated dissociated" dislocation
            # 4 partial dislocs are .L.L, .L.R, .R.L, .R.R
            # (L = left_dislocation, R = right_dislocation)

            for key, prop in [
                ("burgers_vectors", "burgers"),
                ("dislocation_types", "name"),
                ("dislocation_classes", "__class__")
            ]:
                quad_disloc.info[key] = [
                    getattr(self.left_dislocation.left_dislocation, prop),
                    getattr(self.left_dislocation.right_dislocation, prop),
                    getattr(self.right_dislocation.left_dislocation, prop),
                    getattr(self.right_dislocation.right_dislocation, prop)
                ]

            # Type conversions
            quad_disloc.info["burgers_vectors"] = [list(burgers) for burgers in quad_disloc.info["burgers_vectors"]]
            quad_disloc.info["dislocation_classes"] = [str(name) for name in quad_disloc.info["dislocation_classes"]]
        else:
            # Perfect, non-dissociated dislocation, only show values for single dislocation
            
            quad_disloc.info["core_positions"] = [list(core_pos_1), list(core_pos_2)]
            quad_disloc.info["burgers_vectors"] = [list(self.left_dislocation.burgers), list(self.right_dislocation.burgers)]
            quad_disloc.info["dislocation_types"] = [self.left_dislocation.name, self.right_dislocation.name]
            quad_disloc.info["dislocation_classes"] = [str(self.left_dislocation.__class__), str(self.right_dislocation.__class__)]



        return quad_bulk, quad_disloc
    
    def build_glide_quadrupoles(self, nims, invert_direction=False, glide_left=True, glide_right=True, 
                                left_offset=None, right_offset=None, *args, **kwargs):
        '''
        Construct a sequence of quadrupole structures providing an initial guess of the dislocation glide
        trajectory

        Parameters
        ----------
        nims: int
            Number of images to generate
        invert_direction: bool
            Invert the direction of the glide
            invert_direction=False (default) glides in the +x direction
            invert_direction=True glides in the -x direction
        glide_left, glide_right: bool
            Flags for toggling whether the left/right cores
            are allowed to glide
        *args, **kwargs
            Fed to self.build_quadrupole()
        
        '''
        
        glide_offsets = np.linspace(0, self.glide_distance, nims, endpoint=True)

        if invert_direction:
            glide_offsets *= -1

        if left_offset is not None:
            _loff = left_offset
        else:
            _loff = np.zeros(3)

        if right_offset is not None:
            _roff = right_offset
        else:
            _roff = np.zeros(3)

        images = []

        for i in range(nims):
            left_offset = np.array([glide_offsets[i], 0, 0]) if glide_left else np.zeros(3)
            right_offset = np.array([glide_offsets[i], 0, 0]) if glide_right else np.zeros(3)

            left_offset += _loff
            right_offset += _roff

            images.append(self.build_quadrupole(
                left_offset=left_offset,
                right_offset=right_offset,
                *args,
                **kwargs
            )[1])
        return images
    
    def build_kink_quadrupole(self, z_reps, layer_tol=1e-1, invert_direction=False, *args, **kwargs):
        '''
        Construct a quadrupole structure providing an initial guess of the dislocation kink
        mechanism. Produces a periodic array of kinks, where each kink is 
        z_reps * self.unit_cell[2, 2] Ang long

        Parameters
        ----------
        z_reps: int
            Number of replications of self.unit_cell to use per kink
            Should be >1
        layer_tol: float
            Tolerance for trying to determine the top atomic layer of the cell
            (required in order to complete periodicity)
            Top layer is defined as any atom with z component >= max(atom_z) - layer_tol
        invert_direction: bool
            Invert the direction of the glide
            invert_direction=False (default) kink in the +x direction
            invert_direction=True kink in the -x direction
        *args, **kwargs
            Fed to self.build_quadrupole() & self.build_glide_quadrupoles
        
        '''
        assert z_reps > 1

        direction = -1 if invert_direction else 1

        # Need both cores to move for the infinite kink structure
        reps = self.build_glide_quadrupoles(z_reps, glide_left=True, glide_right=True, 
                                            invert_direction=invert_direction, *args, **kwargs)

        kink_struct = reps[0]

        for image in reps[1:]:
            kink_struct = stack(kink_struct, image)
        
        cell = kink_struct.cell[:, :]

        cell[2, 0] += self.glide_distance * direction

        kink_struct.set_cell(cell)
        kink_struct.wrap()

        glide_parity = (-direction) % self.glides_per_unit_cell

        if glide_parity:
            # Cell won't be periodic if multiple glides needed to complete periodicity
            # glide_parity gives number of layers required to remove

            for i in range(glide_parity):
                atom_heights = kink_struct.get_positions()[:, 2]
                top_atom = np.max(atom_heights)

                layer_mask = atom_heights >= top_atom - layer_tol

                avg_layer_pos = np.average(atom_heights[layer_mask])

                kink_struct = kink_struct[~layer_mask]
                cell[2, 2] = avg_layer_pos
                kink_struct.set_cell(cell)
                kink_struct.wrap()
        return kink_struct
    
    def view_quad(self, system, *args, **kwargs):
        '''
        Specialised wrapper of view_cyl for showing quadrupoles, as the quadrupole cell 
        causes the plot to be rotated erroneously
        
        Takes the same args and kwargs as view_cyl
        '''

        view = self.view_cyl(system, hide_arrows=True, *args, **kwargs)

        cell = system.cell[:, :]
        
        # Default rotation has cell[0, :] pointing in x
        # Rotate such that cell[0, :] has angle theta to x
        # cell[1, :] has -theta to x
        dot = (cell[0, :] @ cell[1, :]) / (np.linalg.norm(cell[0, :]) * np.linalg.norm(cell[1, :]))
        rot_angle = -np.arccos(dot)/2 + np.pi

        # Rotate system by rot_angle about [0, 0, 1]
        view.control.spin([0, 0, 1], rot_angle)
        return view
   
# TODO: If non-cubic dislocation classes are implemented, need to build an
# interface to make "Quadrupole" work for both
Quadrupole = CubicCrystalDislocationQuadrupole


class BCCScrew111Dislocation(CubicCrystalDislocation):
    crystalstructure = "bcc"
    axes = np.array([[1, 1, -2],
                     [-1, 1, 0],
                     [1, 1, 1]])
    burgers_dimensionless = np.array([1, 1, 1]) / 2.0
    unit_cell_core_position_dimensionless = np.array([np.sqrt(6.)/6.0, np.sqrt(2.)/6.0, 0])
    glide_distance_dimensionless = np.sqrt(6) / 3.0
    name = "1/2<111> screw"


class BCCEdge111Dislocation(CubicCrystalDislocation):
    crystalstructure = "bcc"
    axes = np.array([[1, 1, 1],
                     [1, -1, 0],
                     [1, 1, -2]])
    burgers_dimensionless = np.array([1, 1, 1]) / 2.0
    unit_cell_core_position_dimensionless =  np.array([(1.0/3.0) * np.sqrt(3.0)/2.0, 0.25 * np.sqrt(2.0), 0])
    glide_distance_dimensionless = np.sqrt(3) / 3.0
    n_planes = 6
    name = "1/2<111> edge"

class BCCEdge111barDislocation(CubicCrystalDislocation):
    crystalstructure = "bcc"
    axes = np.array([[1, 1, -1],
                     [1, 1, 2],
                     [1, -1, 0]])
    burgers_dimensionless = np.array([-1, -1, 1]) / 2.0
    unit_cell_core_position_dimensionless =  np.array([(1.0/3.0) * np.sqrt(3.0)/2.0, 0.25 * np.sqrt(2.0), 0])
    glide_distance_dimensionless = np.sqrt(3) / 3.0
    n_planes = 1
    name = "1/2<11-1> edge"

class BCCMixed111Dislocation(CubicCrystalDislocation):
    crystalstructure = "bcc"
    axes = np.array([[1, -1, -2],
                     [1, 1, 0],
                     [1, -1, 1]])
    burgers_dimensionless = np.array([1, -1, -1]) / 2.0
    glide_distance_dimensionless = np.sqrt(6) / 3.0
    # middle of the right edge of the first upward triangle
    # half way between (1/6, 1/2, 0) and (1/3, 0, 0) in fractional coords
    unit_cell_core_position_dimensionless = np.array([1/6, 1/6, 0])
    name = "1/2<111> mixed"


class BCCEdge100Dislocation(CubicCrystalDislocation):
    crystalstructure = "bcc"
    axes = np.array([[1, 0, 0],
                     [0, 0, -1],
                     [0, 1, 0]])
    burgers_dimensionless = np.array([1, 0, 0])
    unit_cell_core_position_dimensionless = np.array([1/4, 1/4, 0])
    glide_distance_dimensionless = 1.0
    n_planes = 2
    name = "<100>{110} edge"


class BCCEdge100110Dislocation(CubicCrystalDislocation):
    crystalstructure = "bcc"
    axes = np.array([[1, 0, 0],
                     [0, 1, 1],
                     [0, -1, 1]])
    burgers_dimensionless = np.array([1, 0, 0])
    unit_cell_core_position_dimensionless = np.array([0.5, np.sqrt(2) / 4.0, 0])
    glide_distance_dimensionless = 0.5
    n_planes = 2
    name = "<100>{001} edge"


class DiamondGlide30degreePartial(CubicCrystalDislocation):
    crystalstructure="diamond"
    axes = np.array([[1, 1, -2],
                     [1, 1, 1],
                     [1, -1, 0]])
    burgers_dimensionless = np.array([1, -2, 1.]) / 6.
    glide_distance_dimensionless = np.sqrt(6) / 4.0
    # 1/4 + 1/2 * (1/3 - 1/4) - to be in the middle of the glide set
    unit_cell_core_position_dimensionless = np.array([np.sqrt(6)/12, 7*np.sqrt(3)/24, 0])
    n_planes = 2
    # There is very small distance between
    # atomic planes in glide configuration.
    # Due to significant anisotropy application of the self consistent
    # displacement field leads to deformation of the atomic planes.
    # This leads to the cut plane crossing one of the atomic planes and
    # thus breaking the stacking fault.
    self_consistent = False
    name = "1/6<112> 30 degree partial"


class DiamondGlide90degreePartial(CubicCrystalDislocation):
    crystalstructure = "diamond"
    axes = np.array([[1, 1, -2],
                     [1, 1, 1],
                     [1, -1, 0]])
    # 1/4 + 1/2 * (1/3 - 1/4) - to be in the middle of the glide set
    unit_cell_core_position_dimensionless = np.array([np.sqrt(6)/12, 7 * np.sqrt(3) / 24, 0])
    burgers_dimensionless = np.array([1., 1., -2.]) / 6.
    glide_distance_dimensionless = np.sqrt(6)/4
    n_planes = 2
    # There is very small distance between
    # atomic planes in glide configuration.
    # Due to significant anisotropy application of the self consistent
    # displacement field leads to deformation of the atomic planes.
    # This leads to the cut plane crossing one of the atomic planes and
    # thus breaking the stacking fault.
    self_consistent = False
    name = "1/6<112> 90 degree partial"


class DiamondGlideScrew(CubicCrystalDissociatedDislocation):
    burgers_dimensionless = np.array([1, -1, 0]) / 2
    new_left_burgers = np.array([2., -1., -1.]) / 6
    left_dislocation = DiamondGlide30degreePartial
    right_dislocation = DiamondGlide30degreePartial
    name = "1/2<110> screw"


class DiamondGlide60Degree(CubicCrystalDissociatedDislocation):
    burgers_dimensionless = np.array([1, 0, -1]) / 2
    new_left_burgers = np.array([2., -1., -1.]) / 6
    left_dislocation = DiamondGlide30degreePartial
    right_dislocation = DiamondGlide90degreePartial
    name = "1/2<110> 60 degree screw"


class FCCScrewShockleyPartial(CubicCrystalDislocation):
    crystalstructure="fcc"
    axes = np.array([[1, 1, -2],
                     [1, 1, 1],
                     [1, -1, 0]])
    burgers_dimensionless = np.array([1, -2, 1.]) / 6.
    glide_distance_dimensionless = np.sqrt(6) / 4.0
    n_planes = 2
    unit_cell_core_position_dimensionless = np.array([5/6, 1/9, 0])
    name = "1/6<112> screw Shockley partial"


class FCCEdgeShockleyPartial(CubicCrystalDislocation):
    crystalstructure = "fcc"
    axes = np.array([[1, -1, 0],
                     [1, 1, 1],
                     [-1, -1, 2]])
    burgers_dimensionless = np.array([1, -2, 1.]) / 6
    unit_cell_core_position_dimensionless = np.array([0, 1/6 , 0])
    glide_distance_dimensionless = np.sqrt(2)/4
    n_planes = 6
    name = "1/2<110> edge Shockley partial"


class FCCScrew110Dislocation(CubicCrystalDissociatedDislocation):
    burgers_dimensionless = np.array([1, -1, 0]) / 2
    new_left_burgers = np.array([2., -1., -1.]) / 6
    left_dislocation = FCCScrewShockleyPartial
    right_dislocation = FCCScrewShockleyPartial
    name = "1/2<110>{111} screw"


class FCCEdge110Dislocation(CubicCrystalDissociatedDislocation):
    crystalstructure = "fcc"
    burgers_dimensionless = np.array([1, -1, 0]) / 2
    new_left_burgers = np.array([2., -1., -1.]) / 6
    left_dislocation = FCCEdgeShockleyPartial
    right_dislocation = FCCEdgeShockleyPartial
    name = "1/2<110> edge"


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

    import warnings

    msg = f"gamma_line is depreciated. Use of matscipy.gamma_surface.StackingFault is preferred"
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

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
                "Initial max force is smaller than fmax!" +
                "Check surface direction")

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
