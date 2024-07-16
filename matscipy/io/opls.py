#
# Copyright 2016-2017, 2020 Andreas Klemenz (Fraunhofer IWM)
#           2020 Thomas Reichenbach (Fraunhofer IWM)
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

import time
import copy
import re

from packaging.version import Version

import numpy as np
import ase
import ase.data
import ase.io
import ase.io.lammpsrun
import ase.calculators.lammpsrun
import matscipy.neighbours

import matscipy.opls

try:
    import ase.version
    ase_version_str = ase.version.version
except:
    ase_version_str = ase.__version__


def read_extended_xyz(fileobj):
    """
    Read an extended xyz file with labeled atoms. The number of atoms
    should be given in the first line, the second line contains the cell
    dimensions and the definition of the columns. The file should contain
    the following columns: element (1 or 2 characters), x(float), y(float),
    z (float), molecule id (int), name (1 or 2 characters). A full
    description of the extended xyz format can be found for example in the
    ASE documentation. An example for a file defining an H2 molecule is
    given below.
    ::
      2
      Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3:molid:I:1:type:S:1
      H 4.5 5.0 5.0 1 H1
      H 5.5 5.0 5.0 1 H1


    Parameters
    ----------
    filename : str
        Name of the file to read from.

    Returns
    -------
    matscipy.opls.OPLSStructure
        Atomic structure as defined in the input file.
    """
    atoms = ase.io.read(fileobj)
    opls_struct = matscipy.opls.OPLSStructure(atoms)
    opls_struct.arrays = atoms.arrays

    types = opls_struct.get_array('type')
    opls_struct.types = np.unique(types)

    tags = np.zeros(len(opls_struct), dtype=int)
    for it, type in enumerate(opls_struct.types):
        tags[types == type] = it
    opls_struct.set_tags(tags)

    return opls_struct


def read_block(filename, name):
    """
    Read a named data block from a parameter file for a non-reactive
    potential. Blocks begin with ``# name`` and are terminated by empty
    lines. Using the function on a file named ``parameterfile`` which
    contains a block named ``Bonds`` and the following entries
    ::
      # Bonds
      C1-C1 10.0 1.0
      H1-H1 20.0 2.0

    will return the following dictionary:
    ::
      {'C1-C1': [10., 1.], 'H1-H1': [20., 2.]}

    Parameters
    ----------
    filename : str
        Name of the file to read from.
    name : str
        Name of the data block to search for.

    Returns
    -------
    dict
        Name-Value pairs. Each value is a list of arbitrary length.

    Raises
    ------
    RuntimeError
        If data block ``name`` is not found in the file.

    """
    data = {}
    if isinstance(filename, str):
        with open(filename, 'r') as fileobj:
            block = False
            for line in fileobj.readlines():
                line = line.split()

                # find data block
                if len(line) >= 2:
                    if line[1] == name:
                        block = True

                # end of data block
                if block == True and len(line) == 0:
                    block = False

                # read data
                if block:
                    if line[0][0] == '#':
                        continue
                    else:
                        symbol = line[0]
                        data[symbol] = []
                        for word in line[1:]:
                            if word[0] == '#':
                                break
                            else:
                                data[symbol].append(float(word))
                        if len(data[symbol]) == 1:
                            data[symbol] = data[symbol][0]

            if len(data) == 0:
                raise RuntimeError('Data block \"%s\" not found in file \"%s\"' % (name, filename))

    return data


def read_cutoffs(filename):
    """
    Read the cutoffs for construction of a non-reactive system from a
    file. Comments in the file begin with ``#``, the file should be
    structured like this:
    ::
      # Cutoffs
      C1-C1 1.23  # name, cutoff (A)
      H1-H1 4.56  # name, cutoff (A)
      C1-H1 7.89  # name, cutoff (A)

    Parameters
    ----------
    filename : str
        Name of the file to read from.

    Returns
    -------
    matscipy.opls.CutoffList
        Cutoffs.
    """
    cutoffs = matscipy.opls.CutoffList(read_block(filename, 'Cutoffs'))
    return cutoffs


def read_parameter_file(filename):
    """
    Read the parameters of a non-reactive potential from a file. An
    example for the file structure is given below. The blocks are
    separated by empty lines, comments begin with ``#``. For more
    information about the potentials, refer to the documentation of
    the LAMMPS commands ``bond_style harmonic``,
    ``angle_style harmonic``, ``dihedral_style harmonic``. The default
    global cutoffs for Lennard-Jones and Coulomb interactions are 10.0
    and 7.4 A. They can be overridden with the optional
    ``Cutoffs-LJ-Coulomb`` block. By default, geometric mixing is
    applied between Lennard-Jones parameters of different particle types
    and the global cutoff is used for all pairs. This behavior can be
    overridden using the optional ``LJ-pairs`` block.
    ::
      # Element
      C1 0.001 3.5 -0.01  # name, LJ-epsilon (eV), LJ-sigma (A), charge (e)
      H1 0.001 2.5  0.01  # name, LJ-epsilon (eV), LJ-sigma (A), charge (e)

      # Cutoffs-LJ-Coulomb (this block is optional)
      LJ 10.0  # distance (A)
      C  10.0  # distance (A)
    
      # LJ-pairs (this block is optional)
      C1-H1 0.002 2.1 12.0  # name, epsilon (eV), sigma (A), cutoff (A)

      # Bonds
      C1-C1 10.0 1.0  # name, spring constant*2 (eV/A**2), distance (A)

      # Angles
      H1-C1-C1 1.0 100.0  # name, spring constant*2 (eV), equilibrium angle

      # Dihedrals
      H1-C1-C1-H1 0.0 0.0 0.01 0.0  # name, energy (eV), energy (eV), ...

      # Cutoffs
      C1-C1 1.85  # name, cutoff (A)
      C1-H1 1.15  # name, cutoff (A)

    Parameters
    ----------
    filename : str
        Name of the file to read from.

    Returns
    -------
    cutoffs : matscipy.opls.CutoffList
        Cutoffs.
    ljq : matscipy.opls.LJQData
        Lennard-Jones data and atomic charges.
    bonds : matscipy.opls.BondData
        Bond coefficients, i.e. spring constants and
        equilibrium distances.
    angles : matscipy.opls.AnglesData
        Angle coefficients.
    dihedrals : matscipy.opls.DihedralsData
        Dihedral coefficients.
    """
    ljq = matscipy.opls.LJQData(read_block(filename, 'Element'))

    try:
        ljq_cut = read_block(filename, 'Cutoffs-LJ-Coulomb')
        ljq.lj_cutoff = ljq_cut['LJ']
        ljq.c_cutoff = ljq_cut['C']
    except:
        pass

    try:
        ljq.lj_pairs = read_block(filename, 'LJ-pairs')
    except:
        pass

    bonds     = matscipy.opls.BondData(read_block(filename, 'Bonds'))
    angles    = matscipy.opls.AnglesData(read_block(filename, 'Angles'))
    dihedrals = matscipy.opls.DihedralsData(read_block(filename, 'Dihedrals'))
    cutoffs   = matscipy.opls.CutoffList(read_block(filename, 'Cutoffs'))

    return cutoffs, ljq, bonds, angles, dihedrals

def write_lammps(prefix, atoms):
    """
    Convenience function. The functions
    :func:`matscipy.io.opls.write_lammps_in`,
    :func:`matscipy.io.opls.write_lammps_atoms` and
    :func:`matscipy.io.opls.write_lammps_definitions`
    are usually called at the same time. This function
    combines them, filenames will be ``prefix.in``,
    ``prefix.atoms`` and ``prefix.opls``.

    Parameters
    ----------
    prefix : str
        Prefix for filenames.
    atoms : matscipy.opls.OPLSStructure
        The atomic structure to be written.
    """
    write_lammps_in(prefix)
    write_lammps_atoms(prefix, atoms)
    write_lammps_definitions(prefix, atoms)


def write_lammps_in(prefix):
    """
    Writes a simple LAMMPS input script for a structure optimization
    using a non-reactive potential. The name of the resulting script
    is ``prefix.in``, while the atomic structure is defined in
    ``prefix.atoms`` and the definition of the atomic interaction in
    ``prefix.opls``.

    Parameters
    ----------
    prefix : str
        Prefix for filename.
    """
    if isinstance(prefix, str):
        with open(prefix + '.in', 'w') as fileobj:
            fileobj.write('# LAMMPS relaxation\n\n')

            fileobj.write('units           metal\n')
            fileobj.write('atom_style      full\n')
            fileobj.write('boundary        p p p\n\n')

            fileobj.write('read_data       %s.atoms\n' % (prefix))
            fileobj.write('include         %s.opls\n' % (prefix))
            fileobj.write('kspace_style    pppm 1e-5\n\n')

            fileobj.write('neighbor        1.0 bin\n')
            fileobj.write('neigh_modify    delay 0 every 1 check yes\n\n')

            fileobj.write('thermo          1000\n')
            fileobj.write('thermo_style    custom step temp press cpu pxx pyy pzz pxy pxz pyz ke pe etotal vol lx ly lz atoms\n\n')

            fileobj.write('dump            1 all xyz 1000 dump_relax.xyz\n')
            fileobj.write('dump_modify     1 sort id\n\n')

            fileobj.write('restart         100000 test_relax\n\n')

            fileobj.write('min_style       fire\n')
            fileobj.write('minimize        1.0e-14 1.0e-5 100000 100000\n')


def write_lammps_atoms(prefix, atoms, units='metal'):
    """
    Write atoms input for LAMMPS. Filename will be ``prefix.atoms``.

    Parameters
    ----------
    prefix : str
        Prefix for filename.
    atoms : matscipy.opls.OPLSStructure
        The atomic structure to be written.
    units : str, optional
        The units to be used.
    """
    if isinstance(prefix, str):
        with open(prefix + '.atoms', 'w') as fileobj:
            # header
            fileobj.write(fileobj.name + ' (by write_lammps_atoms)\n\n')
            fileobj.write('%d atoms\n' % (len(atoms)))
            fileobj.write('%d atom types\n' % (len(atoms.types)))

            blist = atoms.bond_list
            if len(blist):
                btypes = atoms.bond_types
                fileobj.write('%d bonds\n' % (len(blist)))
                fileobj.write('%d bond types\n' % (len(btypes)))

            alist = atoms.ang_list
            if len(alist):
                atypes = atoms.ang_types
                fileobj.write('%d angles\n' % (len(alist)))
                fileobj.write('%d angle types\n' % (len(atypes)))

            dlist = atoms.dih_list
            if len(dlist):
                dtypes = atoms.dih_types
                fileobj.write('%d dihedrals\n' % (len(dlist)))
                fileobj.write('%d dihedral types\n' % (len(dtypes)))

            # cell
            if Version(ase_version_str) > Version('3.11.0'):
                p = ase.calculators.lammpsrun.Prism(atoms.get_cell())
            else:
                p = ase.calculators.lammpsrun.prism(atoms.get_cell())

            xhi, yhi, zhi, xy, xz, yz = ase.calculators.lammpsrun.convert(
                p.get_lammps_prism(), 'distance', 'ASE', units
                )
            fileobj.write('\n0.0 %f  xlo xhi\n' % xhi)
            fileobj.write('0.0 %f  ylo yhi\n' % yhi)
            fileobj.write('0.0 %f  zlo zhi\n' % zhi)

            # write tilt factors for non-orthogonal cells
            if np.abs(xy) > 1e-10 or np.abs(xz) > 1e-10 or np.abs(yz) > 1e-10:
                fileobj.write('\n%f %f %f  xy xz yz\n' % (xy, xz, yz))

            # atoms
            fileobj.write('\nAtoms\n\n')
            tags  = atoms.get_tags()
            types = atoms.types
            if atoms.has('molid'):
                molid = atoms.get_array('molid')
            else:
                molid = [1] * len(atoms)

            pos = ase.calculators.lammpsrun.convert(atoms.get_positions(), 'distance', 'ASE', units)
            if Version(ase_version_str) > Version('3.17.0'):
                positions_lammps_str = p.vector_to_lammps(pos).astype(str)
            elif Version(ase_version_str) > Version('3.13.0'):
                positions_lammps_str = p.positions_to_lammps_strs(pos)
            else:
                positions_lammps_str = map(p.pos_to_lammps_str, pos)

            for i, r in enumerate(positions_lammps_str):
                q = ase.calculators.lammpsrun.convert(atoms.atom_data[types[tags[i]]][2], 'charge', 'ASE', units)
                fileobj.write('%6d %3d %3d %s %s %s %s' % ((i + 1, molid[i],
                                                            tags[i] + 1,
                                                            q)
                                                           + tuple(r)))
                fileobj.write(' # ' + atoms.types[tags[i]] + '\n')

            # velocities
            velocities = ase.calculators.lammpsrun.convert(atoms.get_velocities(), 'velocity', 'ASE', units)
            if velocities is not None:
                fileobj.write('\nVelocities\n\n')
                for i, v in enumerate(velocities):
                    fileobj.write('%6d %g %g %g\n' %
                                  (i + 1, v[0], v[1], v[2]))

            # masses
            masses = ase.calculators.lammpsrun.convert(atoms.get_masses(), 'mass', 'ASE', units)
            tags   = atoms.get_tags()

            fileobj.write('\nMasses\n\n')
            for i, type, tag in zip(range(len(atoms.types)), atoms.types, np.unique(tags)):
                fileobj.write('%6d %g # %s\n' %
                              (i + 1,
                               masses[tags == tag][0],
                               type))

            # bonds
            if len(blist):
                fileobj.write('\nBonds\n\n')
                for ib, bvals in enumerate(blist):
                    fileobj.write('%8d %6d %6d %6d ' %
                                  (ib + 1, bvals[0] + 1, bvals[1] + 1,
                                   bvals[2] + 1))
                    try:
                        fileobj.write('# ' + btypes[bvals[0]])
                    except:
                        pass
                    fileobj.write('\n')

            # angles
            if len(alist):
                fileobj.write('\nAngles\n\n')
                for ia, avals in enumerate(alist):
                    fileobj.write('%8d %6d %6d %6d %6d ' %
                                  (ia + 1, avals[0] + 1,
                                   avals[1] + 1, avals[2] + 1, avals[3] + 1))
                    try:
                        fileobj.write('# ' + atypes[avals[0]])
                    except:
                        pass
                    fileobj.write('\n')

            # dihedrals
            if len(dlist):
                fileobj.write('\nDihedrals\n\n')
                for i, dvals in enumerate(dlist):
                    fileobj.write('%8d %6d %6d %6d %6d %6d ' %
                                  (i + 1, dvals[0] + 1,
                                   dvals[1] + 1, dvals[2] + 1,
                                   dvals[3] + 1, dvals[4] + 1))
                    try:
                        fileobj.write('# ' + dtypes[dvals[0]])
                    except:
                        pass
                    fileobj.write('\n')


def write_lammps_definitions(prefix, atoms):
    """
    Write force field definitions for LAMMPS.
    Filename will be ``prefix.opls``.

    Parameters
    ----------
    prefix : str
        Prefix for filename.
    atoms : matscipy.opls.OPLSStructure
        The atomic structure for which the force field definitions
        should be written. Must contain at least Lennard-Jones and
        charge data :class:`matscipy.opls.LJQData`. Bond-, angle-
        and dihedral-potentials are optional and are written if they
        are present:
        :class:`matscipy.opls.BondData`,
        :class:`matscipy.opls.AnglesData`,
        :class:`matscipy.opls.DihedralsData`.
    """
    if isinstance(prefix, str):
        with open(prefix + '.opls', 'w') as fileobj:
            fileobj.write('# OPLS potential\n')
            fileobj.write('# write_lammps ' +
                          str(time.asctime(
                        time.localtime(time.time()))))

            # bonds
            if len(atoms.bond_types):
                fileobj.write('\n# bonds\n')
                fileobj.write('bond_style      harmonic\n')
                for ib, btype in enumerate(atoms.bond_types):
                    fileobj.write('bond_coeff %6d' % (ib + 1))
                    itype, jtype = btype.split('-')
                    name, values = atoms.bonds.name_value(itype, jtype)
                    for value in values:
                        fileobj.write(' ' + str(value))
                    fileobj.write(' # ' + name + '\n')

            # angles
            if len(atoms.ang_types):
                fileobj.write('\n# angles\n')
                fileobj.write('angle_style      harmonic\n')
                for ia, atype in enumerate(atoms.ang_types):
                    fileobj.write('angle_coeff %6d' % (ia + 1))
                    itype, jtype, ktype = atype.split('-')
                    name, values = atoms.angles.name_value(itype, jtype, ktype)
                    for value in values:
                        fileobj.write(' ' + str(value))
                    fileobj.write(' # ' + name + '\n')

            # dihedrals
            if len(atoms.dih_types):
                fileobj.write('\n# dihedrals\n')
                fileobj.write('dihedral_style      opls\n')
                for id, dtype in enumerate(atoms.dih_types):
                    fileobj.write('dihedral_coeff %6d' % (id + 1))
                    itype, jtype, ktype, ltype = dtype.split('-')
                    name, values = atoms.dihedrals.name_value(itype, jtype, ktype, ltype)
                    for value in values:
                        fileobj.write(' ' + str(value))
                    fileobj.write(' # ' + name + '\n')

            # Lennard Jones settings
            fileobj.write('\n# L-J parameters\n')
            fileobj.write('pair_style lj/cut/coul/long %10.8f %10.8f\n' %
                          (atoms.atom_data.lj_cutoff, atoms.atom_data.c_cutoff))
            fileobj.write('special_bonds lj/coul 0.0 0.0 0.5\n')
            for ia, atype in enumerate(atoms.types):
                for ib, btype in enumerate(atoms.types):
                    if len(atype) < 2:
                        atype = atype + ' '
                    if len(btype) < 2:
                        btype = btype + ' '
                    pair = atype + '-' + btype
                    if pair in atoms.atom_data.lj_pairs:
                        if ia < ib:
                            fileobj.write('pair_coeff %3d %3d' % (ia + 1, ib + 1))
                        else:
                            fileobj.write('pair_coeff %3d %3d' % (ib + 1, ia + 1))
                        for value in atoms.atom_data.lj_pairs[pair]:
                            fileobj.write(' ' + str(value))
                        fileobj.write(' # ' + pair + '\n')
                    elif atype == btype:
                        fileobj.write('pair_coeff %3d %3d' % (ia + 1, ib + 1))
                        for value in atoms.atom_data[atype][:2]:
                            fileobj.write(' ' + str(value))
                        fileobj.write(' # ' + atype + '\n')

            fileobj.write('pair_modify shift yes mix geometric\n')

            # Charges
            fileobj.write('\n# charges\n')
            for ia, atype in enumerate(atoms.types):
                if len(atype) < 2:
                    atype = atype + ' '
                fileobj.write('set type ' + str(ia + 1))
                fileobj.write(' charge ' + str(atoms.atom_data[atype][2]))
                fileobj.write(' # ' + atype + '\n')


def read_lammps_definitions(filename):
    """
    Reads force field definitions from a LAMMPS parameter file and
    stores the parameters in :class:`matscipy.opls.LJQData`,
    :class:`matscipy.opls.BondData`, :class:`matscipy.opls.AnglesData`
    and :class:`matscipy.opls.DihedralsData` objects. The 'number'
    of the particles, pairs, ... for the corresponding interaction
    parameters is not included in these objects and is output in
    dicts. Note that there is an offset of one between LAMMPS and
    python numbering.

    Parameter file:
    ::
      bond_style      harmonic
      bond_coeff      1 1.2 3.4 # AA-AA
      bond_coeff      2 5.6 7.8 # AA-BB

    Returned dictionary:
    ::
      bond_type_index[0] = 'AA-AA'
      bond_type_index[1] = 'AA-BB'

    Parameters
    ----------
    filename : str
        Name of the file to read from.

    Returns
    -------
    ljq_data : matscipy.opls.LJQData
        Lennard-Jones and charge data.
    bond_data : matscipy.opls.BondData
        Parameters of the harmonic bond potentials.
    ang_data : matscipy.opls.AnglesData
        Parameters of the harmonic angle potentials.
    dih_data : matscipy.opls.DihedralsData
        Parameters of the OPLS dihedral potentials.
    particle_type_index : dict
        Indicees of particle types as used by LAMMPS.
    bond_type_index : dict
        Indicees of bond types as used by LAMMPS.
    ang_type_index : dict
        Indicees of angle types as used by LAMMPS.
    dih_type_index : dict
        Indicees of dihedral types as used by LAMMPS.
    """
    with open(filename, 'r') as fileobj:
        bond_nvh = {}
        ang_nvh  = {}
        dih_nvh  = {}

        particle_type_index = {}
        bond_type_index     = {}
        ang_type_index      = {}
        dih_type_index      = {}

        ljq_data = matscipy.opls.LJQData({})

        for line in fileobj.readlines():
            re_lj_cut = re.match('^pair_style\s+lj/cut/coul/long\s+(\d+\.?\d*)\s+(\d+\.?\d*)$', line)
            if re_lj_cut:
                ljq_data.lj_cutoff = float(re_lj_cut.groups()[0])
                ljq_data.c_cutoff  = float(re_lj_cut.groups()[1])

            re_pc     = re.match('^pair_coeff\s+(\d+)\s+(\d+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+\#\s+(\S+)$', line)
            re_pc_cut = re.match('^pair_coeff\s+(\d+)\s+(\d+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(\d+\.?\d*)\s+\#\s+(\S+)$', line)
            if re_pc_cut:
                lj_pair_type = re_pc_cut.groups()[5]
                lj_pair_p1   = float(re_pc_cut.groups()[2])
                lj_pair_p2   = float(re_pc_cut.groups()[3])
                lj_pair_p3   = float(re_pc_cut.groups()[4])
                ljq_data.lj_pairs[lj_pair_type] = [lj_pair_p1, lj_pair_p2, lj_pair_p3]

                t1, t2 = lj_pair_type.split('-')
                if t1 == t2 and t1 not in ljq_data:
                    ljq_data[t1] = [lj_pair_p1, lj_pair_p2]

            if re_pc:
                lj_type = re_pc.groups()[4]
                lj_p1   = float(re_pc.groups()[2])
                lj_p2   = float(re_pc.groups()[3])

                if not lj_type in ljq_data:
                    ljq_data[lj_type] = [lj_p1, lj_p2]
                else:
                    ljq_data[lj_type] = [lj_p1, lj_p2, ljq_data[lj_type][-1]]

            re_q = re.match('^set\s+type\s+(\d+)\s+charge\s+(-?\d+\.?\d*)\s+\#\s+(\S+)$', line)
            if re_q:
                q_type  = re_q.groups()[2]
                q_index = int(re_q.groups()[0]) - 1
                q_p1    = float(re_q.groups()[1])

                if not q_type in ljq_data:
                    ljq_data[q_type] = [q_p1]
                else:
                    ljq_data[q_type] = [ljq_data[q_type][0], ljq_data[q_type][1], q_p1]
                particle_type_index[q_index] = q_type


            re_bond_coeff = re.match('^bond_coeff\s+(\d+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+\#\s+(\S+)$', line)
            if re_bond_coeff:
                bond_type  = re_bond_coeff.groups()[3]
                bond_index = int(re_bond_coeff.groups()[0]) - 1
                bond_p1    = float(re_bond_coeff.groups()[1])
                bond_p2    = float(re_bond_coeff.groups()[2])

                bond_nvh[bond_type] = [bond_p1, bond_p2]
                bond_type_index[bond_index] = bond_type

            re_ang_coeff = re.match('^angle_coeff\s+(\d+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+\#\s+(\S+)$', line)
            if re_ang_coeff:
                ang_type  = re_ang_coeff.groups()[3]
                ang_index = int(re_ang_coeff.groups()[0]) - 1
                ang_p1    = float(re_ang_coeff.groups()[1])
                ang_p2    = float(re_ang_coeff.groups()[2])

                ang_nvh[ang_type] = [ang_p1, ang_p2]
                ang_type_index[ang_index] = ang_type

            re_dih_coeff = re.match('^dihedral_coeff\s+(\d+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+\#\s+(\S+)$', line)
            if re_dih_coeff:
                dih_type  = re_dih_coeff.groups()[5]
                dih_index = int(re_dih_coeff.groups()[0]) - 1
                dih_p1    = float(re_dih_coeff.groups()[1])
                dih_p2    = float(re_dih_coeff.groups()[2])
                dih_p3    = float(re_dih_coeff.groups()[3])
                dih_p4    = float(re_dih_coeff.groups()[4])

                dih_nvh[dih_type] = [dih_p1, dih_p2, dih_p3, dih_p4]
                dih_type_index[dih_index] = dih_type

    bond_data = matscipy.opls.BondData(bond_nvh)
    ang_data  = matscipy.opls.AnglesData(ang_nvh)
    dih_data  = matscipy.opls.DihedralsData(dih_nvh)

    return (ljq_data, bond_data, ang_data, dih_data,
            particle_type_index, bond_type_index, ang_type_index, dih_type_index)


def read_lammps_data(filename, filename_lammps_params=None):
    """
    Read positions, bonds, angles and dihedrals from a LAMMPS file.
    Optionally, a LAMMPS parameter file can be specified to restore
    all interactions from a preceding simulation.

    Parameters
    ----------
    filename : str
        Name of the file to read the atomic configuration from.
    filename_lammps_params : str, optional
        Name of the file to read the interactions from.

    Returns
    -------
    matscipy.opls.OPLSStructure
        Atomic structure as defined in the input file.

    """
    atoms = ase.io.read(filename, format='lammps-data', Z_of_type=None,
                        style='full', sort_by_id=False, units='metal')

    tags = copy.deepcopy(atoms.numbers)

    # try to guess the atomic numbers from the particle masses
    atomic_numbers = np.empty(len(atoms), dtype=int)
    ams = ase.data.atomic_masses[:]
    ams[np.isnan(ams)] = 0
    for i, mass in enumerate(atoms.get_masses()):
        m2 = (ams - mass)**2
        atomic_numbers[i] = m2.argmin()
    atoms.numbers = atomic_numbers

    opls_struct = matscipy.opls.OPLSStructure(atoms)
    opls_struct.charges = opls_struct.get_array('initial_charges')
    opls_struct.set_tags(tags)
    opls_struct.set_array('molid', atoms.get_array('mol-id'))

    if filename_lammps_params:
        if 'bonds' in atoms.arrays:
            bond_list = []
            for bond_i, bond in enumerate(atoms.get_array('bonds')):
                for item in bond.split(','):
                    re_bond = re.match('(\d+)\((\d+)\)', item)
                    if re_bond:
                        bond_j        = int(re_bond.groups()[0])
                        bond_type_num = int(re_bond.groups()[1])-1
                        bond_list.append([bond_type_num, bond_i, bond_j])
            opls_struct.bond_list = np.array(bond_list)
        else:
            opls_struct.bond_list = []

        if 'angles' in atoms.arrays:
            ang_list = []
            for ang_j, ang in enumerate(atoms.get_array('angles')):
                for item in ang.split(','):
                    re_ang = re.match('(\d+)-(\d+)\((\d+)\)', item)
                    if re_ang:
                        ang_i        = int(re_ang.groups()[0])
                        ang_k        = int(re_ang.groups()[1])
                        ang_type_num = int(re_ang.groups()[2])-1
                        ang_list.append([ang_type_num, ang_i, ang_j, ang_k])
            opls_struct.ang_list = np.array(ang_list)
        else:
            opls_struct.ang_list = []

        if 'dihedrals' in atoms.arrays:
            dih_list = []
            for dih_i, dih in enumerate(atoms.get_array('dihedrals')):
                for item in dih.split(','):
                    re_dih = re.match('(\d+)-(\d+)-(\d+)\((\d+)\)', item)
                    if re_dih:
                        dih_j        = int(re_dih.groups()[0])
                        dih_k        = int(re_dih.groups()[1])
                        dih_l        = int(re_dih.groups()[2])
                        dih_type_num = int(re_dih.groups()[3])-1
                        dih_list.append([dih_type_num, dih_i, dih_j, dih_k, dih_l])
            opls_struct.dih_list = np.array(dih_list)
        else:
            opls_struct.dih_list = []


        # further settings require data in 'filename_lammps_params'
        lammps_params = read_lammps_definitions(filename_lammps_params)


        opls_struct.set_atom_data(lammps_params[0])

        part_type_index = lammps_params[4]

        part_types = np.full(len(opls_struct), None)
        for i, part_type in enumerate(atoms.get_array('type') - 1):
            part_types[i] = part_type_index[part_type]
        opls_struct.set_types(part_types)


        if 'bonds' in atoms.arrays:
            opls_struct.bonds = lammps_params[1]

            bond_type_index = lammps_params[5]

            bond_types = []
            for bond_type_num in np.unique(opls_struct.bond_list.T[0]):
                bond_types.append(bond_type_index[bond_type_num])
            opls_struct.bond_types = bond_types
        else:
            opls_struct.bond_types = []


        if 'angles' in atoms.arrays:
            opls_struct.angles = lammps_params[2]

            ang_type_index = lammps_params[6]

            ang_types = []
            for ang_type_num in np.unique(opls_struct.ang_list.T[0]):
                ang_types.append(ang_type_index[ang_type_num])
            opls_struct.ang_types = ang_types
        else:
            opls_struct.ang_types = []


        if 'dihedrals' in atoms.arrays:
            opls_struct.dihedrals = lammps_params[3]

            dih_type_index = lammps_params[7]

            dih_types = []
            for dih_type_num in np.unique(opls_struct.dih_list.T[0]):
                dih_types.append(dih_type_index[dih_type_num])
            opls_struct.dih_types = dih_types
        else:
            opls_struct.dih_types = []

    return opls_struct


def update_from_lammps_dump(atoms, filename, check=True):
    """
    Read simulation cell, positions and velocities from a LAMMPS
    dump file and use them to update an existing configuration.

    Parameters
    ----------
    atoms : matscipy.opls.OPLSStructure
        Atomic structure to be updated.
    filename : str
        Name of the file to read the atomic configuration from.
    check : bool, optional
        Make sure that the particle types in the input structure
        and the read-in file are the same.

    Returns
    -------
    matscipy.opls.OPLSStructure
        Atomic structure as defined in the input file.

    """
    atoms_dump = ase.io.lammpsrun.read_lammps_dump(filename)

    if len(atoms_dump) != len(atoms):
        raise RuntimeError('Structure in ' + filename +
                           ' has wrong length: %d != %d' %
                           (len(atoms_dump), len(atoms)))

    if check:
        for a, b in zip(atoms, atoms_dump):
            # check that the atom types match
            if not (a.tag + 1 == b.number):
                raise RuntimeError('Atoms index %d are of different '
                                   'type (%d != %d)'
                                   % (a.index, a.tag + 1, b.number))

    atoms.set_cell(atoms_dump.get_cell())
    atoms.set_positions(atoms_dump.get_positions())
    if atoms_dump.get_velocities() is not None:
        atoms.set_velocities(atoms_dump.get_velocities())

    return atoms
