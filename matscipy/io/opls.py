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
import sys
import distutils.version

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
    """Read extended xyz file with labeled atoms."""
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
    data = {}
    if isinstance(filename, str):
        fileobj = open(filename, 'r')

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
        print('Error: Data block \"%s\" not found in file \"%s\"' % (name, filename))
        sys.exit()

    fileobj.close()
    return data


def read_cutoffs(filename):
    cutoffs = matscipy.opls.CutoffList(read_block(filename, 'Cutoffs'))
    return cutoffs


def read_parameter_file(filename):
    one       = read_block(filename, 'Element')
    bonds     = matscipy.opls.BondData(read_block(filename, 'Bonds'))
    angles    = matscipy.opls.AnglesData(read_block(filename, 'Angles'))
    dihedrals = matscipy.opls.DihedralsData(read_block(filename, 'Dihedrals'))
    cutoffs   = matscipy.opls.CutoffList(read_block(filename, 'Cutoffs'))

    return cutoffs, one, bonds, angles, dihedrals

def write_lammps(prefix, atoms):
    write_lammps_in(prefix)
    write_lammps_atoms(prefix, atoms)
    write_lammps_definitions(prefix, atoms)

def write_lammps_in(prefix):
    if isinstance(prefix, str):
        fileobj = open(prefix + '.in', 'w')
    fileobj.write("""# LAMMPS relaxation (written by ASE)

units           metal
atom_style      full
boundary        p p p
#boundary       p p f

""")
    fileobj.write('read_data ' + prefix + '.atoms\n')
    fileobj.write('include  ' + prefix + '.opls\n')
    fileobj.write("""
kspace_style    pppm 1e-5
#kspace_modify  slab 3.0

neighbor        1.0 bin
neigh_modify    delay 0 every 1 check yes

thermo          1000
thermo_style    custom step temp press cpu pxx pyy pzz pxy pxz pyz ke pe etotal vol lx ly lz atoms

dump            1 all xyz 1000 dump_relax.xyz
dump_modify     1 sort id

restart         100000 test_relax

min_style       fire
minimize        1.0e-14 1.0e-5 100000 100000
""")
    fileobj.close()


def write_lammps_atoms(prefix, atoms):
    """Write atoms input for LAMMPS"""
    if isinstance(prefix, str):
        fileobj = open(prefix + '.atoms', 'w')

    # header
    fileobj.write(fileobj.name + ' (by write_lammps_atoms)\n\n')
    fileobj.write(str(len(atoms)) + ' atoms\n')
    fileobj.write(str(len(atoms.types)) + ' atom types\n')

    blist = atoms.bond_list
    if len(blist):
        btypes = atoms.bond_types
        fileobj.write(str(len(blist)) + ' bonds\n')
        fileobj.write(str(len(btypes)) + ' bond types\n')

    alist = atoms.ang_list
    if len(alist):
        atypes = atoms.ang_types
        fileobj.write(str(len(alist)) + ' angles\n')
        fileobj.write(str(len(atypes)) + ' angle types\n')

    dlist = atoms.dih_list
    if len(dlist):
        dtypes = atoms.dih_types
        fileobj.write(str(len(dlist)) + ' dihedrals\n')
        fileobj.write(str(len(dtypes)) + ' dihedral types\n')

    # cell
    if distutils.version.LooseVersion(ase_version_str) > distutils.version.LooseVersion('3.11.0'):
        p = ase.calculators.lammpsrun.Prism(atoms.get_cell())
    else:
        p = ase.calculators.lammpsrun.prism(atoms.get_cell())

    xhi, yhi, zhi, xy, xz, yz = p.get_lammps_prism()
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

    if distutils.version.LooseVersion(ase_version_str) > distutils.version.LooseVersion('3.17.0'):
        positions_lammps_str = p.vector_to_lammps(atoms.get_positions()).astype(str)
    elif distutils.version.LooseVersion(ase_version_str) > distutils.version.LooseVersion('3.13.0'):
        positions_lammps_str = p.positions_to_lammps_strs(atoms.get_positions())
    else:
        positions_lammps_str = map(p.pos_to_lammps_str, atoms.get_positions())

    for i, r in enumerate(positions_lammps_str):
        q = atoms.atom_data[types[tags[i]]][2]
        fileobj.write('%6d %3d %3d %s %s %s %s' % ((i + 1, molid[i],
                                                    tags[i] + 1, 
                                                    q)
                                                   + tuple(r)))
        fileobj.write(' # ' + atoms.types[tags[i]] + '\n')

    # velocities
    velocities = atoms.get_velocities()
    if velocities is not None:
        fileobj.write('\nVelocities\n\n')
        for i, v in enumerate(velocities):
            fileobj.write('%6d %g %g %g\n' %
                          (i + 1, v[0], v[1], v[2]))

    # masses
    masses = atoms.get_masses()
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
    """Write force field definitions for LAMMPS."""

    if isinstance(prefix, str):
        fileobj = open(prefix + '.opls', 'w')

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
    fileobj.write('pair_style lj/cut/coul/long 10.0 7.4' +
                  ' # consider changing these parameters\n')
    fileobj.write('special_bonds lj/coul 0.0 0.0 0.5\n')
    for ia, atype in enumerate(atoms.types):
        if len(atype) < 2:
            atype = atype + ' '
        fileobj.write('pair_coeff ' + str(ia + 1) + ' ' + str(ia + 1))
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


def read_lammps_data(filename):
    """Read positions, connectivities, etc."""
    
    if isinstance(filename, str):
        fileobj = open(filename, 'r')

    lines = fileobj.readlines()
    lines.pop(0)

    def next_entry():
        line = lines.pop(0).strip()
        if(len(line) > 0):
            lines.insert(0, line)

    def next_key():
        while(len(lines)):
            line = lines.pop(0).strip()
            if(len(line) > 0):
                lines.pop(0)
                return line
        return None

    next_entry()
    header = {}
    while(True):
        line = lines.pop(0).strip()
        if len(line):
            w = line.split()
            if len(w) == 2:
                header[w[1]] = int(w[0])
            else:
                header[w[1] + ' ' + w[2]] = int(w[0])
        else:
            break


    # read box
    next_entry()
    cell = np.zeros(3)
    for i in range(3):
        line = lines.pop(0).strip()
        cell[i] = float(line.split()[1])


    while(not lines.pop(0).startswith('Atoms')):
        pass
    lines.pop(0)


    natoms = header['atoms']
    molid     = np.ones(natoms, dtype=int)
    tags      = np.ones(natoms, dtype=int)
    charges   = np.zeros(natoms, dtype=float)
    positions = np.zeros([natoms,3])
    types     = ['']*header['atom types']
    inconsistent = False

    for line in lines[:natoms]:
        w = line.split()
        i = int(w[0])-1
        molid[i]        = int(w[1])
        tags[i]         = int(w[2])-1
        charges[i]      = float(w[3])
        positions[i][0] = float(w[4])
        positions[i][1] = float(w[5])
        positions[i][2] = float(w[6])

        # try to read atom type from comment
        if len(w) >= 8:
            type = ''.join(w[8:])
            if types[tags[i]] == type:
                pass
            elif types[tags[i]] == '':
                types[tags[i]] = type
            else:
                inconsistent = True

    if inconsistent:
        print('WARNING: Inconsistency between particle descriptions and particle tags found.')
        types = []
        for type in np.unique(tags):
            types.append(str(type))


    opls_struct = matscipy.opls.OPLSStructure(str(natoms)+'H', positions=positions, cell=cell)
    opls_struct.set_tags(tags)
    opls_struct.set_array('molid', molid)

    opls_struct.atom_data = {}
    opls_struct.types = types
    for tag, type in zip(np.unique(tags), types):
        opls_struct.atom_data[type] = [0.0, 0.0, charges[tags == tag][0]]

    del lines[:natoms]

    key = next_key()


    velocities = np.zeros([natoms,3])
    if key == 'Velocities':
        for line in lines[:natoms]:
            w = line.split()
            i = int(w[0])-1
            velocities[i][0] = float(w[1])
            velocities[i][1] = float(w[2])
            velocities[i][2] = float(w[3])

        del lines[:natoms]

        key = next_key()


    if key == 'Masses':
        ntypes = len(opls_struct.atom_data)
        masses = np.empty((ntypes))
        for line in lines[:ntypes]:
            w = line.split()
            i = int(w[0])-1
            masses[i] = float(w[1])

        del lines[:ntypes]

        opls_struct.set_masses(masses[tags])
        opls_struct.set_velocities(velocities)

        # get the elements from the masses
        # this ensures that we have the right elements
        # even when reading from a lammps dump file
        def newtype(element, types):
            if len(element) > 1:
                # can not extend, we are restricted to
                # two characters
                return element
            count = 0
            for type in types:
                if type[0] == element:
                    count += 1
            label = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            return (element + label[count])

        atomic_numbers = np.empty(ntypes, dtype=int)
        ams = ase.data.atomic_masses[:]
        ams[np.isnan(ams)] = 0
        for i, mass in enumerate(masses):
            m2 = (ams - mass)**2
            atomic_numbers[i] = m2.argmin()

        opls_struct.set_atomic_numbers(atomic_numbers[tags])

        key = next_key()
                


    if key != 'Bonds':
        bond_list = np.empty([0,3], dtype=int)
        bond_types = np.empty(0, dtype=str)
    else:
        nbonds = header['bonds']
        bond_list  = np.empty([nbonds,3], dtype=int)
        bond_types = ['']*header['bond types']
        inconsistent = False

        for line in lines[:nbonds]:
            w = line.split()
            i = int(w[0])-1
            bond_list[i][0] = int(w[1])-1
            bond_list[i][1] = int(w[2])-1
            bond_list[i][2] = int(w[3])-1

            # try to read bond type info from comment
            if len(w) >= 5:
                bond_type = ''.join(w[5:])
                if bond_types[bond_list[i][0]-1] == bond_type:
                    pass
                elif bond_types[bond_list[i][0]-1] == '':
                    bond_types[bond_list[i][0]-1] = bond_type
                else:
                    inconsistent = True

        if inconsistent:
            print('WARNING: Inconsistency between bond descriptions and bond type numbers found.')
            bond_types = ['']*header['bond types']

        del lines[:nbonds]

        key = next_key()

    opls_struct.bond_types = bond_types
    opls_struct.bond_list = bond_list



    if key != 'Angles':
        ang_list = np.empty([0,4], dtype=int)
        ang_types = np.empty(0, dtype=str)
    else:
        nangles = header['angles']
        ang_list  = np.empty([nangles,4], dtype=int)
        ang_types = ['']*header['angle types']
        inconsistent = False

        for line in lines[:nangles]:
            w = line.split()
            i = int(w[0])-1
            ang_list[i][0] = int(w[1])-1
            ang_list[i][1] = int(w[2])-1
            ang_list[i][2] = int(w[3])-1
            ang_list[i][3] = int(w[4])-1

            # try to read angle type info from comment
            if len(w) >= 5:
                ang_type = ''.join(w[6:])
                if ang_types[ang_list[i][0]-1] == ang_type:
                    pass
                elif ang_types[ang_list[i][0]-1] == '':
                    ang_types[ang_list[i][0]-1] = ang_type
                else:
                    inconsistent = True

        if inconsistent:
            print('WARNING: Inconsistency between angle descriptions and angle type numbers found.')
            ang_types = ['']*header['angle types']

        del lines[:nangles]

        key = next_key()

    opls_struct.ang_types = ang_types
    opls_struct.ang_list = ang_list



    if key != 'Dihedrals':
        dih_list = np.empty([0,5], dtype=int)
        dih_types = np.empty(header['dihedral types'], dtype=str)
    else:
        ndihedrals = header['dihedrals']
        dih_list = np.empty([ndihedrals,5], dtype=int)
        dih_types = ['']*header['dihedral types']
        inconsistent = False

        for line in lines[:ndihedrals]:
            w = line.split()
            i = int(w[0])-1
            dih_list[i][0] = int(w[1])-1
            dih_list[i][1] = int(w[2])-1
            dih_list[i][2] = int(w[3])-1
            dih_list[i][3] = int(w[4])-1
            dih_list[i][4] = int(w[5])-1

            # try to read dihedral type info from comment
            if len(w) >= 7:
                dih_type = ''.join(w[7:])
                if dih_types[dih_list[i][0]-1] == dih_type:
                    pass
                elif dih_types[dih_list[i][0]-1] == '':
                    dih_types[dih_list[i][0]-1] = dih_type
                else:
                    inconsistent = True

        if inconsistent:
            print('WARNING: Inconsistency between dihedral descriptions and dihedral type numbers found.')
            dih_types = ['']*header['dihedral types']

        del lines[:ndihedrals]

    opls_struct.dih_types = dih_types
    opls_struct.dih_list = dih_list


    return opls_struct


def update_from_lammps_dump(atoms, filename, check=True):
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


