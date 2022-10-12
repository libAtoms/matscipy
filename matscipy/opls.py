#
# Copyright 2016-2017 Andreas Klemenz (Fraunhofer IWM)
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
import numpy as np

import ase
import ase.data
import ase.io
import ase.io.lammpsrun

import matscipy.neighbours
import sys


def twochar(name):
    if len(name) > 1:
        return name[:2]
    else:
        return name + ' '


class BondData:
    def __init__(self, name_value_hash=None):
        if name_value_hash:
            self.nvh = name_value_hash
            self.set_names(name_value_hash.keys())

    def set_names(self, names):
        if not hasattr(self, 'names'):
            self.names = set()
        for name in names:
            aname, bname = name.split('-')
            name1 = twochar(aname) + '-' + twochar(bname)
            name2 = twochar(bname) + '-' + twochar(aname)
            if not name1 in self.names and not name2 in self.names:
                self.names.add(name1)

    def get_name(self, aname, bname):
        name1 = twochar(aname) + '-' + twochar(bname)
        name2 = twochar(bname) + '-' + twochar(aname)
        if name1 in self.names:
            return name1
        elif name2 in self.names:
            return name2
        else:
            return None
    
    def name_value(self, aname, bname):
        name1 = twochar(aname) + '-' + twochar(bname)
        name2 = twochar(bname) + '-' + twochar(aname)
        if name1 in self.nvh:
            return name1, self.nvh[name1]
        if name2 in self.nvh:
            return name2, self.nvh[name2]
        return None, None

    def get_value(self, aname, bname):
        return self.name_value(aname, bname)[1]
        

class CutoffList(BondData):
    def max(self):
        return max(self.nvh.values())


class AnglesData:
    def __init__(self, name_value_hash=None):
        if name_value_hash:
            self.nvh = name_value_hash
            self.set_names(name_value_hash.keys())

    def set_names(self, names):
        if not hasattr(self, 'names'):
            self.names = set()
        for name in names:
            aname, bname, cname = name.split('-')
            name1 = twochar(aname) + '-' + twochar(bname) + '-' + twochar(cname)
            name2 = twochar(cname) + '-' + twochar(bname) + '-' + twochar(aname)
            if not name1 in self.names and not name2 in self.names:
                self.names.add(name1)

    def add_name(self, aname, bname, cname):
        if not hasattr(self, 'names'):
            self.names = set()
        name1 = twochar(aname) + '-' + twochar(bname) + '-' + twochar(cname)
        name2 = twochar(cname) + '-' + twochar(bname) + '-' + twochar(aname)
        if not name1 in self.names and not name2 in self.names:
            self.names.add(name1)

    def get_name(self, aname, bname, cname):
        if not hasattr(self, 'names'):
            return None
        name1 = twochar(aname) + '-' + twochar(bname) + '-' + twochar(cname)
        name2 = twochar(cname) + '-' + twochar(bname) + '-' + twochar(aname)
        if name1 in self.names:
            return name1
        elif name2 in self.names:
            return name2
        else:
            return None
    
    def name_value(self, aname, bname, cname):
        for name in [
            (twochar(aname) + '-' + twochar(bname) + '-' + twochar(cname)),
            (twochar(cname) + '-' + twochar(bname) + '-' + twochar(aname))]:
            if name in self.nvh:
                return name, self.nvh[name]
        return None, None
    

class DihedralsData:
    def __init__(self, name_value_hash=None):
        if name_value_hash:
            self.nvh = name_value_hash
            self.set_names(name_value_hash.keys())

    def set_names(self, names):
        if not hasattr(self, 'names'):
            self.names = set()
        for name in names:
            aname, bname, cname, dname = name.split('-')
            name1 = twochar(aname) + '-' + twochar(bname) + '-' + twochar(cname) + '-' + twochar(dname)
            name2 = twochar(dname) + '-' + twochar(cname) + '-' + twochar(bname) + '-' + twochar(aname)
            if not name1 in self.names and not name2 in self.names:
                self.names.add(name1)

    def add_name(self, aname, bname, cname, dname):
        if not hasattr(self, 'names'):
            self.names = set()
        name1 = twochar(aname) + '-' + twochar(bname) + '-' + twochar(cname) + '-' + twochar(dname)
        name2 = twochar(dname) + '-' + twochar(cname) + '-' + twochar(bname) + '-' + twochar(aname)
        if not name1 in self.names and not name2 in self.names:
            self.names.add(name1)

    def get_name(self, aname, bname, cname, dname):
        if not hasattr(self, 'names'):
            return None
        name1 = twochar(aname) + '-' + twochar(bname) + '-' + twochar(cname) + '-' + twochar(dname)
        name2 = twochar(dname) + '-' + twochar(cname) + '-' + twochar(bname) + '-' + twochar(aname)
        if name1 in self.names:
            return name1
        elif name2 in self.names:
            return name2
        else:
            return None
    
    def name_value(self, aname, bname, cname, dname):
        for name in [
            (twochar(aname) + '-' + twochar(bname) + '-' +
             twochar(cname) + '-' + twochar(dname)),
            (twochar(dname) + '-' + twochar(cname) + '-' + 
             twochar(bname) + '-' + twochar(aname))]:
            if name in self.nvh:
                return name, self.nvh[name]
        return None, None



class OPLSStructure(ase.Atoms):
    default_map = {
        'BR': 'Br',
        'Be': 'Be',
        'C0': 'Ca',
        'Li': 'Li',
        'Mg': 'Mg',
        'Al': 'Al',
        'Ar': 'Ar',
        }

    def __init__(self, *args, **kwargs):
        ase.Atoms.__init__(self, *args, **kwargs)
        if len(self) == 0:
            self.types = None
        else:
            types = np.array(self.get_chemical_symbols())
            self.types = np.unique(types)

            tags = np.zeros(len(self), dtype=int)
            for it, type in enumerate(self.types):
                tags[types == type] = it
            self.set_tags(tags)

        self._combinations = {}
        self._combinations[0] = []
        self._combinations[1] = []
        self._combinations[2] = [(0, 1)]
        self._combinations[3] = [(0, 1), (0, 2), (1, 2)]
        self._combinations[4] = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
        self._combinations[5] = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4), (3, 4)]
        self._combinations[6] = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4), (3, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]
        self._combinations[7] = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4), (3, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6)]
        self._combinations[8] = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4), (3, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)]

    def _get_combinations(self, n):
        r = range(n)
        i = np.tile(r,n)
        j = np.repeat(r,n)
        return zip(i[i < j], j[i < j])

    def append(self, atom):
        """Append atom to end."""
        self.extend(ase.Atoms([atom]))

    def set_types(self, types):
        types = np.array(types)
        self.types = np.unique(types)

        tags = np.zeros(len(self), dtype=int)
        for it, type in enumerate(self.types):
            tags[types == type] = it
        self.set_tags(tags)

    def get_types(self):
        return self.types

    def set_cutoffs(self, cutoffs):
        self.cutoffs = cutoffs

    def get_neighbors(self):
        atoms = ase.Atoms(self)
        types = np.array(self.get_types())
        tags = atoms.get_tags()

        cut = self.cutoffs.max()
        ni, nj, dr = matscipy.neighbours.neighbour_list('ijd', atoms, cut)

        tags2cutoffs = np.full([len(types), len(types)], -1.)
        for i, itype in enumerate(types):
            for j, jtype in enumerate(types):
                tags2cutoffs[i,j] = self.cutoffs.get_value(itype, jtype)

        cutoff_undef = np.where(tags2cutoffs < 0.)
        if np.shape(cutoff_undef)[1] > 0:
            for i in range(np.shape(cutoff_undef)[1]):
                iname = types[cutoff_undef[0][i]]
                jname = types[cutoff_undef[1][i]]
                print('ERROR: Cutoff %s-%s not found' % (iname, jname))
                sys.exit()
        cut = tags2cutoffs[tags[ni], tags[nj]]

        self.ibond = ni[dr <= cut]
        self.jbond = nj[dr <= cut]


    def set_atom_data(self, atom_data):
        self.atom_data = atom_data


    def get_charges(self):
        types = self.types[self.get_tags()]
        self.charges = np.zeros(len(self))

        for i, itype in enumerate(self.types):
            self.charges[self.get_tags() == i] = self.atom_data[itype][2]

        return self.charges


    def get_bonds(self, bonds=None):
        types = np.array(self.get_types())
        tags = self.get_tags()

        if bonds:
            self.bonds = bonds

        if not hasattr(self, 'ibond'):
            self.get_neighbors()
        ibond = self.ibond
        jbond = self.jbond

        # remove duplicates from neighbor list
        mask = jbond <= ibond
        ibond = ibond[mask]
        jbond = jbond[mask]


        tags2bond_names = np.full([len(types), len(types)], '', dtype=object)
        for i, itype in enumerate(types):
            for j, jtype in enumerate(types):
                name = self.cutoffs.get_name(itype, jtype)
                if not name:
                    print('ERROR: Cutoff %s-%s not found' % (itype, jtype))
                    sys.exit()
                tags2bond_names[i,j] = name

        names = tags2bond_names[tags[ibond], tags[jbond]]

        self.bond_types = np.unique(names)

        self.bond_list = np.empty([0,3], dtype=int)
        for n,bond_type in enumerate(self.bond_types):
            mask = names == bond_type
            bond_list_n = np.empty([len(np.where(mask)[0]),3], dtype=int)
            bond_list_n[:,0] = n
            bond_list_n[:,1] = ibond[np.where(mask)]
            bond_list_n[:,2] = jbond[np.where(mask)]
            self.bond_list = np.append(self.bond_list, bond_list_n, axis=0)

        if hasattr(self, 'bonds'):
            potential_unknown = False
            for nb, bond_type in enumerate(self.bond_types):
                itype, jtype = bond_type.split('-')
                if not self.bonds.get_value(itype, jtype):
                    print('ERROR: Pair potential %s-%s not found' % (itype, jtype))
                    print('List of affected bonds:')
                    mask = self.bond_list.T[0] == nb
                    print(self.bond_list[mask].T[1:].T)
                    potential_unknown = True
            if potential_unknown:
                sys.exit()

        return self.bond_types, self.bond_list


    def get_angles(self, angles=None):
        types = np.array(self.get_types())
        tags = self.get_tags()
         
        self.ang_list = []
        self.ang_types = []

        if not hasattr(self, 'ibond'):
            self.get_neighbors()
        ibond = self.ibond
        jbond = self.jbond

        if angles:
            self.angles = angles

        if not hasattr(self, 'angles'):
            self.angles = AnglesData()
            for itype in types:
                for jtype in types:
                    for ktype in types:
                        self.angles.add_name(itype, jtype, ktype)



        angles_undef = AnglesData()
        angles_undef_lists = {}

        for i in range(len(self)):
            iname = types[tags[i]]

            ineigh = jbond[ibond == i]
            n_ineigh = np.shape(ineigh)[0]

            if n_ineigh not in self._combinations.keys():
                self._combinations[n_ineigh] = self._get_combinations(n_ineigh)

            for nj,nk in self._combinations[len(ineigh)]:
                j = ineigh[nj]
                k = ineigh[nk]

                jname = types[tags[j]]
                kname = types[tags[k]]
                name = self.angles.get_name(jname, iname, kname)

                if hasattr(self, 'angles') and not name:
                    # Angle found without matching parameter definition
                    # Add to list anyway to get meaningful error messages
                    if not angles_undef.get_name(jname, iname, kname):
                        angles_undef.add_name(jname, iname, kname)
                        angles_undef_lists[angles_undef.get_name(jname, iname, kname)] = [[j, i, k]]
                    else:
                        angles_undef_lists[angles_undef.get_name(jname, iname, kname)].append([j, i, k])
                    continue

                if name not in self.ang_types:
                    self.ang_types.append(name)
                self.ang_list.append([self.ang_types.index(name), j, i, k])


        if len(angles_undef_lists) > 0:
            for name in angles_undef_lists:
                print('ERROR: Angular potential %s not found' % (name))
                print('List of affected angles:')
                for angle in angles_undef_lists[name]:
                    print(angle)
            sys.exit()

        return self.ang_types, self.ang_list



    def get_dihedrals(self, dihedrals=None, full_output=False):
        types = self.get_types()
        tags = self.get_tags()

        if dihedrals:
            self.dihedrals = dihedrals

        if not hasattr(self, 'dihedrals'):
            self.dihedrals = DihedralsData()
            for itype in types:
                for jtype in types:
                    for ktype in types:
                        for ltype in types:
                            self.dihedrals.add_name(itype, jtype, ktype, ltype)


        self.dih_list = []
        self.dih_types = []

        if not hasattr(self, 'ibond'):
            self.get_neighbors()
        ibond = self.ibond
        jbond = self.jbond


        dihedrals_undef_lists = {}

        for j,k in zip(ibond, jbond):
            if j < k:
                jname = types[tags[j]]
                kname = types[tags[k]]

                i_dihed = jbond[ibond == j]
                l_dihed = jbond[ibond == k]
                i_dihed = i_dihed[i_dihed != k]
                l_dihed = l_dihed[l_dihed != j]

                for i in i_dihed:
                    iname = types[tags[i]]
                    for l in l_dihed:
                        lname = types[tags[l]]

                        name = self.dihedrals.get_name(iname, jname, kname, lname)

                        if hasattr(self, 'dihedrals') and not name:
                            # Dihedral found without matching parameter definition
                            # Add to list anyway to get meaningful error messages
                            name = iname + '-' + jname + '-' + kname + '-' + lname
                            if name not in dihedrals_undef_lists:
                                dihedrals_undef_lists[name] = [[i, j, k, l]]
                            else:
                                dihedrals_undef_lists[name].append([i, j, k, l])
                            continue

                        if name not in self.dih_types:
                            self.dih_types.append(name)
                        self.dih_list.append([self.dih_types.index(name), i, j, k, l])


        if len(dihedrals_undef_lists) > 0:
            # "dihedrals_undef_lists" might contain duplicates, i.e. A-B-C-D and D-C-B-A.
            # This could be avoided by using a "DihedralsData" object to store dihedral
            # names, similar to the way the "AnglesData" object is used in the
            # "get_angles()" method. For performance reasons, this is not done here.
            for name in dihedrals_undef_lists:
                print('WARNING: Dihedral potential %s not found' % (name))
                if not full_output:
                    print('Example for affected atoms: %s' % (str(dihedrals_undef_lists[name][0])))
                else:
                    print('Full list of affected atoms:')
                    for dihed in dihedrals_undef_lists[name]:
                        print(dihed)

        return self.dih_types, self.dih_list


