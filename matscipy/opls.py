#
# Copyright 2016-2017, 2023 Andreas Klemenz (Fraunhofer IWM)
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
import ase
import ase.data
import ase.io
import ase.io.lammpsrun
import matscipy.neighbours


def twochar(name):
    """
    Ensures that the particle names have a length of exactly two
    characters.

    Parameters
    ----------
    name : str
        Particle name.

    Returns
    -------
    str
        Particle name with exactly 2 characters.
    """
    if len(name) > 1:
        return name[:2]
    else:
        return name + ' '


class LJQData(dict):
    """
    Store Lennard-Jones parameters and charges for each particle type. In
    the simplest version, each particle type has one set of Lennard-Jones
    parameters, with geometric mixing applied between parameters of
    different types. Parameters for individual pairs of particle types can
    be specified in the ``lj_pairs`` dictionary.

    Example:
    Set the Lennard-Jones and Coulomb cutoffs to 12 and 8 Angstroms, geometric
    mixing of the Lennard-Jones parameters for particles ``C1`` and ``C2`` and
    between ``C2`` and ``C3``, custom parameters and cutoff for the interaction
    between ``C1`` and ``C3``:
    ::
      LJQData.lj_cutoff = 12.0
      LJQData.c_cutoff  =  8.0

      LJQData['C1'] = [LJ-epsilon (eV), LJ-sigma (A), charge (e)]
      LJQData['C2'] = [LJ-epsilon (eV), LJ-sigma (A), charge (e)]
      LJQData['C3'] = [LJ-epsilon (eV), LJ-sigma (A), charge (e)]

      LJQData.lj_pairs['C1-C3'] = [epsilon (eV), sigma (A), cutoff (A)]
    """
    def __init__(self, args):
        dict.__init__(self, args)

        # default cutoffs
        self.lj_cutoff = 10.0
        self.c_cutoff = 7.4

        self.lj_pairs = {}


class BondData:
    """
    Store spring constants and equilibrium distances for harmonic potentials
    and ensure correct handling of permutations. See documentation of the
    LAMMPS ``bond_style harmonic`` command for details.
    """

    def __init__(self, name_value_hash=None):
        """
        Parameters
        ----------
        name_value_hash : dict, optional
            Bond names and corresponding potential parameters, e.g.
            ::
              {'AA-BB': [10., 1.], 'AA-CC': [20., 2.], ...}.
        """
        if name_value_hash:
            self.nvh = name_value_hash
            self.set_names(name_value_hash.keys())

    def set_names(self, names):
        """
        Create a list of participating particles from a list of bond
        types.

        Parameters
        ----------
        names : list
            List of bond type names, e.g.
            ::
              ['AA-BB', 'AA-CC', ...]
        """
        if not hasattr(self, 'names'):
            self.names = set()
        for name in names:
            aname, bname = name.split('-')
            name1 = twochar(aname) + '-' + twochar(bname)
            name2 = twochar(bname) + '-' + twochar(aname)
            if name1 not in self.names and name2 not in self.names:
                self.names.add(name1)

    def get_name(self, aname, bname):
        """
        Returns the name of the bond type between two particle types
        as it is defined internally. If one particle is named ``AA``
        and the other is named ``BB``, the bond type between them could
        be either ``AA-BB`` or ``BB-AA``. The parameters would be the same
        and are stored only once.

        Parameters
        ----------
        aname : str
            Name of first particle.
        bname : str
            Name of second particle.

        Returns
        -------
        str
            Bond-name, either ``aname-bname`` or ``bname-aname``.
        """
        name1 = twochar(aname) + '-' + twochar(bname)
        name2 = twochar(bname) + '-' + twochar(aname)
        if name1 in self.names:
            return name1
        elif name2 in self.names:
            return name2
        else:
            return None

    def name_value(self, aname, bname):
        """
        Returns the name of a bond type between two particles of type
        ``aname`` and ``bname`` as stored internally and the corresponding
        potential parameters, i.e. the spring constant and the equilibrium
        distance.

        Parameters
        ----------
        aname : str
            Name of first particle.
        bname : str
            Name of second particle.

        Returns
        -------
        name : str
            Bond-name, either ``aname-bname`` or ``bname-aname``.
        parameters : list
            Potential parameters, i.e. spring constant and
            equilibrium distance.
        """
        name1 = twochar(aname) + '-' + twochar(bname)
        name2 = twochar(bname) + '-' + twochar(aname)
        if name1 in self.nvh:
            return name1, self.nvh[name1]
        if name2 in self.nvh:
            return name2, self.nvh[name2]
        return None, None

    def get_value(self, aname, bname):
        """
        Returns the potential parameters for a bond between two
        particles of type ``aname`` and ``bname``, i.e. the spring
        constant and the equilibrium distance.

        Parameters
        ----------
        aname : str
            Name of first particle.
        bname : str
            Name of second particle.

        Returns
        -------
        list
            Potential parameters, i.e. spring constant and
            equilibrium distance.
        """
        return self.name_value(aname, bname)[1]


class CutoffList(BondData):
    """
    Store cutoffs for pair interactions and ensure correct handling of
    permutations. Cutoffs can be used to automatically find all interacting
    atoms of a :class:`matscipy.opls.OPLSStructure` object based on a simple
    distance criterion.
    """

    def max(self):
        return max(self.nvh.values())


class AnglesData:
    """
    Store spring constants and equilibrium angles for harmonic potentials
    and ensure correct handling of permutations. See documentation of the
    LAMMPS ``angle_style harmonic`` command for details.
    """

    def __init__(self, name_value_hash=None):
        """
        Parameters
        ----------
        name_value_hash : dict, optional
            Angle names and corresponding potential parameters, e.g.
            ::
              {'AA-BB-CC': [10., 1.], 'AA-CC-BB': [20., 2.], ...}
        """
        if name_value_hash:
            self.nvh = name_value_hash
            self.set_names(name_value_hash.keys())

    def set_names(self, names):
        """
        Create a list of participating particles from a list of angle
        types.

        Parameters
        ----------
        names : list
            List of angle type names, e.g.
            ::
              ['AA-BB-CC', 'AA-CC-BB', ...]
        """
        if not hasattr(self, 'names'):
            self.names = set()
        for name in names:
            aname, bname, cname = name.split('-')
            name1 = (twochar(aname) + '-' +
                     twochar(bname) + '-' +
                     twochar(cname))
            name2 = (twochar(cname) + '-' +
                     twochar(bname) + '-' +
                     twochar(aname))
            if name1 not in self.names and name2 not in self.names:
                self.names.add(name1)

    def add_name(self, aname, bname, cname):
        """
        Add an angle type to the internal list if not already present.

        Parameters
        ----------
        aname : str
            Name of first particle.
        bname : str
            Name of second particle.
        cname : str
            Name of third particle.
        """
        if not hasattr(self, 'names'):
            self.names = set()
        name1 = (twochar(aname) + '-' +
                 twochar(bname) + '-' +
                 twochar(cname))
        name2 = (twochar(cname) + '-' +
                 twochar(bname) + '-' +
                 twochar(aname))
        if name1 not in self.names and name2 not in self.names:
            self.names.add(name1)

    def get_name(self, aname, bname, cname):
        """
        Returns the name of the angle type between three particle
        types as it is defined internally. If the particles are named
        ``AA``, ``BB``, ``CC``, the angle type could be ``AA-BB-CC``
        or ``CC-BB-AA``. The parameters would be the same and are
        stored only once.

        Parameters
        ----------
        aname : str
            Name of first particle.
        bname : str
            Name of second particle.
        cname : str
            Name of third particle.

        Returns
        -------
        str
            Angle-name, either ``aname-bname-cname``
            or ``cname-bname-aname``.
        """
        if not hasattr(self, 'names'):
            return None
        name1 = (twochar(aname) + '-' +
                 twochar(bname) + '-' +
                 twochar(cname))
        name2 = (twochar(cname) + '-' +
                 twochar(bname) + '-' +
                 twochar(aname))
        if name1 in self.names:
            return name1
        elif name2 in self.names:
            return name2
        else:
            return None

    def name_value(self, aname, bname, cname):
        """
        Returns the name of an angle type between three particles of
        type ``aname``, ``bname`` and ``cname`` as stored internally
        and the corresponding potential parameters.

        Parameters
        ----------
        aname : str
            Name of first particle.
        bname : str
            Name of second particle.
        cname : str
            Name of third particle.

        Returns
        -------
        name : str
            Angle-name, either ``aname-bname-cname``
            or ``cname-bname-aname``.
        parameters : list
            Potential parameters.
        """
        for name in [(twochar(aname) + '-' +
                      twochar(bname) + '-' +
                      twochar(cname)),
                     (twochar(cname) + '-' +
                      twochar(bname) + '-' +
                      twochar(aname))
                     ]:
            if name in self.nvh:
                return name, self.nvh[name]
        return None, None


class DihedralsData:
    """
    Store energy constants for dihedral potentials and ensure correct handling
    of permutations. See documentation of the LAMMPS ``dihedral_style opls``
    command for details.
    """

    def __init__(self, name_value_hash=None):
        """
        Parameters
        ----------
        name_value_hash : dict, optional
            Dihedral names and corresponding potential parameters, e.g.
            ::
              {'AA-BB-CC-DD': [1., 1., 1., 1.],
               'AA-BB-AA-BB': [2., 2., 2., 2.], ...}
        """
        if name_value_hash:
            self.nvh = name_value_hash
            self.set_names(name_value_hash.keys())

    def set_names(self, names):
        """
        Create a list of participating particles from a list of
        dihedral types.

        Parameters
        ----------
        names : list
            List of dihedral type names, e.g.
            ::
              ['AA-BB-CC-DD', 'AA-BB-AA-BB', ...]
        """
        if not hasattr(self, 'names'):
            self.names = set()
        for name in names:
            aname, bname, cname, dname = name.split('-')
            name1 = (twochar(aname) + '-' + twochar(bname) + '-' +
                     twochar(cname) + '-' + twochar(dname))
            name2 = (twochar(dname) + '-' + twochar(cname) + '-' +
                     twochar(bname) + '-' + twochar(aname))
            if name1 not in self.names and name2 not in self.names:
                self.names.add(name1)

    def add_name(self, aname, bname, cname, dname):
        """
        Add a dihedral type to the internal list if not already
        present.

        Parameters
        ----------
        aname : str
            Name of first particle.
        bname : str
            Name of second particle.
        cname : str
            Name of third particle.
        dname : str
            Name of fourth particle.
        """
        if not hasattr(self, 'names'):
            self.names = set()
        name1 = (twochar(aname) + '-' + twochar(bname) + '-' +
                 twochar(cname) + '-' + twochar(dname))
        name2 = (twochar(dname) + '-' + twochar(cname) + '-' +
                 twochar(bname) + '-' + twochar(aname))
        if name1 not in self.names and name2 not in self.names:
            self.names.add(name1)

    def get_name(self, aname, bname, cname, dname):
        """
        Add a dihedral type to the internal list if not already
        present.

        Parameters
        ----------
        aname : str
            Name of first particle.
        bname : str
            Name of second particle.
        cname : str
            Name of third particle.
        dname : str
            Name of fourth particle.
        """
        if not hasattr(self, 'names'):
            return None
        name1 = (twochar(aname) + '-' + twochar(bname) + '-' +
                 twochar(cname) + '-' + twochar(dname))
        name2 = (twochar(dname) + '-' + twochar(cname) + '-' +
                 twochar(bname) + '-' + twochar(aname))
        if name1 in self.names:
            return name1
        elif name2 in self.names:
            return name2
        else:
            return None

    def name_value(self, aname, bname, cname, dname):
        """
        Returns the name of a dihedral type between four particles of
        type ``aname``, ``bname``, ``cname`` and ``dname`` as stored
        internally and the corresponding potential parameters.

        Parameters
        ----------
        aname : str
            Name of first particle.
        bname : str
            Name of second particle.
        cname : str
            Name of third particle.
        dname : str
            Name of fourth particle.

        Returns
        -------
        name : str
            Angle-name, either ``aname-bname-cname-dname``
            or ``dname-cname-bname-aname``.
        parameters : list
            Potential parameters.
        """
        for name in [(twochar(aname) + '-' + twochar(bname) + '-' +
                      twochar(cname) + '-' + twochar(dname)),
                     (twochar(dname) + '-' + twochar(cname) + '-' +
                      twochar(bname) + '-' + twochar(aname))
                     ]:
            if name in self.nvh:
                return name, self.nvh[name]
        return None, None


class OPLSStructure(ase.Atoms):
    """
    Extension of the :class:`ase.Atoms` class for non-reactive simulations.
    """

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
        """
        Set a type for each atom to specify the interaction with its neighbors.
        This enables atoms of the same element to have different interaction
        potentials. During initialization, the types are initially derived from
        the chemical symbols of the atoms.
        """
        ase.Atoms.__init__(self, *args, **kwargs)
        if len(self) == 0:
            self.types = None
        else:
            types = np.array(self.get_chemical_symbols())
            self.types = np.unique(types)

            tags = np.zeros(len(self), dtype=int)
            for itype, type_str in enumerate(self.types):
                tags[types == type_str] = itype
            self.set_tags(tags)

        # Angle lists are generated from neighbor lists. Assume an atom with the
        # number 2 has the three neighbors [4, 6, 9]. Then the following angles
        # with 2 in the middle are possible: (4-2-6), (4-2-9), (6-2-9) and the
        # equivalent angles (6-2-4), (9-2-4) and (9-2-6). self._combinations
        # contains predefined combinations of indices of the neighbor lists for
        # the most frequently occurring numbers of nearest neighbors. With these,
        # the list of occurring angles can be determined much faster than if the
        # combinations had to be calculated in each step.
        self._combinations = {}
        self._combinations[0] = []
        self._combinations[1] = []
        self._combinations[2] = [(0, 1)]
        self._combinations[3] = [(0, 1), (0, 2), (1, 2)]
        self._combinations[4] = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3),
                                 (2, 3)]
        self._combinations[5] = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3),
                                 (2, 3), (0, 4), (1, 4), (2, 4), (3, 4)]
        self._combinations[6] = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3),
                                 (2, 3), (0, 4), (1, 4), (2, 4), (3, 4),
                                 (0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]
        self._combinations[7] = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3),
                                 (2, 3), (0, 4), (1, 4), (2, 4), (3, 4),
                                 (0, 5), (1, 5), (2, 5), (3, 5), (4, 5),
                                 (0, 6), (1, 6), (2, 6), (3, 6), (4, 6),
                                 (5, 6)]
        self._combinations[8] = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3),
                                 (2, 3), (0, 4), (1, 4), (2, 4), (3, 4),
                                 (0, 5), (1, 5), (2, 5), (3, 5), (4, 5),
                                 (0, 6), (1, 6), (2, 6), (3, 6), (4, 6),
                                 (5, 6), (0, 7), (1, 7), (2, 7), (3, 7),
                                 (4, 7), (5, 7), (6, 7)]

    def _get_combinations(self, n):
        """
        Fallback for a large number of neighbors for which the
        possible combinations are not included in self._combinations

        Parameters
        ----------
        n : int
            Number of next neighbors of an atom.

        Returns
        -------
        list
            See documentation of self._combinations for details.

        """
        r = range(n)
        i = np.tile(r, n)
        j = np.repeat(r, n)
        return zip(i[i < j], j[i < j])

    def append(self, atom):
        """
        Append atom to the end.

        Parameters
        ----------
        atom : ase.Atoms
            The particle to append.
        """
        self.extend(ase.Atoms([atom]))

    def set_types(self, types):
        """
        Set a type for each atom to specify the interaction with its
        neighbors. This enables atoms of the same element to have
        different interaction potentials.

        Parameters
        ----------
        types : list
            A list of strings that specify the type of each atom.
        """
        types = np.array(types)
        self.types = np.unique(types)

        tags = np.zeros(len(self), dtype=int)
        for itype, type_str in enumerate(self.types):
            tags[types == type_str] = itype
        self.set_tags(tags)

    def get_types(self):
        """
        Returns a unique list of atom types.

        Returns
        -------
        numpy.ndarray
            Particle types. Each element is a :class:`str` with two
            characters.
        """
        return self.types

    def set_cutoffs(self, cutoffs):
        """
        Add a CutoffList object to the structure. This allows the
        :meth:`matscipy.opls.OPLSStructure.get_neighbors` method
        to find all interacting atoms of the structure based on a
        simple distance criterion.

        Parameters
        ----------
        cutoffs : opls.CutoffList
            Cutoffs.
        """
        self.cutoffs = cutoffs

    def get_neighbors(self):
        """
        Find all atoms which might interact with each
        other based on a simple distance criterion.
        """
        atoms = ase.Atoms(self)
        types = np.array(self.get_types())
        tags = atoms.get_tags()

        cut = self.cutoffs.max()
        ni, nj, dr = matscipy.neighbours.neighbour_list('ijd', atoms, cut)

        tags2cutoffs = np.full([len(types), len(types)], -1.)
        for i, itype in enumerate(types):
            for j, jtype in enumerate(types):
                cutoff = self.cutoffs.get_value(itype, jtype)
                if cutoff is not None:
                    tags2cutoffs[i, j] = self.cutoffs.get_value(itype, jtype)

        cutoff_undef = np.where(tags2cutoffs < 0.)
        if np.shape(cutoff_undef)[1] > 0:
            for i in range(np.shape(cutoff_undef)[1]):
                iname = types[cutoff_undef[0][i]]
                jname = types[cutoff_undef[1][i]]
                raise RuntimeError(f'Cutoff {iname}-{jname} not found')
        cut = tags2cutoffs[tags[ni], tags[nj]]

        self.ibond = ni[dr <= cut]
        self.jbond = nj[dr <= cut]

    def set_atom_data(self, atom_data):
        """
        Set Lennard-Jones parameters and atomic charges. Notice that each
        atom has exactly one set of Lennard-Jones parameters. Parameters
        for interactions between different types of atoms are calculated
        by geometric mixing. See documentation of the LAMMPS
        ``pair_modify`` command for details.

        Parameters
        ----------
        atom_data : dict
            Dictionary containing Lennard-Jones parameters and charges for
            each particle type. key: ``Particle type``, one or two
            characters, value: ``[LJ-epsilon, LJ-sigma, charge]``.
        """
        self.atom_data = atom_data

    def get_charges(self):
        """
        Return an array of atomic charges. Same functionality as the
        :meth:`ase.Atoms.get_charges` method, but atomic charges are
        taken from a user definition instead of the result of a
        calculation.

        Returns
        -------
        numpy.ndarray
            Particle charges.
        """
        self.charges = np.zeros(len(self), dtype=float)

        for i, itype in enumerate(self.types):
            self.charges[self.get_tags() == i] = self.atom_data[itype][2]

        return self.charges

    def get_bonds(self, bonds=None):
        """
        Returns an array of all bond types and an array of all bonds
        in the system. This method also checks if potential parameters
        for all found bonds are present.

        Parameters
        ----------
        bonds : opls.BondData, optional
            Pairwise potential parameters. Can be set here or elsewhere
            in the code, e.g. by setting the attribute
            :attr:`matscipy.opls.OPLSStructure.bonds`. If it is present,
            this method runs a test to check if interaction parameters
            are defined for all present bonds.

        Returns
        -------
        bond_types : numpy.ndarray
            Array of strings characterizing all present bond types.
            Example: A system consists of particles with the types
            ``A1`` and ``A2``. If all particles interact with each
            other, ``bond_types`` will be
            ::
              ['A1-A1', 'A1-A2', 'A2-A2']
            If there were no interactions between the types ``A1``
            and ``A2``, ``bond_types`` would be
            ::
              ['A1-A1', 'A2-A2']
        bond_list : numpy.ndarray
            ``bond_list.shape = (n, 3)`` where ``n`` is the number
            of particles in the system. Contains arrays of 3 integers
            for each bond in the system. First number: interaction
            type, index of ``bond_types``, second and third numbers:
            indicees of participating particles.
            Example: A system consists of 3 particles of type ``AA``,
            all particles are interacting. bond_types would be 
            ::
              ['AA-AA']
            and bond_list would be
            ::
              [[0, 0, 1], [0, 0, 2], [0, 1, 2]]

        Raises
        ------
        RuntimeError
            If ``self.bonds`` is present and bonds are found for which
            no parameters are defined. In this case a warning a full
            list of all affected bonds will be printed on ``STDOUT``.
        """
        types = np.array(self.get_types())
        tags = self.get_tags()

        if bonds is not None:
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
                if name is None:
                    raise RuntimeError(f'Cutoff {itype}-{jtype} not found')
                tags2bond_names[i, j] = name

        names = tags2bond_names[tags[ibond], tags[jbond]]

        self.bond_types = np.unique(names)

        self.bond_list = np.empty([0, 3], dtype=int)
        for n, bond_type in enumerate(self.bond_types):
            mask = names == bond_type
            bond_list_n = np.empty([len(np.where(mask)[0]), 3], dtype=int)
            bond_list_n[:, 0] = n
            bond_list_n[:, 1] = ibond[np.where(mask)]
            bond_list_n[:, 2] = jbond[np.where(mask)]
            self.bond_list = np.append(self.bond_list, bond_list_n, axis=0)

        if hasattr(self, 'bonds'):
            potential_unknown = False
            for nb, bond_type in enumerate(self.bond_types):
                itype, jtype = bond_type.split('-')
                if self.bonds.get_value(itype, jtype) is None:
                    print('ERROR: Pair potential %s-%s not found' %
                          (itype, jtype))
                    print('List of affected bonds:')
                    mask = self.bond_list.T[0] == nb
                    print(self.bond_list[mask].T[1:].T)
                    potential_unknown = True
            if potential_unknown:
                raise RuntimeError('Undefined pair potentials.')

        return self.bond_types, self.bond_list

    def get_angles(self, angles=None):
        """
        Returns an array of all angle types and an array of all
        angles in the system. This method also checks if potential
        parameters for all found angles are present.

        Parameters
        ----------
        angles : opls.AnglesData, optional
            Potential parameters. Can be set here or elsewhere in the
            code, e.g. by setting the attribute
            :attr:`matscipy.opls.OPLSStructure.angles`.

        Returns
        -------
        ang_types : list
            Array of strings characterizing all present angle types.
            Example: A system consists of atoms of types ``A1`` and
            ``A2``, all conceivable angle types are present in the
            system. ``ang_types`` would be
            ::
              ['A1-A1-A1', 'A1-A1-A2', 'A1-A2-A1', 'A1-A2-A2', 'A2-A1-A2', 'A2-A2-A2']
        ang_list : list
            ``len(ang_list) = n`` where ``n`` is the number of particles
            in the system. Each list entry is a list of 4 integers,
            characterizing the angles present in the system.
            Example:
            A system contains 7 atoms, ``(0,1,2)`` of type ``A1`` and
            ``(3,4,5,6)`` of type ``A2``. If there are angles between
            ``(0,1,2)``, ``(0,3,4)`` and ``(0,5,6)``, ``ang_list``
            would be
            ::
              ['A1-A1-A1', 'A2-A1-A2']
            and ``ang_list`` would be
            ::
              [[0, 0, 1, 2], [1, 0, 3, 4], [1, 0, 5, 6]]

        Raises
        ------
        RuntimeError
            If ``self.angles`` is present and angles are found for which
            no parameters are defined. In this case a warning a full list
            of all affected angles will be printed on ``STDOUT``.
        """
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

            for nj, nk in self._combinations[len(ineigh)]:
                j = ineigh[nj]
                k = ineigh[nk]

                jname = types[tags[j]]
                kname = types[tags[k]]
                name = self.angles.get_name(jname, iname, kname)

                if hasattr(self, 'angles') and name is None:
                    # Angle found without matching parameter definition
                    # Add to list anyway to get meaningful error messages
                    if not angles_undef.get_name(jname, iname, kname):
                        angles_undef.add_name(jname, iname, kname)
                        angles_undef_lists[
                            angles_undef.get_name(jname, iname, kname)
                            ] = [[j, i, k]]
                    else:
                        angles_undef_lists[
                            angles_undef.get_name(jname, iname, kname)
                            ].append([j, i, k])
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
            raise RuntimeError('Undefined angular potentials.')

        return self.ang_types, self.ang_list

    def get_dihedrals(self, dihedrals=None, full_output=False):
        """
        Returns an array of all dihedral types and an array of all
        dihedrals in the system. This method also checks if potential
        parameters for all found dihedrals are present.

        Parameters
        ----------
        dihedrals : opls.DihedralsData, optional
            Potential parameters. Can be set here or elsewhere in the
            code, e.g. by setting the attribute
            :attr:`matscipy.opls.OPLSStructure.dihedrals`.
        full_output : bool, optional
            Print a full list of all found dihedrals on ``STDOUT`` for
            which no potential parameters are defined. By default, only
            one example is printed. Having the full list is sometimes
            helpful for debugging.

        Returns
        -------
        dih_types : list
            Array of strings characterizing all present dihedral types.
            Example: Consider a system consisting of one benzene molecule.
            There are three possible types of dihedrals and ``dih_type``
            would be
            ::
              ['C-C-C-C', 'C-C-C-H', 'H-C-C-H']
        dih_list : list
            ``len(dih_list) = n`` where ``n`` is the number of particles in the
            system. Each list entry is a list of 5 integers, characterizing the
            dihedrals present in the system.
            Example: Consider a system consisting of one benzene molecule with
            the ``C`` atoms ``(0,1,2,3,4,5)`` and the ``H`` atoms
            ``(6,7,8,9,10,11)``. ``dih_type`` would be
            ::
              ['C-C-C-C', 'C-C-C-H', 'H-C-C-H']
            and ``dih_list`` would be
            ::
              [[0, 0, 1, 2, 3], [0, 1, 2, 3, 4], ... , [1, 6, 0, 1, 2],
               [1, 7, 1, 2, 3], ... , [2, 6, 0, 1, 7], [2, 7, 1, 2, 8], ...]
        """
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

        for j, k in zip(ibond, jbond):
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

                        name = self.dihedrals.get_name(iname, jname,
                                                       kname, lname)

                        if hasattr(self, 'dihedrals') and not name:
                            # Dihedral found without matching parameter
                            # definition. Add to list anyway to get
                            # meaningful error messages.
                            name = (iname + '-' + jname + '-' +
                                    kname + '-' + lname)
                            if name not in dihedrals_undef_lists:
                                dihedrals_undef_lists[name] = [[i, j, k, l]]
                            else:
                                dihedrals_undef_lists[name].append([i, j,
                                                                    k, l])
                            continue

                        if name not in self.dih_types:
                            self.dih_types.append(name)
                        self.dih_list.append([self.dih_types.index(name),
                                              i, j, k, l])

        if len(dihedrals_undef_lists) > 0:
            # "dihedrals_undef_lists" might contain duplicates,
            # i.e. A-B-C-D and D-C-B-A. This could be avoided by
            # using a "DihedralsData" object to store dihedral
            # names, similar to the way the "AnglesData" object
            # is used in the "get_angles()" method. For performance
            # reasons, this is not done here.
            for name in dihedrals_undef_lists:
                print('WARNING: Dihedral potential %s not found' % (name))
                if not full_output:
                    print('Example for affected atoms: %s' %
                          (str(dihedrals_undef_lists[name][0])))
                else:
                    print('Full list of affected atoms:')
                    for dihed in dihedrals_undef_lists[name]:
                        print(dihed)

        return self.dih_types, self.dih_list
