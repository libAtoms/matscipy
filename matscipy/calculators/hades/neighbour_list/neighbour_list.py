from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

from math import sqrt
from .neighbour_list_base import NeighborListBase


class NeighborListHysteretic(NeighborListBase):
    """Neighbor list object.

    cutoffs: list of float
        List of cutoff radii - one for each atom. If the spheres (defined by
        their cutoff radii) of two atoms overlap, they will be counted as
        neighbors.
    skin: float
        If no atom has moved more than the skin-distance since the
        last call to the ``update()`` method, then the neighbor list
        can be reused.  This will save some expensive rebuilds of
        the list, but extra neighbors outside the cutoff will be
        returned.
    sorted: bool
        Wheter the neighbours should be sorted by distance
    self_interaction: bool
        Should an atom return itself as a neighbor?
    bothways: bool
        Return all neighbors.  Default is to return only "half" of
        the neighbors.
    max_neighbours: list of floats
        Maximum neighbours for each atoms, if exceeded, the furthest neighbous
        will be deleted
    hysteretic_break_factor: float
        If atoms are connected, the link will break only of they move apart
        further than cutoff * hysteretic_break_factor
    bondfile_name: str
        Name of the lammps structure file if a list is to be build from it

    Example::

      nl = NeighborList([2.3, 1.7])
      nl.update(atoms)
      indices, offsets = nl.get_neighbours(0)

    """

    def __init__(self, atoms, cutoffs, skin=0.3, sorted=False, self_interaction=True,
                 bothways=False, max_neighbours=None, hysteretic_break_factor=None,
                 bondfile_name=None):

        self.cutoffs = np.asarray(cutoffs)
        self.skin = skin
        self.sorted = sorted
        self.self_interaction = self_interaction
        self.bothways = bothways
        self.nupdates = 0

        # Additional data
        self.old_neighbours = None
        self.hysteretic_break_factor = hysteretic_break_factor
        self.max_neighbours = max_neighbours

        self.init_neighbor_list(atoms, bondfile_name)

    def init_neighbor_list(self, atoms=None, bondfile_name=None):
        """Initialize the neighbor list, either from a LAMMPS structure file
        or from atomic positions

        Parameters
        ----------
        atoms : ase.Atoms
            atoms to initialize the list from
        bondfile_name : str
            Name of the lammps structure file if a list is to be build from it

        """
        if (bondfile_name is not None):
            self.build_from_file(atoms, bondfile_name)
        elif (atoms is not None):
            self.build_from_positions(atoms)
        else:
            raise ValueError("either an atoms object or a bondfile must be specified!")

    def update(self, atoms):
        """Make sure the list is up to date.

        Parameters
        ----------
        atoms : ase.Atoms
            atoms to initialize the list from

        Returns
        -------
        bool
            True of the update was sucesfull
        """
        if self.nupdates == 0:
            self.init_neighbor_list(atoms)
            return True

        if ((self.pbc != atoms.get_pbc()).any() or
            (self.cell != atoms.get_cell()).any() or
            ((self.positions - atoms.get_positions())**2).sum(1).max() >
                self.skin**2):
            self.do_update(atoms)
            return True

        return False

    def do_update(self, atoms):
        """Update the neighbor list by evaluating new distances

        Parameters
        ----------
        atoms : ase.Atoms
            atoms to initialize the list from

        Raises
        ------
        AttributeError
            Must initialize the list first by init_neighbour_list!
        """
        if self.nupdates == 0:
            raise AttributeError("Must initialize the list first by init_neighbour_list!")

        for aIndex in range(len(atoms)):
            if len(self.neighbors[aIndex]) == 0:
                continue
            # disp = atoms.positions[self.neighbors[aIndex]] - atoms.positions[aIndex]
            # dist = np.linalg.norm(disp, ord=2, axis=1)
            dist = atoms.get_distances(aIndex, self.neighbors[aIndex], mic=True)
            cutoffBonds = []
            for metaNIndex, nIndex in enumerate(self.neighbors[aIndex]):
                maxDist = self.cutoffs[aIndex] * self.hysteretic_break_factor[aIndex] +\
                    self.cutoffs[nIndex] * self.hysteretic_break_factor[nIndex]
                if (dist[metaNIndex] > maxDist):
                    cutoffBonds.append(metaNIndex)

            self.neighbors[aIndex] = np.delete(self.neighbors[aIndex], cutoffBonds, axis=0)

    def build_from_file(self, atoms, bondfile_name):
        """Read the nrighbour list from file

        Parameters
        ----------
        atoms : ase.Atoms
            atoms to initialize the list from
        bondfile_name : str
            Name of the lammps structure file if a list is to be build from it

        Raises
        ------
        IOError
            Not all the bond data filled-in, check the structure file formatting!
        """
        with open(bondfile_name, "r") as fp:
            lines = fp.readlines()
            for line in lines:
                if "bonds" in line:
                    nBonds = int(line.split()[0])
                elif "Bonds" in line:
                    bondIndex = lines.index(line)
                    break

            bondlist = np.zeros((nBonds, 2), dtype=int)
            index = 0

            for i in range(bondIndex, bondIndex + nBonds + 10):
                try:
                    a = int(lines[i].split()[2]) - 1
                    b = int(lines[i].split()[3]) - 1
                    bondlist[index, 0] = a
                    bondlist[index, 1] = b
                    index += 1
                except IndexError:
                    pass

                if (index == nBonds):
                    break
            else:
                raise IOError("Not all the bond data filled-in, check the structure file formatting!")

        self.populate_from_bondlist(atoms, bondlist)
        self.nupdates += 1

    def populate_from_bondlist(self, atoms, bondlist):
        """Populate the neighbor list from bondlist

        Parameters
        ----------
        atoms : ase.Atoms
            atoms to initialize the list from
        bondfile_name : str
            Name of the lammps structure file if a list is to be build from it
        """

        self.positions = atoms.get_positions()
        self.pbc = atoms.get_pbc()
        self.cell = atoms.get_cell()

        natoms = len(atoms)
        self.neighbors = [np.empty(0, int) for a in range(natoms)]
        self.displacements = [np.empty((0, 3), int) for a in range(natoms)]

        for bond in bondlist:
            self.neighbors[bond[0]] = np.concatenate((self.neighbors[bond[0]], [bond[1]]))
            self.neighbors[bond[1]] = np.concatenate((self.neighbors[bond[1]], [bond[0]]))

    def build_from_positions(self, atoms):
        """Build the neighbour list based on pairwise distances.

        Parameters
        ----------
        atoms : ase.Atoms
            atoms to initialize the list from

        Raises
        ------
        ValueError
            Must specify cutoff radii for all atoms
        """
        self.positions = atoms.get_positions()
        self.pbc = atoms.get_pbc()
        self.cell = atoms.get_cell()

        if len(self.cutoffs) != len(atoms):
            raise ValueError('Wrong number of cutoff radii: {0} != {1}'
                             .format(len(self.cutoffs), len(atoms)))

        if len(self.cutoffs) > 0:
            rcmax = self.cutoffs.max()
        else:
            rcmax = 0.0

        icell = np.linalg.pinv(self.cell)
        scaled = np.dot(self.positions, icell)
        scaled0 = scaled.copy()

        N = []
        for i in range(3):
            if self.pbc[i]:
                scaled0[:, i] %= 1.0
                v = icell[:, i]
                h = 1 / sqrt(np.dot(v, v))
                n = int(2 * rcmax / h) + 1
            else:
                n = 0
            N.append(n)

        offsets = (scaled0 - scaled).round().astype(int)
        positions0 = atoms.positions + np.dot(offsets, self.cell)
        natoms = len(atoms)
        indices = np.arange(natoms)

        self.nneighbors = 0
        self.npbcneighbors = 0
        self.neighbors = [np.empty(0, int) for a in range(natoms)]
        self.displacements = [np.empty((0, 3), int) for a in range(natoms)]
        for n1 in range(0, N[0] + 1):
            for n2 in range(-N[1], N[1] + 1):
                for n3 in range(-N[2], N[2] + 1):
                    if n1 == 0 and (n2 < 0 or n2 == 0 and n3 < 0):
                        continue
                    displacement = np.dot((n1, n2, n3), self.cell)
                    for a in range(natoms):
                        d = positions0 + displacement - positions0[a]
                        i = indices[(d**2).sum(1) <
                                    (self.cutoffs + self.cutoffs[a])**2]
                        if n1 == 0 and n2 == 0 and n3 == 0:
                            if self.self_interaction:
                                i = i[i >= a]
                            else:
                                i = i[i > a]
                        self.nneighbors += len(i)
                        self.neighbors[a] = np.concatenate(
                            (self.neighbors[a], i))
                        disp = np.empty((len(i), 3), int)
                        disp[:] = (n1, n2, n3)
                        disp += offsets[i] - offsets[a]
                        self.npbcneighbors += disp.any(1).sum()
                        self.displacements[a] = np.concatenate(
                            (self.displacements[a], disp))

        if self.bothways:
            neighbors2 = [[] for a in range(natoms)]
            displacements2 = [[] for a in range(natoms)]
            for a in range(natoms):
                for b, disp in zip(self.neighbors[a], self.displacements[a]):
                    neighbors2[b].append(a)
                    displacements2[b].append(-disp)
            for a in range(natoms):
                nbs = np.concatenate((self.neighbors[a], neighbors2[a]))
                disp = np.array(list(self.displacements[a]) +
                                displacements2[a])
                # Force correct type and shape for case of no neighbors:
                self.neighbors[a] = nbs.astype(int)
                self.displacements[a] = disp.astype(int).reshape((-1, 3))

        if self.sorted:
            for a, i in enumerate(self.neighbors):
                mask = (i < a)
                if mask.any():
                    j = i[mask]
                    offsets = self.displacements[a][mask]
                    for b, offset in zip(j, offsets):
                        self.neighbors[b] = np.concatenate(
                            (self.neighbors[b], [a]))
                        self.displacements[b] = np.concatenate(
                            (self.displacements[b], [-offset]))
                    mask = np.logical_not(mask)
                    self.neighbors[a] = self.neighbors[a][mask]
                    self.displacements[a] = self.displacements[a][mask]

        self.nupdates += 1

    def get_neighbours(self, a):
        """Return neighbors of atom number a.

        A list of indices to neighboring atoms is
        returned.

        Parameters
        ----------
        a : int
            atomic index


        Returns
        -------
        np.array
            array of neighbouring indices

        """

        return self.neighbors[a]
