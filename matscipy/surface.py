#
# Copyright 2014, 2020 James Kermode (Warwick U.)
#           2019 James Brixey (Warwick U.)
#           2015 Punit Patel (Warwick U.)
#           2014 Lars Pastewka (U. Freiburg)
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

import itertools
import functools

import numpy as np
from numpy.linalg import norm, inv

def gcd(a, b):
    """Calculate the greatest common divisor of a and b"""
    while b:
        a, b = b, a%b
    return a

class MillerIndex(np.ndarray):
    """
    Representation of a three of four index Miller direction or plane

    A :class:`MillerIndex` can be constructed from vector or parsed from a string::

        x = MillerIndex('-211')
        y = MillerIndex('111', type='plane')
        z = x.cross(y)
        print x # prints "[-211]"
        print y # prints "(111)", note round brackets denoting a plane
        print z.latex()
        assert(angle_between(x,y) == pi/2.)
        assert(angle_between(y,z) == pi/2.)
        assert(angle_between(x,z) == pi/2.)
    """

    __array_priority__ = 101.0

    brackets = {'direction': '[]',
                'direction_family': '<>',
                'plane': '()',
                'plane_family': '{}'}

    all_brackets = list(itertools.chain(*brackets.values()))

    def __new__(cls, v=None, type='direction'):
        if isinstance(v, str):
            v = MillerIndex.parse(v)
        if len(v) == 3 or len(v) == 4:
            self = np.ndarray.__new__(cls, len(v))
            self[:] = v
        else:
            raise ValueError('%s input v should be of length 3 or 4' % cls.__name__)
        self.type = type
        self.simplify()
        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.type = getattr(obj, 'type', 'direction')

    def __repr__(self):
        return ('%s(['+'%d'*len(self)+'])') % ((self.__class__.__name__,) + tuple(self))
        
    def __str__(self):
        bopen, bclose = MillerIndex.brackets[self.type]
        return (bopen+'%d'*len(self)+bclose) % tuple(self)

    def latex(self):
        """
        Format this :class:`MillerIndex` as a LaTeX string
        """
        s = '$'
        bopen, bclose = MillerIndex.brackets[self.type]
        s += bopen
        for component in self:
            if component < 0:
                s += r'\bar{%d}' % abs(component)
            else:
                s += '%d' % component
        s += bclose
        s += '$'
        return s

    @classmethod
    def parse(cls, s):
        r"""
        Parse a Miller index string

        Negative indices can be denoted by:
         1. leading minus sign, e.g. ``[11-2]``
         2. trailing ``b`` (for 'bar'), e.g. ``112b``
         3. LaTeX ``\bar{}``, e.g. ``[11\bar{2}]`` (which renders as :math:`[11\bar{2}]` in LaTeX)

        Leading or trailing brackets of various kinds are ignored.
        i.e. ``[001]``, ``{001}``, ``(001)``, ``[001]``, ``<001>``, ``001`` are all equivalent.

        Returns an array of components (i,j,k) or (h,k,i,l)
        """

        if not isinstance(s, str):
            raise TypeError("Can't parse from %r of type %r" % (s, type(s)))

        orig_s = s
        for (a, b) in [(r'\bar{','-')] + [(b,'') for b in MillerIndex.all_brackets]:
            s = s.replace(a, b)

        L = list(s)
        components = np.array([1,1,1,1]) # space for up to 4 elements
        i = 3 # parse backwards from end of string
        while L:
            if i < 0:
                raise ValueError('Cannot parse Miller index from string "%s", too many components found' % orig_s)
            c = L.pop()
            if c == '-':
                if i == 3:
                    raise ValueError('Miller index string "%s" cannot end with a minus sign' % orig_s)
                components[i+1] *= -1
            elif c == 'b':
                components[i] *= -1
            elif c in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                components[i] *= int(c)
                i -= 1
            else:
                raise ValueError('Unexpected character "%s" in miller index string "%s"' % (c, orig_s))
                
        if i == 0:
            return components[1:]
        elif i == -1:
            return components
        else:
            raise ValueError('Cannot parse Miller index from string %s, too few components found' % orig_s)
        
        self.simplify()

    def simplify(self):
        """
        Simplify by dividing through by greatest common denominator
        """
        d = abs(functools.reduce(gcd, self))
        self[:] /= d

    def simplified(self):
        copy = self.copy()
        copy.simplify()
        return copy

    def norm(self):
        return np.linalg.norm(self)

    def normalised(self):
        a = self.as3()
        return np.array(a, dtype=float)/a.norm()

    hat = normalised

    def cross(self, other):
        a = self.as3()
        b = MillerIndex(other).as3()
        return np.cross(a, b).view(MillerIndex).simplified()

    def cosine(self, other):
        other = MillerIndex(other)
        return np.dot(self.normalised(), other.normalised())

    def angle(self, other):
        return np.arccos(self.cosine(other))

    def as4(self):
        if len(self) == 4:
            return self
        else:
            h, k, l = self
            i = -(h+l)
            return MillerIndex((h,k,i,l))

    def as3(self):
        if len(self) == 3:
            return self
        else:
            h, k, i, l = self
            return MillerIndex((h, k, l))

    def plane_spacing(self, a):
        return a/self.as3().norm()

def MillerPlane(v):
   """Special case of :class:`MillerIndex` with ``type="plane"``"""
   return MillerIndex(v, 'plane')

def MillerDirection(v):
   """Special case of :class:`MillerIndex` with ``type="direction"`` (the default)"""
   return MillerIndex(v, 'direction')


def angle_between(a, b):
    """Angle between crystallographic directions between a=[ijk] and b=[lmn], in radians."""
    return MillerIndex(a).angle(b)


def make_unit_slab(unit_cell, axes):   
    """
    General purpose unit slab creation routine

    Only tested with cubic unit cells.

    Code translated from quippy.structures.unit_slab()
        https://github.com/libAtoms/QUIP/blob/public/src/libAtoms/Structures.f95

    Arguments
    ---------
        unit_cell : Atoms
            Atoms object containing primitive unit cell
        axes: 3x3 array
            Miller indices of desired slab, as columns

    Returns
    -------
        slab : Atoms
            Output slab, with axes aligned with x, y, z.
    """
    a1 = axes[:,0]
    a2 = axes[:,1]
    a3 = axes[:,2]
    rot = np.zeros((3,3))
    rot[0,:] = a1/norm(a1)
    rot[1,:] = a2/norm(a2)
    rot[2,:] = a3/norm(a3)
    
    pos = unit_cell.get_positions().T
    lattice = unit_cell.get_cell().T
    lattice = np.dot(rot, lattice)
    
    at = unit_cell.copy()
    at.set_positions(np.dot(rot, pos).T)
    at.set_cell(lattice.T)

    sup = at * (5,5,5)
    sup.positions[...] -= sup.positions.mean(axis=0)
    
    sup_lattice = np.zeros((3,3))
    for i in range(3):
        sup_lattice[:,i] = (axes[0,i]*lattice[:,0] + 
                            axes[1,i]*lattice[:,1] + 
                            axes[2,i]*lattice[:,2])
    
    sup.set_cell(sup_lattice.T, scale_atoms=False)
    
    # Form primitive cell by discarding atoms with 
    # lattice coordinates outside range [-0.5,0.5]
    d = [0.01,0.02,0.03]  # Small shift to avoid conincidental alignments
    i = 0
    g = inv(sup_lattice)
    sup_pos = sup.get_positions().T
    while True:
        t = np.dot(g, sup_pos[:, i] + d)
        if (t <= -0.5).any() | (t >= 0.5).any():
            del sup[i]
            sup_pos = sup.get_positions().T
            i -= 1 # Retest since we've removed an atom
        if i == len(sup)-1:
            break
        i += 1

    sup.set_scaled_positions(sup.get_scaled_positions())
    return sup    
