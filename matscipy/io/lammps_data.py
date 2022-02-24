#
# Copyright 2022 Lucas Frérot (U. Freiburg)
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

"""Helper class LAMMPSData to read/write LAMMPS text files."""

from functools import wraps, reduce
from operator import mul
from io import TextIOBase
from os import PathLike
import typing as ts
import numpy as np

from ..molecules import Molecules


FileDescriptor = ts.Union[str, PathLike, TextIOBase]


def read_molecules_from_lammps_data(fd: FileDescriptor, style="full"):
    """Read molecules information from LAMMPS data file.

    Parameters
    ----------
    fd :
        File descriptor, i.e. file path or text stream.
    style :
        LAMMPS atomic style.

    Returns
    -------
    Molecules
        An object containing the molecule connectivity data.

    Notes
    -----
    For the connectivity data to make sense with atoms ids read by ASE, the
    ``sort_by_id`` flag of ``read()`` must be set to ``True``.
    """
    data = LAMMPSData(style=style)
    data.read(fd)
    return Molecules(
        bonds_connectivity=data["bonds"]["atoms"] - 1,
        bonds_types=data["bonds"]["type"],
        angles_connectivity=data["angles"]["atoms"] - 1,
        angles_types=data["angles"]["type"],
        dihedrals_connectivity=data["dihedrals"]["atoms"] - 1,
        dihedrals_types=data["dihedrals"]["type"],
    )


def check_legal_name(func):
    """Check proper dataset name."""
    @wraps(func)
    def inner(*args, **kwargs):
        data_key = args[1]
        legal_names = args[0]._legal_names

        if data_key not in legal_names:
            raise Exception(
                f'Requested data "{args[1]}" is not recognized by LAMMPSData')
        return func(*args, **kwargs)
    return inner


def column_size(dtype: np.dtype):
    """Compute number of columns from dtype."""
    if dtype.fields is None:
        return 1

    numcols = 0
    for tup, _ in dtype.fields.values():
        if tup.subdtype is None:
            numcols += 1
        else:
            _, shape = tup.subdtype
            numcols += reduce(mul, shape, 1)
    return numcols


class LAMMPSData:
    """Main class to interact with LAMMPS text files."""

    _data_names = [
        # Sections which have types
        "atoms",
        "bonds",
        "angles",
        "dihedrals",

        # Sections without types
        "velocities",
        "masses",
    ]

    _header_data_names = _data_names[0:4]

    _type_names = {
        s[:-1] + ' types': s for s in _header_data_names
    }

    __headers = {s: s.capitalize() for s in _data_names}

    _legal_names = _data_names + list(_type_names.keys())

    _write_formats = {
        "bonds": "%d %d %d",
        "angles": "%d %d %d %d",
        "dihedrals": "%d %d %d %d %d",
        "velocities": "%.18e %.18e %.18e",
        "masses": "%.18e",
    }

    _atom_write_formats = {
        "atomic": "%d %.18e %.18e %.18e",
        "bond": "%d %d %.18e %.18e %.18e",
        "angle": "%d %d %.18e %.18e %.18e",
        "charge": "%d %.18e %.18e %.18e %.18e",
        "full": "%d %d %.18e %.18e %.18e %.18e",
    }

    _dtypes = {
        "bonds": np.dtype([('type', np.int32), ('atoms', np.int32, 2)]),
        "angles": np.dtype([('type', np.int32), ('atoms', np.int32, 3)]),
        "dihedrals": np.dtype([('type', np.int32), ('atoms', np.int32, 4)]),
        "velocities": np.dtype([('vel', np.double, 3)]),
        "masses": np.double,
    }

    _atom_dtypes = {
        "atomic": np.dtype([('type', np.int32), ('pos', np.double, 3)]),
        "bond": np.dtype([('mol', np.int32),
                          ('type', np.int32),
                          ('pos', np.double, 3)]),
        "angle": np.dtype([('mol', np.int32),
                           ('type', np.int32),
                           ('pos', np.double, 3)]),
        "charge": np.dtype([('type', np.int32),
                            ('charge', np.double),
                            ('pos', np.double, 3)]),
        "full": np.dtype([('mol', np.int32),
                          ('type', np.int32),
                          ('charge', np.double),
                          ('pos', np.double, 3)]),
    }

    def __init__(self, style="atomic", image_flags=False):
        """Initialize data object with atom style."""
        self.style = style

        # Add a flags field to atoms array
        if image_flags:
            for k, dtype in self._atom_dtypes.items():
                dtype_dict = dict(dtype.fields)
                dtype_dict['image_flags'] = (np.dtype("3<i8"), dtype.itemsize)
                self._atom_dtypes[k] = np.dtype(dtype_dict)
                self._atom_write_formats[k] += " %d %d %d"

        self._dtypes['atoms'] = self._atom_dtypes[style]
        self._write_formats['atoms'] = self._atom_write_formats[style]

        self.__data = {k: np.array([], dtype=self._dtypes[k])
                       for k in self._data_names}
        self.ranges = []

    @check_legal_name
    def __getitem__(self, name: str):
        """Get data component."""
        if name in self._type_names:
            name = self._type_names[name]
            return self.__data[name]['type']
        elif name in self._data_names:
            return self.__data[name]

    @check_legal_name
    def __setitem__(self, name: str, value: ts.Any):
        """Set data component."""
        if name in self._type_names:
            name = self._type_names[name]
            self.__data[name].resize(len(value))
            self.__data[name]['type'] = value
        elif name in self._data_names:
            self.__data[name].resize(len(value))
            data = self.__data[name]
            try:
                data[data.dtype.names[-1]] = np.array(value)
            except TypeError:
                data[:] = np.array(value)

    def write(self, fd: FileDescriptor):
        """Write data to text file or stream."""
        if isinstance(fd, (str, PathLike)):
            with open(fd, 'w') as stream:
                return self.write(stream)
        if not isinstance(fd, TextIOBase):
            raise TypeError("File should be path or text stream.")

        def null_filter(measure, generator):
            return filter(lambda t: t[1] != 0,
                          map(lambda x: (x[0], measure(x[1])), generator))

        # Writer header
        fd.write('\n')
        # Write data numbers
        for key, value in null_filter(len, [
                (k, self[k])
                for k in self._header_data_names
        ]):
            fd.write(f'{value} {key}\n')
        # Write unique type numbers (for non-zero types)
        for key, value in null_filter(
                lambda x: len(set(x)),
                [(k, self[k]) for k in self._type_names]
        ):
            fd.write(f'{value} {key}\n')
        fd.write('\n\n')

        # Write system size
        for span, label in zip(self.ranges, "xyz"):
            fd.write('{0} {1} {2}lo {2}hi\n'.format(*span, label))
        fd.write('\n')

        # Write masses
        fd.write('Masses\n\n')
        for i, m in enumerate(self['masses']):
            fd.write(f'{i+1} {m}\n')
        fd.write('\n')

        # Write data categories
        for label, header in self.__headers.items():
            if not len(self[label]) or label == "masses":
                continue

            if label == "atoms":
                fd.write(header + f'  # {self.style}\n\n')
            else:
                fd.write(header + '\n\n')

            for i, line in enumerate(self[label]):
                flatline = []
                for component in line:
                    if isinstance(component, np.ndarray):
                        flatline += component.tolist()
                    else:
                        flatline.append(component)
                fd.write('%d ' % (i+1)
                         + self._write_formats[label] % tuple(flatline))
                fd.write('\n')
            fd.write('\n')

    def read(self, fd: FileDescriptor):
        """Read data from text file or stream."""
        if isinstance(fd, (str, PathLike)):
            with open(fd, 'r') as stream:
                return self.read(stream)
        if not isinstance(fd, TextIOBase):
            raise TypeError("File should be path or text stream.")

        def header_info(fd, names):
            counts = {}
            box = [[], [], []]
            bounds = ['xlo xhi', 'ylo yhi', 'zlo zhi']
            has_bounds = False
            for line in fd:
                if line == "\n" and has_bounds:
                    break

                for label in filter(lambda x: x in line, names):
                    counts[label] = int(line.split()[0])

                for i, label in filter(lambda x: x[1] in line,
                                       enumerate(bounds)):
                    box[i] = [
                        float(x) for x in
                        line.replace(label, "").strip().split()
                    ]

                    has_bounds = True

            return counts, box

        counts, self.ranges = header_info(
            fd, self._data_names + list(self._type_names.keys()))

        data_counts = {k: v for k, v in counts.items()
                       if k in self._data_names}
        type_counts = {k: v for k, v in counts.items()
                       if k in self._type_names}

        # If velocities are present
        data_counts['velocities'] = data_counts['atoms']

        for linum, line in enumerate(fd):
            if 'Masses' in line:
                ntypes = type_counts['atom types']
                self['masses'].resize(ntypes)
                self['masses'][:] = \
                    np.genfromtxt(fd, skip_header=1,
                                  max_rows=ntypes, usecols=(1,))

            else:
                for label in self._data_names:
                    if self.__headers[label] in line:
                        nlines = data_counts[label]
                        self[label].resize(nlines)
                        dtype = self[label].dtype

                        raw_dtype = np.dtype([('num', np.int32)] + [
                            (k, v[0]) for k, v in dtype.fields.items()
                        ])
                        raw_data = \
                            np.genfromtxt(fd,
                                          skip_header=1,
                                          max_rows=nlines,
                                          dtype=raw_dtype)

                        # Correct for 1-numbering
                        raw_data['num'] -= 1
                        self[label][raw_data['num']] = \
                            raw_data[list(dtype.fields)]
