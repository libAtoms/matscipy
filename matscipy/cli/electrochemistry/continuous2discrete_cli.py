#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2019) Johannes Hoermann, University of Freiburg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ======================================================================
"""Generate discrete coordinate sets from continuous distributions.
Export as atom positions in .xyz of LAMMPS data file format.

Copyright 2019 IMTEK Simulation
University of Freiburg

Authors:
  Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>
  Lukas Elflein <elfleinl@cs.uni-freiburg.de>
"""
import argparse
import logging
import os
import os.path
import sys

import numpy as np

import ase
import ase.io
import scipy.constants as sc
from scipy import interpolate, integrate

from matscipy.electrochemistry import continuous2discrete

logger = logging.getLogger(__name__)


def main():
    """Generate discrete coordinate sets from continuous distributions.
    Export as atom positions in .xyz of LAMMPS data file format.
    Plot continuous and discrete distributions if wanted.

    ATTENTION: LAMMPS data file export (atom style 'full') requires
    ase>=3.19.0b1 (> 6th Nov 2019) due to recently reseolved issue"""

    # in order to have both:
    # * preformatted help text and ...
    # * automatic display of defaults
    class ArgumentDefaultsAndRawDescriptionHelpFormatter(
            argparse.ArgumentDefaultsHelpFormatter,
            argparse.RawDescriptionHelpFormatter):
        pass

    class StoreAsNumpyArray(argparse._StoreAction):
        def __call__(self, parser, namespace, values, option_string=None):
            values = np.array(values, ndmin=1)
            return super().__call__(parser, namespace, values, option_string)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=ArgumentDefaultsAndRawDescriptionHelpFormatter)

    parser.add_argument('infile', metavar='IN', nargs='?',
                        help='binary numpy .npz or plain text .txt input file')
    parser.add_argument('outfile', metavar='OUT', nargs='?',
                        help='.xyz format output file')

    parser.add_argument('--box', '-b', default=[50.0e-9, 50.0e-9, 100.0e-9], nargs=3,
                        action=StoreAsNumpyArray,
                        metavar=('X', 'Y', 'Z'), required=False, type=float,
                        dest="box", help='Box dimensions')

    parser.add_argument('--names', default=['Na', 'Cl'], type=str, nargs='+',
                        metavar=('NAME'), required=False, dest="names",
                        help='Atom names')

    parser.add_argument('--charges', default=[1, -1], type=float, nargs='+',
                        action=StoreAsNumpyArray,
                        metavar=('NAME'), required=False, dest="charges",
                        help='Atom charges')

    parser.add_argument('--mol-id-offset', default=[-1, -1],
                        type=int, nargs='+',
                        action=StoreAsNumpyArray,
                        metavar=('OFFSET'), required=False,
                        dest="mol_id_offset",
                        help=('When storing as LAMMPS data, this tool uses'
                              ' atom sytle "full", assigning a molecule id to'
                              ' each atom. Per default, a unique 1-indexed'
                              ' molecule id corrsponding to its atom id is'
                              ' assigned to each atom. Specify a species-wise'
                              ' offset here. Specifying 0 (zero) for each'
                              ' species will result in moleucules containing '
                              ' one atom per species.'))

    # sampling
    parser.add_argument('--ngridpoints', default=np.nan, type=float, nargs='+',
                        action=StoreAsNumpyArray,
                        metavar=('N'), required=False, dest="n_gridpoints",
                        help=('Number of grid points for discrete support. '
                              'Continuous support for all sampes per default. '
                              'Specify "NaN" explicitly for continuous support '
                              'in particular species, i.e. '
                              '"--n_gridpoints 100 NaN 50"'))
    parser.add_argument('--sample-size', default=np.nan, type=float, nargs='+',
                        action=StoreAsNumpyArray,
                        metavar=('N'), required=False, dest="sample_size",
                        help=('Sample size. Specify '
                              'multiple values for specific number of atom '
                              'positions for each species. Per default, infer '
                              'sample size from distributions, assuming '
                              'concentrations in SI units (i.e. mM or mol / m^3).'
                              'Specify "NaN" explicitly for inference in certain '
                              'species only, i.e. '
                              '"--sample-size 100 NaN 50"'))

    # output
    parser.add_argument('--nbins', default=100, type=int,
                        metavar=('N'), required=False, dest="nbins",
                        help='Number of bins for histogram plots')

    parser.add_argument('--debug', default=False, required=False,
                        action='store_true', dest="debug", help='debug flag')
    parser.add_argument('--verbose', default=False, required=False,
                        action='store_true', dest="verbose", help='verbose flag')
    parser.add_argument('--log', required=False, nargs='?', dest="log",
                        default=None, const='c2d.log', metavar='LOG',
                        help='Write log file c2d.log, optionally specify log file name')

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
        # This supports bash autocompletion. To enable this, pip install
        # argcomplete, activate global completion, or add
        #      eval "$(register-python-argcomplete lpad)"
        # into your .bash_profile or .bashrc
    except ImportError:
        pass

    args = parser.parse_args()

    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING

    logformat = "[ %(filename)s:%(lineno)s - %(funcName)s() ]: %(message)s"

    logging.basicConfig(level=loglevel,
                        format=logformat)

    # explicitly modify the root logger (necessary?)
    logger = logging.getLogger()
    logger.setLevel(loglevel)

    # remove all handlers
    for h in logger.handlers:
        logger.removeHandler(h)

    # create and append custom handles
    ch = logging.StreamHandler()
    formatter = logging.Formatter(logformat)
    ch.setFormatter(formatter)
    ch.setLevel(loglevel)
    logger.addHandler(ch)

    if args.log:
        fh = logging.FileHandler(args.log)
        fh.setFormatter(formatter)
        fh.setLevel(loglevel)
        logger.addHandler(fh)

    logger.info('This is `{}` : `{}`.'.format(__file__, __name__))

    if not isinstance(args.box, np.ndarray):
        args.box = np.array(args.box, ndmin=1)
    if not isinstance(args.sample_size, np.ndarray):
        args.sample_size = np.array(args.sample_size, ndmin=1)
    if not isinstance(args.n_gridpoints, np.ndarray):
        args.n_gridpoints = np.array(args.n_gridpoints, ndmin=1)

    if not args.infile:
        infile = sys.stdin
        infile_format = '.txt'
    else:
        infile = args.infile
        _, infile_format = os.path.splitext(infile)

    if infile_format == '.npz':
        file = np.load(infile)
        x = file['x']
        u = file['u']
        c = file['c']
    else:  # elif infile_format == 'txt'
        data = np.loadtxt(infile, unpack=True)
        x = data[0, :]
        u = data[1, :]
        c = data[2:, :]

    if c.ndim > 1:
        C = [c[k, :] for k in range(c.shape[0])]
    else:
        C = [c]

    del c

    logger.info('Read {:d} concentration distributions.'.format(len(C)))
    sample_size = args.sample_size
    sample_size = (sample_size.repeat(len(C)) if sample_size.shape == (1,)
                   else sample_size)

    # distribution functions from concentrations;
    D = [interpolate.interp1d(x, c) for c in C]

    # infer sample size from integral over concentration distribution if
    # no explicit sample size given
    # TODO: over-estimates sample size when distribution highly nonlinear
    for i, s in enumerate(sample_size):
        if np.isnan(s):
            # average concentration in distribution over interval
            cave, _ = integrate.quad(D[i], 0, args.box[-1]) / args.box[-1]  # z direction
            # [V] = m^3, [c] = mol / m^3, [N_A] = 1 / mol
            sample_size[i] = int(np.round(args.box.prod()*cave * sc.Avogadro))
            logger.info('Inferred {} samples on interval [{},{}] m'.format(
                sample_size[i], 0, args.box[-1]))
            logger.info('for average concentration {} mM.'.format(cave))

    n_gridpoints = args.n_gridpoints  # assume n_gridpoints is np.ndarray
    n_gridpoints = n_gridpoints.repeat(len(C)) if n_gridpoints.shape == (1,) else n_gridpoints

    logger.info('Generating {} positions on {} support for species {}.'.format(
        sample_size, n_gridpoints, args.names))

    logger.info('Generating structure from distribution ...')
    struc = [continuous2discrete(
                distribution=d,
                box=args.box, count=sample_size[k],
                n_gridpoints=n_gridpoints[k]) for k, d in enumerate(D)]

    logger.info('Generated {:d} coordinate sets.'.format(len(struc)))

    logger.info('Creating ase.Atom objects ...')
    system = ase.Atoms(
        cell=args.box/sc.angstrom,
        pbc=[1, 1, 0])

    # 1-indexed LAMMPS data molecule id within ASE's io module.
    # We assume here that 'mol-id' is 0-indexed internally but converted to a
    system.new_array('mol-id', [], dtype=int)

    for i, s in enumerate(struc):
        logger.info('{:d} samples in coordinate set {:d}.'.format(len(s), i))
        new_species = ase.Atoms(
            symbols=args.names[i]*int(sample_size[i]),
            charges=[args.charges[i]]*int(sample_size[i]),
            positions=s/sc.angstrom)

        # per default, consecutive numbering, otherwise custom offest:
        mol_id_offset = len(system) \
            if args.mol_id_offset[i] < 0 else args.mol_id_offset[i]

        # without offset, LAMMPS molecule ids are 1-indexed
        # mol-id 0 is reserved for 'non-bonded' atoms
        new_species.new_array(
            'mol-id', 1 + mol_id_offset + np.arange(
                start=0, stop=len(new_species), step=1, dtype=int), dtype=int)

        system += new_species

    logger.info('Writing output file ...')

    if not args.outfile:
        outfile = sys.stdout
        outfile_format = '.xyz'
    else:
        outfile = args.outfile
        _, outfile_format = os.path.splitext(outfile)

    logger.info('Output format {} to {}.'.format(outfile_format, outfile))

    if outfile_format == '.lammps':
        ase.io.write(
            outfile, system,
            format='lammps-data',
            units="real",
            atom_style='full',
            specorder=args.names)
    # specorder shoudl make sure ASE assigns types in the same order as
    # species names have been specified on command line.

    else:  # elif outfile_format == '.xyz'
        ase.io.write(outfile, system, format='extxyz')

    logger.info('Done.')


if __name__ == '__main__':
    # Execute everything else
    main()
