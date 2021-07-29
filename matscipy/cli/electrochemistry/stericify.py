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
"""
Applies steric correction to coordiante sample.

Copyright 2020 IMTEK Simulation
University of Freiburg

Authors:

    Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>

Examples:

    Apply one steric radius for all species:

        stericify.py --verbose --radii 2.0 -- input.lammps output.lammps

    Apply per-species steric radii (for a system of supposedly two species):

        stericify.py --verbose --radii 2.0 5.0 -- input.lammps output.lammps

    Overrides other parameters otherwise infered from input file:

        stericify.py --verbose --box 48 48 196 --names Na Cl --charges 1 -1 \\
            --radii 2.0 5.0 -- input.lammps output.lammps
"""
import logging
import os
import sys
import time

import ase
import ase.io
import numpy as np

try:
    import json
    import urllib.parse
except ImportError:
    pass

from matscipy.electrochemistry.steric_correction import apply_steric_correction
from matscipy.electrochemistry.steric_correction import scipy_distance_based_closest_pair


def main():
    """Applies steric correction to coordiante sample. Assures a certain
    minimum pairwiese distance between points in sample and
    between points and box boundary.

    ATTENTION: LAMMPS data file export (atom style 'full') requires ase>3.20.0
    """
    logger = logging.getLogger()

    import argparse

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
            return super().__call__(
                parser, namespace, values, option_string)

    class StoreAsDict(argparse._StoreAction):
        def __call__(self, parser, namespace, value, option_string=None):
            if 'json' not in sys.modules or 'urllib' not in sys.modules:
                raise ModuleNotFoundError(
                    "Modules 'json' and 'urllib' required for parsing dicts.")
            try:
                parsed_value = json.loads(urllib.parse.unquote(value))
            except json.decoder.JSONDecodeError as exc:
                logger.error("Failed parsing '{}'".format(value))
                raise exc

            return super().__call__(
                parser, namespace, parsed_value, option_string)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=ArgumentDefaultsAndRawDescriptionHelpFormatter)

    parser.add_argument('infile', metavar='IN', nargs='?',
                        help='.xyz or .lammps (LAMMPS data) format input file')
    parser.add_argument('outfile', metavar='OUT', nargs='?',
                        help='.xyz or .lammps (LAMMPS data) format output file')

    parser.add_argument('--radii', '-r', default=[2.0], type=float, nargs='+',
                        action=StoreAsNumpyArray,
                        metavar=('R'), required=False, dest="radii",
                        help=('Steric radii, either one for all or '
                              'species-wise. Same units as distances in input.'))

    parser.add_argument('--box', '-b', default=None, nargs=3,
                        action=StoreAsNumpyArray,
                        metavar=('X', 'Y', 'Z'), required=False, type=float,
                        dest="box", help=('Bounding box, overrides cell from'
                                          'input. Same units as distances in input.'))

    parser.add_argument('--names', default=None, type=str, nargs='+',
                        metavar=('NAME'), required=False, dest="names",
                        help='Atom names, overrides names from input')

    parser.add_argument('--charges', default=None, type=float, nargs='+',
                        action=StoreAsNumpyArray,
                        metavar=('Z'), required=False, dest="charges",
                        help='Atom charges, overrides charges from input')

    parser.add_argument('--method', type=str,
                        metavar=('METHOD'), required=False,
                        dest="method",
                        default='L-BFGS-B',
                        help='Scipy minimizer')

    parser.add_argument('--options', type=str,
                        action=StoreAsDict,
                        metavar=('JSON DICT'), required=False,
                        dest="options",
                        default={
                            'gtol':    1.e-12,
                            'ftol':    1.e-12,
                            'maxiter': 100,
                            'disp':    False,
                        },
                        help=(
                            'Convergence options for scipy minimier.'
                            ' Pass as JSON-formatted key:value dict. See'
                            ' https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html'
                            ' for minimizer-specific options.'))

    parser.add_argument('--debug', default=False, required=False,
                        action='store_true', dest="debug", help='debug flag')
    parser.add_argument('--verbose', default=False, required=False,
                        action='store_true', dest="verbose",
                        help='verbose flag')
    parser.add_argument('--log', required=False, nargs='?', dest="log",
                        default=None, const='c2d.log', metavar='LOG',
                        help=(
                            'Write log file c2d.log, optionally specify log'
                            ' file name'))

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
        # This supports bash autocompletion.
        # To enable this, 'pip install argcomplete',
        # then activate global completion.
    except ImportError:
        pass

    args = parser.parse_args()

    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING

    # logformat  = ''.join(("%(asctime)s",
    #  "[ %(filename)s:%(lineno)s - %(funcName)s() ]: %(message)s"))
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

    logger.debug('Args: {}'.format(args))

    # input validation
    if not args.infile:
        infile = sys.stdin
        infile_format = 'xyz'
    else:
        infile = args.infile
        _, infile_format = os.path.splitext(infile)

    logger.info('Input format {} from {}.'.format(infile_format, infile))

    if infile_format == '.lammps':
        system = ase.io.read(
            infile, format='lammps-data', units="real", style='full')
    else:  # elif outfile_format == '.xyz'
        system = ase.io.read(infile, format='extxyz')

    n = len(system)  # total number of particles
    logger.info('Read "{}" system within bounding box'.format(system.symbols))
    for l in system.cell:
        logger.info('    [ {:> 8.2e},{:> 8.2e},{:> 8.2e} ]'.format(*l))

    species_atomic_numbers = np.unique(system.get_atomic_numbers())
    species_symbols = [
        ase.data.chemical_symbols[i] for i in species_atomic_numbers]

    n_species = len(species_atomic_numbers)
    logger.info('    containing {:d} particles of {:d} species'.format(
        n, n_species))
    if not isinstance(args.radii, np.ndarray):
        args.radii = np.array(args.radii, ndmin=1)

    r = np.zeros(n)
    if len(args.radii) == 1:
        logger.info('Applying steric radius r = {:.2e} to all species.'.format(
            args.radii[0]))
        r[:] = args.radii[0]

    elif len(args.radii) == n_species:
        for i, (a, s) in enumerate(zip(species_atomic_numbers, species_symbols)):
            logger.info(
                'Applying steric radius r = {:.2e} for species {:s}.'.format(
                    args.radii[i], s))
            r[system.get_atomic_numbers() == a] = args.radii[i]
    else:
        raise ValueError(
            """Steric radii must either be one value for all species or one value
            per species, i.e. {:d} values in your case.""".format(n_species))

    if args.box is not None:
        if not isinstance(args.box, np.ndarray):
            args.box = np.array(args.box, ndmin=1)

        logger.info('Box specified on command line')
        logger.info('   [ {:> 8.2e},{:> 8.2e},{:> 8.2e} ]'.format(*args.box))
        logger.info('overides box from input data.')
        system.set_cell(args.box)

    if args.charges is not None:
        logger.info('Charges specified on command line reassign input charges to')
        new_charges = np.ndarray(n, dtype=int)

        for a, s, c in zip(species_atomic_numbers, species_symbols, args.charges):
            logger.info(
                '    {:s} -> {}'.format(s, c))
            new_charges[system.get_atomic_numbers() == a] = c
        system.set_initial_charges(new_charges)

    if args.names is not None:
        new_species_symbols = args.names
        new_species_atomic_numbers = [
            ase.data.atomic_numbers[name] for name in new_species_symbols]
        logger.info('Species specified on command line reassign input species to')
        new_atomic_numbers = np.ndarray(n, dtype=int)
        for aold, anew, sold, snew in zip(species_atomic_numbers,
                                          new_species_atomic_numbers,
                                          species_symbols, new_species_symbols):
            logger.info(
                '    {:s} -> {:s}'.format(sold, snew))

            new_atomic_numbers[system.get_atomic_numbers() == aold] = anew
        system.set_atomic_numbers(new_atomic_numbers)
        species_atomic_numbers = new_species_atomic_numbers
        species_symbols = new_species_symbols

        specorder = args.names  # assure type ordering as specified on cmdline
    else:
        specorder = None  # keep ordering as is

    # prepare for minimization
    x0 = system.get_positions()

    # only works for orthogonal box
    box3 = np.array(system.get_cell_lengths_and_angles())[0:3]
    box6 = np.array([[0., 0., 0], box3])  # needs lower corner

    # n = x0.shape[0], set above
    # dim = x0.shape[1]

    # benchmakr methods
    mindsq, (p1, p2) = scipy_distance_based_closest_pair(x0)
    pmin = np.min(x0, axis=0)
    pmax = np.max(x0, axis=0)
    mind = np.sqrt(mindsq)
    logger.info("Minimum pair-wise distance in initial sample: {}".format(mind))
    logger.info("First sample point in pair:    ({:8.4e},{:8.4e},{:8.4e})".format(*p1))
    logger.info("Second sample point in pair    ({:8.4e},{:8.4e},{:8.4e})".format(*p2))
    logger.info("Box lower boundary:            ({:8.4e},{:8.4e},{:8.4e})".format(*box6[0]))
    logger.info("Minimum coordinates in sample: ({:8.4e},{:8.4e},{:8.4e})".format(*pmin))
    logger.info("Maximum coordinates in sample: ({:8.4e},{:8.4e},{:8.4e})".format(*pmax))
    logger.info("Box upper boundary:            ({:8.4e},{:8.4e},{:8.4e})".format(*box6[1]))

    t0 = time.perf_counter()
    x1, res = apply_steric_correction(x0, box=box6, r=r,
                                      method=args.method, options=args.options)
    # use default method and options

    t1 = time.perf_counter()
    dt = t1 - t0
    logger.info("{} s runtime".format(dt))

    mindsq, (p1, p2) = scipy_distance_based_closest_pair(x1)
    mind = np.sqrt(mindsq)
    pmin = np.min(x1, axis=0)
    pmax = np.max(x1, axis=0)

    logger.info("Finished with status = {}, success = {}, #it = {}".format(
        res.status, res.success, res.nit))
    logger.info("    message = '{}'".format(res.message))
    logger.info("Minimum pair-wise distance in final configuration: {:8.4e}".format(mind))
    logger.info("First sample point in pair:    ({:8.4e},{:8.4e},{:8.4e})".format(*p1))
    logger.info("Second sample point in pair    ({:8.4e},{:8.4e},{:8.4e})".format(*p2))
    logger.info("Box lower boundary:            ({:8.4e},{:8.4e},{:8.4e})".format(*box6[0]))
    logger.info("Minimum coordinates in sample: ({:8.4e},{:8.4e},{:8.4e})".format(*pmin))
    logger.info("Maximum coordinates in sample: ({:8.4e},{:8.4e},{:8.4e})".format(*pmax))
    logger.info("Box upper boundary:            ({:8.4e},{:8.4e},{:8.4e})".format(*box6[1]))

    diff = x1 - x0
    n_diff = np.count_nonzero(diff)
    diffnorm = np.linalg.norm(diff)
    logger.info(
      '{:d} coords. differ numerically in final and initial config.'.format(
        n_diff
      ))
    logger.info('Norm of difference between final and initial config')
    logger.info('    || x1 - x0 || = {:.4e}'.format(diffnorm))

    system.set_positions(x1)

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
            format='lammps-data', units="real", atom_style='full',
            specorder=specorder)
    else:  # elif outfile_format == '.xyz'
        ase.io.write(outfile, system, format='extxyz')

    logger.info('Done.')


if __name__ == '__main__':
    # Execute everything else
    main()
