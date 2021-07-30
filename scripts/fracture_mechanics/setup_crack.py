#
# Copyright 2015, 2021 Lars Pastewka (U. Freiburg)
#           2017 James Kermode (Warwick U.)
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

import ase.optimize
from ase.units import GPa

import matscipy.fracture_mechanics.crack as crack
from matscipy import has_parameter, parameter
from matscipy.elasticity import fit_elastic_constants
from matscipy.hydrogenate import hydrogenate
from matscipy.logger import screen
from matscipy.neighbours import neighbour_list

###

def setup_crack(logger=screen):
    calc = parameter('calc')

    cryst = parameter('cryst').copy()
    cryst.set_pbc(True)

    # Double check elastic constants. We're just assuming this is really a periodic
    # system. (True if it comes out of the cluster routines.)

    compute_elastic_constants = parameter('compute_elastic_constants', False)
    elastic_fmax = parameter('elastic_fmax', 0.01)
    elastic_symmetry = parameter('elastic_symmetry', 'triclinic')
    elastic_optimizer = parameter('elastic_optimizer', ase.optimize.FIRE)

    if compute_elastic_constants:
        cryst.set_calculator(calc)
        log_file = open('elastic_constants.log', 'w')
        C, C_err = fit_elastic_constants(cryst, verbose=False,
                                         symmetry=elastic_symmetry,
                                         optimizer=elastic_optimizer,
                                         logfile=log_file,
                                         fmax=elastic_fmax)
        log_file.close()
        logger.pr('Measured elastic constants (in GPa):')
        logger.pr(np.round(C*10/GPa)/10)

        crk = crack.CubicCrystalCrack(parameter('crack_surface'),
                                      parameter('crack_front'),
                                      Crot=C/GPa)
    else:
        if has_parameter('C'):
            crk = crack.CubicCrystalCrack(parameter('crack_surface'),
                                          parameter('crack_front'),
                                          C=parameter('C'))
        else:
            crk = crack.CubicCrystalCrack(parameter('crack_surface'),
                                          parameter('crack_front'),
                                          parameter('C11'), parameter('C12'),
                                          parameter('C44'))


    logger.pr('Elastic constants used for boundary condition (in GPa):')
    logger.pr(np.round(crk.C*10)/10)

    # Get Griffith's k1.
    k1g = crk.k1g(parameter('surface_energy'))
    logger.pr('Griffith k1 = %f' % k1g)

    # Apply initial strain field.
    tip_x = parameter('tip_x', cryst.cell.diagonal()[0]/2)
    tip_y = parameter('tip_y', cryst.cell.diagonal()[1]/2)

    bondlength = parameter('bondlength', 2.7)
    bulk_nn = parameter('bulk_nn', 4)

    a = cryst.copy()
    a.set_pbc([False, False, True])

    hydrogenate_flag = parameter('hydrogenate', False)
    hydrogenate_crack_face_flag = parameter('hydrogenate_crack_face', True)

    if hydrogenate_flag and not hydrogenate_crack_face_flag:
        # Get surface atoms of cluster with crack
        a = hydrogenate(cryst, bondlength, parameter('XH_bondlength'), b=a)
        g = a.get_array('groups')
        g[a.numbers==1] = -1
        a.set_array('groups', g)
        cryst = a.copy()

    k1 = parameter('k1')
    try:
      k1 = k1[0]
    except:
      pass
    ux, uy = crk.displacements(cryst.positions[:,0], cryst.positions[:,1],
                               tip_x, tip_y, k1*k1g)
    a.positions[:len(cryst),0] += ux
    a.positions[:len(cryst),1] += uy

    # Center notched configuration in simulation cell and ensure enough vacuum.
    oldr = a[0].position.copy()
    vacuum = parameter('vacuum')
    a.center(vacuum=vacuum, axis=0)
    a.center(vacuum=vacuum, axis=1)
    tip_x += a[0].x - oldr[0]
    tip_y += a[0].y - oldr[1]

    # Choose which bond to break.
    bond1, bond2 = \
        parameter('bond', crack.find_tip_coordination(a, bondlength=bondlength, bulk_nn=bulk_nn))

    if parameter('center_crack_tip_on_bond', False):
        tip_x, tip_y, dummy = (a.positions[bond1]+a.positions[bond2])/2

    # Hydrogenate?
    coord = np.bincount(neighbour_list('i', a, bondlength), minlength=len(a))
    a.set_array('coord', coord)

    if parameter('optimize_full_crack_face', False):
        g = a.get_array('groups')
        gcryst = cryst.get_array('groups')
        coord = a.get_array('coord')
        g[coord!=4] = -1
        gcryst[coord!=4] = -1
        a.set_array('groups', g)
        cryst.set_array('groups', gcryst)

    if hydrogenate_flag and hydrogenate_crack_face_flag:
        # Get surface atoms of cluster with crack
        exclude = np.logical_and(a.get_array('groups')==1, coord!=4)
        a.set_array('exclude', exclude)
        a = hydrogenate(cryst, bondlength, parameter('XH_bondlength'), b=a,
                        exclude=exclude)
        g = a.get_array('groups')
        g[a.numbers==1] = -1
        a.set_array('groups', g)
        basename = parameter('basename', 'energy_barrier')
        ase.io.write('{0}_hydrogenated.xyz'.format(basename), a,
                     format='extxyz')

    # Move reference crystal by same amount
    cryst.set_cell(a.cell)
    cryst.set_pbc([False, False, True])
    cryst.translate(a[0].position - oldr)

    # Groups mark the fixed region and the region use for fitting the crack tip.
    g = a.get_array('groups')
    gcryst = cryst.get_array('groups')

    logger.pr('Opening bond {0}--{1}, initial bond length {2}'.
          format(bond1, bond2, a.get_distance(bond1, bond2, mic=True)))

    # centre vertically on the opening bond
    if parameter('center_cell_on_bond', True):
      a.translate([0., a.cell[1,1]/2.0 -
                      (a.positions[bond1, 1] +
                       a.positions[bond2, 1])/2.0, 0.])

    a.info['bond1'] = bond1
    a.info['bond2'] = bond2

    return a, cryst, crk, k1g, tip_x, tip_y, bond1, bond2, g==0, gcryst==0, g==1
