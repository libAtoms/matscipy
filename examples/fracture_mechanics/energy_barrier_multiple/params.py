#! /usr/bin/env python

from math import log

import numpy as np

import ase.io

from matscipy.fracture_mechanics.clusters import diamond_110_001

import atomistica

###

# Interaction potential
calc = atomistica.KumagaiScr()

# Fundamental material properties
el              = 'Si'
a0              = 5.429
C11             = 166.   # GPa
C12             = 65.    # GPa
C44             = 77.    # GPa
surface_energy  = 1.08  * 10    # GPa*A = 0.1 J/m^2

# Crack system
n               = [ 6, 6, 1 ]
crack_surface   = [ 1, 1, 0 ]
crack_front     = [ 0, 0, 1 ]

vacuum          = 6.0

# Simulation control
bonds = [( 58, 59 ), (61, 84)]
        
k1              = 1.00
bond_lengths    = np.linspace(2.5, 4.5, 41)

fmax            = 0.001

# Setup crack system
cryst = diamond_110_001(el, a0, n, crack_surface, crack_front)
ase.io.write('cryst.cfg', cryst)

optimize_tip_position = True
