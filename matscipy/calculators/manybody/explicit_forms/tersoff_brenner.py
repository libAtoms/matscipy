#
# Copyright 2021 Lars Pastewka (U. Freiburg)
#           2021 Jan Griesser (U. Freiburg)
#           2021 griesserj@griesserjs-MacBook-Pro.local
#           2020 Jonas Oldenstaedt (U. Freiburg)
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

from math import sqrt

import numpy as np

import ase.data as data

# The parameter sets are compatible with Atomistica.
# See: https://github.com/Atomistica/atomistica/blob/master/src/python/atomistica/parameters.py

def pair_index(i, j, maxval):
    return np.minimum(i + j * maxval, j + i * maxval) - np.minimum(i * (i + 1) // 2, j * (j + 1) // 2)


def triplet_index(i, maxval):
    return k + maxval * (j + maxval * i)


# Mixing rules
def mix(p, key, rule):
    nel = len(p['el'])
    for i in range(nel):
        for j in range(i + 1, nel):
            ii = pair_index(i, i, nel)
            jj = pair_index(j, j, nel)
            ij = pair_index(i, j, nel)
            p[key][ij] = rule(p[key][ii], p[key][jj])


def mix_arithmetic(p, key):
    mix(p, key, lambda x,y: (x+y)/2)


def mix_geometric(p, key):
    mix(p, key, lambda x,y: sqrt(x*y))


#
# Parameter sets
# The '__ref__' dictionary entry is the journal reference
#

Tersoff_PRB_39_5566_Si_C = {
    '__ref__':  'Tersoff J., Phys. Rev. B 39, 5566 (1989)',
    'style':    'Tersoff',
    'el':       [  'C',   'Si'  ],
    'A':        [  1.3936e3,    sqrt(1.3936e3*1.8308e3),  1.8308e3  ],
    'B':        [  3.4674e2,    sqrt(3.4674e2*4.7118e2),  4.7118e2  ],
    'chi':      [  1.0,         0.9776e0,                 1.0       ],
    'lambda':   [  3.4879e0,    (3.4879e0+2.4799e0)/2,    2.4799e0  ],
    'mu':       [  2.2119e0,    (2.2119e0+1.7322e0)/2,    1.7322e0  ],
    'lambda3':  [  0.0,         0.0,                      0.0       ],
    'beta':     [  1.5724e-7,   1.1000e-6   ],
    'n':        [  7.2751e-1,   7.8734e-1   ],
    'c':        [  3.8049e4,    1.0039e5    ],
    'd':        [  4.3484e0,    1.6217e1    ],
    'h':        [  -5.7058e-1,  -5.9825e-1  ],
    'r1':       [  1.80,        sqrt(1.80*2.70),          2.70      ],
    'r2':       [  2.10,        sqrt(2.10*3.00),          3.00      ],
}


Goumri_Said_ChemPhys_302_135_Al_N = {
    '__ref__':  'Goumri-Said S., Kanoun M.B., Merad A.E., Merad G., Aourag H., Chem. Phys. 302, 135 (2004)',
    'el':       [  'Al',   'N'  ],
    'r1':       [  3.20,       2.185,     1.60     ],
    'r2':       [  3.60,       2.485,     2.00     ],
    'A':        [  746.698,    3000.214,  636.814  ],
    'B':        [  40.451,     298.81,    511.76   ],
    'chi':      [  1.0,        1.0,       1.0      ],
    'lambda':   [  2.4647,     3.53051,   5.43673  ],
    'mu':       [  0.9683,     1.99995,   2.7      ],
    'beta':     [  1.094932,   5.2938e-3  ],
    'n':        [  6.085605,   1.33041    ],
    'c':        [  0.074836,   2.0312e4   ],
    'd':        [  19.569127,  20.312     ],
    'h':        [  -0.659266,  -0.56239   ]
}


Matsunaga_Fisher_Matsubara_Jpn_J_Appl_Phys_39_48_B_C_N = {
  '__ref__':  'Matsunaga K., Fisher C., Matsubara H., Jpn. J. Appl. Phys. 39, 48 (2000)',
    'el':       [  'C', 'N', 'B' ],
    'style':    'Tersoff',
    'A':        [  1.3936e3,    -1.0,     -1.0,     1.1e4,      -1.0,     2.7702e2  ],
    'B':        [  3.4674e2,    -1.0,     -1.0,     2.1945e2,   -1.0,     1.8349e2  ],
    'chi':      [  1.0,         0.9685,   1.0025,   1.0,        1.1593,   1.0       ],
    'lambda':   [  3.4879,      -1.0,     -1.0,     5.7708,     -1.0,     1.9922    ],
    'mu':       [  2.2119,      -1.0,     -1.0,     2.5115,     -1.0,     1.5856    ],
    'omega':    [  1.0,         0.6381,   1.0,      1.0,        1.0,      1.0       ],
    'lambda3':  [  0.0,         0.0,      0.0,      0.0,        0.0,      0.0       ],
    'r1':       [  1.80,        -1.0,     -1.0,     2.0,         -1.0,    1.8       ],
    'r2':       [  2.10,        -1.0,     -1.0,     2.3,         -1.0,    2.1       ],
    'beta':     [  1.5724e-7,   1.0562e-1,   1.6e-6     ],
    'n':        [  7.2751e-1,   12.4498,     3.9929     ],
    'c':        [  3.8049e4,    7.9934e4,    5.2629e-1  ],
    'd':        [  4.3484e0,    1.3432e2,    1.5870e-3  ],
    'h':        [  -5.7058e-1,  -0.9973,     0.5        ],
}
# Apply mixing rules
mix_geometric(Matsunaga_Fisher_Matsubara_Jpn_J_Appl_Phys_39_48_B_C_N, 'A')
mix_geometric(Matsunaga_Fisher_Matsubara_Jpn_J_Appl_Phys_39_48_B_C_N, 'B')
mix_arithmetic(Matsunaga_Fisher_Matsubara_Jpn_J_Appl_Phys_39_48_B_C_N, 'lambda')
mix_arithmetic(Matsunaga_Fisher_Matsubara_Jpn_J_Appl_Phys_39_48_B_C_N, 'mu')
mix_geometric(Matsunaga_Fisher_Matsubara_Jpn_J_Appl_Phys_39_48_B_C_N, 'r1')
mix_geometric(Matsunaga_Fisher_Matsubara_Jpn_J_Appl_Phys_39_48_B_C_N, 'r2')


Erhart_PRB_71_035211_SiC = {
    '__ref__':  'Erhart P., Albe K., Phys. Rev. B 71, 035211 (2005)',
    'style':    'Brenner',
    'el':       [  'C',   'Si' ],
    'D0':       [  6.00,      4.36,       3.24      ],
    'r0':       [  1.4276,    1.79,       2.232     ],
    'S':        [  2.167,     1.847,      1.842     ],
    'beta':     [  2.0099,    1.6991,     1.4761    ],
    'gamma':    [  0.11233,   0.011877,   0.114354  ],
    'c':        [  181.910,   273987.0,   2.00494   ],
    'd':        [  6.28433,   180.314,    0.81472   ],
    'h':        [  0.5556,    0.68,       0.259     ],
    'mu':       [  0.0,       0.0,        0.0       ],
    'n':        [  1.0,       1.0,        1.0       ],
    'r1':       [  1.85,      2.20,       2.68      ],
    'r2':       [  2.15,      2.60,       2.96      ]
}


Erhart_PRB_71_035211_Si = {
    '__ref__':  'Erhart P., Albe K., Phys. Rev. B 71, 035211 (2005)',
    'style':    'Brenner',
    'el':       [  'Si' ],
    'D0':       [  3.24     ],
    'r0':       [  2.222    ],
    'S':        [  1.57     ],
    'beta':     [  1.4760   ],
    'gamma':    [  0.09253  ],
    'c':        [  1.13681  ],
    'd':        [  0.63397  ],
    'h':        [  0.335    ],
    'mu':       [  0.0      ],
    'n':        [  1.0      ],
    'r1':       [  2.75     ],
    'r2':       [  3.05     ]
}


Albe_PRB_65_195124_PtC = {
    '__ref__':  'Albe K., Nordlund K., Averback R. S., Phys. Rev. B 65, 195124 (2002)',
    'style':    'Brenner',
    'el':       [  'Pt',   'C' ],
    'D0':       [  3.683,     5.3,        6.0       ],
    'r0':       [  2.384,     1.84,       1.39      ],
    'S':        [  2.24297,   1.1965,     1.22      ],
    'beta':     [  1.64249,   1.836,      2.1       ],
    'gamma':    [  8.542e-4,  9.7e-3,     2.0813e-4 ],
    'c':        [  34.0,      1.23,       330.0     ],
    'd':        [  1.1,       0.36,       3.5       ],
    'h':        [  1.0,       1.0,        1.0       ],
    'mu':       [  1.335,     0.0,        0.0       ],
    'n':        [  1.0,       1.0,        1.0       ],
    'r1':       [  2.9,       2.5,        1.7       ],
    'r2':       [  3.3,       2.8,        2.0       ]
}


Henriksson_PRB_79_114107_FeC = {
    '__ref__': 'Henriksson K.O.E., Nordlund K., Phys. Rev. B 79, 144107 (2009)',
    'style':    'Brenner',
    'el':      [  'Fe', 'C'  ],
    'D0':      [  1.5,         4.82645134,   6.0          ],
    'r0':      [  2.29,        1.47736510,   1.39         ],
    'S':       [  2.0693109,   1.43134755,   1.22         ],
    'beta':    [  1.4,         1.63208170,   2.1          ],
    'gamma':   [  0.0115751,   0.00205862,   0.00020813   ],
    'c':       [  1.2898716,   8.95583221,   330.0        ],
    'd':       [  0.3413219,   0.72062047,   3.5          ],
    'h':       [ -0.26,        0.87099874,   1.0          ],
    'mu':      [  0.0,         0.0,          0.0          ],
    'n':       [  1.0,         1.0,          1.0          ],
    'r1':      [  2.95,        2.3,          1.70         ],
    'r2':      [  3.35,        2.7,          2.00         ]
}


Kioseoglou_PSSb_245_1118_AlN = {
    '__ref__':  'Kioseoglou J., Komninou Ph., Karakostas Th., Phys. Stat. Sol. (b) 245, 1118 (2008)',
    'style':    'Brenner',
    'el':       [  'N',   'Al' ],
    'D0':       [  9.9100,     3.3407,     1.5000   ],
    'r0':       [  1.1100,     1.8616,     2.4660   ],
    'S':        [  1.4922,     1.7269,     2.7876   ],
    'beta':     [  2.05945,    1.7219,     1.0949   ],
    'gamma':    [  0.76612,    1.1e-6,     0.3168   ],
    'c':        [  0.178493,   100390,     0.0748   ],
    'd':        [  0.20172,    16.2170,    19.5691  ],
    'h':        [  0.045238,   0.5980,     0.6593   ],
    'mu':       [  0.0,        0.0,        0.0      ],
    'n':        [  1.0,        0.7200,     6.0865   ],
    'r1':       [  2.00,       2.19,       3.40     ],
    'r2':       [  2.40,       2.49,       3.60     ]
}


# Juslin's W-C-H parameterization
Juslin_JAP_98_123520_WCH = {
    '__ref__': 'Juslin N., Erhart P., Traskelin P., Nord J., Henriksson K.O.E, Nordlund K., Salonen E., Albe K., J. Appl. Phys. 98, 123520 (2005)',
    'style':    'Brenner',
    'el':      [ 'W', 'C', 'H' ],
    'D0':      [ 5.41861,     6.64,      2.748,    0.0,  6.0,         3.6422,      0.0,  3.642,    4.7509  ],
    'r0':      [ 2.34095,     1.90547,   1.727,   -1.0,  1.39,        1.1199,     -1.0,  1.1199,   0.74144 ],
    'S':       [ 1.92708,     2.96149,   1.2489,   0.0,  1.22,        1.69077,     0.0,  1.69077,  2.3432  ],
    'beta':    [ 1.38528,     1.80370,   1.52328,  0.0,  2.1,         1.9583,      0.0,  1.9583,   1.9436  ],
    'gamma':   [ 0.00188227,  0.072855,  0.0054,   0.0,  0.00020813,  0.00020813,  0.0,  12.33,    12.33   ],
    'c':       [ 2.14969,     1.10304,   1.788,    0.0,  330.0,       330.0,       0.0,  0.0,      0.0     ],
    'd':       [ 0.17126,     0.33018,   0.8255,   0.0,  3.5,         3.5,         0.0,  1.0,      1.0     ],
    'h':       [-0.27780,     0.75107,   0.38912,  0.0,  1.0,         1.0,         0.0,  1.0,      1.0     ],
    'n':       [ 1.0,         1.0,       1.0,      0.0,  1.0,         1.0,         0.0,  1.0,      1.0     ],
    'alpha':   [ 0.45876, 0.0, 0.0, 0.45876, 0.0, 0.0, 0.45876, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 0.0, 4.0, 4.0 ],
    'omega':   [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.94586, 4.54415, 1.0, 1.0, 1.0, 1.0, 0.33946, 0.22006, 1.0, 1.0, 1.0 ],
    'r1':      [ 3.20,        2.60,      2.68,     0.0,  1.70,        1.30,        0.0,  1.30,     1.10    ],
    'r2':      [ 3.80,        3.00,      2.96,     0.0,  2.00,        1.80,        0.0,  1.80,     1.70    ],
}


Kuopanportti_CMS_111_525_FeCH = {
    '__ref__' : 'Kuopanportti P., Hayward, N., Fu C., Kuronen A., Nordlund K., Comp. Mat. Sci. 111, 525 (2016)',
    'style':    'Brenner',
    'el':     [ 'Fe', 'C', 'H'],
    'D0':     [ 1.5,      4.82645134,  1.630,    0.0,  6.0,         3.6422,      0.0,  3.642,    4.7509  ],
    'r0':     [ 2.29,     1.47736510,  1.589,   -1.0,  1.39,        1.1199,     -1.0,  1.1199,   0.74144 ],
    'S':      [ 2.0693,   1.43134755,  4.000,    0.0,  1.22,        1.69077,     0.0,  1.69077,  2.3432  ],
    'beta':   [ 1.4,      1.63208170,  1.875,    0.0,  2.1,         1.9583,      0.0,  1.9583,   1.9436  ],
    'gamma':  [ 0.01158,  0.00205862,  0.01332,  0.0,  0.00020813,  0.00020813,  0.0,  12.33,    12.33   ],
    'c':      [ 1.2899,   8.95583221,  424.5,    0.0,  330.0,       330.0,       0.0,  0.0,      0.0     ],
    'd':      [ 0.3413,   0.72062047,  7.282,    0.0,  3.5,         3.5,         0.0,  1.0,      1.0     ],
    'h':      [-0.26,     0.87099874, -0.1091,   0.0,  1.0,         1.0,         0.0,  1.0,      1.0     ],
    'n':      [ 1.0,      1.0,         1.0,      0.0,  1.0,         1.0,         0.0,  1.0,      1.0     ],
    'alpha':  [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 0.0, 4.0, 4.0 ],
    'omega':  [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.94586, 4.54415, 1.0, 1.0, 1.0, 1.0, 0.33946, 0.22006, 1.0, 1.0, 1.0 ],
    'r1':     [ 2.95,     2.30,        2.2974,   0.0,  1.70,        1.30,        0.0,  1.30,     1.10    ],
    'r2':     [ 3.35,     2.70,        2.6966,   0.0,  2.00,        1.80,        0.0,  1.80,     1.70    ],
}


Brenner_PRB_42_9458_C_I = {
    '__ref__':  'Brenner D., Phys. Rev. B 42, 9458 (1990) [potential I]',
    'style':    'Brenner',
    'el':       [  'C'  ],
    'D0':       [  6.325            ],
    'r0':       [  1.315            ],
    'S':        [  1.29             ],
    'beta':     [  1.5              ],
    'gamma':    [  0.011304         ],
    'c':        [  19.0             ],
    'd':        [  2.5              ],
    'h':        [  1.0              ],
    'mu':       [  0.0              ],
    'n':        [  1.0/(2*0.80469)  ],
    'r1':       [  1.70             ],
    'r2':       [  2.00             ]
}


Brenner_PRB_42_9458_C_II = {
    '__ref__':  'Brenner D., Phys. Rev. B 42, 9458 (1990) [potential II]',
    'style':    'Brenner',
    'el':       [  'C'  ],
    'D0':       [  6.0          ],
    'r0':       [  1.39         ],
    'S':        [  1.22         ],
    'beta':     [  2.1          ],
    'gamma':    [  0.00020813   ],
    'c':        [  330.0        ],
    'd':        [  3.5          ],
    'h':        [  1.0          ],
    'mu':       [  0.0          ],
    'n':        [  1.0/(2*0.5)  ],
    'r1':       [  1.70         ],
    'r2':       [  2.00         ]
}


def _a(x):
    '''
    Compute absolute value (norm) of an array of vectors
    '''
    return np.linalg.norm(x, axis=1)


def _o(x, y):
    """Outer product"""
    return x.reshape(-1, 3, 1) * y.reshape(-1, 1, 3)


def TersoffBrenner(parameters):
    """
    Implementation of the function form for Abell-Tersoff-Brenner potentials.

    Reference
    ------------
    J. Tersoff, Physical review B 39.8 (1989): 5566.
    """

    style = parameters['style'].lower()

    el = parameters['el']

    nb_elements = len(el)
    nb_pairs = nb_elements * (nb_elements + 1) // 2

    c = np.array(parameters['c'])
    d = np.array(parameters['d'])
    h = np.array(parameters['h'])
    r1 = np.array(parameters['r1'])
    r2 = np.array(parameters['r2'])
    if style == 'tersoff':
        # These are Tersoff-style parameters. The symbols follow the notation in
        # Tersoff J., Phys. Rev. B 39, 5566 (1989)
        #
        # In particular, pair terms are characterized by A, B, lam, mu and parameters for the three body terms ijk
        # depend only on the type of atom i
        A = np.array(parameters['A'])
        B = np.array(parameters['B'])
        lam = np.array(parameters['lambda'])
        mu = np.array(parameters['mu'])
        beta = np.array(parameters['beta'])
        lambda3 = np.array(parameters['lambda3'])
        chi = np.array(parameters['chi'])
        n = np.array(parameters['n'])
        
        # Consistency check
        assert len(A) == nb_pairs
        assert len(B) == nb_pairs
        assert len(lam) == nb_pairs
        assert len(mu) == nb_pairs
        assert len(beta) == nb_elements
        assert len(lambda3) == nb_pairs
        assert len(chi) == nb_pairs
        assert len(n) == nb_elements
        assert len(c) == nb_elements
        assert len(d) == nb_elements
        assert len(h) == nb_elements
        assert len(r1) == nb_pairs
        assert len(r2) == nb_pairs
    elif style == 'brenner':
        # These are Brenner/Erhart-Albe-style parameters. The symbols follow the notation in
        # Brenner D., Phys. Rev. B 42, 9458 (1990) and
        # Erhart P., Albe K., Phys. Rev. B 71, 035211 (2005)
        #
        # In particular, pairs terms are characterized by D0, S, beta, r0, the parameters n, chi are always unity and
        # parameters for the three body terms ijk depend on the type of the bond ij
        _D0 = np.array(parameters['D0'])
        _S = np.array(parameters['S'])
        _r0 = np.array(parameters['r0'])
        _beta = np.array(parameters['beta'])
        _mu = np.array(parameters['mu'])
        gamma = np.array(parameters['gamma'])

        # Convert to Tersoff parameters
        lambda3 = 2 * _mu
        lam = _beta * np.sqrt(2 * _S)
        mu = _beta * np.sqrt(2 / _S)
        A = _D0 / (_S - 1) * np.exp(lam * _r0)
        B = _S * _D0 / (_S - 1) * np.exp(mu * _r0)

        # Consistency check
        assert len(A) == nb_pairs
        assert len(B) == nb_pairs
        assert len(lam) == nb_pairs
        assert len(mu) == nb_pairs
        assert len(gamma) == nb_pairs
        assert len(lambda3) == nb_pairs
        assert len(c) == nb_pairs
        assert len(d) == nb_pairs
        assert len(h) == nb_pairs
        assert len(r1) == nb_pairs
        assert len(r2) == nb_pairs
    else:
        raise ValueError(f'Unknown parameter style {style}')

    # Number of elements in parameter set. We will assign a consecutive internal element number.
    nb_elements = len(el)
    atomic_numbers = [data.atomic_numbers[e] for e in el]
    atomic_number_to_internal_type = np.zeros(np.max(atomic_numbers)+1, dtype=int)
    atomic_number_to_internal_type[atomic_numbers] = np.arange(len(atomic_numbers))

    # Assign internal element number given the atomic number
    atom_type = lambda n: atomic_number_to_internal_type[n]

    # Combine two internal element number into an index for a pair property
    pair_type = lambda i, j: pair_index(i, j, nb_elements)

    f = lambda r, p: np.where(
        r < r1[p],
        np.ones_like(r),
        np.where(r > r2[p],
                 np.zeros_like(r),
                 (1 + np.cos((np.pi * (r - r1[p]) / (r2[p] - r1[p])))) / 2
                 )
    )
    df = lambda r, p: np.where(
        r < r1[p],
        np.zeros_like(r),
        np.where(r > r2[p],
                 np.zeros_like(r),
                 -np.pi * np.sin(np.pi * (r - r1[p]) / (r2[p] - r1[p])) / (2 * (r2[p] - r1[p]))
                 )
    )
    ddf = lambda r, p: np.where(
        r < r1[p],
        np.zeros_like(r),
        np.where(r > r2[p],
                 np.zeros_like(r),
                 -np.pi ** 2 * np.cos(np.pi * (r - r1[p]) / (r2[p] - r1[p])) / (2 * (r2[p] - r1[p]) ** 2)
                 )
    )

    fR = lambda r, p: A[p] * np.exp(-lam[p] * r)
    dfR = lambda r, p: -lam[p] * fR(r, p)
    ddfR = lambda r, p: lam[p] ** 2 * fR(r, p)

    fA = lambda r, p: -B[p] * np.exp(-mu[p] * r)
    dfA = lambda r, p: -mu[p] * fA(r, p)
    ddfA = lambda r, p: mu[p] ** 2 * fA(r, p)

    if style == 'tersoff':
        b = lambda xi, i, p: \
            chi[p] * (1 + (beta[i] * xi) ** n[i]) ** (-1 / (2 * n[i]))
        db = lambda xi, i, p: \
            chi[p] * np.where(xi == 0.0, 0.0, -0.5 * beta[i] * np.power(beta[i] * xi, n[i] - 1, where=xi != 0.0)
                              * (1 + (beta[i] * xi) ** n[i]) ** (-1 - 1 / (2 * n[i])))
        ddb = lambda xi, i, p: \
            chi[p] * np.where(xi == 0.0, 0.0, -0.5 * beta[i] ** 2 * (n[i] - 1)
                              * np.power(beta[i] * xi, n[i] - 2, where=xi != 0.0)
                              * np.power(1 + (beta[i] * xi) ** n[i], -1 - 1 / (2 * n[i]))
                              - 0.5 * beta[i] ** 2 * n[i] * np.power(beta[i] * xi, -2 + 2 * n[i], where=xi != 0.0)
                              * (-1 - 1 / (2 * n[i])) * np.power(1 + (beta[i] * xi) ** n[i], -2 - 1 / (2 * n[i])))

        g = lambda cost, i, p:\
            1 + c[i] ** 2 / d[i] ** 2 - c[i] ** 2 / (d[i] ** 2 + (h[i] - cost) ** 2)
        dg = lambda cost, i, p:\
            -2 * c[i] ** 2 * (h[i] - cost) / (d[i] ** 2 + (h[i] - cost) ** 2) ** 2
        ddg = lambda cost, i, p:\
            2 * c[i] ** 2 / (d[i] ** 2 + (h[i] - cost) ** 2) ** 2 \
            - 8 * c[i] ** 2 * (h[i] - cost) ** 2 / (d[i] ** 2 + (h[i] - cost) ** 2) ** 3
    else:
        b = lambda xi, i, p: np.power(1 + gamma[p] * xi, -0.5)
        db = lambda xi, i, p: -0.5 * gamma[p] * np.power(1 + gamma[p] * xi, -1.5)
        ddb = lambda xi, i, p: 0.75 * (gamma[p] ** 2) * np.power(1 + gamma[p] * xi, -2.5)

        g = lambda cost, i, p:\
            1 + c[p] ** 2 / d[p] ** 2 - c[p] ** 2 / (d[p] ** 2 + (h[p] + cost) ** 2)
        dg = lambda cost, i, p:\
            2 * c[p] ** 2 * (h[p] + cost) / (d[p] ** 2 + (h[p] + cost) ** 2) ** 2
        ddg = lambda cost, i, p:\
            2 * c[p] ** 2 / (d[p] ** 2 + (h[p] + cost) ** 2) ** 2\
            - 8 * c[p] ** 2 * (h[p] + cost) ** 2 / (d[p] ** 2 + (h[p] + cost) ** 2) ** 3

    hf = lambda rij, rik, ij, ik: \
        f(_a(rik), ik) * np.exp(lambda3[ik] * (_a(rij) - _a(rik)))
    d1h = lambda rij, rik, ij, ik: \
        lambda3[ik] * hf(rij, rik, ij, ik)
    d2h = lambda rij, rik, ij, ik: \
        -lambda3[ik] * hf(rij, rik, ij, ik) + df(_a(rik), ik) * np.exp(lambda3[ik] * (_a(rij) - _a(rik)))
    d11h = lambda rij, rik, ij, ik: \
        lambda3[ik] ** 2 * hf(rij, rik, ij, ik)
    d12h = lambda rij, rik, ij, ik: \
        (df(_a(rik), ik) * lambda3[ik] * np.exp(lambda3[ik] * (_a(rij) - _a(rik)))
         - lambda3[ik] * hf(rij, rik, ij, ik))
    d22h = lambda rij, rik, ij, ik: \
        (ddf(_a(rik), ik) * np.exp(lambda3[ik] * (_a(rij) - _a(rik)))
         + 2 * lambda3[ik] * np.exp(lambda3[ik] * (_a(rij) - _a(rik))) * df(_a(rik), ik)
         + lambda3[ik] ** 2 * hf(rij, rik, ij, ik))

    # Derivatives of F
    F = lambda r, xi, i, p: \
        f(r, p) * (fR(r, p) + b(xi, i, p) * fA(r, p))
    d1F = lambda r, xi, i, p: \
        df(r, p) * (fR(r, p) + b(xi, i, p) * fA(r, p)) \
        + f(r, p) * (dfR(r, p) + b(xi, i, p) * dfA(r, p))
    d2F = lambda r, xi, i, p: \
        f(r, p) * fA(r, p) * db(xi, i, p)
    d11F = lambda r, xi, i, p: \
        f(r, p) * (ddfR(r, p) + b(xi, i, p) * ddfA(r, p)) \
        + 2 * df(r, p) * (dfR(r, p) + b(xi, i, p) * dfA(r, p)) + ddf(r, p) * (fR(r, p) + b(xi, i, p) * fA(r, p))
    d22F = lambda r, xi, i, p: \
        f(r, p) * fA(r, p) * ddb(xi, i, p)
    d12F = lambda r, xi, i, p: \
        f(r, p) * dfA(r, p) * db(xi, i, p) + fA(r, p) * df(r, p) * db(xi, i, p)

    # Helping functions
    costh = lambda rij, rik: np.sum(rij * rik, axis=1) / (_a(rij) * _a(rik))

    c1 = lambda rij, rik: ((rik.T / _a(rik) - rij.T / _a(rij) * costh(rij, rik)) / _a(rij)).T
    c2 = lambda rij, rik: ((rij.T / _a(rij) - rik.T / _a(rik) * costh(rij, rik)) / _a(rik)).T

    dc11 = lambda rij, rik: \
        ((- _o(c1(rij, rik), rij) - _o(rij, c1(rij, rik))
          - (costh(rij, rik) * (np.eye(3) - (_o(rij, rij).T / _a(rij) ** 2).T).T).T).T / _a(rij) ** 2).T
    dc22 = lambda rij, rik:\
        ((- _o(c2(rij, rik), rik) - _o(rik, c2(rij, rik))
          - (costh(rij, rik) * (np.eye(3) - (_o(rik, rik).T / _a(rik) ** 2).T).T).T).T / _a(rik) ** 2).T
    dc12 = lambda rij, rik: \
        (((np.eye(3) - (_o(rij, rij).T / _a(rij) ** 2).T).T / _a(rij) - _o(c1(rij, rik), rik).T / _a(rik)) / _a(rik)).T

    Dh1 = lambda rij, rik, ij, ik: (d1h(rij, rik, ij, ik) * rij.T / _a(rij)).T
    Dh2 = lambda rij, rik, ij, ik: (d2h(rij, rik, ij, ik) * rik.T / _a(rik)).T

    Dg1 = lambda rij, rik, i, ij: (dg(costh(rij, rik), i, ij) * c1(rij, rik).T).T
    Dg2 = lambda rij, rik, i, ij: (dg(costh(rij, rik), i, ij) * c2(rij, rik).T).T

    # Derivatives of G
    G = lambda rij, rik, i, ij, ik: g(costh(rij, rik), i, ij) * hf(rij, rik, ij, ik)

    d1G = lambda rij, rik, i, ij, ik: (
                Dh1(rij, rik, ij, ik).T * g(costh(rij, rik), i, ij) + hf(rij, rik, ij, ik) * Dg1(rij, rik, i, ij).T).T
    d2G = lambda rij, rik, i, ij, ik: (
                Dh2(rij, rik, ij, ik).T * g(costh(rij, rik), i, ij) + hf(rij, rik, ij, ik) * Dg2(rij, rik, i, ij).T).T

    d11G = lambda rij, rik, i, ij, ik: \
        _o(Dg1(rij, rik, i, ij), Dh1(rij, rik, ij, ik)) + _o(Dh1(rij, rik, ij, ik), Dg1(rij, rik, i, ij)) \
        + (g(costh(rij, rik), i, ij) * Dh11(rij, rik, ij, ik).T).T + (hf(rij, rik, ij, ik) * Dg11(rij, rik, i, ij).T).T

    Dh11 = lambda rij, rik, ij, ik: \
        (d11h(rij, rik, ij, ik) * _o(rij, rij).T / _a(rij) ** 2
         + d1h(rij, rik, ij, ik) * ((np.eye(3) - (_o(rij, rij).T / _a(rij) ** 2).T).T / _a(rij))).T

    Dg11 = lambda rij, rik, i, ij: \
        (ddg(costh(rij, rik), i, ij) * _o(c1(rij, rik), c1(rij, rik)).T
         + dg(costh(rij, rik), i, ij) * dc11(rij, rik).T).T

    d22G = lambda rij, rik, i, ij, ik: \
        _o(Dg2(rij, rik, i, ij), Dh2(rij, rik, ij, ik)) + _o(Dh2(rij, rik, ij, ik), Dg2(rij, rik, i, ij)) \
        + ((g(costh(rij, rik), i, ij) * Dh22(rij, rik, ij, ik).T).T
           + (hf(rij, rik, ij, ik) * Dg22(rij, rik, i, ij).T).T)

    Dh22 = lambda rij, rik, ij, ik: \
        (d22h(rij, rik, ij, ik) * _o(rik, rik).T / _a(rik) ** 2
         + d2h(rij, rik, ij, ik) * ((np.eye(3) - (_o(rik, rik).T / _a(rik) ** 2).T).T / _a(rik))).T

    Dg22 = lambda rij, rik, i, ij: \
        (ddg(costh(rij, rik), i, ij) * _o(c2(rij, rik), c2(rij, rik)).T
         + dg(costh(rij, rik), i, ij) * dc22(rij, rik).T).T

    d12G = lambda rij, rik, i, ij, ik: \
        _o(Dg1(rij, rik, i, ij), Dh2(rij, rik, ij, ik)) + _o(Dh1(rij, rik, ij, ik), Dg2(rij, rik, i, ij)) \
        + ((g(costh(rij, rik), i, ij) * Dh12(rij, rik, ij, ik).T).T
           + (hf(rij, rik, ij, ik) * Dg12(rij, rik, i, ij).T).T)

    Dh12 = lambda rij, rik, ij, ik: \
        (d12h(rij, rik, ij, ik) * _o(rij, rik).T / (_a(rij) * _a(rik))).T

    Dg12 = lambda rij, rik, i, ij: \
        (ddg(costh(rij, rik), i, ij) * _o(c1(rij, rik), c2(rij, rik)).T
         + dg(costh(rij, rik), i, ij) * dc12(rij, rik).T).T

    return {
        'atom_type': atom_type,
        'pair_type': pair_type,
        'F': F,
        'G': G,
        'd1F': d1F,
        'd2F': d2F,
        'd11F': d11F,
        'd12F': d12F,
        'd22F': d22F,
        'd1G': d1G,
        'd2G': d2G,
        'd11G': d11G,
        'd22G': d22G,
        'd12G': d12G,
        'cutoff': r2,
    }
