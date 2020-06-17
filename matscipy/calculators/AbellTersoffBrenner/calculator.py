import numpy as np
from ase.calculators.calculator import Calculator
from matscipy.neighbours import neighbour_list, first_neighbours, triplet_list


class AbellTersoffBrenner(Calculator):
    implemented_properties = ['energy']  # , 'stress', 'forces', "hessian"]
    default_parameters = {}
    name = 'ThreeBodyPotential'

    def __init__(self, g_a, g_o, g, f_c, xi_ij, b_ij, psi_ij):
        Calculator.__init__(self)
        self.g_a = g_a
        self.g_o = g_o
        self.g = g
        self.f_c = np.vectorize(f_c)
        self.xi_ij = xi_ij
        self.b_ij = b_ij
        self.psi_ij = psi_ij
        self.cos_theta = lambda abs_dr_ij, abs_dr_ik, dr_ijc, dr_ikc: np.einsum('ij,ij->i', dr_ijc, dr_ikc)/(abs_dr_ij*abs_dr_ik)
        # Test: theta(np.array([1, 1]), np.array([1, 2]), np.array([[0,1], [0,1]]), np.array([[-1, 0],[0,2]])) == [0,1]

    def __call__(self):
        return self.psi_ij

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # build pair, first_i, triplet and first_ij lists
        i_n, j_n, abs_dr_n, dr_nc = neighbour_list('ijdD', atoms=atoms,
                                                   cutoff=self.cutoff)
        first_i = first_neighbours(i_n)
        ij_t, ik_t = triplet_list(first_i)
        first_ij = first_neighbours(ij_t)

        # mask results to match the triplet absolut distance lists
        # for r_ij and r_ik
        self.r_ij = abs_dr_n[ij_t]
        self.r_ik = abs_dr_n[ik_t]
        # safe the absolut distances to r_ij_pair
        self.r_ij_pair = abs_dr_n

        # calculate the cos_theat values from the geometric identity
        self.cos_theta_n = self.cos_theta(self.r_ij, self.r_ik,
                                          dr_nc[ij_t], dr_nc[ik_t])

        # calculate the cutoff function values for all pairs and triplets
        f_c_trip = self.f_c(self.r_ij)
        f_c_pair = self.f_c(self.r_ij_pair)

        g_n = self.g(self.g_o(self.cos_theta_n), self.g_a(self.cos_theta_n))

        xi_ij_ik = self.xi_ij(self.r_ij, self.r_ik, f_c_trip, g_n)

        # calculate subsums over k for each pair ij (k \neq ij)
        xi_ij = np.add.reduceat(xi_ij_ik, first_ij[:-1])

        b_ij = self.b_ij(xi_ij)

        psi_ij = self.psi_ij(self.r_ij_pair, f_c_pair, b_ij)

        epot = 1/2*np.sum(psi_ij)

        self.results = {'energy': epot}


class KumagaiTersoff(AbellTersoffBrenner):
    def __init__(self, A, B, lambda_1, lambda_2, eta, delta, alpha, beta, c_1,
                 c_2, c_3, c_4, c_5, h, R_1, R_2, __ref__=None, el=None):
        self.__ref__ = __ref__
        self.cutoff = R_2

        super().__init__(
            g_a=lambda cos_theta: (1 + c_4*np.exp(-c_5*(h - cos_theta)**2)),
            g_o=lambda cos_theta: ((c_2*(h - cos_theta)**2)/(c_3 + (h - cos_theta)**2)),
            g=lambda g_o, g_a: c_1 + g_o*g_a,
            f_c=lambda r_ij: (0.0+(r_ij <= R_1)) if (r_ij >= R_2 or r_ij <= R_1) else (1/2+(9/16)*np.cos(np.pi*(r_ij - R_1)/(R_2 - R_1)) - (1/16)*np.cos(3*np.pi*(r_ij-R_1)/(R_2-R_1))),
            xi_ij=lambda r_ij, r_ik, f_c, g: f_c * g * np.exp(alpha*(r_ij-r_ik)**beta),
            b_ij=lambda xi_ij:  (1+xi_ij**eta)**(-delta),
            psi_ij_rep=lambda r_ij, f_c: f_c*(A*np.exp(-lambda_1 * r_ij)),
            psi_ij_att=lambda r_ij, f_c, b_ij: -f_c*b_ij*B*np.exp(-lambda_2 * r_ij),
            psi_ij=lambda r_ij, f_c, b_ij: f_c*(A*np.exp(-lambda_1 * r_ij) - b_ij*B*np.exp(-lambda_2 * r_ij))
            )

        def get_cutoff(self):
            return self.cutoff

Kumagai_CompMaterSci_39_457_Si_py = {
    "__ref__":  "T. Kumagai, S. Izumi, S. Hara, and S. Sakai, Comput. Mater. Sci. 39, 457 (2007)",  # TODO: cite
    "el":       [ "Si"        ],
    "A":        [ 3281.5905   ],
    "B":        [ 121.00047   ],
    "lambda_1": [ 3.2300135   ],
    "lambda_2": [ 1.3457970   ],
    "eta":      [ 1.0000000   ],
    "delta":    [ 0.53298909  ],
    "alpha":    [ 2.3890327   ],
    "beta":     [ 1           ],
    "c_1":      [ 0.20173476  ],
    "c_2":      [ 730418.72   ],
    "c_3":      [ 1000000.0   ],
    "c_4":      [ 1.0000000   ],
    "c_5":      [ 26.000000   ],
    "h":        [ -0.36500000 ],
    "R_1":      [ 2.70        ],
    "R_2":      [ 3.30        ],
  }

Kumagai_CompMaterSci_39_457_Si_py_var = {
    "__ref__":  "T. Kumagai, S. Izumi, S. Hara, and S. Sakai, Comput. Mater. Sci. 39, 457 (2007)",  # TODO: cite
    "el":        "Si"        ,
    "A":         3281.5905   ,
    "B":         121.00047   ,
    "lambda_1":  3.2300135   ,
    "lambda_2":  1.3457970   ,
    "eta":       1.0000000   ,
    "delta":     0.53298909  ,
    "alpha":     2.3890327   ,
    "beta":      1           ,
    "c_1":       0.20173476  ,
    "c_2":       730418.72   ,
    "c_3":       1000000.0   ,
    "c_4":       1.0000000   ,
    "c_5":       26.000000   ,
    "h":         -0.36500000 ,
    "R_1":       2.70        ,
    "R_2":       3.30        ,
  }

    
