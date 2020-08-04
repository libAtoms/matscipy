import numpy as np
from ase.calculators.calculator import Calculator
from matscipy.neighbours import neighbour_list, first_neighbours, triplet_list


class AbellTersoffBrenner(Calculator):
    implemented_properties = ['energy', 'forces', 'hessian']
    default_parameters = {}
    name = 'ThreeBodyPotential'

    def __init__(self, g, f_c, xi, b, psi_rep, psi_att, exp,
                 df_c_r, dpsi_rep_r, dpsi_att_r, db_xi, dg_costheta,
                 dexp_r_ij, dexp_r_ik):
        Calculator.__init__(self)
        self.g = g
        self.f_c = f_c
        self.xi = xi
        self.b = b
        self.psi_rep = psi_rep
        self.psi_att = psi_att
        self.cos_theta = lambda abs_dr_ij, abs_dr_ik, dr_ijc, dr_ikc: \
            np.einsum('ij,ij->i', dr_ijc, dr_ikc)/(abs_dr_ij*abs_dr_ik)
        # forces
        self.exp = exp
        self.dg_costheta = dg_costheta
        self.df_c_r = df_c_r
        self.dpsi_rep_r = dpsi_rep_r
        self.dpsi_att_r = dpsi_att_r
        self.dexp_r_ij = dexp_r_ij
        self.dexp_r_ik = dexp_r_ik
        self.db_xi = db_xi

    def __call__(self):
        return None

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        nat = len(self.atoms)

        # get neighbour and triplet lists
        i_n, j_n, self.r_p, dr_pc = neighbour_list('ijdD', atoms=atoms,
                                                   cutoff=self.cutoff)
        first_i = first_neighbours(nat, i_n)

        ij_t, ik_t = triplet_list(first_i)

        # calculate energy

        # pair terms
        f_c_p = self.f_c(self.r_p)

        U_p = self.psi_att(self.r_p, f_c_p)
        V_p = self.psi_rep(self.r_p, f_c_p)

        # triplet terms
        if ij_t.size > 0:
            self.cos_theta_t = self.cos_theta(self.r_p[ij_t], self.r_p[ik_t],
                                              dr_pc[ij_t], dr_pc[ik_t])
            g_t = self.g(self.cos_theta_t)

            xi_t = self.xi(self.r_p[ij_t], self.r_p[ik_t], f_c_p[ik_t], g_t)
            xi_p = np.bincount(ij_t, weights=xi_t, minlength=nat)
            b_p = self.b(xi_p)
        else:
            b_p = np.ones_like(i_n)

        e_n = V_p + b_p * U_p

        epot = 1/2*np.sum(e_n)

        # calculate forces

        # '-' is due to a different definition of \v{r}_ij
        norm_pc = -dr_pc/self.r_p.reshape(-1, 1)

        # pair terms
        df_c_r_p = self.df_c_r(self.r_p)

        f_att_p = self.dpsi_att_r(f_c_p, self.r_p, df_c_r_p)
        f_rep_p = self.dpsi_rep_r(f_c_p, self.r_p, df_c_r_p)

        f_pc = norm_pc*(f_rep_p + b_p*f_att_p).reshape(-1, 1)

        # triplet terms

        if ij_t.size > 0:

            dha = lambda r_ij, r_ik: self.f_c(r_ik)*self.dexp_r_ij(r_ij, r_ik)
            dhb = lambda r_ij, r_ik: self.f_c(r_ik)*self.dexp_r_ik(r_ij, r_ik)\
                + self.df_c_r(r_ik) * self.exp(r_ij, r_ik)

            db_p = self.db_xi(xi_p)

            f_aa_p = np.bincount(ij_t, weights=(g_t*U_p[ij_t]*db_p[ij_t] *
                                                dha(self.r_p[ij_t],
                                                    self.r_p[ik_t])))
            f_ab_p = np.bincount(ij_t, weights=(g_t*U_p[ik_t]*db_p[ik_t] *
                                                dhb(self.r_p[ik_t],
                                                    self.r_p[ij_t])))

            f_a_p_c = norm_pc*(f_aa_p + f_ab_p).reshape(-1, 1)

            h = lambda r_ij, r_ik:  self.f_c(r_ik) * self.exp(r_ij, r_ik)

            c_tc = (norm_pc[ik_t] - self.cos_theta_t.reshape(-1, 1)
                    * norm_pc[ij_t])/(self.r_p[ij_t].reshape(-1, 1))

            dg_t = self.dg_costheta(self.cos_theta_t)

            f_ba_tc = c_tc*(dg_t*U_p[ij_t]*db_p[ij_t]*h(self.r_p[ij_t],
                            self.r_p[ik_t])).reshape(-1, 1)
            f_bb_tc = c_tc*(dg_t*U_p[ik_t]*db_p[ik_t]*h(self.r_p[ik_t],
                            self.r_p[ij_t])).reshape(-1, 1)
            # TODO: use symmetry; proabably not existent
            f_b_tc = f_ba_tc + f_bb_tc

            fx_b_p = np.bincount(ij_t, weights=f_b_tc[:, 0])
            fy_b_p = np.bincount(ij_t, weights=f_b_tc[:, 1])
            fz_b_p = np.bincount(ij_t, weights=f_b_tc[:, 2])

            fx_p = f_pc[:, 0] + f_a_p_c[:, 0] + fx_b_p
            fy_p = f_pc[:, 1] + f_a_p_c[:, 1] + fy_b_p
            fz_p = f_pc[:, 2] + f_a_p_c[:, 2] + fz_b_p

            fx_n = -1/2*(np.bincount(i_n, weights=fx_p) -
                         np.bincount(j_n, weights=fx_p))
            fy_n = -1/2*(np.bincount(i_n, weights=fy_p) -
                         np.bincount(j_n, weights=fy_p))
            fz_n = -1/2*(np.bincount(i_n, weights=fz_p) -
                         np.bincount(j_n, weights=fz_p))

        f_n = np.transpose([fx_n, fy_n, fz_n])

        self.results = {'energy': epot, 'forces': f_n}


class KumagaiTersoff(AbellTersoffBrenner):
    def __init__(self, A, B, lambda_1, lambda_2, eta, delta, alpha, beta,
                 c_1, c_2, c_3, c_4, c_5, h, R_1, R_2, __ref__=None, el=None):
        self.__ref__ = __ref__
        self.cutoff = R_2
        super().__init__(
            g=lambda cos_theta: c_1 + (1 + c_4*np.exp(-c_5*(h-cos_theta)**2)) *\
                                ((c_2*(h-cos_theta)**2)/(c_3 + (h-cos_theta)**2)),

            f_c=lambda r: np.where(r <= R_1, 1.0,  np.where(r >= R_2, 0.0, (1/2+(9/16) \
                                                *np.cos(np.pi*(r - R_1)/(R_2 - R_1)) - (1/16) \
                                                    *np.cos(3*np.pi*(r-R_1)/(R_2-R_1))))),

            # f_c =  lambda r : np.where(r <= R_1, 1.0,  np.where(r >= R_2, 0.0, \
            #                    (1/2*(1+np.cos(np.pi*(r - R_1)/(R_2 - R_1)))))),


            xi=lambda r_ij, r_ik, f_c, g: f_c * g * np.exp(alpha*(r_ij-r_ik)**beta),
            exp=lambda r_ij, r_ik: np.exp(alpha*(r_ij-r_ik)**beta),
            b=lambda xi_ij:  1/((1+xi_ij**eta)**(delta)),
            psi_rep=lambda r_ij, f_c: f_c*(A*np.exp(-lambda_1 * r_ij)),
            psi_att=lambda r_ij, f_c: -f_c*(B*np.exp(-lambda_2 * r_ij)),

            # forces
            df_c_r=lambda r: np.where(r >= R_2, 0.0,\
                                      np.where(r <= R_1, 0.0,\
                                               (3*np.pi*(3*np.sin(np.pi *\
                                                         (R_1 - r)/(R_1 - R_2))\
                                        - np.sin(3*np.pi*(R_1 - r)/(R_1 - R_2))))/(16*(R_1 - R_2)))),

            # df_c_r = lambda r: np.where(r >= R_2, 0.0, np.where(r <= R_1, 0.0, np.pi*np.sin(np.pi*(R_1 - r)/(R_1 - R_2))/(2*(R_1 - R_2)))),

            dpsi_rep_r=lambda f_c, r, df_c_r:  A*np.exp(-lambda_1*r)*(df_c_r - f_c*lambda_1),

            dpsi_att_r=lambda f_c, r, df_c_r: -B*np.exp(-lambda_2*r)*(df_c_r - f_c*lambda_2),

            db_xi=lambda xi: -delta*eta*xi**(eta-1)*(xi**eta+1)**(-delta-1),

            dg_costheta=lambda cos_theta: 2*c_2*(cos_theta - h)*((c_3 + (cos_theta - h)**2)* \
                                                        (-c_4*c_5*(cos_theta - h)**2 + c_4 + \
                                                         np.exp(c_5*(cos_theta - h)**2)) - \
                                                        (c_4 + np.exp(c_5*(cos_theta - h)**2)) \
                                                        *(cos_theta - h)**2)*np.exp(-c_5*(cos_theta - \
                                                                        h)**2)/(c_3 + (cos_theta - h)**2)**2,
            dexp_r_ij=lambda r_ij, r_ik: (alpha*beta*(r_ij-r_ik)**(beta-1))*np.exp(alpha*(r_ij-r_ik)**beta),
            dexp_r_ik=lambda r_ij, r_ik: -(alpha*beta*(r_ij-r_ik)**(beta-1))*np.exp(alpha*(r_ij-r_ik)**beta),
        )

        def get_cutoff(self):
            return self.cutoff

Kumagai_CompMaterSci_39_457_Si_py_var = {
    "__ref__":  "T. Kumagai, S. Izumi, S. Hara, and S. Sakai, \
            Comput. Mater. Sci. 39, 457 (2007)",
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
