import numpy as np

def ab(x):
    """Compute absolute value (norm) of an array of vectors"""
    return np.linalg.norm(x, axis=1)

def KumagaiTersoff():
    A = 3281.5905
    B = 121.00047
    lambda_1 = 3.2300135
    lambda_2 = 1.3457970
    eta = 1.0000000
    delta = 0.53298909
    alpha = 2.3890327
    c_1 = 0.20173476
    c_2 = 730418.72
    c_3 = 1000000.0
    c_4 = 1.0000000
    c_5 = 26.000000
    h = -0.36500000
    R_1 = 2.70
    R_2 = 3.30

    f = lambda r: np.where(
            r <= R_1, 1.0,
            np.where(r >= R_2, 0.0,
                     (1/2+(9/16) * np.cos(np.pi*(r - R_1)/(R_2 - R_1))
                      - (1/16) * np.cos(3*np.pi*(r - R_1)/(R_2 - R_1)))
                     )
                           )
    df = lambda r: np.where(
            r >= R_2, 0.0,
            np.where(r <= R_1, 0.0,
                     (3*np.pi*(3*np.sin(np.pi * (R_1 - r) / (R_1 - R_2))
                      - np.sin(3*np.pi*(R_1 - r) / (R_1 - R_2))))/(16*(R_1 - R_2))
                     )
                            )
    ddf = lambda r: np.where(
            r >= R_2, 0.0,
            np.where(r <= R_1, 0.0,
                     ((9*np.pi**2*(np.cos(3*np.pi*(R_1 - r)/(R_1 - R_2))
                       - np.cos(np.pi*(R_1 - r)/(R_1 - R_2))))/(16*(R_1 - R_2)**2))
                     )
                            )

    fR = lambda r:  A*np.exp(-lambda_1 * r)
    dfR = lambda r: -lambda_1 * fR(r)
    ddfR = lambda r: lambda_1**2 * fR(r)

    fA = lambda r: -B*np.exp(-lambda_2 * r)
    dfA = lambda r: -lambda_2 * fA(r)
    ddfA = lambda r: lambda_2**2 * fA(r)

    b = lambda xi: 1/((1+xi**eta)**(delta))
    db = lambda xi: -delta*eta*xi**(eta-1)*(xi**eta+1)**(-delta-1)
    ddb = lambda xi: delta*eta*xi**(eta - 1)*(delta + 1)*(xi**eta + 1)**(-delta - 2)

    F = lambda r, xi: f(r) * (fR(r) + b(xi) * fA(r))
    d1F = lambda r, xi: df(r) * (fR(r) + b(xi) * fA(r)) + f(r) * (dfR(r) + b(xi) * dfA(r))
    d2F = lambda r, xi: f(r) * fA(r) * db(xi)
    d11F = lambda r, xi: f(r) * (ddfR(r) + b(xi) * ddfA(r)) + 2 * df(r) * (dfR(r) + b(xi) * dfA(r)) + ddf(r) * (fR(r) + b(xi) * fA(r))
    d22F = lambda r, xi:  f(r) * fA(r) * ddb(xi)
    d12F = lambda r, xi: f(r) * dfA(r) * db(xi) + fA(r) * df(r) * db(xi)

    g = lambda cost: c_1 + (1 + c_4*np.exp(-c_5*(h-cost)**2)) * \
                           ((c_2*(h-cost)**2)/(c_3 + (h-cost)**2))
    dg = lambda cost: 2*c_2*(cost - h)*(
            (c_3 + (cost - h)**2) *
            (-c_4*c_5*(cost - h)**2 + c_4 +
             np.exp(c_5*(cost - h)**2)) -
            (c_4 + np.exp(c_5*(cost - h)**2))
            * (cost - h)**2) * np.exp(-c_5*(cost - h)**2)/(c_3 + (cost - h)**2)**2
    ddg = lambda cos_theta: \
        (2*c_2*((c_3 + (cos_theta - h)**2)**2
                * (2*c_4*c_5**2*(cos_theta - h)**4
                - 5*c_4*c_5*(cos_theta - h)**2 + c_4
                + np.exp(c_5*(cos_theta - h)**2))
                + (c_3 + (cos_theta - h)**2)*(cos_theta - h)**2
                * (4*c_4*c_5*(cos_theta - h)**2
                - 5*c_4 - 5*np.exp(c_5*(cos_theta - h)**2))
                + 4*(c_4 + np.exp(c_5*(cos_theta - h)**2))*(cos_theta - h)**4)
              * np.exp(-c_5*(cos_theta - h)**2)/(c_3 + (cos_theta - h)**2)**3
         )

    hf = lambda rij, rik: f(ab(rik)) * np.exp(alpha * (ab(rij) - ab(rik)))
    d1h = lambda rij, rik: alpha * hf(rij, rik)
    d2h = lambda rij, rik: \
        - alpha * hf(rij, rik) \
        + df(ab(rik)) * np.exp(alpha * (ab(rij) - ab(rik)))
    d11h = lambda rij, rik: alpha**2 * hf(rij, rik)
    d12h = lambda rij, rik: alpha * d2h(rij, rik)
    d22h = lambda rij, rik: \
         - alpha * ( 2 * df(ab(rik)) * np.exp(alpha * (ab(rij) - ab(rik))) \
         - alpha * hf(rij, rik)) \
         + ddf(ab(rik)) * np.exp(alpha * (ab(rij) - ab(rik)))

    costh = lambda rij, rik: np.sum(rij*rik, axis=1) / (ab(rij)*ab(rik))
    c1q = lambda rij, rik, q: (rik[:, q]/ab(rik) - rij[:, q]/ab(rij) * costh(rij, rik)) / ab(rij)
    c2q = lambda rij, rik, q: (rij[:, q]/ab(rij) - rik[:, q]/ab(rik) * costh(rij, rik)) / ab(rik)
    
    dc1q1t = lambda rij, rik, q, t: \
        (- c1q(rij, rik, q) * rij[:, t] \
         - rij[:, q] * c1q(rij, rik, t) \
         - costh(rij, rik) * (int(q == t) - rij[:, q]*rij[:, t]/ab(rij)**2) \
        )/ab(rij)**2
    dc2q2t = lambda rij, rik, q, t: \
        (- c2q(rij, rik, q) * rik[:, t] \
         - rik[:, q] * c2q(rij, rik, t) \
         - costh(rij, rik) * (int(q == t) - rik[:, q]*rik[:, t]/ab(rik)**2) \
        )/ab(rik)**2
    dc1q2t = lambda rij, rik, q, t: \
        ((int(q == t) - rij[:, q]*rij[:, t]/ab(rij)**2)/ab(rij)
         - c1q(rij, rik, q) * rik[:, t]/ab(rik) \
        )/ab(rik)

    # new
    c1 = lambda rij, rik: ((rik.T/ab(rik) - rij.T/ab(rij) * costh(rij, rik)) / ab(rij)).T
    c2 = lambda rij, rik: ((rij.T/ab(rij) - rik.T/ab(rik) * costh(rij, rik)) / ab(rik)).T

    dc11 = lambda rij, rik: \
        ((- c1(rij, rik).reshape(-1, 3, 1) * rij.reshape(-1, 1, 3) \
         - rij.reshape(-1, 3, 1) * c1(rij, rik).reshape(-1, 1, 3) \
         - (costh(rij, rik) * (np.eye(3) -  ((rij.reshape(-1, 1, 3)*rij.reshape(-1, 3, 1)).T/ab(rij)**2).T).T).T \
        ).T/ab(rij)**2).T
    dc22 = lambda rij, rik: \
        ((- c2(rij, rik).reshape(-1, 3, 1) * rik.reshape(-1, 1, 3) \
         - rik.reshape(-1, 3, 1) * c2(rij, rik).reshape(-1, 1, 3) \
         - (costh(rij, rik) * (np.eye(3) -  ((rik.reshape(-1, 1, 3)*rik.reshape(-1, 3, 1)).T/ab(rik)**2).T).T).T \
        ).T/ab(rik)**2).T
    
    dc12 = lambda rij, rik: \
        (((np.eye(3) -  ((rij.reshape(-1, 1, 3)*rij.reshape(-1, 3, 1)).T/ab(rij)**2).T).T/ab(rij)
         - (c1(rij, rik).reshape(-1, 3, 1) * rik.reshape(-1, 1, 3)).T/ab(rik) \
        )/ab(rik)).T


    Dh1q = lambda rij, rik, q: d1h(rij, rik) * (rij[:, q] / ab(rij))
    Dh2q = lambda rij, rik, q: d2h(rij, rik) * (rik[:, q] / ab(rik))
    
    Dh2 = lambda rij, rik: (d2h(rij, rik) * rik.T / ab(rik)).T
    Dh1 = lambda rij, rik: (d1h(rij, rik) * rij.T / ab(rij)).T

    Dh1q1t = lambda rij, rik, q, t: \
        d11h(rij, rik) * rij[:, q]/ab(rij) * rij[:, t]/ab(rij) \
         + d1h(rij, rik) * (int(q == t) - rij[:, q]/ab(rij) * rij[:, t]/ab(rij))/ab(rij)
    Dh2q2t = lambda rij, rik, q, t: \
        d22h(rij, rik) * rik[:, q]/ab(rik) * rik[:, t]/ab(rik) \
         + d2h(rij, rik) * (int(q == t) - rik[:, q]/ab(rik) * rik[:, t]/ab(rik))/ab(rik)
    Dh1q2t = lambda rij, rik, q, t: \
        d12h(rij, rik) * rij[:, q]/ab(rij) * rik[:, t]/ab(rik)

    Dh11 = lambda rij, rik: \
        (d11h(rij, rik) * (((rij.reshape(-1, 3, 1) * rij.reshape(-1, 1, 3)).T/ab(rij)**2).T).T \
         + d1h(rij, rik) * ((np.eye(3) - ((rij.reshape(-1, 3, 1) * rij.reshape(-1, 1, 3)).T/ab(rij)**2).T).T/ab(rij))).T
    Dh22 = lambda rij, rik: \
        (d22h(rij, rik) * (((rik.reshape(-1, 3, 1) * rik.reshape(-1, 1, 3)).T/ab(rik)**2).T).T \
         + d2h(rij, rik) * ((np.eye(3) - ((rik.reshape(-1, 3, 1) * rik.reshape(-1, 1, 3)).T/ab(rik)**2).T).T/ab(rik))).T
    Dh12 = lambda rij, rik: \
        (d12h(rij, rik) * (rij.reshape(-1, 3, 1) * rik.reshape(-1, 1, 3)).T/(ab(rij)*ab(rik))).T

    # d12G depends on
    Dg1q = lambda rij, rik, q: dg(costh(rij, rik)) * c1q(rij, rik, q)
    Dg2q = lambda rij, rik, q: dg(costh(rij, rik)) * c2q(rij, rik, q)    

    Dg1q1t = lambda rij, rik, q, t: \
        (ddg(costh(rij, rik)) * c1q(rij, rik, q) * c1q(rij, rik, t)
         + dg(costh(rij, rik)) * dc1q1t(rij, rik, q, t))
    Dg2q2t = lambda rij, rik, q, t: \
        (ddg(costh(rij, rik)) * c2q(rij, rik, q) * c2q(rij, rik, t)
         + dg(costh(rij, rik)) * dc2q2t(rij, rik, q, t))
    Dg1q2t = lambda rij, rik, q, t: \
        (ddg(costh(rij, rik)) * c1q(rij, rik, q) * c2q(rij, rik, t)
         + dg(costh(rij, rik)) * dc1q2t(rij, rik, q, t))
    
    ###############

    Dg2 = lambda rij, rik: (dg(costh(rij, rik)) * c2(rij, rik).T).T
    Dg1 = lambda rij, rik: (dg(costh(rij, rik)) * c1(rij, rik).T).T

    Dg11 = lambda rij, rik: \
        (ddg(costh(rij, rik)) * (c1(rij, rik).reshape(-1, 3, 1) * c1(rij, rik).reshape(-1, 1, 3)).T
         + dg(costh(rij, rik)) * dc11(rij, rik).T).T
    
    Dg22 = lambda rij, rik: \
        (ddg(costh(rij, rik)) * (c2(rij, rik).reshape(-1, 3, 1) * c2(rij, rik).reshape(-1, 1, 3)).T
         + dg(costh(rij, rik)) * dc22(rij, rik).T).T
    Dg12 = lambda rij, rik: \
        (ddg(costh(rij, rik)) * (c1(rij, rik).reshape(-1, 3, 1) * c2(rij, rik).reshape(-1, 1, 3)).T
         + dg(costh(rij, rik)) * dc12(rij, rik).T).T    
    
    G = lambda rij, rik: g(costh(rij, rik)) * hf(rij, rik)
    
    d1G = lambda rij, rik: (Dh1(rij, rik).T * g(costh(rij, rik)) + hf(rij, rik) * Dg1(rij, rik).T).T
    d2G = lambda rij, rik: (Dh2(rij, rik).T * g(costh(rij, rik)) + hf(rij, rik) * Dg2(rij, rik).T).T

    # TODO
    d1q2tG = lambda rij, rik, q, t: \
        Dg1q(rij, rik, q) * Dh2q(rij, rik, t) + Dh1q(rij, rik, q) * Dg2q(rij, rik, t) \
        + g(costh(rij, rik)) * Dh1q2t(rij, rik, q, t) + hf(rij, rik) * Dg1q2t(rij, rik, q, t)

    d11G = lambda rij, rik: \
        Dg1(rij, rik).reshape(-1, 3, 1) * Dh1(rij, rik).reshape(-1, 1, 3) + Dh1(rij, rik).reshape(-1, 3, 1) * Dg1(rij, rik).reshape(-1, 1, 3) \
        + ((g(costh(rij, rik)) * Dh11(rij, rik).T).T + (hf(rij, rik) * Dg11(rij, rik).T).T)
    d22G = lambda rij, rik: \
        Dg2(rij, rik).reshape(-1, 3, 1) * Dh2(rij, rik).reshape(-1, 1, 3) + Dh2(rij, rik).reshape(-1, 3, 1) * Dg2(rij, rik).reshape(-1, 1, 3) \
        + ((g(costh(rij, rik)) * Dh22(rij, rik).T).T + (hf(rij, rik) * Dg22(rij, rik).T).T)
    d12G = lambda rij, rik: \
        Dg1(rij, rik).reshape(-1, 3, 1) * Dh2(rij, rik).reshape(-1, 1, 3) + Dh1(rij, rik).reshape(-1, 3, 1) *0* Dg2(rij, rik).reshape(-1, 1, 3) \
        + ((g(costh(rij, rik)) * Dh12(rij, rik).T).T + (hf(rij, rik) * Dg12(rij, rik).T).T)

    return {
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
        'd1x2xG': lambda rij, rik: d1q2tG(rij, rik, 0, 0),
        'd1y2yG': lambda rij, rik: d1q2tG(rij, rik, 1, 1),
        'd1z2zG': lambda rij, rik: d1q2tG(rij, rik, 2, 2),
        'd1y2zG': lambda rij, rik: d1q2tG(rij, rik, 1, 2),
        'd1x2zG': lambda rij, rik: d1q2tG(rij, rik, 0, 2),
        'd1x2yG': lambda rij, rik: d1q2tG(rij, rik, 0, 1),
        'd1z2yG': lambda rij, rik: d1q2tG(rij, rik, 2, 1),
        'd1z2xG': lambda rij, rik: d1q2tG(rij, rik, 2, 0),
        'd1y2xG': lambda rij, rik: d1q2tG(rij, rik, 1, 0),
        'cutoff': R_2,
    }
