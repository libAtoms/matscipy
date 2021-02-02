import numpy as np


def ab(x):
    """Compute absolute value (norm) of an array of vectors"""
    return np.linalg.norm(x, axis=1)

def TersoffIII():
    """
    Implementation of the T3 potential for silicon.

    Reference
    ------------
    J. Tersoff, Physical review B 39.8 (1989): 5566.

    """

    A = 1.8308e3
    B = 4.7118e2
    chi = 1.0
    lam = 2.4799e0
    mu = 1.7322e0
    beta = 1.1000e-6
    n = 7.8734e-1
    c = 1.0039e5
    d = 1.6217e1
    h = -5.9825e-1
    R_1 = 2.70
    R_2 = 3.00
    #lam3 = 5.19745
    lam3 = 0.0
    delta = 3

    f = lambda r: np.where(
        r < R_1,
        np.ones_like(r),
        np.where(r > R_2,
            np.zeros_like(r),
            (1 + np.cos((np.pi*(r-R_1)/(R_2-R_1))))/2
            )
        )
    df = lambda r: np.where(
        r < R_1,
        np.zeros_like(r),
        np.where(r > R_2,
            np.zeros_like(r),
            -np.pi * np.sin(np.pi * (r - R_1)/(R_2 - R_1)) / (2*(R_2 - R_1))
            )
        )
    ddf = lambda r: np.where(
        r < R_1,
        np.zeros_like(r),
        np.where(r > R_2,
            np.zeros_like(r),
            -np.pi**2 * np.cos(np.pi * (r - R_1)/(R_2 - R_1)) / (2*(R_2 - R_1)**2)
            )
        )

    fR = lambda r: A * np.exp(-lam * r)
    dfR = lambda r: -lam * fR(r)
    ddfR = lambda r: lam**2 * fR(r)

    fA = lambda r: -B * np.exp(-mu * r)
    dfA = lambda r: -mu * fA(r)
    ddfA = lambda r: mu**2 * fA(r)

    b = lambda xi: (1 + (beta * xi)**n)**(-1 / (2 * n))
    db = lambda xi: -0.5 * beta * (beta * xi)**(-1 + n) * (1 + (beta * xi)**n)**(-1-1/(2*n))
    ddb = lambda xi: -0.5 * beta**2 * (n-1) * (beta * xi)**(-2 + n) * (1 + (beta * xi)**n)**(-1-1/(2*n)) - \
        0.5 * beta**2 * n * (beta * xi)**(-2 + 2*n) * ( -1 - 1/(2*n)) * (1 + (beta * xi)**n)**(-2-1/(2*n))

    g = lambda cost: 1 + c**2 / d**2 - c**2 / (d**2 + (h - cost)**2)
    dg = lambda cost: -2 * c**2 * (h - cost) / (d**2 + (h - cost)**2)**2
    ddg = lambda cost: 2 * c**2 / (d**2 + (h - cost)**2)**2 - 8 * c**2 * (h - cost)**2 / (d**2 + (h - cost)**2)**3        

    hf = lambda rij, rik: f(ab(rik)) * np.exp(lam3*(ab(rij)-ab(rik))**delta)
    d1h = lambda rij, rik: lam3 * hf(rij, rik)
    d2h = lambda rij, rik: -lam3 * hf(rij, rik) + \
        df(ab(rik)) * np.exp(lam3*(ab(rij)-ab(rik))**delta)
    d11h = lambda rij, rik: lam3**2*hf(rij, rik)
    d12h = lambda rij, rik: (df(ab(rik))*(lam3*delta)
                             * np.exp(lam3*(ab(rij)-ab(rik))**delta)
                             - lam3 * hf(rij, rik))
    d22h = lambda rij, rik: \
        (ddf(ab(rik))*np.exp(lam3*(ab(rij)-ab(rik))**delta)
         + 2 * (lam3*delta)*np.exp(lam3*(ab(rij)-ab(rik))**delta)*df(ab(rik))
         + lam3**2 * hf(rij, rik))


    F = lambda r, xi: f(r) * (fR(r) + b(xi) * fA(r))
    d1F = lambda r, xi: df(r) * (fR(r) + b(xi) * fA(r)) + f(r) * (dfR(r) + b(xi) * dfA(r))
    d2F = lambda r, xi: f(r) * fA(r) * db(xi)
    d11F = lambda r, xi: f(r) * (ddfR(r) + b(xi) * ddfA(r)) + 2 * df(r) * (dfR(r) + b(xi) * dfA(r)) + ddf(r) * (fR(r) + b(xi) * fA(r))
    d22F = lambda r, xi:  f(r) * fA(r) * ddb(xi)
    d12F = lambda r, xi: f(r) * dfA(r) * db(xi) + fA(r) * df(r) * db(xi)


    G = lambda rij, rik: g(costh(rij, rik)) * hf(rij, rik)
    
    d1G = lambda rij, rik: (Dh1(rij, rik).T * g(costh(rij, rik)) + hf(rij, rik) * Dg1(rij, rik).T).T
    d2G = lambda rij, rik: (Dh2(rij, rik).T * g(costh(rij, rik)) + hf(rij, rik) * Dg2(rij, rik).T).T

    Dh1 = lambda rij, rik: (d1h(rij, rik) * rij.T / ab(rij)).T
    Dh2 = lambda rij, rik: (d2h(rij, rik) * rik.T / ab(rik)).T

    Dg1 = lambda rij, rik: (dg(costh(rij, rik)) * c1(rij, rik).T).T
    Dg2 = lambda rij, rik: (dg(costh(rij, rik)) * c2(rij, rik).T).T

   
    d11G = lambda rij, rik: \
        Dg1(rij, rik).reshape(-1, 3, 1) * Dh1(rij, rik).reshape(-1, 1, 3) + Dh1(rij, rik).reshape(-1, 3, 1) * Dg1(rij, rik).reshape(-1, 1, 3) \
        + ((g(costh(rij, rik)) * Dh11(rij, rik).T).T + (hf(rij, rik) * Dg11(rij, rik).T).T)

    Dh11 = lambda rij, rik: \
        (d11h(rij, rik) * (((rij.reshape(-1, 3, 1) * rij.reshape(-1, 1, 3)).T/ab(rij)**2).T).T \
         + d1h(rij, rik) * ((np.eye(3) - ((rij.reshape(-1, 3, 1) * rij.reshape(-1, 1, 3)).T/ab(rij)**2).T).T/ab(rij))).T

    Dg11 = lambda rij, rik: \
        (ddg(costh(rij, rik)) * (c1(rij, rik).reshape(-1, 3, 1) * c1(rij, rik).reshape(-1, 1, 3)).T
         + dg(costh(rij, rik)) * dc11(rij, rik).T).T


    d22G = lambda rij, rik: \
        Dg2(rij, rik).reshape(-1, 3, 1) * Dh2(rij, rik).reshape(-1, 1, 3) + Dh2(rij, rik).reshape(-1, 3, 1) * Dg2(rij, rik).reshape(-1, 1, 3) \
        + ((g(costh(rij, rik)) * Dh22(rij, rik).T).T + (hf(rij, rik) * Dg22(rij, rik).T).T)       

    Dh22 = lambda rij, rik: \
        (d22h(rij, rik) * (((rik.reshape(-1, 3, 1) * rik.reshape(-1, 1, 3)).T/ab(rik)**2).T).T \
         + d2h(rij, rik) * ((np.eye(3) - ((rik.reshape(-1, 3, 1) * rik.reshape(-1, 1, 3)).T/ab(rik)**2).T).T/ab(rik))).T

    Dg22 = lambda rij, rik: \
        (ddg(costh(rij, rik)) * (c2(rij, rik).reshape(-1, 3, 1) * c2(rij, rik).reshape(-1, 1, 3)).T
         + dg(costh(rij, rik)) * dc22(rij, rik).T).T


    d12G = lambda rij, rik: \
        Dg1(rij, rik).reshape(-1, 3, 1) * Dh2(rij, rik).reshape(-1, 1, 3) + Dh1(rij, rik).reshape(-1, 3, 1) * Dg2(rij, rik).reshape(-1, 1, 3) \
        + ((g(costh(rij, rik)) * Dh12(rij, rik).T).T + (hf(rij, rik) * Dg12(rij, rik).T).T)

    Dh12 = lambda rij, rik: \
        (d12h(rij, rik) * (rij.reshape(-1, 3, 1) * rik.reshape(-1, 1, 3)).T/(ab(rij)*ab(rik))).T
   
    Dg12 = lambda rij, rik: \
        (ddg(costh(rij, rik)) * (c1(rij, rik).reshape(-1, 3, 1) * c2(rij, rik).reshape(-1, 1, 3)).T
         + dg(costh(rij, rik)) * dc12(rij, rik).T).T


    # 
    costh = lambda rij, rik: np.sum(rij*rik, axis=1) / (ab(rij)*ab(rik))

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
        'cutoff': R_2,
    }
