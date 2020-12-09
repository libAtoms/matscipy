import numpy as np

def TersoffIII():
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
    r1 = 2.70
    r2 = 3.00
    #lam3 = 5.19745
    lam3 = 0.0
    delta = 3

    f = lambda r: np.where(
        r < r1,
        np.ones_like(r),
        np.where(r > r2,
            np.zeros_like(r),
            (1 + np.cos((np.pi*(r-r1)/(r2-r1))))/2
            )
        )
    df = lambda r: np.where(
        r < r1,
        np.zeros_like(r),
        np.where(r > r2,
            np.zeros_like(r),
            -np.pi * np.sin(np.pi * (r - r1)/(r2 - r1)) / (2*(r2 - r1))
            )
        )
    ddf = lambda r: np.where(
        r < r1,
        np.zeros_like(r),
        np.where(r > r2,
            np.zeros_like(r),
            -np.pi**2 * np.cos(np.pi * (r - r1)/(r2 - r1)) / (2*(r2 - r1)**2)
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

    F = lambda r, xi: f(r) * (fR(r) + b(xi) * fA(r))
    d1F = lambda r, xi: df(r) * (fR(r) + b(xi) * fA(r)) + f(r) * (dfR(r) + b(xi) * dfA(r))
    d2F = lambda r, xi: f(r) * fA(r) * db(xi)
    d11F = lambda r, xi: f(r) * (ddfR(r) + b(xi) * ddfA(r)) + 2 * df(r) * (dfR(r) + b(xi) * dfA(r)) + ddf(r) * (fR(r) + b(xi) * fA(r))
    d22F = lambda r, xi:  f(r) * fA(r) * ddb(xi)
    d12F = lambda r, xi: f(r) * dfA(r) * db(xi) + fA(r) * df(r) * db(xi)

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

    Dh1q = lambda rij, rik, q: d1h(rij, rik) * (rij[:, q] / ab(rij))
    Dh2q = lambda rij, rik, q: d2h(rij, rik) * (rik[:, q] / ab(rik))

    Dh1q1t = lambda rij, rik, q, t: \
        d11h(rij, rik) * rij[:, q]/ab(rij) * rij[:, t]/ab(rij) \
         + d1h(rij, rik) * (int(q == t) - rij[:, q]/ab(rij) * rij[:, t]/ab(rij))/ab(rij)
    Dh2q2t = lambda rij, rik, q, t: \
        d22h(rij, rik) * rik[:, q]/ab(rik) * rik[:, t]/ab(rik) \
         + d2h(rij, rik) * (int(q == t) - rik[:, q]/ab(rik) * rik[:, t]/ab(rik))/ab(rik)
    Dh1q2t = lambda rij, rik, q, t: \
        d12h(rij, rik) * rij[:, q]/ab(rij) * rik[:, t]/ab(rik)

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

    G = lambda rij, rik: g(costh(rij, rik)) * hf(rij, rik)
    d1qG = lambda rij, rik, q: Dh1q(rij, rik, q) * g(costh(rij, rik)) + hf(rij, rik) * Dg1q(rij, rik, q)
    d2qG = lambda rij, rik, q: Dh2q(rij, rik, q) * g(costh(rij, rik)) + hf(rij, rik) * Dg2q(rij, rik, q)

    d1q1tG = lambda rij, rik, q, t: \
        Dg1q(rij, rik, q) * Dh1q(rij, rik, t) + Dg1q(rij, rik, t) * Dh1q(rij, rik, q) \
        + g(costh(rij, rik)) * Dh1q1t(rij, rik, q, t) + hf(rij, rik) * Dg1q1t(rij, rik, q, t)
    d2q2tG = lambda rij, rik, q, t: \
        Dg2q(rij, rik, q) * Dh2q(rij, rik, t) + Dg2q(rij, rik, t) * Dh2q(rij, rik, q) \
        + g(costh(rij, rik)) * Dh2q2t(rij, rik, q, t) + hf(rij, rik) * Dg2q2t(rij, rik, q, t)
    d1q2tG = lambda rij, rik, q, t: \
        Dg1q(rij, rik, q) * Dh2q(rij, rik, t) + Dg2q(rij, rik, t) * Dh1q(rij, rik, q) \
        + g(costh(rij, rik)) * Dh1q2t(rij, rik, q, t) + hf(rij, rik) * Dg1q2t(rij, rik, q, t)

    return {
        'F': F,
        'G': G,
        'd1F': d1F,
        'd2F': d2F,
        'd11F': d11F,
        'd12F': d12F,
        'd22F': d22F,
        'd1xG': lambda rij, rik: d1qG(rij, rik, 0),
        'd1yG': lambda rij, rik: d1qG(rij, rik, 1),
        'd1zG': lambda rij, rik: d1qG(rij, rik, 2),
        'd2xG': lambda rij, rik: d2qG(rij, rik, 0),
        'd2yG': lambda rij, rik: d2qG(rij, rik, 1),
        'd2zG': lambda rij, rik: d2qG(rij, rik, 2),
        'd1x1xG': lambda rij, rik: d1q1tG(rij, rik, 0, 0),
        'd1y1yG': lambda rij, rik: d1q1tG(rij, rik, 1, 1),
        'd1z1zG': lambda rij, rik: d1q1tG(rij, rik, 2, 2),
        'd1y1zG': lambda rij, rik: d1q1tG(rij, rik, 1, 2),
        'd1x1zG': lambda rij, rik: d1q1tG(rij, rik, 0, 2),
        'd1x1yG': lambda rij, rik: d1q1tG(rij, rik, 0, 1),
        'd2x2xG': lambda rij, rik: d2q2tG(rij, rik, 0, 0),
        'd2y2yG': lambda rij, rik: d2q2tG(rij, rik, 1, 1),
        'd2z2zG': lambda rij, rik: d2q2tG(rij, rik, 2, 2),
        'd2y2zG': lambda rij, rik: d2q2tG(rij, rik, 1, 2),
        'd2x2zG': lambda rij, rik: d2q2tG(rij, rik, 0, 2),
        'd2x2yG': lambda rij, rik: d2q2tG(rij, rik, 0, 1),
        'd1x2xG': lambda rij, rik: d1q2tG(rij, rik, 0, 0),
        'd1y2yG': lambda rij, rik: d1q2tG(rij, rik, 1, 1),
        'd1z2zG': lambda rij, rik: d1q2tG(rij, rik, 2, 2),
        'd1y2zG': lambda rij, rik: d1q2tG(rij, rik, 1, 2),
        'd1x2zG': lambda rij, rik: d1q2tG(rij, rik, 0, 2),
        'd1x2yG': lambda rij, rik: d1q2tG(rij, rik, 0, 1),
        'd1z2yG': lambda rij, rik: d1q2tG(rij, rik, 2, 1),
        'd1z2xG': lambda rij, rik: d1q2tG(rij, rik, 2, 0),
        'd1y2xG': lambda rij, rik: d1q2tG(rij, rik, 1, 0),
        'cutoff': r2
    }
