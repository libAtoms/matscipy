import numpy as np

def ab(x):
    """Compute absolute value (norm) of an array of vectors"""
    return np.linalg.norm(x, axis=1)

def StillingerWeber2():
    """
    Implementation of the Stillinger-Weber potential for silicon.

    Reference
    ------------
    F. Stillinger and T. Weber, Physical review B 31.8 5262 (1985)

    """

    epsilon = 2.1683         
    sigma = 2.0951         
    costheta0 = 0.333333333333
    A = 7.049556277    
    B = 0.6022245584   
    p = 4              
    q = 0              
    a = 1.80           
    lambda1 = 21.0
    gamma = 1.20   
    cutoff = a*sigma

    U2 = lambda r: np.where(r < cutoff, A * epsilon * (B*np.power(sigma/r,p) - 1) * np.exp(sigma/(r-a*sigma)), 0)   
    dU2 = lambda r: np.where(r < cutoff, -A * epsilon * (sigma/np.power(r-a*sigma,2) * (B*np.power(sigma/r, p) - 1) + B*p/r*np.power(sigma/r,p))* np.exp(sigma/(r-a*sigma)), 0)
    ddU2_1 = lambda r: np.where(r < cutoff, A * B * p * epsilon * ((sigma /(r * np.power(r-a*sigma, 2)) + (p + 1)/np.power(r, 2)) * np.power(sigma/r, p)) * np.exp(sigma/(r-a*sigma)), 0)
    ddU2_2 = lambda r: np.where(r < cutoff, A * epsilon * sigma * (2/np.power(r-a*sigma, 3) + sigma/np.power(r-a*sigma, 4)) * np.exp(sigma/(r-a*sigma)), 0)
    ddU2_3 = lambda r: np.where(r < cutoff, A * B * epsilon * sigma * np.power(sigma/r, p) * (p/(r*np.power(r-a*sigma,2)) + 2/np.power(r-a*sigma,3) + sigma/np.power(r-a*sigma, 4)) * np.exp(sigma/(r-a*sigma)), 0)
    ddU2 = lambda r: ddU2_1(r) - ddU2_2(r) + ddU2_3(r)

    """
    ddU2 = lambda r: np.where(r < cutoff, A * B * p * epsilon * ((sigma /(r * np.power(r-a*sigma, 2)) + (p + 1)/np.power(r, 2)) * np.power(sigma/r, p)) * np.exp(sigma/(r-a*sigma)) - \
                     A * epsilon * sigma * (2/np.power(r-a*sigma, 3) + sigma/np.power(r-a*sigma, 4)) * np.exp(sigma/(r-a*sigma)) + \
                     A * B * epsilon * sigma * np.power(sigma/r, p) * (p/(r*np.power(r-a*sigma,2)) + 2/np.power(r-a*sigma,3) + sigma/np.power(r-a*sigma, 4)) * np.exp(sigma/(r-a*sigma)), 0)
    """

    b = lambda xi: xi
    db = lambda xi: xi**0

    g = lambda cost: np.power(cost+costheta0, 2)
    dg = lambda cost: 2 * (cost + costheta0)

    hf = lambda rij, rik: np.where(ab(rik)<cutoff, epsilon * np.exp(gamma*sigma/(ab(rij)-a*sigma)) * np.exp(gamma*sigma/(ab(rik)-a*sigma)), 0) 
    d1h = lambda rij, rik: np.where(ab(rik)<cutoff, - gamma * sigma / np.power(ab(rij)-a*sigma, 2) * hf(rij, rik), 0)
    d2h = lambda rij, rik: np.where(ab(rik)<cutoff, - gamma * sigma / np.power(ab(rik)-a*sigma, 2) * hf(rij, rik), 0)  
    d11h = lambda rij, rik: np.where(ab(rik)<cutoff, -2 * gamma * sigma / np.power(ab(rij)-a*sigma, 3) * hf(rij, rik) - gamma * sigma / np.power(ab(rij)-a*sigma, 2) * d1h(rij, rik), 0)
    d22h = lambda rij, rik: np.where(ab(rik)<cutoff, gamma * sigma / np.power(ab(rik)-a*sigma, 3) * hf(rij, rik) - gamma * sigma / np.power(ab(rik)-a*sigma, 2) * d2h(rij, rik), 0) 
    d12h = lambda rij, rik: np.where(ab(rik)<cutoff, -gamma * sigma / np.power(ab(rij)-a*sigma, 2) * d2h(rij, rik), 0)

    F = lambda r, xi: np.where(r < cutoff, U2(r) + lambda1 * b(xi), 0)
    d1F = lambda r, xi: np.where(r < cutoff, dU2(r), 0) 
    d2F = lambda r, xi: np.where(r < cutoff, lambda1*r**0, 0)
    d11F = lambda r, xi: np.where(r < cutoff, ddU2(r), 0) 
    d12F = lambda r, xi: 0*r
    d22F = lambda r, xi: 0*r

    G = lambda rij, rik: np.where((ab(rij)<cutoff) & (ab(rik)<cutoff), hf(rij, rik) * g(costh(rij, rik)), 0)

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
        (2 * (c1(rij, rik).reshape(-1, 3, 1) * c1(rij, rik).reshape(-1, 1, 3)).T
         + dg(costh(rij, rik)) * dc11(rij, rik).T).T

    d22G = lambda rij, rik: \
        Dg2(rij, rik).reshape(-1, 3, 1) * Dh2(rij, rik).reshape(-1, 1, 3) + Dh2(rij, rik).reshape(-1, 3, 1) * Dg2(rij, rik).reshape(-1, 1, 3) \
        + ((g(costh(rij, rik)) * Dh22(rij, rik).T).T + (hf(rij, rik) * Dg22(rij, rik).T).T)

    Dg22 = lambda rij, rik: \
        (2 * (c2(rij, rik).reshape(-1, 3, 1) * c2(rij, rik).reshape(-1, 1, 3)).T
        + dg(costh(rij, rik)) * dc22(rij, rik).T).T

    Dh22 = lambda rij, rik: \
        (d22h(rij, rik) * (((rik.reshape(-1, 3, 1) * rik.reshape(-1, 1, 3)).T/ab(rik)**2).T).T \
         + d2h(rij, rik) * ((np.eye(3) - ((rik.reshape(-1, 3, 1) * rik.reshape(-1, 1, 3)).T/ab(rik)**2).T).T/ab(rik))).T

    d12G = lambda rij, rik: \
        Dg1(rij, rik).reshape(-1, 3, 1) * Dh2(rij, rik).reshape(-1, 1, 3) + Dh1(rij, rik).reshape(-1, 3, 1) *0* Dg2(rij, rik).reshape(-1, 1, 3) \
        + ((g(costh(rij, rik)) * Dh12(rij, rik).T).T + (hf(rij, rik) * Dg12(rij, rik).T).T)

    Dh12 = lambda rij, rik: \
        (d12h(rij, rik) * (rij.reshape(-1, 3, 1) * rik.reshape(-1, 1, 3)).T/(ab(rij)*ab(rik))).T

    Dg12 = lambda rij, rik: \
         (2 * (c1(rij, rik).reshape(-1, 3, 1) * c2(rij, rik).reshape(-1, 1, 3)).T
         + dg(costh(rij, rik)) * dc12(rij, rik).T).T    

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
        (2 * c1q(rij, rik, q) * c1q(rij, rik, t)
         + dg(costh(rij, rik)) * dc1q1t(rij, rik, q, t))
    Dg2q2t = lambda rij, rik, q, t: \
        (2 * c2q(rij, rik, q) * c2q(rij, rik, t)
         + dg(costh(rij, rik)) * dc2q2t(rij, rik, q, t))
    Dg1q2t = lambda rij, rik, q, t: \
        (2 * c1q(rij, rik, q) * c2q(rij, rik, t)
         + dg(costh(rij, rik)) * dc1q2t(rij, rik, q, t))
    d1q2tG = lambda rij, rik, q, t: \
        Dg1q(rij, rik, q) * Dh2q(rij, rik, t) + Dh1q(rij, rik, q) * Dg2q(rij, rik, t) \
        + g(costh(rij, rik)) * Dh1q2t(rij, rik, q, t) + hf(rij, rik) * Dg1q2t(rij, rik, q, t)

    # Helping functions 
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

    costh = lambda rij, rik: np.sum(rij*rik, axis=1) / (ab(rij)*ab(rik))    
    c1q = lambda rij, rik, q: (rik[:, q]/ab(rik) - rij[:, q]/ab(rij) * costh(rij, rik)) / ab(rij)
    c2q = lambda rij, rik, q: (rij[:, q]/ab(rij) - rik[:, q]/ab(rik) * costh(rij, rik)) / ab(rik)

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
        'cutoff': a*sigma,
    }
