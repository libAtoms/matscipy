import numpy as np

def ab(x):
    """Compute absolute value (norm) of an array of vectors"""
    return np.linalg.norm(x, axis=1)

def StillingerWeber():
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

    fR = lambda r: A * epsilon * (B*np.power(sigma/r, p) - 1) * np.exp(sigma/(r-a*sigma))
    dfR = lambda r: - A*epsilon*B*p/r * np.power(sigma/r, p) * np.exp(sigma/(r-a*sigma)) - sigma/np.power(r-a*sigma, 2)*fR(r)
    ddfR = lambda r: A*B*p*epsilon/r * np.power(sigma/r, p) * np.exp(sigma/(r-a*sigma)) * (sigma/np.power(r-a*sigma, 2) + (p+1)/r) \
                     + 2 * sigma / np.power(r-a*sigma, 3) * fR(r) - sigma / np.power(r-a*sigma, 2) * dfR(r)

    fA = lambda r: np.exp(gamma*sigma/(r-a*sigma))
    dfA = lambda r: - gamma * sigma / np.power(r - a*sigma, 2) * fA(r)
    ddfA = lambda r: 2 * gamma * sigma / np.power(r-a*sigma, 3) * fA(r) - gamma * sigma / np.power(r-a*sigma, 2) * dfA(r)

    hf = lambda rik: np.where(ab(rik)<cutoff, np.exp(gamma*sigma/(ab(rik)-a*sigma)), 0) 
    d2h = lambda rik: -gamma * sigma / np.power(ab(rik)-a*sigma, 2) * hf(rik)
    d22h = lambda rik: 2 * gamma * sigma / np.power(ab(rik)-a*sigma, 3) * hf(rik) - gamma * sigma / np.power(ab(rik)-a*sigma, 2) * d2h(rik)

    g = lambda cost: np.power(cost + costheta0, 2)
    dg = lambda cost: 2 * (cost + costheta0)
    ddg = lambda cost: 2*cost**0

    F = lambda r, xi: np.where(r < cutoff, fR(r) + lambda1 * epsilon * fA(r) * xi , 0)
    d1F = lambda r, xi: np.where(r < cutoff, dfR(r) + lambda1 * epsilon * xi * dfA(r), 0)
    d2F = lambda r, xi: np.where(r < cutoff, lambda1 * epsilon * fA(r), 0)
    d11F = lambda r, xi: np.where(r < cutoff, ddfR(r) + lambda1 * epsilon * xi * ddfA(r), 0)
    d22F = lambda r, xi: np.where(r < cutoff, 0 * r, 0 )
    d12F = lambda r, xi: np.where(r < cutoff, lambda1 * epsilon * dfA(r), 0) 

    G = lambda rij, rik: hf(rik) * g(costh(rij, rik))

    d1G = lambda rij, rik: (hf(rik) * Dg1(rij, rik).T).T 
    d2G = lambda rij, rik: (Dh2(rik).T * g(costh(rij, rik)) + hf(rik) * Dg2(rij, rik).T).T

    Dh2 = lambda rik: (d2h(rik) * rik.T / ab(rik)).T 

    Dg1 = lambda rij, rik: (dg(costh(rij, rik)) * c1(rij, rik).T).T 
    Dg2 = lambda rij, rik: (dg(costh(rij, rik)) * c2(rij, rik).T).T

    d11G = lambda rij, rik: \
        ((hf(rik) * Dg11(rij, rik).T).T)

    Dg11 = lambda rij, rik: \
        (ddg(costh(rij, rik)) * (c1(rij, rik).reshape(-1, 3, 1) * c1(rij, rik).reshape(-1, 1, 3)).T
         + dg(costh(rij, rik)) * dc11(rij, rik).T).T

    d22G = lambda rij, rik: \
        Dg2(rij, rik).reshape(-1, 3, 1) * Dh2(rik).reshape(-1, 1, 3) + Dh2(rik).reshape(-1, 3, 1) * Dg2(rij, rik).reshape(-1, 1, 3) \
        + ((g(costh(rij, rik)) * Dh22(rij, rik).T).T + (hf(rik) * Dg22(rij, rik).T).T)

    Dg22 = lambda rij, rik: \
        (ddg(costh(rij, rik)) * (c2(rij, rik).reshape(-1, 3, 1) * c2(rij, rik).reshape(-1, 1, 3)).T
        + dg(costh(rij, rik)) * dc22(rij, rik).T).T

    Dh22 = lambda rij, rik: \
        (d22h(rik) * (((rik.reshape(-1, 3, 1) * rik.reshape(-1, 1, 3)).T/ab(rik)**2).T).T \
         + d2h(rik) * ((np.eye(3) - ((rik.reshape(-1, 3, 1) * rik.reshape(-1, 1, 3)).T/ab(rik)**2).T).T/ab(rik))).T

    d12G = lambda rij, rik: \
        Dg1(rij, rik).reshape(-1, 3, 1) * Dh2(rik).reshape(-1, 1, 3) + ((hf(rik) * Dg12(rij, rik).T).T)

    Dg12 = lambda rij, rik: \
         (ddg(costh(rij, rik)) * (c1(rij, rik).reshape(-1, 3, 1) * c2(rij, rik).reshape(-1, 1, 3)).T
         + dg(costh(rij, rik)) * dc12(rij, rik).T).T    

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
        'cutoff': a*sigma,
    }
