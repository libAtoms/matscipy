import numpy as np
from collections import namedtuple 

SW_parameters  = namedtuple("SW_parameters", ["epsilon", "sigma", "costheta0", "A", "B", "p", "a", "lambda1", "gamma"])

# Original parametrization: F. Stillinger and T. Weber, Physical review B 31.8 5262 (1985)
original_SW = SW_parameters(epsilon=2.1683, sigma=2.0951, costheta0=0.333333333333, A=7.049556277, B=0.6022245584,
                            p=4, a=1.80, lambda1=21.0, gamma=1.20)

# D. Holland and M. Marder Physical Review Letters 80.4 (1998): 746.
brittle_fracture_SW = SW_parameters(epsilon=2.1683, sigma=2.0951, costheta0=0.333333333333, A=7.049556277, B=0.6022245584,
                            p=4, a=1.80, lambda1=42.0, gamma=1.20)

# RLC Vink et al. Journal of non-crystalline solids 282.2-3 (2001): 248-255.
aSi_SW = SW_parameters(epsilon=1.64833, sigma=2.0951, costheta0=0.333333333333, A=7.049556277, B=0.6022245584,
                            p=4, a=1.80, lambda1=31.5, gamma=1.20)

# J. Russo et. al. Physical Review X 8.2 (2018): 021040.
# D. Richard et al. Physical Review Letters 125.8 (2020): 085502.
glassforming_SW = SW_parameters(epsilon=1.64833, sigma=2.0951, costheta0=0.333333333333, A=7.049556277, B=0.6022245584,
                            p=4, a=1.80, lambda1=18.75, gamma=1.20)

def ab(x):
    """Compute absolute value (norm) of an array of vectors"""
    return np.linalg.norm(x, axis=1)

def StillingerWeber(parameters=original_SW):
    """
    Implementation of the functional form of the Stillinger-Weber potential.

    Reference
    ------------
    F. Stillinger and T. Weber, Physical review B 31.8 5262 (1985)

    """
    try:
        epsilon = parameters.epsilon
        sigma = parameters.sigma
        costheta0 = parameters.costheta0
        A = parameters.A
        B = parameters.B
        p = parameters.p
        a = parameters.a
        lambda1 = parameters.lambda1
        gamma = parameters.gamma
    except AttributeError:
        raise AttributeError("One or some necessary parameters are missing!")

    fR = lambda r: A * epsilon * (B*np.power(sigma/r, p) - 1) * np.exp(sigma/(r-a*sigma))
    dfR = lambda r: - A*epsilon*B*p/r * np.power(sigma/r, p) * np.exp(sigma/(r-a*sigma)) - sigma/np.power(r-a*sigma, 2)*fR(r)
    ddfR = lambda r: A*B*p*epsilon/r * np.power(sigma/r, p) * np.exp(sigma/(r-a*sigma)) * (sigma/np.power(r-a*sigma, 2) + (p+1)/r) \
                     + 2 * sigma / np.power(r-a*sigma, 3) * fR(r) - sigma / np.power(r-a*sigma, 2) * dfR(r)

    fA = lambda r: np.exp(gamma*sigma/(r-a*sigma))
    dfA = lambda r: - gamma * sigma / np.power(r - a*sigma, 2) * fA(r)
    ddfA = lambda r: 2 * gamma * sigma / np.power(r-a*sigma, 3) * fA(r) - gamma * sigma / np.power(r-a*sigma, 2) * dfA(r)

    hf = lambda rik: np.where(ab(rik)<a*sigma, np.exp(gamma*sigma/(ab(rik)-a*sigma)), 0) 
    d2h = lambda rik: -gamma * sigma / np.power(ab(rik)-a*sigma, 2) * hf(rik)
    d22h = lambda rik: 2 * gamma * sigma / np.power(ab(rik)-a*sigma, 3) * hf(rik) - gamma * sigma / np.power(ab(rik)-a*sigma, 2) * d2h(rik)

    g = lambda cost: np.power(cost + costheta0, 2)
    dg = lambda cost: 2 * (cost + costheta0)
    ddg = lambda cost: 2*cost**0

    """
    F = lambda r, xi: np.where(r < a*sigma, fR(r) + lambda1 * epsilon * fA(r) * xi , 0)
    d1F = lambda r, xi: np.where(r < a*sigma, dfR(r) + lambda1 * epsilon * xi * dfA(r), 0)
    d2F = lambda r, xi: np.where(r < a*sigma, lambda1 * epsilon * fA(r), 0)
    d11F = lambda r, xi: np.where(r < a*sigma, ddfR(r) + lambda1 * epsilon * xi * ddfA(r), 0)
    d22F = lambda r, xi: np.zeros_like(r)
    d12F = lambda r, xi: np.where(r < a*sigma, lambda1 * epsilon * dfA(r), 0) 
    """
    def F(r, xi):
        mask = (r < a*sigma)
        F_n = np.zeros_like(r)
        F_n[mask] = fR(r[mask]) + lambda1 * epsilon * fA(r[mask]) * xi[mask]
        return F_n
    def d1F(r, xi):
        mask = (r < a*sigma)
        d1F_n = np.zeros_like(r)
        d1F_n[mask] = dfR(r[mask]) + lambda1 * epsilon * xi[mask] * dfA(r[mask])
        return d1F_n
    def d2F(r, xi):
        mask = (r < a*sigma)
        d2F_n = np.zeros_like(r)
        d2F_n[mask] = lambda1 * epsilon * fA(r[mask])     
        return d2F_n
    def d11F(r, xi):
        mask = (r < a*sigma)
        d11F_n = np.zeros_like(r)
        d11F_n[mask] = ddfR(r[mask]) + lambda1 * epsilon * xi[mask] * ddfA(r[mask])   
        return d11F_n
    def d22F(r, xi):
        return np.zeros_like(r)
    def d12F(r, xi):
        mask = (r < a*sigma)
        d12F_n = np.zeros_like(r)
        d12F_n[mask] = lambda1 * epsilon * dfA(r[mask])
        return d12F_n

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
