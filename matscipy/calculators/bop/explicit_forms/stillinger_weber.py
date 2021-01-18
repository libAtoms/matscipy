import numpy as np

def ab(x):
    """Compute absolute value (norm) of an array of vectors"""
    return np.linalg.norm(x, axis=1)

def StillingerWeber():
    """

    Reference
    ------------
    F. Stillinger and T. Weber, Physical review B 31.8 5262 (1985)

    """
    epsilon = 2.1683         
    sigma = 2.0951         
    costheta0 = -0.333333333333
    A = 7.049556277    
    B = 0.6022245584   
    p = 4              
    q = 0              
    a = 1.80           
    lambda1 = 21.0           
    gamma = 1.20   

    f = lambda r: np.where(r <= a*sigma, 1.0, 0)
    df = lambda r: 0 * r

    F = lambda r, xi: f(r) * (U2(r) + lambda1 * b(xi))
    d1F = lambda r, xi: f(r) * dU2(r) + df(r) * (U2(r) + lambda*b(xi))
    d2F = lambda r, xi: lambda * f(r) * 1

    G = lambda rij, rik: hf(rij, rik) * g(costh(rij, rik))
    d1G = lambda rij, rik: (Dh1(rij, rik).T * g(costh(rij, rik)) + hf(rij, rik) * Dg1(rij, rik).T).T 
    d2G = lambda rij, rik: (Dh2(rij, rik).T * g(costh(rij, rik)) + hf(rij, rik) * Dg2(rij, rik).T).T

    Dh1 = lambda rij, rik: (d1h(rij, rik) * rij.T / ab(rij)).T
    Dh2 = lambda rij, rik: (d2h(rij, rik) * rik.T / ab(rik)).T 

    Dg1 = lambda rij, rik: (dg(costh(rij, rik)) * c1(rij, rik).T).T 
    Dg2 = lambda rij, rik: (dg(costh(rij, rik)) * c2(rij, rik).T).T

    U2 = lambda r: A * epsilon * (B*np.power(sigma/r,p) - np.power(sigma/r,q)) * np.exp(gamma*sigma/(r-a*sigma))   
    dU2 = lambda r: -A * epsilon * (sigma/np.power(r-a*sigma,2) * (B*np.power(sigma/r, p) - 1) + B*p/r*np.power(sigma/r, p))* np.exp(sigma/(r-a*sigma))

    b = lambda xi: xi
    db = lambda xi: xi**0

    g = lambda cost: np.power(cost+costheta0, 2)
    dg = lambda cost: 2 * (cost + costheta0)

    hf = lambda rij, rik: epsilon * f(ab(rik)) * np.exp(gamma*sigma/(ab(rij)-a*sigma) * np.exp(gamma*sigma/(ab(rik)-a*sigma))) 
    d1h = lambda rij, rik: - gamma * sigma / np.power(ab(rij)-a*sigma, 2) * hf(rij, rik)
    d2h = lambda rij, rik: - gamma * sigma / np.power(ab(rik)-a*sigma, 2) * hf(rij, rik) + epsilon*np.exp(gamma*sigma/(ab(rij) - a*sigma) * np.exp(gamm*sigma/(ab(rik)-a*sigma)))*df(ab(rik))


    # Helping functions 
    c1 = lambda rij, rik: ((rik.T/ab(rik) - rij.T/ab(rij) * costh(rij, rik)) / ab(rij)).T
    c2 = lambda rij, rik: ((rij.T/ab(rij) - rik.T/ab(rik) * costh(rij, rik)) / ab(rik)).T

    costh = lambda rij, rik: np.sum(rij*rik, axis=1) / (ab(rij)*ab(rik))    

    return {
        'F': F,
        'G': G,
        'd1F': d1F,
        'd2F': d2F,
        'd1G': d1G,
        'd2G': d2G,
        'cutoff': a*sigma,
    }
