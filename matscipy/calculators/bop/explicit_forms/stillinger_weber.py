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

    F = lambda r, xi: f(r) * (U2(r) + lambda1 * b(xi))

    G = lambda rij, rik: f(ab(rik)) * U3(rij,rik)
  

    U2 = lambda r: A * epsilon * (B*np.power(sigma/r,p) - np.power(sigma/r,q)) * np.exp(gamma*sigma/(r-a*sigma))   

    U3 = lambda rij, rik: epsilon*(costh(rij,rik)-costheta0)**2 * np.exp(gamma*sigma/(ab(rij)-a*sigma)) * np.exp(gamma*sigma/(ab(rik)-a*sigma))

    b = lambda xi: xi

    costh = lambda rij, rik: np.sum(rij*rik, axis=1) / (ab(rij)*ab(rik))    

    f = lambda r: np.where(r <= a*sigma, 1.0, 0)
    df = lambda r: 0 * r

    return {
        'F': F,
        'G': G,
        'cutoff': a*sigma,
    }
