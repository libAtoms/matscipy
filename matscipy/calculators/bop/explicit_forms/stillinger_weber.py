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
    lambda_1 = 21.0           
    gamma = 1.20           







