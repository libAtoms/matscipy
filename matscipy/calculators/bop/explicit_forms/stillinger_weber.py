import numpy as np

def ab(x):
    """Compute absolute value (norm) of an array of vectors"""
    return np.linalg.norm(x, axis=1)

def StillingerWeber():
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
