#!/usr/bin/python
import numpy as np
import scipy.special as sc

def mymvtdist(x, mu, df, Sigma):
    x = np.atleast_2d(x) # requires x as 2d
    nD = Sigma.shape[0] # dimensionality

    numerator = sc.gamma(1.0 * (nD + df) / 2.0)

    denominator = (
            np.exp(sc.gammaln(1.0 * df / 2.0)) * 
            np.power(df * np.pi, 1.0 * nD / 2.0) *  
            np.power(np.linalg.det(Sigma), 1.0 / 2.0) * 
            np.power(
                1.0 + (1.0 / df) *
                np.diagonal(
                    np.dot( np.dot(x - mu, np.linalg.inv(Sigma)), (x - mu).T)
                ), 
                1.0 * (nD + df) / 2.0
                )
            )

    return 1.0 * numerator / denominator