#!/usr/bin/python
from numpy import *
from math import gamma

def mymvtdist(x, df, mu, Sigma):
    x = atleast_2d(x) # requires x as 2d
    nD = Sigma.shape[0] # dimensionality

    numerator = gamma(1.0 * (nD + df) / 2.0)

    denominator = (
            gamma(1.0 * df / 2.0) * 
            power(df * pi, 1.0 * nD / 2.0) *  
            power(linalg.det(Sigma), 1.0 / 2.0) * 
            power(
                1.0 + (1.0 / df) *
                diagonal(
                    dot( dot(x - mu, linalg.inv(Sigma)), (x - mu).T)
                ), 
                1.0 * (nD + df) / 2.0
                )
            )

    return 1.0 * numerator / denominator