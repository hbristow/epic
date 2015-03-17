import numpy as np


class DiscreteMax(object):
    """Compute the discrete maximum of the data fidelity term

    Finds the globally optimal solution to the following objective
    via exhaustive search:

        argmin  D(dx) + rho/2 || dx - v ||^2_2
           dx
    """

    def __init__(self, unary):
        M,N,P,Q = unary.shape
        self.unary = unary
        self.x, self.y = np.meshgrid(np.arange(Q), np.arange(P))

    def argmin(self, v, rho):
        """Compute the global solution to the proximal problem"""
        M,N,P,Q = self.unary.shape
        x,y = self.x, self.y
        vx,vy = v[0], v[1]
        distance = rho/2.0*((vx[...,None,None]-x)**2 + (vy[...,None,None]-y)**2)
        score = self.unary - distance
        argmax = score.reshape(M,N,-1).argmax(-1)
        vy,vx = np.unravel_index(argmax, (P,Q))
        return np.row_stack((vx[None,...], vy[None,...]))

    def eval(self, x):
        """Evaluate the discrete function at the given points"""
        M,N,P,Q = self.unary.shape
        x,y = x[0], x[1]
        MN  = M*N
        M,N = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')

        confidence = self.unary.flat[ x + y*Q + N*P*Q + M*N*P*Q ]
        return confidence.sum() / MN
