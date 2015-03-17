from __future__ import division
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import numpy as np
import itertools
import time
import sys


# ----------------------------------------------------------------------------
# Meanshift Helpers
# ----------------------------------------------------------------------------
def infnorm(x):
    return np.abs(x).max()

def no_op(x,y):
    return x,y


# ----------------------------------------------------------------------------
# Meanshift
# ----------------------------------------------------------------------------
def meanshift(unary, alpha=0.1, sigma=0.2, max_iters=100, eps=1e-3):
    """Perform mean-shift over probabilistic unary, with Gaussian kernel

    Args:
        unary: The data-term of shape M,N,P,Q

    Keyword Args:
        sigma: The Gaussian variance
        max_iters: The maximum number of allowable iterates
        eps: The flow tolerance convergence criteria
    """

    # get the unary size
    unary = np.array(unary, copy=False, ndmin=4)
    unary = np.exp(unary/alpha)
    M,N,P,Q = unary.shape
    axis  = (-1,-2)

    # initialize the flow vectors to be spread across the second image
    x,y   = np.meshgrid(np.arange(Q), np.arange(P))
    ux,uy = np.linspace(0,Q-1,N), np.linspace(0,P-1,M)
    ux,uy = np.meshgrid(ux,uy)
    ux0,uy0 = ux,uy

    # allocate the initial flow
    fx,fy = np.zeros((M,N)), np.zeros((M,N))

    # iterate until convergence
    fx0,fy0,i = np.inf, np.inf, 0
    while i < max_iters and (infnorm(fx-fx0) > eps or infnorm(fy-fy0) > eps):

        # compute the kernel
        gaussian = np.exp( -np.sqrt((ux[...,None,None]-x)**2 + (uy[...,None,None]-y)**2)/sigma )
        kernel   = unary*gaussian
        norm     = kernel.sum(axis=axis)

        # compute the meanshift update
        ux = (x*kernel).sum(axis=axis) / norm
        uy = (y*kernel).sum(axis=axis) / norm

        # compute the flow estimate
        fx,fx0 = ux-ux0,fx
        fy,fy0 = uy-uy0,fy
        i += 1

    # return the flow
    return fx,fy,kernel


def discreteshift(unary, regularizer=no_op, wsize=51, alpha=0.1, sigma=0.2, max_iters=100, eps=1.0):

    # get the window size
    unary = np.array(unary, copy=False, ndmin=4)
    M,N,Mw,Nw = unary.shape
    Mwh,Nwh = np.floor((Mw/2, Nw/2))
    axis = (-2,-1)

    # initialize the points to be spread across the image
    x,y = np.meshgrid(np.arange(Nw), np.arange(Mw))
    x0,y0 = np.meshgrid(np.linspace(0,Nw-1,N), np.linspace(0,Mw-1,M))
    fx,fy = np.zeros((M,N)), np.zeros((M,N))

    # precompute the unary and distance kernels
    ku = np.exp(-unary/alpha)
    kx,ky = np.meshgrid(np.linspace(-1.0,1.0,wsize), np.linspace(-1.0,1.0,wsize))
    kd = np.exp(-np.sqrt(kx**2 + ky**2)/sigma)

    # precompute the indexing values
    Mk = np.floor(kd.shape[0]/2)
    Nk = np.floor(kd.shape[1]/2)

    # debug
    d = np.empty(unary.shape)

    # iterate until convergence
    for iters in xrange(max_iters):

        d.fill(0.0)
        maxflow = 0.0
        absflow = 0.0
        meanflow = 0.0
        for m in range(M):
            for n in range(N):

                # compute the nearest grid centroid
                xmn,ymn = round(x0[m,n]+fx[m,n]), round(y0[m,n]+fy[m,n])
                yn = min(ymn,Mk)
                yp = min(Mw-ymn,Mk+1)
                xn = min(xmn,Nk)
                xp = min(Nw-xmn,Nk+1)

                dmn  = d[m,n,ymn-yn:ymn+yp,xmn-xn:xmn+xp]
                kumn = ku[m,n,ymn-yn:ymn+yp,xmn-xn:xmn+xp]*kd[Mk-yn:Mk+yp,Nk-xn:Nk+xp]
                kdmn = kd[Mk-yn:Mk+yp,Nk-xn:Nk+xp]

                dmn[:] = kumn*kdmn
                norm = dmn.sum()

                diffx = np.sum(dmn*x[ymn-yn:ymn+yp,xmn-xn:xmn+xp]) / norm - xmn
                diffy = np.sum(dmn*y[ymn-yn:ymn+yp,xmn-xn:xmn+xp]) / norm - ymn
                maxflow = max(maxflow,abs(diffx),abs(diffy))
                absflow += abs(diffx) + abs(diffy)
                meanflow += diffx**2 + diffy**2

                fx[m,n] += diffx
                fy[m,n] += diffy

        # apply the regularizer to the gradient
        fx,fy = regularizer(fx,fy)

        # convergence criteria
        #print maxflow, absflow/(Mw*Nw), meanflow/(Mw*Nw)
        if maxflow < eps:
            print('Converged after {} iterations'.format(iters))
            break
    else:
        print('Maximum iterations exceeded')

    return fx,fy,d
