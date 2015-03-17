# gradient.py
# Comparison of gradient estimation techniques for Discretely Sampled Functions
from __future__ import division
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import RectBivariateSpline
import numpy as np

# ----------------------------------------------------------------------------
# Synthetic Response Generator
# ----------------------------------------------------------------------------
def noise(shape, beta=-2):
    from numpy import pi, sin, cos, fft, append, arange, ceil, floor
    """Generate spatial noise with given power spectrum

    Args:
        shape: The shape of the output array

    Keyword Args:
        beta: The spectral distribution
            0: White noise
           -1: Pink noise
           -2: Brownian noise
    """
    M,N = shape[-2:]

    # compute the frequency grid
    u,v = np.meshgrid(
        append(arange(0, floor(M/2)), -arange(ceil(M/2)-1, -1, -1))/M,
        append(arange(0, floor(N/2)), -arange(ceil(N/2)-1, -1, -1))/N,
        indexing='ij'
    )

    # generate the power spectrum
    mag = (u**2 + v**2)**(beta/2)
    mag[mag == np.inf] = 0

    # generate a grid of random phase shifts
    phi = np.random.standard_normal(shape)

    # inverse transform to obtain the spatial pattern
    x = fft.ifft2(mag**0.5 * (cos(2*pi*phi) + 1j*sin(2*pi*phi)))
    return x.real

def rescale(x, min=0.0, max=1.0):
    """Recale an array into the range [min, max]

    Args:
        min,max: The range of the output array. The output is always returned
            in floating point format.
    """
    range = max - min
    xnorm = (x-x.min()) / (x.max()-x.min())
    return range*xnorm + min

def logistic(x, sigma=1.0):
    """Logistic function of the inputs"""
    return 1.0 / (1.0 + np.exp(sigma*x))

def synthetic_function(shape):
    """Create a synthetic non-linear classifier response"""
    x = noise(shape)
    x = rescale(x, 0, 10)
    x = 2*logistic(x)
    return x

def argmax(x):
    shape = x.shape
    idx = x.reshape(shape[:-2]+(-1,)).argmax(axis=-1)
    idx = np.unravel_index(idx, shape[-2:])
    return idx

def initialize(x):
    idx = argmax(x)


# ----------------------------------------------------------------------------
# Gradient Estimators
# ----------------------------------------------------------------------------
def cutoff(sigma, eps=1e-3):
    return np.ceil(sigma*np.sqrt(-np.log(eps)))

def infnorm(x):
    return np.abs(x).max()

class Gradient(object):
    def compute(self, f, x):
        raise NotImplementedError

class Meanshift(Gradient):
    def __init__(self, shape, sigma=1.0):
        self.sigma = sigma
        M,N,P,Q = shape
        self.x, self.y = np.meshgrid(np.arange(Q), np.arange(P))

    def compute(self, f, x):

        # compute the kernel
        ux,uy = x[...,0], x[...,1]
        axis = (-2,-1)
        gaussian = np.exp( -np.sqrt((ux[...,None,None]-self.x)**2 + \
                                    (uy[...,None,None]-self.y)**2)/self.sigma**2 )
        kernel = f*gaussian
        norm   = kernel.sum(axis=axis)
        self.kernel = kernel
        self.norm = norm

        # compute the meanshift update
        ux = (self.x*kernel).sum(axis=axis) / norm - ux
        uy = (self.y*kernel).sum(axis=axis) / norm - uy

        return -np.dstack((ux,uy))


class MeanshiftFixed(Gradient):
    def __init__(self, shape, sigma=1.0):
        self.sigma = sigma
        self.W = W = cutoff(sigma)
        x,y = np.meshgrid(np.linspace(-W,W,2*W+1), np.linspace(-W,W,2*W+1))
        self.gaussian = np.exp( -np.sqrt(x**2 + y**2)/sigma**2 )

    def compute(self, f, x):

        # compute the kernel
        axis = (-2,-1)
        W = self.W
        M,N,P,Q = f.shape
        ux,uy = x[...,0], x[...,1]
        dx = np.zeros_like(ux)
        dy = np.zeros_like(uy)
        for i in range(M):
            for j in range(N):
                x,y = np.round(ux[i,j]), np.round(uy[i,j])
                x,y = np.linspace(x-W,x+W,2*W+1), np.linspace(y-W,y+W,2*W+1)
                xs,ys = np.meshgrid(x,y)
                x = np.abs(x) - (x > Q)*x%(Q-2)
                y = np.abs(y) - (y > P)*y%(P-2)
                xr,yr = np.meshgrid(x.astype(int), y.astype(int))

                # compute the response
                kernel = f[i,j,yr,xr]*self.gaussian
                norm   = kernel.sum(axis=axis)

                # compute the meanshift update
                dx[i,j] = (xs*kernel).sum(axis=axis) / norm - ux[i,j]
                dy[i,j] = (ys*kernel).sum(axis=axis) / norm - uy[i,j]

        return -np.dstack((dx,dy))


        #axis = (-2,-1)
        #gaussian = np.exp( -np.sqrt((ux[...,None,None]-self.x)**2 + \
        #                            (uy[...,None,None]-self.y)**2)/self.sigma**2 )
        #self.kernel = kernel = f*gaussian
        #gx = ((1/self.sigma**2)*(ux[...,None,None]-self.x)*kernel).sum(axis=axis)
        #gy = ((1/self.sigma**2)*(uy[...,None,None]-self.y)*kernel).sum(axis=axis)

        #return np.dstack((gx,gy))


class Direct(Gradient):
    def __init__(self, shape, sigma=1.0):
        self.sigma = sigma
        self.x, self.y = np.meshgrid(np.arange(Q), np.arange(P))

    def compute(self, f, x):

        # split the components
        ux,uy = x[...,0], x[...,1]

        # compute the basis
        gaussian = np.exp( -np.sqrt((self.x-ux[...,None,None])**2 + \
                                    (self.y-uy[...,None,None])**2)/self.sigma**2 )
        bx = (self.x-ux[...,None,None])*gaussian/self.sigma**2
        by = (self.y-uy[...,None,None])*gaussian/self.sigma**2



class FiniteDifferences(Gradient):
    def __init__(self, shape, sigma=1.0):
        M,N,P,Q = shape
        self.sigma = sigma
        self.x = np.arange(Q)
        self.y = np.arange(P)

    def compute(self, f, x):

        # compute the smoothed signal
        ux,uy = x[...,0], x[...,1]
        M,N,P,Q = f.shape
        sigma = self.sigma
        f = gaussian_filter(f, sigma)#1,1,sigma,sigma))

        # compute the gradient
        dx = np.zeros_like(ux)
        dy = np.zeros_like(uy)
        for i in range(M):
            for j in range(N):
                #fi = RectBivariateSpline(self.y, self.x, f[i,j])
                fi = f[i,j]
                xi,yi = np.round(ux[i,j]),np.round(uy[i,j])
                dx[i,j] = fi[yi,xi+1]-fi[yi,xi-1]
                dy[i,j] = fi[yi+1,xi]-fi[yi-1,xi]
                #dx[i,j] = fi(yi,xi+1)-fi(yi,xi-1)
                #dy[i,j] = fi(yi+1,xi)-fi(yi-1,xi)

        return -np.dstack((dx,dy))


class Simoncelli(Gradient):
    pass

def gradient_descent(estimator, f, x0, accelerated=True,
        tau=0.01,tol=1e-3,max_iters=1000):

    # initialize the optimization variables
    x,x0,t0,t,iters = x0,0,1,1,0

    # solve!
    while infnorm(x-x0) > tol and iters < max_iters:

        # compute the momentum
        y = x + (t0-1)/t*(x-x0)

        # compute the gradient
        g = estimator(f,y)
        x,x0 = x - tau*g, x

        # update the momentum size
        if accelerated:
            t,t0 = (1 + np.sqrt(1+4*t**2))/2, t
        iters += 1

        print g, x, iters, infnorm(x-x0)

    return x


# ----------------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------------
class Solver(object):
    def solve(self, f, x0):
        raise NotImplementedError
