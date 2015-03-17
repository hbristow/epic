from __future__ import division
import numpy as np
from scipy.ndimage.filters import correlate1d as correlate
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg import toeplitz

from epic import decorator, vis


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def squared_norm(x):
    return np.linalg.norm(x)**2


# ----------------------------------------------------------------------------
# Rigid Regularizers
# ----------------------------------------------------------------------------
class Affine(object):
    """Compute a rigid update to the points:

    argmin || J*J^T*x - x ||^2_2 + rho/2 || x - v ||^2_2
        x
    """

    def argmin(self, v, rho):

        # get the separate x,y components of the proximal point
        xy  = v.reshape(-1)
        M,N = v.shape[1:]

        # get the jacobian and inverse Jacobian
        J,Jinv = self.jacobian(M,N)

        # compute the rigid solution
        xyr = J.dot(Jinv.dot(xy))

        # compute the proximal solution
        xyp = (xyr + rho*xy)/(1.0 + rho)
        return xyp.reshape(2,M,N)

    @decorator.cached
    def jacobian(self, M, N):

        x,y = np.meshgrid(np.arange(N), np.arange(M))
        x,y = np.ravel(x), np.ravel(y)
        one = np.ones((x.size,))
        zero = np.zeros((x.size,))
        J = np.vstack((
            np.column_stack((x, y, one, zero, zero, zero)),
            np.column_stack((zero, zero, zero, x, y, one))
        ))
        Jinv = np.linalg.pinv(J)
        return J, Jinv

class Affine2(object):
    """Compute a rigid update to the points:

    argmin || J*J^T*x - x ||^2_2 + rho/2 || x - v ||^2_2
        x
    """

    def argmin(self, v, rho):

        # get the separate x,y components of the proximal point
        xy  = v.reshape(-1)
        M,N = v.shape[1:]

        # get the jacobian and inverse Jacobian
        A,B = self.jacobian(M,N)

        # compute the solution to the normal system of equations
        xyr = np.linalg.solve(A.T.dot(A) + rho*B.T.dot(B), rho*B.T.dot(xy))
        return xyr[:2*M*N].reshape(2,M,N)

    @decorator.cached
    def jacobian(self, M, N):

        x,y = np.meshgrid(np.arange(N), np.arange(M))
        x,y = np.ravel(x), np.ravel(y)
        one = np.ones((x.size,))
        zero = np.zeros((x.size,))
        J = np.vstack((
            np.column_stack((x, y, one, zero, zero, zero)),
            np.column_stack((zero, zero, zero, x, y, one))
        ))
        I = np.identity(2*M*N)
        A = np.hstack((-I, J))
        B = np.hstack((I, np.zeros((2*M*N, 6))))

        return A,B


# ----------------------------------------------------------------------------
# Smooth Flow
# ----------------------------------------------------------------------------
class L2Smooth(object):
    def __init__(self, shape, rho=0.1):

        # create the Toeplitz matrix rows and columns
        M,N = shape
        c = np.zeros((max(M,N),)); c[1] = -1.0
        r = np.zeros((max(M,N),)); r[1] = 1.0

        # create the filter matrices
        Gm = toeplitz(c[:M], r[:M])
        Gn = toeplitz(c[:N], r[:N])

        # forward differences at the endpoints
        Gm[0,0] = -1.0
        Gn[0,0] = -1.0
        Gm[-1,-1] = 1.0
        Gn[-1,-1] = 1.0

        # store factorizations of the matrices
        self.Lm = cho_factor(Gm.T.dot(Gm) + rho*np.identity(M))
        self.Ln = cho_factor(Gn.T.dot(Gn) + rho*np.identity(N))
        self.rho = rho

    def argmin(self, x, y, w=None):
        return cho_solve(self.Ln, self.rho*x.T).T,\
               cho_solve(self.Lm, self.rho*y)


# ----------------------------------------------------------------------------
# Variational Flow
# ----------------------------------------------------------------------------
class TVNormBase(object):
    def __init__(self, l=-np.inf, u=np.inf):
        """Initialize a TVNorm operator

        Args:
            l (float): The lower bound on the array (pixel) values
            u (float): The upper bound on the array (pixel) values
        """
        try:
            # compute the type of box constraints being used
            self.proj = {
                (True,  True,  True): lambda x: ((l<x)*(x<u))*x + (x>=u)*u + (x<=l)*l,
                (False, True,  True): lambda x: (x<u)*x + (x>=u)*u,
                (True,  False, True): lambda x: (l<x)*x + (x<=l)*l,
                (False, False, True): lambda x: x
            }[(-np.inf < l, u < np.inf, l < u)]
        except KeyError:
            msg = 'Box constraint (-inf <= l < u <= inf) is not satisfied for l={0},u={1}'.format(l,u)
            raise ValueError(msg)

    def gradient(self, x):
        """first difference gradient operator"""
        dx = np.zeros(x.shape)
        dy = np.zeros(x.shape)
        dx[:,:-1] = x[:,1:] - x[:,:-1]
        dy[:-1,:] = x[1:,:] - x[:-1,:]
        return dx, dy

    def divergence(self, x, y):
        """2D divergence of an array (image)"""
        dx = np.hstack((x[:,0:1], x[:,1:-1]-x[:,0:-2], -x[:,-2:-1]))
        dy = np.vstack((y[0:1,:], y[1:-1,:]-y[0:-2,:], -y[-2:-1,:]))
        return dx + dy

    def argmin(self, v, rho):
        """Minimize the TV Norm objective

        This involves solving a gradient-based FISTA objective
        """
        rx = np.zeros(v.shape)
        ry = np.zeros(v.shape)
        max_iters, tol, i, t0, obj, p, q = 100, 1e-3, 0, 1.0, np.inf, 0.0, 0.0

        while i < max_iters:
            # project the solution onto the box constraints
            x = self.proj(v - 1.0/rho*self.divergence(rx, ry))

            # check for convergence
            obj, obj0 = self(x) + np.sum(rho/2.0*(x - v)**2), obj
            #obj, obj0 = self(x) + rho/2.0*squared_norm(x - v), obj
            if np.abs(obj - obj0)/obj < tol:
                break

            # step along the -gradient direction
            dx,dy = self.gradient(x)
            rx -= (rho/8.0) * dx
            ry -= (rho/8.0) * dy

            # update the dual weights
            wx, wy = self.dual_weights(rx, ry)
            p, p0 = rx / wx, p
            q, q0 = ry / wy, q

            # compute the momentum
            t  = (1.0+np.sqrt(4.0*t0*t0))/2.0
            rx = p + (t0-1.0)/t * (p - p0)
            ry = q + (t0-1.0)/t * (q - q0)
            t0 = t
            i += 1

        return x


class TVL2Norm(TVNormBase):
    """Isotropic Total Variation (TV) Norm
    ::

        f(x) = || x ||_TV(L2)

    The proximal operator for the TV-norm is presented in:

        - A. Beck, M. Teboulle, "Fast Gradient-Based Algorithms for Constrained
            Total Variation Image Denoising and Deblurring Problems"
    """
    def __call__(self, x):
        dx, dy = self.gradient(x)
        return np.sqrt(dx*dx + dy*dy).sum()

    def dual_weights(self, x, y):
        w = np.fmax(np.sqrt(x*x + y*y), 1.0)
        return w, w


class TVL1Norm(TVNormBase):
    """Anisotropic Total Variation L1 (TVL1) Norm
    ::

        f(x) = || x ||_TV(L1)

    The proximal operator for the anisotropic TV-norm is presented in:

        - A. Beck, M. Teboulle, "Fast Gradient-Based Algorithms for Constrained
            Total Variation Image Denoising and Deblurring Problems"
    """
    def __call__(self, x):
        dx, dy = self.gradient(x)
        return np.abs(dx).sum() + np.abs(dy).sum()

    def dual_weights(self, x, y):
        wx = np.fmax(np.abs(x), 1.0)
        wy = np.fmax(np.abs(y), 1.0)
        return wx, wy

class TVL1(object):
    def __init__(self, rho=0.1, l=-np.inf, u=np.inf):
        self.tv = TVL1Norm(l, u)
        self.rho = rho

    def argmin(self, x, y, w=None):
        rho = 1.0 if w is None else w
        return self.tv.argmin(w*x, self.rho), self.tv.argmin(w*y, self.rho)


# ----------------------------------------------------------------------------
# Generalized Variation
# ----------------------------------------------------------------------------
def infnorm(x):
    return np.max(np.abs(x))
    return np.sqrt(np.max(np.sum(x**2, axis=2)))

def dx(x):
    return correlate(x, (0,-1.0,1.0), axis=1, mode='nearest')
def dxt(x):
    return correlate(x, (-1.0,1.0,0), axis=1, mode='nearest')

def dy(x):
    return correlate(x, (0,-1.0,1.0), axis=0, mode='nearest')
def dyt(x):
    return correlate(x, (-1.0,1.0,0), axis=0, mode='nearest')

def div(x):
    return np.dstack((
        dx(x[:,:,0]) + dy(x[:,:,1]),
        dx(x[:,:,1]) + dy(x[:,:,2])
    ))

def div2(x):
    return dxt(dx(x[:,:,0])) + dyt(dy(x[:,:,2])) + \
            dyt(dx(x[:,:,1])) + dxt(dy(x[:,:,1]))

def symm(u):
    return np.dstack((
        dxt(u[:,:,0]),
        (dyt(u[:,:,0])+dxt(u[:,:,1]))/2,
        dyt(u[:,:,1])
    ))

def symm2(u):
    return np.dstack((
        dxt(dx(u)),
        (dyt(dx(u)) + dxt(dy(u)))/2,
        dyt(dy(u))
    ))

def shrink(x,rho):
    return np.sign(x)*np.fmax(np.abs(x)-rho, 0.0)


class TGVADMM(object):

    def __init__(self, a0=0.1, a1=0.05):
        self.a0 = a0
        self.a1 = a1

    def dx(self, x):
        return correlate(x, (-0.5,0,0.5), axis=1)
    def dx(self, x):
        return correlate(x, (-0.5,0,0.5), axis=0)

    def grad(self, x):
        return np.vstack((
            self.dx(u),
            self.dy(u)
        ))

    def symm(self, v):
        pass

class TGVL2(object):

    def __init__(self, a0=0.1, a1=0.01, tau=0.01):
        self.a0 = a0
        self.a1 = a1
        self.tau = tau
        self.sigma = 1.0/8.0

    def proximal(self, f):

        M,N = f.shape
        self.nu = np.random.standard_normal((M,N,2))
        v,v0,t0,t,tol,max_iters,iters = np.random.standard_normal((M,N,3)),1,1,1,1e-3,1000,0
        while infnorm(v-v0) > tol and iters < max_iters:
            y = v + (t0-1)/t*(v-v0)
            v,v0 = self.proj(y + self.tau*(symm2(f - div2(y)))),v
            t,t0 = (1 + np.sqrt(1+4*t**2))/2,t
            iters += 1
            print np.linalg.norm(div2(v)), infnorm(v-v0)
        return f - div2(v)

    def proj(self, v):

        M,N,K = v.shape
        nu,nu0,t0,t,tol,max_iters,iters = self.nu,1,1,1,1e-4,100,0
        while infnorm(nu-nu0) > tol and iters < max_iters:
            y = nu + (t0-1)/t*(nu-nu0)
            q = shrink(v + symm(y), self.a0)
            nu,nu0 = shrink(y + self.sigma*div(v + symm(y)-q), self.a1),nu
            t,t0 = (1 + np.sqrt(1+4*t**2))/2,t
            iters += 1
        self.nu = nu
        return v - q + symm(nu)

    def argmin(self, x, y, w=None):
        xmin,xrange = x.min(), x.max()-x.min()
        ymin,yrange = y.min(), y.max()-y.min()
        x = self.proximal(vis.rescale(x))
        y = self.proximal(vis.rescale(y))
        return x*xrange+xmin, y*yrange+ymin
        return self.proximal(x), self.proximal(y)
