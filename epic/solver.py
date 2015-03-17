from __future__ import division
import os
import siftflow
import numpy as np
from scipy.misc import imread, imresize

from epic import _epic
from epic import admm, convolution, decorator, fidelity, lda, regularizers, vis

modulepath = os.path.dirname(__file__)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def stretchgrid(I1, I2):
    """Create a mesh the size of the first image that spans the second image

    Args:
        I1, I2: The two images (ndarray) or image shapes (tuple). The output
            meshes x,y are the same shape as I1, and numerically span the
            shape of I2.
    """
    M1,N1 = I1[:2] if isinstance(I1, tuple) else I1.shape[:2]
    M2,N2 = I2[:2] if isinstance(I2, tuple) else I2.shape[:2]
    x,y   = np.meshgrid(np.linspace(0,N2-1,N1), np.linspace(0,M2-1,M1))
    return x,y

def convex_combination(y):
    """Initialize the points as a combination of span and optimal value

    Initialize x0,y0 as a convex combination of the grid span and max
    of the probability distribution. The weighting, theta, is dictated by
    the confidence. More confident points are initialized closer to their
    optima, and less confident points are initialized closer to the span:

        x0 = theta*x_opt + (1-theta)*x_span
    """
    M,N,P,Q = y.shape
    argmax = y.reshape(M,N,-1).argmax(-1)
    theta  = y.max(axis=(-1,-2))
    x0, y0 = stretchgrid((M,N), (P,Q))
    ym, xm = np.unravel_index(argmax, (P,Q))

    return theta*xm + (1-theta)*x0, \
           theta*ym + (1-theta)*y0


# ----------------------------------------------------------------------------
# Every Pixel is a Classifier (EPIC) Solver
# ----------------------------------------------------------------------------
class EPICSolver(object):
    """Convenience class for solving the semantic correspondence optimization

    Solve the sematnic correspondence optimization problem:

        argmin  D(x) + beta*R(x)
            x
    """
    @decorator.autoassign
    def __init__(self,
            # image features
            image_transform=siftflow.sift,
            image_size=(125,125),
            # detector
            detector_statistics=os.path.join(modulepath,'..','data','statistics_sift_imagenet.npz'),
            detector_factorized=False,
            detector_size=(5,5),
            # probability transform
            #logistic_prior=-2.5,
            #logistic_sigma=0.06,
            logistic_prior=-1.815,
            logistic_sigma=0.044,
            # solver
            fidelity=fidelity.DiscreteMax,
            regularizer=regularizers.Affine,
            admm_beta=0.1,
            admm_rho=0.01,
            admm_max_iters=100,
            admm_eps=0.75):

        # preload the statistics
        self.precompute()

    def precompute(self):
        """Load the LDA statistics and precompute the inverse covariance"""
        self.mean,g = lda.load(self.detector_statistics)
        self.cov    = lda.materialize(g, self.detector_size)
        self.covinv = lda.inverse(self.cov, factorized=self.detector_factorized)

    def transform(self, I):
        """Apply the image transform to an image"""
        ratio = (self.image_size[0] / I.shape[0],
                 self.image_size[1] / I.shape[1])
        return self.image_transform(imresize(I, min(ratio))).astype(float)

    def probability(self, x):
        """Compute the logistic function of an input"""
        return 1.0 / (1.0 + np.exp( -(self.logistic_prior + self.logistic_sigma*x) ))

    def likelihood(self, F1, F2):
        """Compute the raw likelihood of matches, pixelwise"""

        # compute the data fidelity term
        return convolution.gemm(F1, F2, self.mean, self.covinv, padding='edge')


    def argmax(self, scores, x0=None, y0=None):
        """Solve the argmax optimization problem given the posterior probability"""

        # initialize the prox operators
        fidelity = self.fidelity(scores)
        regularizer = self.regularizer()

        # compute the initial correspondence
        M,N,P,Q = scores.shape
        if x0 is None and y0 is None:
            x0,y0 = stretchgrid((M,N),(P,Q))

        # solve the system
        return admm.solve(fidelity, regularizer, x0, y0,
                self.admm_beta, self.admm_rho, self.admm_max_iters, self.admm_eps)

    def max(self, scores, x, y):
        """Compute the max and objective at the given points (assumed argmax)"""
        M,N,P,Q = scores.shape
        M,N = np.meshgrid(np.arange(M), np.arange(N))

        confidence = scores.flat[ x + y*Q + Q*N + Q*N*M ]
        obj = confidence.sum() / (M*N)
        return obj, confidence

    def solve(self, I1, I2):
        """Compute the correspondence from I1 --> I2"""

        # compute the image features
        F1 = self.transform(I1)
        F2 = self.transform(I2)

        # compute the data fidelity
        scores = self.probability(self.likelihood(F1, F2))

        # compute the argmax constrained correspondence vectors
        x, y = self.argmax(scores)
        obj,confidence = self.max(scores, x, y)

        return fx,fy,obj,confidence
