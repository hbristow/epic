from __future__ import division
import sys
import time
import bisect
import datetime
import numpy as np
from scipy.ndimage import imread
from scipy.sparse.linalg import eigsh
from scipy.linalg import cho_factor, cho_solve
try:
    import pyfftw.interfaces.numpy_fft as fft
except ImportError:
    import numpy.fft as fft

# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------
def load(filename):
    """Load the mean and statistics matrix from a npz archive

    Args:
        filename: Path to a .npz archive file containing the mean
            and covariance matrices in the fields 'mean' and 'cov'. The
            covariance matrix is stored as its compressed statistics.

    Returns:
        mean: The length K covariance mean
        cov: The PxQxKxK covariance statistics
    """
    with np.load(filename) as f:
        return f['mean'], f['cov']

def save(filename, mean, cov):
    """Save the mean and statistics matrix to a npz archive

    Args:
        filename: Path to a .npz archive
        mean: The length K covariance mean
        cov: The PxQxKxK covariance statistics
    """
    np.savez(filename, mean=mean, cov=cov)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def collect(images, channels=None, bandwidth=None, transform=lambda x:x):
    """Collect statistics over a set of images

    Args:
        images: A list of image files or images to collect statistics over
        channels: The number of channels in the multi-channel images
        bandwidth: The bandwidth of the statistics to collect

    Keyword Args:
        transform: A feature transform to apply to the image before
            collecting statistics

    Returns:
        mean: The mean of each channel
        cov: The stationary statistics matrix
    """
    # preallocate the mean and stats
    K,B,F = channels, bandwidth, len(images)
    cov = np.zeros((2*B+1,2*B+1,K,K))
    mean = np.zeros((K,))

    # keep track of the number of pixels and displacements encountered
    displacements = np.zeros((2*B+1,2*B+1,1,1))
    pixels = 0

    # iterate over each image
    start = time.time()
    for f,image in enumerate(images):

        # load the image and transform it into feature space
        image = transform(imagify(image))
        M,N = image.shape[:2]
        if M <= B or N <= B:
            continue

        # compute the displacements
        dn,dm = np.meshgrid(np.arange(-B,B+1), np.arange(-B,B+1))

        # accumulate the mean and displacement count
        mean += image.sum(axis=(0,1))
        pixels += M*N
        displacements[dm,dn,0,0] += (N-np.abs(dn))*(M-np.abs(dm))

        # compute the statistics in the Fourier domain
        If = fft.rfft2(image.transpose(2,0,1), (M+B,N+B))
        Ic = If.conj()
        for k1 in range(K):
            for k2 in range(k1,K):

                #  cross correlate
                corr = fft.irfft2(Ic[k1]*If[k2], (M+B,N+B))
                # accumulate
                cov[dm,dn,k1,k2] += corr[dm,dn]
                if k1 != k2:
                    cov[-dm,-dn,k2,k1] += corr[dm,dn]

        # display progress
        elapsed = time.time() - start
        estimate = datetime.timedelta(seconds=int(elapsed / (f+1)*(F-f-1)))
        progress = '\r\x1b[KEstimated time remaining {} [{:.2f}%]'
        sys.stdout.write(progress.format(estimate, (f+1)/F*100))
        sys.stdout.flush()

    # normalize the mean and covariance
    mean = mean/pixels
    cov  = cov/displacements - np.outer(mean, mean) # expectation property
    return mean, cov

def optimal_fft_size(*shape):
    """Compute the optimal FFT size for a given shape using a LUT heuristic"""
    lut = [8, 16, 32, 52, 64, 88, 108, 128, 180, 256, 300, 384, 512, 768, 1024]
    index = (bisect.bisect_left(lut, n) for n in shape)
    return tuple(lut[m] if m < len(lut) else n for m,n in zip(index,shape))

def imagify(obj):
    """Attempt to load an object as an image file, else return"""
    try:
        return imread(obj)
    except AttributeError:
        return obj


# ----------------------------------------------------------------------------
# Materialize covariance
# ----------------------------------------------------------------------------
def materialize(g, shape):
    """Construct a covariance matrix from the statistics matrix

    Args:
        g: The statistics matrix
        shape: The M,N spatial shape of the desired detector. The number of
            channels are inferred from the statistics matrix
    """
    # get the statistics properties
    M,N = shape
    P,Q,K,K = g.shape
    bwm,bwn = np.floor(P/2), np.floor(Q/2)

    # allocate the full covariance
    cov = np.zeros((M,N,K, M,N,K))

    # iterate over the spatial differences
    for m1 in range(M):
        for n1 in range(N):
            for m2 in range(M):
                for n2 in range(N):
                    if abs(m1-m2) > bwm or abs(n1-n2) > bwn: continue
                    cov[m1,n1,:, m2,n2,:] = g[n2-n1, m2-m1, ...]

    # reshape the output
    return cov.reshape(M*N*K, M*N*K)


# ----------------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------------
def minimum_lambda(cov, tol=1e-1, alpha=1e-2):
    """Minimum diagonal value to promote a sample covariance to be SPD

    Args:
        cov: The materialized sampled covariance matrix

    Keyword Args:
        tol: The residual tolerance of the eigenvalue estimation (default: 0.1)
        alpha: The ratio of the largest eigenvalue to include as well. (default: 0.02)
            Lambda is computed as:
                abs(min(eig_min,0)) + alpha*eig_max
    """
    (smin,smax),v = eigsh(cov, 2, which='BE', tol=tol)
    return abs(min(smin, 0)) + alpha*smax

def inverse(cov, eps=None, factorized=False):
    """Return the inverse of the covariance

    This method returns an object with a 'dot' method that computes the
    solution to the system of equations Ax = b, via x = inverse(A).dot(b)

    Args:
        cov: The covariance matrix to invert

    Keyword Args:
        eps: The value to add to the diagonal to make the covariance matrix
            positive-definite. By default, this value is calculated automatically
            by adding the absolute value of the most-negative eigenvalue of
            the matrix.
        factorized: Compute a factorization of the covariance rather than the
            explicit inverse.
    """

    # compute the minimum eps to make cov positive-definite
    if not eps:
        eps = minimum_lambda(cov)

    # augment the diagonal
    M,N = cov.shape
    cov = cov.copy()
    cov.flat[::N+1] += eps

    # compute the inverse
    return factorized_proxy(cho_factor(cov)[0]) if factorized else np.linalg.inv(cov)

class factorized_proxy(object):
    """Thin factorized array proxy that exposes a dot method

    This class wraps a factorized inverse with a single dot method so that
    x = inverse(A).dot(b) is always defined.
    """
    def __init__(self, array):
        self.array = array
    def dot(self, b):
        return cho_solve((self.array,False), b)
    @property
    def shape(self):
        return self.array.shape


# ----------------------------------------------------------------------------
# Logistic
# ----------------------------------------------------------------------------
def sigmoid(x):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-x))

def logistic(x=None, a=1.0, b=0.0):
    """Compute the logistic function

        y = 1 / (1 + exp(-(a*x + b)))

        Keyword Args:
            x: The signal to compute the (elementwise) logstic for. If a signal
                is not given, the function returns a closure with the bias and
                sigma captured
            a: The logistic sigma (default: 1.0)
            b: The logistic bias (default: 0.0)
    """
    def weighted_logistic(x):
        """y = 1 / (1 + exp(-({a}*x + {b})))"""
        return sigmoid(a*b + b)
    if x is not None:
        return weighted_logistic(x)
    weighted_logistic.__doc__ = weighted_logistic.__doc__.format(a=a,b=b)
    return weighted_logistic

def logistic_regression(X, y, C=1.0, reweighted=True, penalty='l1'):
    """Solve a logistic regression problem

    Args:
        X: The NxD array of input examples
        y: The length N array of output labels

    Keyword Args:
        reweighted: Resample the inputs so that the classes have equal numbers
            of samples (default: True)
        penalty: The fitting norm (default: L1)

    Returns:
        A,b: The parameters that minimize Ax + b in a sigmoidal loss
    """
    from sklearn.linear_model import LogisticRegression
    N = y.size
    D = X.size / N
    # reshape the input in case it's transposed or 1D
    X = X.T if X.shape == (D,N) else X.reshape(N,D)

    # solve
    regressor = LogisticRegression(C=C, penalty=penalty, class_weight='auto' if reweighted else None)
    regressor.fit(X,y)

    # return the weights and intercept
    return regressor.coef_, regressor.intercept_
