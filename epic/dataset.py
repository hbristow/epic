import os
import glob
import numpy as np
from scipy.misc import imread
from scipy.ndimage.filters import maximum_filter

# ----------------------------------------------------------------------------
# Variance of Groundtruth
# ----------------------------------------------------------------------------
def groundtruth_variance(subjects):
    """Compute the groundtruth variance on an image pair for different subjects

    Args:
        subjects: A file list of labellings
    """
    corr = np.array([np.load(subject) for subject in subjects])
    # F subjects, P points, LR left/right (2), D dimensions (2)
    F,P,LR,D = corr.shape
    left,right = corr.transpose((2,1,0,3))

    # compute the mean and covariance of the distributions
    means,covs = [],[]
    for point in right:
        means.append(np.mean(point, axis=0))
        covs.append(np.cov(point.T))

    # return the statistics
    return left[:,0], right, means, covs

def euclidean(x, mean, cov):
    """Compute the Euclidean distance of a set of points to distributions

    Args:
        x: MxNx...x2 set of points in (x,y) space
        mean: 2x1 mean of the distribution
        cov:  2x2 covariance of the distribution (not used)
    """
    return np.sqrt(((x-mean)**2).sum(axis=-1))

def mahalanobis(x, mean, cov):
    """Compute the Mahalanobis distance of a set of points to distributions

    Args:
        x: MxNx...x2 set of points in (x,y) space
        mean: 2x1 mean of the distribution
        cov:  2x2 covariance of the distribution
    """
    shape = x.shape[:-1]
    inv = np.linalg.inv(cov)
    x = x.reshape(-1,2) - mean
    y = np.sqrt(np.einsum('ij,ij->i', x.dot(inv), x))
    return y.reshape(shape)

def cdf(points, means, covs, metric=mahalanobis):
    """Compute the Gaussian CDF of a set of points, given distributions

    Args:
        points: Px2 list of points
        means:  Px2 list of distribution means
        covs:   Px2x2 list of covariance matrices

    Keyword Args:
        metric: The distance metric to use (default: mahalobis)
    """
    # compute the standard distances
    distances = []
    for point, mean, cov in zip(points, means, covs):
        distance = metric(point, mean, cov)
        distances.append(distance)

    # compute the CDF
    x,y = sorted(np.array(distances)), np.linspace(0, 1, len(distances))
    return x,y

def pdf(x, mean, cov):
    """Compute the Gaussian PDF of a set of points

    Args:
        x: MxNx...x2 set of points in (x,y) space
        mean: 2x1 mean of the distribution
        cov:  2x2 covariance of the distribution
    """
    shape = x.shape[:-1]
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    x   = x.reshape(-1,2) - mean
    y   = np.exp(-0.5*np.einsum('ij,ij->i', x.dot(inv), x)) / (2*np.pi*np.sqrt(det))
    return y.reshape(shape)

def gmm(means, covs, shape):
    """Generate a 2D Gaussian mixture model with given spatial shape

    Args:
        mean: The Px2 mean of the mixtures
        cov:  The Px2x2 covariance of the mixtures
        shape: The MxN spatial support of the output
    """
    M,N = shape[:2]
    x = np.dstack(np.meshgrid(np.arange(N), np.arange(M)))

    # add the mixtures
    return np.sum(pdf(x,mean,cov) for mean,cov in zip(means,covs))


# ----------------------------------------------------------------------------
# Mine Groundtruth for Validation Data
# ----------------------------------------------------------------------------
def mine(detector, transform, groundtruth='groundtruth', wpos=25,
        hard_negatives=True, wneg=5, easiest=0, hardest=-2,
        random_negatives=5):
    """Mine validation data for positives and hard negatives

    Args:
        detector: The detector which produces raw likelihood estimates
        transform: The feature transform to apply to the images

    Returns:
        X,Y: The likelihood produced by the detector, and the true label
    """
    sources = sorted(glob.glob(os.path.join(groundtruth, '*_source.*')))
    targets = sorted(glob.glob(os.path.join(groundtruth, '*_target.*')))
    matches = sorted(glob.glob(os.path.join(groundtruth, '*_correspondences.npy')))
    X,Y = [],[]

    for source, target, match in zip(sources, targets, matches):

        # load the images and compute the feature transform
        I1 = imread(source)
        I2 = imread(target)
        F1 = transform(I1)
        F2 = transform(I2)
        (M,N),(P,Q) = F1.shape[:2], F2.shape[:2]
        (R,S),(T,U) = I1.shape[:2], I2.shape[:2]
        W = np.round(np.max((P/wpos,Q/wpos)))
        #print W

        # load and rescale the matches to feature scale space
        match = np.load(match)
        match = match / ((S,R),(U,T)) * ((N,M),(Q,P))

        # compute the likelihoods
        scores  = detector(F1,F2)
        if hard_negatives:
            maximum = maximum_filter(scores, (1,1,P/wneg,Q/wneg))
            maximum = (scores == maximum)

        # iterate over the groundtruth labels
        for (x1,y1),(x2,y2) in match:
            score = scores[y1,x1]

            # positive
            pos = score[max(0,y2-W):min(y2+W,P),max(0,x2-W):min(x2+W,Q)].max()
            X.append(pos)
            Y.append(1.0)

            # hard negatives
            if hard_negatives:
                p,q   = maximum[y1,x1].nonzero()
                hard  = np.sort(np.unique(score[p,q]))
                neg   = hard[easiest:hardest]
                label = -np.ones_like(neg)
                X.extend(neg)
                Y.extend(label)

            # random negatives
            if random_negatives:
                neg = np.random.choice(score.flat, random_negatives)
                label = -np.ones_like(neg)
                X.extend(neg)
                Y.extend(label)

    # convert the lists to arrays
    return np.array(X), np.array(Y)
