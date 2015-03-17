import os
import glob
import numpy as np
from scipy.ndimage import imread
from scipy.ndimage.filters import maximum_filter

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
    sources = glob.glob(os.path.join(groundtruth, '*_source.*'))
    targets = glob.glob(os.path.join(groundtruth, '*_target.*'))
    matches = glob.glob(os.path.join(groundtruth, '*_correspondences.npy'))
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
