import numpy as np

def trajectories(correspondences):
    """Map a sequence of inter-frame correspondences to trajectories

    Args:
        correspondences: [(dx, dy), (dx, dy), ...]
                          I1 -> I2  I2 -> I3
    """

    # get the number of frames and points being tracked
    F = len(correspondences)+1
    M,N = correspondences[0][0].shape

    # get the initial points
    x0,y0 = np.meshgrid(np.arange(N), np.arange(M))

    # initialize the trajectory matrix
    X = np.zeros((F,2,M,N), dtype=int)
    X[0,0,...] = x0
    X[0,1,...] = y0

    for (dx,dy),(x0,y0),(x,y) in zip(correspondences, X, X[1:,...]):
        idx  = x0 + N*y0
        x[:] = dx.flat[idx]
        y[:] = dy.flat[idx]

    return X.transpose(0,2,3,1)
