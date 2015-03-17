from __future__ import division
import numpy as np

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def infnorm(x):
    """Norm infinity of a matrix/vector

    || x ||_inf == max( |x| )
    """
    return np.abs(x).max()

# ----------------------------------------------------------------------------
# Alternating Direction Method
# ----------------------------------------------------------------------------
def solve(fidelity, regularizer, x0, y0, beta=0.1,
    rho=0.01, max_iters=10, eps=0.75):
    """Solve the ADMM correspondence problem with the given fidelity and regularizer

        argmin  D(x) + beta*R(z)
          x,z
        s.t.    x == z
    """

    # initialize the variables
    x  = np.row_stack((x0[None,...], y0[None,...]))
    z  = np.copy(x)
    u  = np.zeros_like(x)
    r  = np.inf

    iters = 0
    while infnorm(r) > eps and iters < max_iters:

        # update the subproblems
        x,x0 = fidelity.argmin(z-u, rho), x
        z,z0 = np.round(regularizer.argmin(x+u, rho/beta)), z

        # update the multipliers
        r = x-z
        u = u+r
        print '%05f, %05f, %05f' % \
            (infnorm(x-z), infnorm(x-x0), infnorm(z-z0))

        #rho,iters = rho*mul, iters+1
        iters = iters+1

    return x[0], x[1]
