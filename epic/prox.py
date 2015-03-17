from __future__ import division
import numpy as np

# ----------------------------------------------------------------------------
# Proximal Helpers
# ----------------------------------------------------------------------------
def infnorm(x):
    return np.abs(x).max()

def no_op(fx,fy,w=None):
    return fx,fy


# ----------------------------------------------------------------------------
# Proximal Gradient Method
# ----------------------------------------------------------------------------
def proximal_gradient(gradient_function, proximal_operator, x0, y0, max_iters=100, eps=1e-3, mul=0.95):
    """Proximal gradient method

    Optimizes the objective:
        argmin_{x,y} f(x,y) + a*g(x,y)

    Where f(x) is a non-convex function and g(x) is a non-smooth regularizer.

    Args:
        gradient_function: callable, returns the gradient of the non-convex
            function at the current estimate (x,y). If a step size or line
            search is required by the gradient function, this should be
            a closure
        proximal_operator: callable, calculates the proximal operator to the
            regularizer g
        x0,y0: The initial estimates of the minima

    Keyword Args:
        eps: The convergence tolerance
    """

    fx,fy,fx0,fy0 = np.zeros(x0.shape), np.zeros(y0.shape), np.inf, np.inf
    iters, t, t0  = 0, 1.0, 1.0

    # iterate until convergence
    while ( infnorm(fx-fx0) > eps or infnorm(fy-fy0) > eps ) and iters < max_iters:

        # compute the gradient
        gx,gy,w = gradient_function.compute(x0+fx, y0+fy)
        #gradient_function.sigma = max(gradient_function.sigma*mul, 0.1)
        #gradient_function.precompute()

        # update the flow vectors
        #gx,gy = proximal_operator(gx,gy,w=w)#np.sqrt(w))
        #fx,fy, fx0,fy0 = fx+gx,fy+gy, fx, fy
        #fx,fy, fx0,fy0 = fx+gx,fy+gy, fx,fy
        #for operator in proximal_operator:
        #    fx,fy = operator(fx,fy,w=w)
        (fx,fy),fx0,fy0 = proximal_operator(fx+gx,fy+gy,w=w),fx,fy

        # update the optimization variables
        iters += 1

    return fx,fy
