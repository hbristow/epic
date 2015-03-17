import functools
from inspect import getargspec, isfunction
from itertools import izip, ifilter, starmap


# ----------------------------------------------------------------------------
# Auto variable assignment
# ----------------------------------------------------------------------------
def autoassign(*names, **kwargs):
    """Automatic attribute assignment decorator

    This decorator automatically assigns all arguments to a function to
    attributes of the class with the same name. Thus,

        @autoassign
        def __init__(self, a, b, c=None):
            pass

    is equivalent to:

        def __init__(self, a, b, c=None):
            self.a = a
            self.b = b
            self.c = c
    """
    if kwargs:
        exclude, f = set(kwargs['exclude']), None
        sieve = lambda l:ifilter(lambda nv: nv[0] not in exclude, l)
    elif len(names) == 1 and isfunction(names[0]):
        f = names[0]
        sieve = lambda l:l
    else:
        names, f = set(names), None
        sieve = lambda l: ifilter(lambda nv: nv[0] in names, l)

    def decorator(f):
        fargnames, _, _, fdefaults = getargspec(f)
        # Remove self from fargnames and make sure fdefault is a tuple
        fargnames, fdefaults = fargnames[1:], fdefaults or ()
        defaults = list(sieve(izip(reversed(fargnames), reversed(fdefaults))))

        @functools.wraps(f)
        def decorated(self, *args, **kwargs):
            assigned = dict(sieve(izip(fargnames, args)))
            assigned.update(sieve(kwargs.iteritems()))
            for _ in starmap(assigned.setdefault, defaults): pass
            self.__dict__.update(assigned)
            return f(self, *args, **kwargs)
        return decorated
    return f and decorator(f) or decorator


# ----------------------------------------------------------------------------
# Memoization/Caching
# ----------------------------------------------------------------------------
class cached(object):
    """Last value memoization for functions of any arguments

    Some proximal operators may be able to take advantage of caching computation
    to speed up iterations. For example, the prox operator to the L2-norm looks
    like
    ::

        argmin 1/2 ||Ax - b||^2_2 + rho/2 || x - v ||^2_2
            x

    Only rho and v change between iterations, which means we can cache a
    factorization of A. Such a factorization involves, for example,
    ::

        L = chol( A^T*A + rho*I )

    Thus the factorization is valid until rho changes. Since most ADMM strategies
    keep rho fixed, this factorization only needs to be computed once ever.

    This class provides a simple mechanism for caching the last computed value.
    In the example above, it could be used as follows:
    ::

        @cached
        def factorize(self, A, rho):
            ...

    factorize will only be called when either A or rho changes, otherwise the
    precomputed value will be returned immediately.
    """
    def __init__(self, f):
        self.f = f
        self.name = '{f}_cached'.format(f=f.__name__)
    def __get__(self, instance, cls=None):
        return functools.update_wrapper(functools.partial(self, instance), self.f)
    def __call__(self, instance, *args):
        key = fast_hash(args)
        try:
            return getattr(instance, self.name)[key]
        except:
            val = self.f(instance, *args)
            setattr(instance, self.name, {key: val})
            return val


def fast_hash(args):
    """Fast recursive hash for numpy arrays

    Numpy arrays are not natively hashable since there is no well-defined
    equivalence property for general arrays. This function computes a hashable
    tuple from an iterable of input items possibly containing numpy arrays.

    Numpy arrays are hashed by ``id()``, so numerically equivalent arrays will
    not hash equal.

    ``fast_hash`` is used internally by ``cached``.
    """
    try:
        return args.__hash__()
    except:
        return tuple(fast_hash(arg) for arg in args)


def cached_property(f):
    """A computed property whose value is only calculated on the first call

    This is used as a decorator in the same manner as the builtin ``property``
    except that the property value is only computed once, then cached
    ::

        @cached_property
        def __doc__(self):
            return dynamically_created_docstring()
    """
    return property(cached(f))
