"""
coeffuncs defines the functions we use when computing coefficients for FD
"""
import numpy as np

def StirlingS1(n):
    """Computes the Stirling numbers

    :param n:
    :type n:
    :return:
    :rtype:
    """
    stirlings = np.eye(n)
    for i in range(n):
        for j in range(i+1):
            stirlings[i, j] = (-1)**(i-j) *( (i-1)*abs(stirlings[i-1, j]) + abs(stirlings[i-1, j-1]))
    return stirlings

def Binomial(n):
    """

    :param n:
    :type n:
    :return:
    :rtype:
    """
    binomials = np.eye(n)
    binomials[:, 0] = 1
    for i in range(2, n):
        if i%2 == 0:
            k = i/2 + 1
        else:
            k = (i+1)/2
        for j in range(int(k)):
            binomials[i, j] = binomials[i, i-j] = binomials[i-1, j-1] + binomials[i-1, j]
    return binomials

# def _StirlingS1(n):
#     """simple recursive definition of the StirlingS1 function in Mathematica
#     implemented at the C level mostly just for fun
#
#     :param n:
#     :type n:
#     :return:
#     :rtype:
#     """
#
#     return _StirlingS1(n)
#
# def Binomial(n):
#     """simple recursive Binomial coefficients up to r, computed all at once to vectorize later ops
#     wastes space, justified by assuming a small-ish value for n
#
#     :param n:
#     :type n:
#     :return:
#     :rtype:
#     """
#     return _Binomial(n)

def GammaBinomial(s, n):
    """Generalized binomial gamma function

    :param s:
    :type s:
    :param n:
    :type n:
    :return:
    :rtype:
    """
    g = np.math.gamma
    g1 = g(s+1)
    g2 = np.array([g(m+1)*g(s-m+1) for m in range(n)])
    g3 = g1/g2
    return g3

def Factorial(n):
    """I was hoping to do this in some built in way with numpy...but I guess it's not possible?
    looks like by default things don't vectorize and just call math.factorial

    :param n:
    :type n:
    :return:
    :rtype:
    """

    base = np.arange(n, dtype=np.int64)
    base[0] = 1
    for i in range(1, n):
        base[i] = base[i]*base[i-1]
    return base

def EvenFiniteDifferenceWeights(m, s, n):
    """Finds the series coefficients for x^s*ln(x)^m centered at x=1. Uses the method:

             Table[
               Sum[
                ((-1)^(r - k))*Binomial[r, k]*
                    Binomial[s, r - j] StirlingS1[j, m] (m!/j!),
                {r, k, n},
                {j, 0, r}
                ],
               {k, 0, n}
               ]
             ]

        which is shown by J.M. here: https://chat.stackexchange.com/transcript/message/49528234#49528234

    :param m: the order of the derivative requested
    :type m:
    :param s: the offset of the point at which the derivative is requested from the left edge of the stencil
    :type s:
    :param n: the number of points used in the stencil
    :type n:
    :return:
    :rtype:
    """

    n = n+1 # in J.M.'s algorithm we go from 0 to n in Mathematica -- which means we have n+1 elements
    stirlings = StirlingS1(n)[:, m]
    bins = Binomial(n)
    sTest = s - int(s)
    if sTest == 0:
        bges = bins[ int(s) ]
    else:
        bges = GammaBinomial(s, n)
    bges = np.flip(bges)
    facs = Factorial(n)
    fcos = facs[m]/facs # factorial coefficient (m!/j!)

    coeffs = np.zeros(n)
    bs = bges
    ss = stirlings
    fs = fcos
    # print(n, m, s, fcos, bges)
    for k in range(n):
        # each of these bits here should go from
        # Binomial[s, r - j] * StirlingS1[j, m] *
        bits = np.zeros(n-k)
        for r in range(k+1, n+1):
            bits[r-k-1] = np.dot(bs[-r:], ss[:r]*fs[:r])

        # (-1)^(r - k))*Binomial[r, k]
        cs = (-1)**(np.arange(n-k)) * bins[k:n, k]
        # print(bits, file=sys.stderr)
        coeffs[k] = np.dot(cs, bits)

    # print(coeffs)

    return coeffs

# _UnevenFiniteDifferenceWeights = None
def UnevenFiniteDifferenceWeights(m, z, x):
    # global _UnevenFiniteDifferenceWeights
    # if _UnevenFiniteDifferenceWeights is None:
    #     try:
    #         from .lib import UnevenFiniteDifferenceWeights as _UnevenFiniteDifferenceWeights
    #     except ImportError:
    #         def _UnevenFiniteDifferenceWeights(m, s, n):
    #
    #             raise NotImplementedError("Need to translate the C implementation to pure python; don't want to use an extension for something so minimal")
    # return _UnevenFiniteDifferenceWeights(m, z, x)

    # We're gonna provide an inefficient imp, but this function is so cheap to start out that who cares


    # Py_ssize_t n = PyObject_Length(xArray) - 1;
    # int dims[2] = {n+1, m+1};
    # PyObject *cArray = _CreateArray(2, dims, "zeros");
    # double *c = _GetDataArray<double>(cArray);

    # this is basically just copied directly from the OG source
    n = len(x) - 1
    dims = [n+1, m+1]
    c = np.zeros(dims)
    dxProdOld = 1.
    dz = x[0] - z
    c[0, 0] = 1.
    for i in range(1, n+1):
        mn = min(i, m)
        dxProd = 1.
        dzOld = dz
        dz = x[i] - z
        for j in range(0, i):
            dx = x[i] - x[j]
            dxProd *= dx
            if j == i-1:
                for k in range(mn, 0, -1):
                    c2 = c[i-1, k-1]
                    c3 = c[i-1, k]
                    weight = (k*c2 - dzOld * c3)*dxProdOld/dxProd
                    c[i, k] = weight
                c[i, 0] = -dzOld*c[i-1, 0]*dxProdOld/dxProd

            for k in range(mn, 0, -1):
                weight = (dz*c[j, k] - k*c[j, k-1])/dx
                c[j, k] = weight
            weight = dz*c[j, 0]/dx
            c[j, 0] = weight
        dxProdOld = dxProd

    return c