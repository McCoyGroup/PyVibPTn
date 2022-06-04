"""
Lazy class for holding force constants and higher derivative tensors pulled from the Gaussian log file
"""
import numpy as np
from ..Numputils import SparseArray

__all__ = [
    "FchkForceConstants",
    "FchkForceDerivatives",
    "FchkDipoleDerivatives",
    "FchkDipoleHigherDerivatives",
    "FchkDipoleNumDerivatives"
]

class FchkForceConstants:
    """
    Holder class for force constants coming out of an fchk file.
    Allows us to construct the force constant matrix in lazy fashion if we want.
    """
    def __init__(self, fcs):
        self.fcs = fcs
        self._n = None

    def __len__(self):
        return len(self.fcs)

    def _get_n(self):
        """
        :return:
        :rtype: int
        """
        if self._n is None:
            self._n = int((-1 + np.sqrt(1 + 8*len(self)))/6) # solving 3n*3n == 2*l - 3n
        return self._n

    @property
    def n(self):
        return self._get_n()
    @property
    def shape(self):
        return (3*self.n, 3*self.n)

    def _get_array(self):
        """Uses the computed n to make and symmetrize an appropriately formatted array from the lower-triangle data
        :return:
        :rtype: np.ndarray
        """
        n = self.n
        full_array = np.zeros((3*n, 3*n))
        full_array[np.tril_indices_from(full_array)] = self.fcs
        full_array = full_array + np.tril(full_array, -1).T
        return full_array

    @property
    def array(self):
        return self._get_array()


class FchkForceDerivatives:
    """Holder class for force constant derivatives coming out of an fchk file"""
    def __init__(self, derivs):
        self.derivs = derivs
        self._n = None

    def __len__(self):
        return len(self.derivs)

    def _get_n(self):
        if self._n is None:
            l = len(self)
            # had to use Mathematica to get this from the cubic poly
            #  2*(3n-6)*(3n)^2 == 2*l - 2*(3n-6)*(3n)
            l_quad = 81*l**2 + 3120*l - 5292
            l_body = (3*np.sqrt(l_quad) - 27*l - 520)
            if l_body > 0:
                l1 = l_body**(1/3)
            else:
                l1 = -(-l_body)**(1/3)
            n = (1/18)*( 10 + (2**(1/3))*( l1 - 86/l1) )
            self._n = int(np.ceil(n)) # precision issues screw this up in python, but not in Mathematica (I think)
        return self._n

    @property
    def n(self):
        return self._get_n()

    def _get_third_derivs(self):
        # fourth and third derivs are same len
        d = self.derivs
        return d[:int(len(d)/2)]

    def _get_fourth_derivs(self):
        # fourth and third derivs are same len
        d = self.derivs
        return d[int(len(d)/2):]

    @property
    def third_derivs(self):
        return self._get_third_derivs()

    @property
    def fourth_derivs(self):
        return self._get_fourth_derivs()
    @staticmethod
    def _fill_3d_tensor(n, derivs):
        """Makes and fills a 3D tensor for our derivatives
        :param n:
        :type n:
        :param derivs:
        :type derivs:
        :return:
        :rtype: np.ndarray
        """
        dim_1 = (3*n)
        mode_n = 3*n-6

        full_array_1 = np.zeros((mode_n, dim_1, dim_1))
        # set the lower triangle
        inds_1, inds_2 = np.tril_indices(dim_1)
        l_per = len(inds_1)
        main_ind = np.broadcast_to(np.arange(mode_n)[:, np.newaxis], (mode_n, l_per)).flatten()
        sub_ind_1 = np.broadcast_to(inds_1, (mode_n, l_per)).flatten()
        sub_ind_2 = np.broadcast_to(inds_2, (mode_n, l_per)).flatten()
        inds = ( main_ind, sub_ind_1, sub_ind_2 )
        full_array_1[inds] = derivs
        # set the upper triangle
        inds2 = ( main_ind, sub_ind_2, sub_ind_1 ) # basically just taking a transpose
        full_array_1[inds2] = derivs

        return full_array_1
    def _get_third_deriv_array(self):
        """we make the appropriate 3D tensor from a bunch of 2D tensors
        :return:
        :rtype: np.ndarray
        """
        n = self.n
        derivs = self.third_derivs
        return self._fill_3d_tensor(n, derivs)
    @property
    def third_deriv_array(self):
        return self._get_third_deriv_array()
    def _get_fourth_deriv_array(self):
        """We'll make our array of fourth derivs exactly the same as the third
        admittedly this should be a 4D tensor, but we only have the diagonal elements so it's just 3D
        I should make it a 4D sparse matrix honestly... Apparently we won't need many terms in the 4D tensor so it might
        make sense to handle that bloop doop bloop in the schmoop
        :return:
        :rtype: np.ndarray
        """
        n = self.n
        derivs = self.fourth_derivs
        return SparseArray.from_diag(self._fill_3d_tensor(n, derivs))
    @property
    def fourth_deriv_array(self):
        return self._get_fourth_deriv_array()

class FchkDipoleDerivatives:
    """Holder class for dipole derivatives coming out of an fchk file"""
    def __init__(self, derivs):
        self.derivs = derivs
        self._n = None

    def _get_n(self):
        """
        :return:
        :rtype: int
        """
        # derivatives with respect to 3N Cartesians...
        if self._n is None:
            self._n = int(len(self.derivs)/9) # solving 3*3n == l
        return self._n
    @property
    def n(self):
        return self._get_n()
    @property
    def shape(self):
        return (3*self.n, 3)
    @property
    def array(self):
        return np.reshape(self.derivs, self.shape)

class FchkDipoleHigherDerivatives:
    """Holder class for dipole derivatives coming out of an fchk file"""
    def __init__(self, derivs):
        self.derivs = derivs
        self._n = None
    def _get_n(self):
        """
        :return:
        :rtype: int
        """
        # numerical derivatives with respect to the 3n-6 normal modes of derivatives with respect to 3N Cartesians...
        # Gaussian gives us stuff out like d^2mu/dQdx and d^3mu/dQ^2dx
        if self._n is None:
            l = len(self.derivs)
            self._n = int(1 + np.sqrt(1 + l/54)) # solving 3n*(3n-6) == l/6
        return self._n
    @property
    def n(self):
        return self._get_n()
    @property
    def shape(self):
        return (3*self.n - 6, 3*self.n, 3)

    @property
    def second_deriv_array(self):
        nels = int(np.prod(self.shape))
        return np.reshape(self.derivs[:nels], self.shape)
    @property
    def third_deriv_array(self):
        nels = int(np.prod(self.shape))
        base_array = np.reshape(self.derivs[nels:], self.shape)
        full_array = np.zeros((3*self.n - 6, 3*self.n - 6, 3*self.n, 3))
        for i in range(3*self.n - 6):
            full_array[i, i] = base_array[i]
        return full_array

class FchkDipoleNumDerivatives:
    """
    Holder class for numerical derivatives coming out of an fchk file.
    Gaussian returns first and second derivatives
    """
    def __init__(self, derivs):
        self.derivs = derivs
        self._n = None
    def _get_n(self):
        """
        Returns the number of _modes_ in the system
        :return:
        :rtype: int
        """
        # derivatives with respect to (3N - 6) modes...
        if self._n is None:
            self._n = len(self.derivs)//6 # solving 2*3*n == l
        return self._n
    @property
    def n(self):
        return self._get_n()
    @property
    def shape(self):
        return (self.n, 3)
    @property
    def first_derivatives(self):
        return np.reshape(self.derivs[:len(self.derivs)//2], self.shape)
    @property
    def second_derivatives(self):
        return np.reshape(self.derivs[len(self.derivs)//2:], self.shape)