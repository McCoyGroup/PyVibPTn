import numpy as np
from .TransformationFunction import TransformationFunction
from ...Numputils import affine_matrix, merge_transformation_matrix, one_pad_vecs

######################################################################################################
##
##                                   AffineTranform Class
##
######################################################################################################

__all__ = [
    "AffineTransform"
]

__reload_hook__ = ['.TransformationFunction', "...Numputils"]

class AffineTransform(TransformationFunction):
    """A simple AffineTranform implementation of the TransformationFunction abstract base class

    """

    def __init__(self, tmat, shift=None):
        """tmat must be a transformation matrix to work properly

        :param shift: the shift for the transformation
        :type shift: np.ndarray | None
        :param tmat: the matrix for the linear transformation
        :type tmat: np.ndarray
        """

        self.transf = affine_matrix(tmat, shift)
        super().__init__()

    @property
    def transform(self):
        return self.transf[:3, :3]

    @property
    def inverse(self):
        """
        Returns the inverse of the transformation
        :return:
        :rtype:
        """
        return self.reverse()

    @property
    def shift(self):
        transf = self.transf
        transf_shape = transf.shape
        if transf_shape[-1] == 4:
            vec = transf[:-1, 3] # hopefully this copies rather than just being a view...
        else:
            vec = None
        return vec

    def merge(self, other):
        """

        :param other:
        :type other: np.ndarray or AffineTransform
        """

        if isinstance(other, AffineTransform):
            other = other.transf
        transf = self.transf

        # I wanted to use type(self) but then I realized that'll fuck me over
        # if I want to merge like a ScalingTransform and a RotationTransform
        return AffineTransform(merge_transformation_matrix(transf, other))


    def reverse(self):
        """Inverts the matrix

        :return:
        :rtype:
        """
        
        inverse = np.linalg.inv(self.transf)
        return AffineTransform(inverse)


    def operate(self, coords, shift=True):
        """

        :param coords: the array of coordinates passed in
        :type coords: np.ndarry
        """
        translate = shift

        # Assumes that we're getting 3D cartesian coordinates...might not be a valid assumption
        coords = np.asarray(coords)
        coord_shape = coords.shape
        if len(coord_shape) == 1:
            adj_coord = coords.reshape((1, coord_shape[0]))
        elif len(coord_shape) > 2:
            nels = np.product(coord_shape[:-1])
            adj_coord = coords.reshape((nels, 3))
        else:
            adj_coord = coords

        tmat = self.transf
        if tmat.shape[-1] == 4:
            if not translate:
                tmat = tmat[:3, :3]
            else:
                adj_coord = one_pad_vecs(adj_coord)
            adj_coord = np.tensordot(adj_coord, tmat, axes=[1, 1])
            if translate:
                adj_coord = adj_coord[..., :3]
            adj_coord = adj_coord.reshape(coord_shape)
        else:
            adj_coord = np.tensordot(adj_coord, tmat, axes=[1, 1])
            adj_coord = adj_coord.reshape(coord_shape)

        return adj_coord

    def __repr__(self):
        ## we'll basically just leverage the ndarray repr:
        return "{}(transformation={}, shift={})".format(type(self).__name__, str(self.transform), str(self.shift))