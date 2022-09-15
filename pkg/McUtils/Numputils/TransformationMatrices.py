
from .VectorOps import vec_normalize, vec_angles
import math, numpy as np

__all__ = [
    "rotation_matrix",
    "translation_matrix",
    "affine_matrix"
]

#######################################################################################################################
#
#                                                 rotation_matrix
#

def rotation_matrix_basic(xyz, theta):
    """rotation matrix about x, y, or z axis

    :param xyz: x, y, or z axis
    :type xyz: str
    :param theta: counter clockwise angle in radians
    :type theta: float
    """

    axis = xyz.lower()
    if axis == "z": # most common case so it comes first
        mat = [
            [ math.cos(theta), -math.sin(theta), 0.],
            [ math.sin(theta),  math.cos(theta), 0.],
            [ 0.,               0.,              1.]
        ]
    elif axis == "y":
        mat = [
            [ math.cos(theta), 0., -math.sin(theta)],
            [ 0.,              1.,               0.],
            [ math.sin(theta), 0.,  math.cos(theta)]
        ]
    elif axis == "x":
        mat = [
            [ 1.,               0.,               0.],
            [ 0.,  math.cos(theta), -math.sin(theta)],
            [ 0.,  math.sin(theta),  math.cos(theta)]
        ]
    else:
        raise Exception("{}: axis '{}' invalid".format('rotation_matrix_basic', xyz))
    return np.array(mat)

def rotation_matrix_basic_vec(xyz, thetas):
    """rotation matrix about x, y, or z axis

    :param xyz: x, y, or z axis
    :type xyz: str
    :param thetas: counter clockwise angle in radians
    :type thetas: float
    """

    thetas = np.asarray(thetas)
    nmats = len(thetas)
    z = np.zeros((nmats,))
    o = np.ones((nmats,))
    c = np.cos(thetas)
    s = np.sin(thetas)
    axis = xyz.lower()
    if axis == "z": # most common case so it comes first
        mat = [
            [ c, s, z],
            [-s, c, z],
            [ z, z, o]
        ]
    elif axis == "y":
        mat = [
            [ c, z, s],
            [ z, o, z],
            [-s, z, c]
        ]
    elif axis == "x":
        mat = [
            [ o, z, z],
            [ z, c, s],
            [ z,-s, c]
        ]
    else:
        raise Exception("{}: axis '{}' invalid".format('rotation_matrix_basic', xyz))
    return np.array(mat).T

#thank you SE for the nice Euler-Rodrigues imp: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
def rotation_matrix_ER(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad),     2 * (bd - ac)    ],
        [2 * (bc - ad),     aa + cc - bb - dd, 2 * (cd + ab)    ],
        [2 * (bd + ac),     2 * (cd - ab),     aa + dd - bb - cc]
    ])

def rotation_matrix_ER_vec(axes, thetas):
    """
    Vectorized version of basic ER
    """

    axes = np.asarray(axes)
    thetas = np.asarray(thetas)
    if len(axes.shape) == 1:
        axes = axes/np.linalg.norm(axes)
        axes = np.broadcast_to(axes, (len(thetas), 3))
    else:
        axes = vec_normalize(axes)

    a = np.cos(thetas/2.0)
    b, c, d = ( -axes * np.reshape(np.sin(thetas / 2.0), (len(thetas), 1)) ).T
    # raise Exception(axes)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad),     2 * (bd - ac)    ],
        [2 * (bc - ad),     aa + cc - bb - dd, 2 * (cd + ab)    ],
        [2 * (bd + ac),    2 * (cd - ab),     aa + dd - bb - cc]
    ]).T

def rotation_matrix_align_vectors(vec1, vec2):
    angles, normals = vec_angles(vec1, vec2)
    return rotation_matrix(normals, angles)

def rotation_matrix(axis, theta):
    """
    :param axis:
    :type axis:
    :param theta: angle to rotate by (or Euler angles)
    :type theta:
    :return:
    :rtype:
    """

    try:
        flen = len(theta)
    except TypeError:
        flen = 0
    extra_shape = None

    if type(axis) == str:
        if flen >0:
            mat_fun = rotation_matrix_basic_vec
        else:
            mat_fun = rotation_matrix_basic
    else:
        if flen > 0:
            axis = np.asanyarray(axis)
            theta = np.asanyarray(theta)

            if axis.ndim == theta.ndim:
                if theta.ndim > 2:
                    extra_shape = theta.shape[:-2]
                    axis = axis.reshape((-1, 3))
                    theta = theta.reshape((-1, 3))
                mat_fun = rotation_matrix_align_vectors
            else:
                if theta.ndim > 1:
                    extra_shape = theta.shape
                    axis = axis.reshape((-1, 3))
                    theta = theta.reshape(-1)
                mat_fun = rotation_matrix_ER_vec
        else:
            mat_fun = rotation_matrix_ER

    mats = mat_fun(axis, theta)
    if extra_shape is not None:
        mats = mats.reshape(extra_shape + (3, 3))
    return mats

#######################################################################################################################
#
#                                                 translation_matrix
#

def translation_matrix(shift):
    share = np.asarray(shift)
    if len(share.shape) == 1:
        ss = share
        zs = 0.
        os = 1.
        mat = np.array(
            [
                [os, zs, zs, ss[0]],
                [zs, os, zs, ss[1]],
                [zs, zs, os, ss[2]],
                [zs, zs, zs, os   ]
            ]
        )
    else:
        zs = np.zeros((share.shape[0],))
        os = np.ones((share.shape[0],))
        ss = share.T
        mat = np.array(
            [
                [os, zs, zs, ss[0]],
                [zs, os, zs, ss[1]],
                [zs, zs, os, ss[2]],
                [zs, zs, zs, os   ]
            ]
        ).T
    return mat

#######################################################################################################################
#
#                                                 affine_matrix
#

def affine_matrix(tmat, shift):
    """Creates an affine transformation matrix from a 3x3 transformation matrix or set of matrices and a shift or set of vecs

    :param tmat: base transformation matrices
    :type tmat: np.ndarray
    :param shift:
    :type shift:
    :return:
    :rtype:
    """

    base_mat = np.asanyarray(tmat)
    if shift is None:
        return base_mat

    if base_mat.ndim > 2:
        shifts = np.asanyarray(shift)
        if shifts.ndim == 1:
            shifts = np.broadcast_to(shifts, (1,)*(base_mat.ndim-2) + shifts.shape)
        shifts = np.broadcast_to(shifts, base_mat.shape[:-2] + (3,))
        shifts = np.expand_dims(shifts, -1)
        mat = np.concatenate([base_mat, shifts], axis=-1)
        padding = np.array([0., 0., 0., 1.])
        padding = np.broadcast_to(
            np.broadcast_to(padding, (1,)*(base_mat.ndim-2) + padding.shape),
            mat.shape[:-2] + (4,)
        )
        padding = np.expand_dims(padding, -2)
        mat = np.concatenate([mat, padding], axis=-2)
    else:
        mat = np.concatenate([base_mat, shift[:, np.newaxis]], axis=-1)
        mat = np.concatenate([mat, [[0., 0., 0., 1.]]], axis=-2)
    return mat