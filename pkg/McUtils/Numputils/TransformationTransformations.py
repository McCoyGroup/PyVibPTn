import numpy as np

__all__ = [
    "merge_transformation_matrix"
    ]

def make_affine_matrix(mat, shift):
    """

    :param mat:
    :type mat: np.ndarray
    :param shift:
    :type shift: np.ndarray
    :return:
    :rtype:
    """
    shift = np.append(shift, [1])
    return np.column_stack(
            (
                np.row_stack(
                    (mat, np.zeros((1, 3)))
                ),
                shift
            )
        )
def merge_transformation_matrix(transf, other):
    """Merges two transformation matrices

    :param transf:
    :type transf: np.ndarray
    :param other:
    :type other: np.ndarray
    :return:
    :rtype:
    """

    other_shape = other.shape
    self_shape = transf.shape

    if other_shape[-1] == 4 and self_shape[-1] == 3:
        transf = make_affine_matrix(transf, np.array([0, 0, 0], dtype=float))
    elif other_shape[-1] == 3 and self_shape[-1] == 4:
        other = make_affine_matrix(other, np.array([0, 0, 0], dtype=float))
    elif other_shape[-1] == 3 and self_shape[-1] == 3:
        transf = make_affine_matrix(transf, np.array([0, 0, 0], dtype=float))
        other = make_affine_matrix(other, np.array([0, 0, 0], dtype=float))
    elif other_shape[-1] != 4 or self_shape[-1] != 4:
        raise ValueError("can't merge affine transforms with shape {} and {}".format(
            transf.shape,
            other.shape
        ))

    return np.dot(transf, other)