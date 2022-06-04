"""
Provides customized set operations based off of the NumPy builtins
to minimize things like excess sorts
"""

import numpy as np
from .Misc import flatten_dtype, unflatten_dtype, recast_permutation, recast_indices, downcast_index_array

__all__ = [
    'unique',
    'intersection',
    'contained',
    'difference',
    'find',
    'argsort',
    'group_by',
    'split_by_regions'
]

coerce_dtype = flatten_dtype
uncoerce_dtype = unflatten_dtype

def argsort(ar):
    ar = np.asanyarray(ar)
    if ar.ndim > 1:
        ar, _, _, _ = coerce_dtype(ar)
    return recast_permutation(np.argsort(ar, kind='mergesort'))

def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=0, sorting=None, minimal_dtype=False):
    """
    A variant on np.unique with default support for `axis=0` and sorting
    """

    ar = np.asanyarray(ar)
    if ar.ndim == 1:
        ret = unique1d(ar, return_index=return_index, return_inverse=return_inverse,
                       return_counts=return_counts, sorting=sorting, minimal_dtype=minimal_dtype)
        return ret

    # axis was specified and not None
    try:
        ar = np.moveaxis(ar, axis, 0)
    except np.AxisError:
        # this removes the "axis1" or "axis2" prefix from the error message
        raise np.AxisError(axis, ar.ndim)

    # coerce the data into the approrpriate shape
    consolidated, dtype, orig_shape, orig_dtype = coerce_dtype(ar)

    output = unique1d(consolidated,
                                return_index=return_index, return_inverse=return_inverse,
                                return_counts=return_counts, sorting=sorting, minimal_dtype=minimal_dtype)
    output = (uncoerce_dtype(output[0], orig_shape, orig_dtype, axis),) + output[1:]
    return output

def unique1d(ar, return_index=False, return_inverse=False,
              return_counts=False, sorting=None, minimal_dtype=False):
    """
    Find the unique elements of an array, ignoring shape.
    """
    ar = np.asanyarray(ar)

    if sorting is None:
        if minimal_dtype:
            sorting = recast_permutation(ar.argsort(kind='mergesort')) # we want to have stable sorts throughout
        else:
            sorting = ar.argsort(kind='mergesort') # we want to have stable sorts throughout
    ar = ar[sorting]

    mask = np.empty(ar.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = ar[1:] != ar[:-1]

    ret = (ar[mask], sorting)
    if return_index:
        ret += (sorting[mask],)
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[sorting] = imask
        ret += (inv_idx,)
    if return_counts:
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)
    return ret

def intersection(ar1, ar2,
                assume_unique=False, return_indices=False,
                sortings=None, union_sorting=None, minimal_dtype=False
                ):

    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)

    if ar1.dtype < ar2.dtype:
        ar1 = ar1.astype(ar2.dtype)
    elif ar1.dtype < ar2.dtype:
        ar2 = ar2.astype(ar1.dtype)

    if ar1.ndim == 1:
        ret = intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices,
                          sortings=sortings, union_sorting=union_sorting, minimal_dtype=minimal_dtype)
        return ret

    ar1, dtype, orig_shape1, orig_dtype1 = coerce_dtype(ar1)
    ar2, dtype, orig_shape2, orig_dtype2 = coerce_dtype(ar2, dtype=dtype)
    output = intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices,
                          sortings=sortings, union_sorting=union_sorting)
    output = (uncoerce_dtype(output[0], orig_shape1, orig_dtype1, None),) + output[1:]
    return output

def intersect1d(ar1, ar2,
                assume_unique=False, return_indices=False,
                sortings=None, union_sorting=None, minimal_dtype=False
                ):
    """
    Find the intersection of two arrays.

    """
    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)

    if not assume_unique:
        if return_indices:
            if sortings is not None:
                ar1, sorting1, ind1 = unique1d(ar1, return_index=True, sorting=sortings[0], minimal_dtype=minimal_dtype)
                ar2, sorting2, ind2 = unique1d(ar2, return_index=True, sorting=sortings[1], minimal_dtype=minimal_dtype)
            else:
                ar1, sorting1, ind1 = unique1d(ar1, return_index=True, minimal_dtype=minimal_dtype)
                ar2, sorting2, ind2 = unique1d(ar2, return_index=True, minimal_dtype=minimal_dtype)
        else:
            if sortings is not None:
                ar1, sorting1 = unique1d(ar1, sorting=sortings[0], minimal_dtype=minimal_dtype)
                ar2, sorting2 = unique1d(ar2, sorting=sortings[1], minimal_dtype=minimal_dtype)
            else:
                ar1, sorting1 = unique1d(ar1)
                ar2, sorting2 = unique1d(ar2)
        sortings = (sorting1, sorting2)

    aux = np.concatenate((ar1, ar2))
    if union_sorting is None:
        aux_sort_indices = np.argsort(aux, kind='mergesort')
        if minimal_dtype:
            aux_sort_indices = recast_permutation(aux_sort_indices)
        aux = aux[aux_sort_indices]
    else:
        aux_sort_indices = union_sorting
        aux = aux[aux_sort_indices]

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - ar1.size
        if not assume_unique:
            ar1_indices = ind1[ar1_indices]
            ar2_indices = ind2[ar2_indices]
        return int1d, sortings, union_sorting, ar1_indices, ar2_indices
    else:
        return int1d, sortings, union_sorting

def contained(ar1, ar2, assume_unique=False, invert=False,
                sortings=None, union_sorting=None, method=None):
    """
    Test whether each element of `ar1` is also present in `ar2`.
    """

    # Ravel both arrays, behavior for the first array could be different
    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)

    if ar1.dtype < ar2.dtype:
        ar1 = ar1.astype(ar2.dtype)
    elif ar2.dtype < ar1.dtype:
        ar2 = ar2.astype(ar1.dtype)

    if ar1.ndim > 1:
        ar1, dtype, orig_shape1, orig_dtype1 = coerce_dtype(ar1)
        ar2, dtype, orig_shape2, orig_dtype2 = coerce_dtype(ar2, dtype=dtype)

    # Check if one of the arrays may contain arbitrary objects
    contains_object = ar1.dtype.hasobject or ar2.dtype.hasobject

    # This code is run when
    # a) the first condition is true, making the code significantly faster
    # b) the second condition is true (i.e. `ar1` or `ar2` may contain
    #    arbitrary objects), since then sorting is not guaranteed to work
    if len(ar2) < 10 * len(ar1) ** 0.145 or contains_object:
        if invert:
            mask = np.ones(len(ar1), dtype=bool)
            for a in ar2:
                mask &= (ar1 != a)
        else:
            mask = np.zeros(len(ar1), dtype=bool)
            for a in ar2:
                mask |= (ar1 == a)
        return mask, sortings, union_sorting

    # Otherwise use sorting
    if assume_unique is not True and assume_unique is not False: # i.e. it's a bool
        assume_unique_1, assume_unique_2 = assume_unique
    else:
        assume_unique_1 = assume_unique_2 = assume_unique
    if not assume_unique_1:
        if sortings is None:
            ar1, sorting1, rev_idx = unique1d(ar1, return_inverse=True)
        else:
            ar1, sorting1, rev_idx = unique1d(ar1, sorting=sortings[0], return_inverse=True)
    else:
        if sortings is not None:
            sorting1 = sortings[0]
        else:
            sorting1 = None

    if not assume_unique_2:
        if sortings is None:
            ar2, sorting2 = unique1d(ar2)
        else:
            ar2, sorting2 = unique1d(ar2, sorting=sortings[1])
    else:
        if sortings is not None:
            sorting2 = sortings[1]
        else:
            sorting2 = None
    sortings = (sorting1, sorting2)

    if method is not None and method == 'find':
        find_pos, _ = find(ar2, ar1, sorting='sorted', check=False) #binary search is fast
        if invert:
            ret = ar2[find_pos] != ar1
        else:
            ret = ar2[find_pos] == ar1
        order = None
    else:
        ar = np.concatenate((ar1, ar2))
        # We need this to be a stable sort, so always use 'mergesort'
        # here. The values from the first array should always come before
        # the values from the second array.
        if union_sorting is None:
            order = ar.argsort(kind='mergesort')
        else:
            order = union_sorting
        sar = ar[order]
        if invert:
            bool_ar = (sar[1:] != sar[:-1])
        else:
            bool_ar = (sar[1:] == sar[:-1])
        flag = np.concatenate((bool_ar, [invert]))
        ret = np.empty(ar.shape, dtype=bool)
        ret[order] = flag

    if assume_unique_1:
        return ret[:len(ar1)], sortings, order
    else:
        return ret[rev_idx], sortings, order

def difference(ar1, ar2, assume_unique=False, sortings=None, method=None, union_sorting=None):
    """
    Calculates set differences over any shape of array
    """

    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)

    if ar1.dtype < ar2.dtype:
        ar1 = ar1.astype(ar2.dtype)
    elif ar2.dtype < ar1.dtype:
        ar2 = ar2.astype(ar1.dtype)

    if ar1.ndim == 1:
        ret = difference1d(ar1, ar2, assume_unique=assume_unique, method=method,
                          sortings=sortings, union_sorting=union_sorting)
        return ret

    ar1, dtype, orig_shape1, orig_dtype1 = coerce_dtype(ar1)
    ar2, dtype, orig_shape2, orig_dtype2 = coerce_dtype(ar2, dtype=dtype)
    output = difference1d(ar1, ar2, assume_unique=assume_unique, method=method,
                          sortings=sortings, union_sorting=union_sorting)
    output = (uncoerce_dtype(output[0], orig_shape1, orig_dtype1, None),) + output[1:]
    return output

def difference1d(ar1, ar2, assume_unique=False, sortings=None, method=None, union_sorting=None):
    """
    Calculates set differences in 1D
    """

    if not assume_unique:
        if sortings is not None:
            ar1, sorting1 = unique(ar1, sorting=sortings[0])
            ar2, sorting2 = unique(ar2, sorting=sortings[0])
        else:
            ar1, sorting1 = unique(ar1)
            ar2, sorting2 = unique(ar2)
        sortings = (sorting1, sorting2)

    in_spec = contained(ar1, ar2, sortings=sortings, union_sorting=union_sorting, assume_unique=True, method=method, invert=True)
    return (ar1[in_spec[0]],) + in_spec[1:]

def find1d(ar, to_find, sorting=None,
           search_space_sorting=None, return_search_space_sorting=False,
           check=True, minimal_dtype=False, missing_val='raise'
           ):
    """
    Finds elements in an array and returns sorting
    """

    presorted = isinstance(sorting, str) and sorting == 'sorted'
    if sorting is None:
        sorting = np.argsort(ar, kind='mergesort')

    if search_space_sorting is None and return_search_space_sorting:
        search_space_sorting = np.argsort(to_find, kind='mergesort')

    if search_space_sorting is not None:
        if isinstance(search_space_sorting, np.ndarray):
            search_space_inverse_sorting = np.argsort(search_space_sorting)
        else:
            search_space_sorting, search_space_inverse_sorting = search_space_sorting

        to_find = to_find[search_space_sorting]

    if presorted:
        vals = np.searchsorted(ar, to_find)
    else:
        vals = np.searchsorted(ar, to_find, sorter=sorting)
    if isinstance(vals, (np.integer, int)):
        vals = np.array([vals])
    # we have the ordering according to the _sorted_ version of `ar`
    # so now we need to invert that back to the unsorted version
    if len(sorting) > 0:
        big_vals = vals == len(ar)
        vals[big_vals] = -1
        if not presorted:
            vals = sorting[vals]
        if check:
            # now because of how searchsorted works, we need to check if the found values
            # truly agree with what we asked for
            bad_vals = ar[vals] != to_find
            if vals.shape == ():
                if bad_vals:
                    vals = -1
            else:
                # print(vals, bad_vals)
                vals[bad_vals] = -1
        else:
            bad_vals = big_vals

    else:
        bad_vals = np.full_like(to_find, True)
        vals = np.full_like(vals, -1)
    if check and bad_vals.any():
        if isinstance(missing_val, str) and missing_val == 'raise':
            raise IndexError("{} not in array".format(to_find[bad_vals]))
        else:
            vals[bad_vals] = missing_val

    if minimal_dtype and not bad_vals.any(): # protecting the missings
        vals = downcast_index_array(vals, ar.shape[-1])

    if search_space_sorting is not None:
        vals = vals[search_space_inverse_sorting]
    ret = (vals, sorting,)
    if return_search_space_sorting:
        ret += ((search_space_sorting, search_space_inverse_sorting),)
    return ret

def find(ar, to_find, sorting=None,
         search_space_sorting=None,
         return_search_space_sorting=False,
         check=True, minimal_dtype=False, missing_val='raise'):
    """
    Finds elements in an array and returns sorting
    """

    ar = np.asanyarray(ar)
    to_find = np.asanyarray(to_find)

    if ar.dtype < to_find.dtype:
        ar = ar.astype(to_find.dtype)
    elif to_find.dtype < ar.dtype:
        to_find = to_find.astype(ar.dtype)

    # print(ar.dtype, to_find.dtype )

    if ar.ndim == 1:
        ret = find1d(ar, to_find, sorting=sorting, check=check,
                     search_space_sorting=search_space_sorting,
                     return_search_space_sorting=return_search_space_sorting,
                     minimal_dtype=minimal_dtype, missing_val=missing_val
                     )
        return ret

    ar, dtype, orig_shape1, orig_dtype1 = coerce_dtype(ar)
    to_find, dtype, orig_shape2, orig_dtype2 = coerce_dtype(to_find, dtype=dtype)
    output = find1d(ar, to_find, sorting=sorting, check=check,
                     search_space_sorting=search_space_sorting,
                     return_search_space_sorting=return_search_space_sorting,
                     minimal_dtype=minimal_dtype, missing_val=missing_val
                    )
    return output

def group_by1d(ar, keys, sorting=None, return_indices=False):
    """
    Splits an array by a keys
    :param ar:
    :type ar:
    :param keys:
    :type keys:
    :param sorting:
    :type sorting:
    :return:
    :rtype:
    """

    uinds, sorting, mask = unique(keys, sorting=sorting, return_inverse=True)
    _, _, inds = unique(mask[sorting], sorting=np.arange(len(mask)), return_index=True)
    groups = np.split(ar[sorting,], inds)[1:]

    ret = ((uinds, groups), sorting)
    if return_indices:
        ret += (inds,)
    return ret

def group_by(ar, keys, sorting=None, return_indices=False):
    """
    Groups an array by keys

    :param ar:
    :type ar:
    :param keys:
    :type keys:
    :param sorting:
    :type sorting:
    :return: group pairs & sorting info
    :rtype:
    """

    ar = np.asanyarray(ar)
    keys = np.asanyarray(keys)

    if keys.ndim == 1:
        ret = group_by1d(ar, keys, sorting=sorting, return_indices=return_indices)
        return ret

    keys, dtype, orig_shape, orig_dtype = coerce_dtype(keys)
    output = group_by1d(ar, keys, sorting=sorting, return_indices=return_indices)
    ukeys, groups = output[0]
    ukeys = uncoerce_dtype(ukeys, orig_shape, orig_dtype, None)
    output = ((ukeys, groups),) + output[1:]
    return output

def split_by_regions1d(ar, regions, sortings=None, return_indices=False):
    """
    :param regions:
    :type regions:
    :param ar1:
    :type ar1:
    :return:
    :rtype:
    """

    if sortings is None:
        sortings = (None, None)
    ar_sorting, region_sorting = sortings
    if ar_sorting is None:
        ar_sorting = argsort(ar)
    ar = ar[ar_sorting]
    if region_sorting is None:
        region_sorting = argsort(regions)

    insertion_spots = np.searchsorted(regions, ar, sorter=region_sorting)
    uinds, _, inds = unique(insertion_spots, sorting=np.arange(len(insertion_spots)), return_index=True)
    groups = np.split(ar, inds)[1:]

    output = (uinds, groups)
    if return_indices:
        return output, inds, sortings
    else:
        return output, sortings

def split_by_regions(ar, regions, sortings=None, return_indices=False):
    """
    Splits an array up by edges defined by regions.
    Operates in 1D but can take compound dtypes using lexicographic
    ordering.
    In that case it is on the user to ensure that lex ordering is what is desired.
    :param ar:
    :type ar:
    :param regions:
    :type regions:
    :param sortings:
    :type sortings:
    :return:
    :rtype:
    """

    ar = np.asanyarray(ar)
    regions = np.asanyarray(regions)

    if ar.ndim == 1:
        ret = split_by_regions1d(regions, ar, sortings=sortings, return_indices=return_indices)
        return ret

    ar, dtype, orig_shape, orig_dtype = coerce_dtype(ar)
    regions, dtype, orig_shape1, orig_dtype1 = coerce_dtype(ar, dtype=dtype)
    output = split_by_regions1d(regions, ar, sortings=sortings, return_indices=return_indices)
    uinds, groups = output[0]
    groups = uncoerce_dtype(groups, orig_shape, orig_dtype, None)
    output = ((uinds, groups),) + output[1:]

    return output
