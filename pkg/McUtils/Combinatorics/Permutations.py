"""
Utilities for working with permutations and permutation indexing
"""

import numpy as np, time, typing, gc, itertools
# import collections, functools as ft
from ..Misc import jit, objmode, prange
from ..Numputils import flatten_dtype, unflatten_dtype, difference as set_difference, unique, contained, group_by, split_by_regions, find, infer_int_dtype
from ..Scaffolding import NullLogger

__all__ = [
    "IntegerPartitioner",
    "UniquePermutations",
    "IntegerPartitionPermutations",
    "SymmetricGroupGenerator",
    "CompleteSymmetricGroupSpace",
    "LatticePathGenerator",
    "PermutationRelationGraph"
]

_infer_dtype = infer_int_dtype

def _infer_nearest_pos_neg_dtype(og_dtype):
    if og_dtype == np.uint8:
        return np.int16
    elif og_dtype == np.uint16:
        return np.int32
    elif og_dtype == np.uint32:
        return np.int64
    elif og_dtype == np.uint64:
        return np.int64
    else:
        return og_dtype

def _as_pos_neg_dtype(ar):
    dt = _infer_nearest_pos_neg_dtype(ar.dtype)
    if dt != ar.dtype:
        return ar.astype(dt)
    else:
        return ar

# _infer_dtype = _infer_dtype#lambda why: 'int64' # makes my life a little easier right now...

class IntegerPartitioner:

    def __init__(self):
        raise NotImplementedError('{} is a singleton class'.format(
            type(self).__name__
        ))

    _partition_counts = None
    @classmethod
    def _manage_counts_array(cls, n, M, l):
        grew = False
        if cls._partition_counts is None:
            grew = True
            # we just initialize this 3D array to be the size we need
            cls._partition_counts = np.full((n, M, l), -1, dtype=int)

        if n > cls._partition_counts.shape[0]:
            grew = True
            # grow in size along this axis
            # we grow by 2X the amount we really need to
            # so as to amortize the cost of the concatenations (thank you Lyle)
            ext_shape = (
                            2 * (n - cls._partition_counts.shape[0]),
                        ) + cls._partition_counts.shape[1:]
            cls._partition_counts = np.concatenate([
                cls._partition_counts,
                np.full(ext_shape, -1, dtype=cls._partition_counts.dtype)
            ],
                axis=0
            )
        if M > cls._partition_counts.shape[1]:
            grew = True
            ext_shape = (
                cls._partition_counts.shape[0],
                2 * (M - cls._partition_counts.shape[1]),
                cls._partition_counts.shape[2]
            )
            cls._partition_counts = np.concatenate([
                cls._partition_counts,
                np.full(ext_shape, -1, dtype=cls._partition_counts.dtype)
            ],
                axis=1
            )
        if l > cls._partition_counts.shape[2]:
            grew = True
            ext_shape = (
                cls._partition_counts.shape[0],
                cls._partition_counts.shape[1],
                2 * (l - cls._partition_counts.shape[2])
            )
            cls._partition_counts = np.concatenate([
                cls._partition_counts,
                np.full(ext_shape, -1, dtype=cls._partition_counts.dtype)
            ],
                axis=2
            )

        if grew:
            for i in range(cls._partition_counts.shape[0]):
                cls._partition_counts[i, i:, 0] = 1
                cls._partition_counts[i, 0, i:] = 1

    @classmethod
    def count_partitions(cls, n, M=None, l=None, manage_counts=True, check=True):
        """
        Uses the recurrence relation written out here
        https://en.wikipedia.org/wiki/Partition_(number_theory)#Partitions_in_a_rectangle_and_Gaussian_binomial_coefficients
        We cache the terms as a 2D list-of-lists because we don't need this
        part of the code to be blazingly fast but would like repeats to not
        do unnecessary work (and because the memory cost is small...)

        :param n:
        :type n:
        :param M:
        :type M:
        :param l:
        :type l:
        :return:
        :rtype:
        """

        if isinstance(n, (int, np.integer)):
            # just simple checks that could cause stuff to break or be overly slow otherwise
            if l is None:
                l = n
            if M is None:
                M = n
            if l > n:
                l = n
            if M > n:
                M = n

            if n == 0 and M ==0 and l == 0:
                return 1
            if M <= 0 or l <= 0:
                return 0
            if l == 1 and M < n:
                return 0

            if manage_counts:
                cls._manage_counts_array(n, M, l)

            counts = cls._partition_counts[n-1, M-1, l-1]
            if counts < 0:
                t1 = cls.count_partitions(n, M, l-1, manage_counts=False)
                t2 = cls.count_partitions(n-l, M-1, l, manage_counts=False)
                counts = t1 + t2
                cls._partition_counts[n-1, M-1, l-1] = counts
        else:
            # we assume we got a numpy array of equal length for each term
            n = np.asanyarray(n)#.astype(int)
            M = np.asanyarray(M)#.astype(int)
            l = np.asanyarray(l)#.astype(int)

            too_long = np.where(l > n)
            if len(too_long) > 0:
                too_long = too_long[0]
                if len(too_long) > 0:
                    l = l.copy()
                    l[too_long] = n[too_long]

            too_big = np.where(M > n)
            if len(too_big) > 0:
                too_big = too_big[0]
                if len(too_big) > 0:
                    M = M.copy()
                    M[too_big] = n[too_big]

            if manage_counts:
                cls._manage_counts_array(np.max(n), np.max(M), np.max(l))


            should_be_1 = n == 0
            should_be_1[should_be_1] = M[should_be_1] == 0
            should_be_1[should_be_1] = l[should_be_1] == 0
            n[should_be_1] = 1
            M[should_be_1] = 1
            l[should_be_1] = 1
            counts = cls._partition_counts[n - 1, M - 1, l - 1]

            # I think this set of conditions might be overkill?
            # in any case we can make this faster by doing the inverse
            should_not_be_0 = l > 0
            should_not_be_0[should_not_be_0] = M[should_not_be_0] > 0
            should_not_be_0[should_not_be_0] = np.logical_or(
                l[should_not_be_0] != 1,
                M[should_not_be_0] >= n[should_not_be_0]
            )
            should_be_0 = np.logical_not(should_not_be_0)
            counts[should_be_0] = 0
            counts[should_be_1] = 1

            if check:
                needs_updates = np.where(counts < 0)

                if len(needs_updates) > 0:
                    needs_updates = needs_updates[0]
                    if len(needs_updates) > 0:
                        # wtf
                        n = n[needs_updates]
                        M = M[needs_updates]
                        l = l[needs_updates]
                        t1 = cls.count_partitions(n, M, l - 1, manage_counts=False)
                        t2 = cls.count_partitions(n - l, M - 1, l, manage_counts=False)
                        counts[needs_updates] = t1 + t2
                        cls._partition_counts[n - 1, M - 1, l - 1] = counts[needs_updates]

        return counts


    @classmethod
    def fill_counts(cls, n, M=None, l=None):
        """
        Fills all counts up to (n, M, l)
        :param n:
        :type n: int
        :param M:
        :type M:
        :param l:
        :type l:
        :return:
        :rtype:
        """
        if M is None:
            M = n
        elif M > n:
            M = n
        if l is None:
            l = n
        elif l > n:
            l = n

        for i in range(1, n+1):
            for m in range(1, M+1):
                for k in range(1, l+1):
                    cls.count_partitions(i, m, k)

    @classmethod
    def count_exact_length_partitions(cls, n, M, l, check=True):
        """
        Unexpectedly common thing to want and a non-obvious formula

        :param n:
        :type n:
        :param M:
        :type M:
        :param l:
        :type l:
        :return:
        :rtype:
        """
        return cls.count_partitions(n - l, M - 1, l, check=check)

    @classmethod
    def count_exact_length_partitions_in_range(cls, n, m, M, l, check=True):
        """
        Returns the partitions with  k > M but length exactly L

        :param n:
        :type n:
        :param M:
        :type M:
        :param l:
        :type l:
        :return:
        :rtype:
        """
        wat1 = cls.count_exact_length_partitions(n, m, l, check=check)
        wat2 = cls.count_exact_length_partitions(n, M, l, check=check)
        return wat1 - wat2

    # @classmethod
    # def count_
    #     return cls.count_partitions(sums - counts, sums - 1, counts) - cls.count_partitions(sums - counts, parts[:, i] - 1,
    #                                                                                  counts)

    @classmethod
    def partitions(cls, n, pad=False, return_lens = False, max_len=None, dtype=None):
        """
        Returns partitions in descending lexicographic order
        Adapted from Kelleher to return terms ordered by length and then second in descending
        lex order which while a computationally suboptimal is very natural for a mapping onto
        physical phenomena (and also it's easier for storage)

        :param n: integer to partition
        :type n: int
        :param return_len: whether to return the length or not
        :type return_len:
        :return:
        :rtype:
        """

        if max_len is None or max_len > n:
            max_len = n
        l = max_len

        if dtype is None:
            dtype = _infer_dtype(n)

        # total_partitions = cls.count_partitions(n)
        count_totals = np.array([cls.count_partitions(n, l=i+1) for i in range(l)])
        counts = np.concatenate([count_totals[:1], np.diff(count_totals)], axis=0)
        # count_totals = np.flip(count_totals)
        if pad:
            storage = np.zeros((count_totals[-1], l), dtype=dtype)
        else:
            storage = [np.zeros((c, i+1), dtype=dtype) for i,c in enumerate(counts)]
            # raise Exception([s.shape for s in storage], counts, count_totals)

        increments = np.zeros(l, dtype=dtype)

        partition = np.ones(n, dtype=dtype)
        partition[0] = n
        k = q = 0 # k tracks the length of the permutation and q tracks where we're writing
        if pad:
            storage[increments[k], :k+1] = partition[:k+1]
        else:
            storage[k][increments[k]] = partition[:k+1]
        increments[k] += 1
        for _ in range(1, cls.count_partitions(n)):
            # everytime we reach this line we have completed a "transition"
            # where we convert an element like a -> a - 1, 1
            # the number of such transitions is hard to calculate a priori?
            if partition[q] == 2:
                # this is a special optimized case where, since the array is initialized to 1
                # when the element we're trying to "transition" is equal to 2 we just set it back to 1
                # and increment the length of our partition
                k += 1
                partition[q] = 1
                q -= 1
            else:
                # perform the transition, decrementing the term at position q
                # and extending the array where necessary
                m = partition[q] - 1
                n1 = k - q + 1
                partition[q] = m
                if n1 >= m:
                    for n1 in range(n1, m-1, -m):
                        q += 1
                        partition[q] = m
                    n1 -= m
                if n1 == 0:
                    k = q
                else:
                    k = q + 1
                    if n1 > 1:
                        q += 1
                        partition[q] = n1

            if k < l-1:
                if pad:
                    storage[count_totals[k-1]+increments[k], :k+1] = partition[:k+1]
                else:
                    storage[k][increments[k]] = partition[:k+1]
                increments[k] += 1
            elif k == l-1:
                if pad:
                    storage[count_totals[k-1]+increments[k], :k+1] = partition[:k+1]
                else:
                    storage[k][increments[k]] = partition[:k+1]
                increments[k] += 1
                if increments[k] == counts[k]:
                    break

        if return_lens:
            if pad:
                lens = np.ones(count_totals[-1], dtype=dtype)
                for i, bounds in enumerate(zip(counts, counts[1:])):
                    a, b = bounds
                    lens[a:b] = i + 1
            else:
                lens = [np.full(c, i + 1, dtype=dtype) for i, c in enumerate(counts)]
            return lens, storage
        else:
            return storage

    @classmethod
    def partition_indices(cls, parts, sums=None, counts=None, check=True):
        """
        Provides a somewhat quick way to get the index of a set of
        integer partitions.
        Parts must be padded so that all parts are the same length.
        If the sums of the partitions are known ahead of time they may be passed
        Similarly if the numbers of non-zero elements in the partitions are known
        ahead of time they may _also_ be passed

        :param parts:
        :type parts: np.ndarray
        :param sums:
        :type sums: np.ndarray
        :return:
        :rtype:
        """

        parts = np.asanyarray(parts)#.astype('int16')
        if not isinstance(parts.dtype, np.integer):
            parts = parts.astype(int)
        smol = parts.ndim == 1
        if smol:
            parts = parts.reshape((1, parts.shape[0]))

        if sums is None:
            sums = np.sum(parts, axis=1)

        # need this to track
        if counts is None:
            counts = np.count_nonzero(parts, axis=1)

        # num_before = np.zeros(parts.shape[0], dtype=int) # where the indices will be in reverse-lex order
        inds = np.arange(parts.shape[0])
        if not check:
            woofs = counts > 1
            num_before = np.zeros(len(sums), dtype=int)
            if woofs.any():
                num_before[woofs] = cls.count_partitions(sums[woofs], sums[woofs], counts[woofs] - 1, check=check)
        else:
            num_before = cls.count_partitions(sums, sums, counts - 1, check=check)

        for i in range(np.max(counts) - 1): # exhaust all elements except the last one where contrib will always be zero
            # now we need to figure out where the counts are greater than 1
            mask = np.where(counts > 1)[0]
            if len(mask) > 0:
                counts = counts[mask]
                inds = inds[mask]
                sums = sums[mask]
                parts = parts[mask]
                if i > 0:
                    subsums = cls.count_exact_length_partitions_in_range(sums, parts[:, i-1], parts[:, i], counts, check=check)
                else:
                    subsums = cls.count_exact_length_partitions_in_range(sums, sums, parts[:, i], counts, check=check)
                #cls.count_partitions(sums - counts, sums - 1, counts) - cls.count_partitions(sums - counts, parts[:, i] - 1, counts)
                num_before[inds] += subsums
                counts -= 1
                sums -= parts[:, i]

        inds = num_before
        if smol:
            inds = inds[0]

        return inds

class UniquePermutations:
    """
    Provides permutations for a _single_ integer partition (very important)
    Also provides a classmethod interface to support the case
    where we don't want to instantiate a permutations object for every partition
    """
    def __init__(self, partition):
        self.part = np.flip(np.sort(partition))
        v, c = np.unique(partition, return_counts=True)
        self.vals = np.flip(v)
        self.counts = np.flip(c)
        self.dim = len(partition)
        self._num = None

    @classmethod
    def get_permutation_class_counts(cls, partition, sort_by_counts=False):
        v, c = np.unique(partition, return_counts=True)
        if sort_by_counts:
            s = np.argsort(c)
            v = v[s]
            c = c[s]
        else:
            v, c = np.flip(v), np.flip(c)
        return v, c

    @property
    def num_permutations(self):
        """
        Counts the number of unique permutations of the partition
        :param counts:
        :type counts:
        :return:
        :rtype:
        """
        if self._num is None:
            self._num = self.count_permutations(self.counts)
        return self._num

    @staticmethod
    def count_permutations(counts):
        """
        Counts the number of unique permutations of the given "counts"
        which correspond to the number of nodes in the unique permutation tree
        :param counts:
        :type counts:
        :return:
        :rtype:
        """
        import math

        subfac = np.prod([math.factorial(x) for x in counts])
        ndim_fac = math.factorial(np.sum(counts))

        return ndim_fac // subfac

    def permutations(self, initial_permutation=None, return_indices=False, num_perms=None):
        """
        Returns the permutations of the input array
        :param initial_permutation:
        :type initial_permutation:
        :param return_indices:
        :type return_indices:
        :param classes:
        :type classes:
        :param counts:
        :type counts:
        :param num_perms:
        :type num_perms:
        :return:
        :rtype:
        """

        if initial_permutation is None:
            initial_permutation = self.part
            if num_perms is None:
                num_perms = self.num_permutations
        else:
            if num_perms is None:
                num_perms = self.num_permutations - self.index_permutations(initial_permutation)

        return self.get_subsequent_permutations(initial_permutation, return_indices=return_indices, num_perms=num_perms)

    @classmethod
    def get_subsequent_permutations(cls, initial_permutation, return_indices=False, classes=None, counts=None, num_perms=None):
        """
        Returns the permutations of the input array
        :return:
        :rtype:
        """

        initial_permutation = np.asanyarray(initial_permutation)
        if num_perms is None:
            # need to determine where we start/how far we have to go
            if counts is None or classes is None:
                classes, counts = np.unique(initial_permutation, return_counts=True)
            total_perms = cls.count_permutations(counts)
            skipped = cls.get_permutation_indices(initial_permutation, classes, counts, num_permutations=total_perms)
            num_perms = total_perms - skipped

        dim = len(initial_permutation)

        storage = np.zeros((num_perms, dim), dtype=initial_permutation.dtype)
        if return_indices:
            inds = np.zeros((num_perms, dim), dtype=_infer_dtype(dim)) # this could potentially be narrower
        else:
            inds = None

        part = initial_permutation.copy()
        if initial_permutation.dtype != np.dtype(object):
            cls._fill_permutations_direct_jit(storage, inds, part, dim)
        else:
            cls._fill_permutations_direct(storage, inds, part, dim)

        if return_indices:
            return inds, storage
        else:
            return storage

    @staticmethod
    @jit(nopython=True, cache=True)
    def _fill_permutations_direct_jit(storage, inds, partition, dim):
        """
        Builds off of this algorithm for generating permutations
        in lexicographic order: https://en.wikipedia.org/wiki/Permutation#Generation_in_lexicographic_order
        Then we adapt it so that it works in _reverse_ lex order since that's how our partitions come in
        This adaption is done just by pretending the the numbers are all negated so all ordering relations
        flip

        :param storage:
        :type storage:
        :param inds:
        :type inds:
        :return:
        :rtype:
        """

        swap = np.arange(len(partition))
        for i in range(len(storage)):
            storage[i] = partition
            if inds is not None:
                inds[i] = swap

            # find largest index such that the next element in the
            # partition is smaller (i.e. find where we need to do our next swap)
            # I'd like to do this with numpy builtins instead of a loop
            # or maybe some partial ordering approach or something
            # but don't have it quite figured out yet
            for i in range(dim-2, -1, -1):
                if partition[i] > partition[i+1]:
                    break
            else:
                break

            # find the next-smallest index such that
            # the partition element is smaller than the one at the swap
            # position
            for j in range(dim-1, i, -1):
                if partition[i] > partition[j]:
                    break

            # swap the terms and reverse the sequence of elements leading
            # up to it
            tmp = partition[j]
            partition[j] = partition[i]
            partition[i] = tmp
            partition[i+1:] = np.flip(partition[i+1:])

            if inds is not None:
                tmp = swap[j]
                swap[j] = swap[i]
                swap[i] = tmp
                swap[i+1:] = np.flip(swap[i+1:])

        return storage

    @classmethod
    def _fill_permutations_direct(cls, storage, inds, partition, dim):
        """
        Builds off of this algorithm for generating permutations
        in lexicographic order: https://en.wikipedia.org/wiki/Permutation#Generation_in_lexicographic_order
        Then we adapt it so that it works in _reverse_ lex order since that's how our partitions come in
        This adaption is done just by pretending the the numbers are all negated so all ordering relations
        flip

        :param storage:
        :type storage:
        :param inds:
        :type inds:
        :return:
        :rtype:
        """

        swap = np.arange(len(partition))
        for i in range(len(storage)):
            storage[i] = partition
            if inds is not None:
                inds[i] = swap

            # find largest index such that the next element in the
            # partition is smaller (i.e. find where we need to do our next swap)
            # I'd like to do this with numpy builtins instead of a loop
            # or maybe some partial ordering approach or something
            # but don't have it quite figured out yet
            for i in range(dim-2, -1, -1):
                if partition[i] > partition[i+1]:
                    break
            else:
                break

            # find the next-smallest index such that
            # the partition element is smaller than the one at the swap
            # position
            for j in range(dim-1, i, -1):
                if partition[i] > partition[j]:
                    break

            # swap the terms and reverse the sequence of elements leading
            # up to it
            partition[(i, j),] = partition[(j, i),]
            partition[i+1:] = np.flip(partition[i+1:])

            if inds is not None:
                swap[(i, j),] = swap[(j, i),]
                swap[i+1:] = np.flip(swap[i+1:])

        return storage

    @staticmethod
    def _subtree_counts(total, ndim, counts, where):
        """
        Computes the number of states in the tree built from decrementing counts[where] by 1
        Is it trivially simple? Yes
        But there's power to having it be named.
        :param total:
        :type total:
        :param ndim:
        :type ndim:
        :param counts:
        :type counts:
        :param where:
        :type where:
        :return:
        :rtype:
        """
        mprod = total * counts[where]
        # if mprod % ndim != 0:
        #     raise ValueError("subtree counts {} don't comport with dimension {}".format(
        #         mprod, ndim
        #     ))
        return mprod // ndim#, dtype=int)

    @staticmethod
    def _reverse_subtree_counts(subtotal, ndim, counts, j):
        """
        Given subtotal = (total * counts[j]) // ndim
              total    = (subtotal * ndim) // counts[j]
        :return:
        :rtype:
        """

        return (subtotal * ndim) // counts[j]

    def index_permutations(self, perms, assume_sorted=False, preserve_ordering=True):
        """
        Gets permutations indices assuming all the data matches the held stuff
        :param perms:
        :type perms:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

        return self.get_permutation_indices(self.vals, self.counts, perms,
                                            assume_sorted=assume_sorted, preserve_ordering=preserve_ordering,
                                            dim=self.dim, num_permutations=self.num_permutations)

    @classmethod
    def get_next_permutation_from_prev(cls, classes, counts, class_map,
                                       ndim,
                                       cur,
                                       prev, prev_index,
                                       prev_dim,
                                       subtree_counts,
                                       ):
        """
        Pulls the next index by reusing as much info as possible from
        previous index
        Less able to be efficient than computing many indices at once so prefer that if
        possible

        :return:
        :rtype:
        """

        raise NotImplementedError("Haven't gotten this working yet")

        if prev is not None:
            diffs = np.not_equal(cur, prev)
            # we reuse as much work as we can by only backtracking where we need to
            agree_pos = np.where(diffs)[0][0]  # where the first disagreement occurs...I'd like this to be less inefficient
            num_diff = ndim - agree_pos  # number of differing states
            if num_diff == 0:  # same state so just reuse the previous value
                return prev_index, prev_dim

            elif agree_pos == 0:
                # no need to actually backtrack when we know what we're gonna get
                cur_dim = ndim - 1
            else:
                # at this point cur_dim gives us the number of trailing
                # digits that are equivalent in the previous permutation
                # so we only need to back-track to where the new state begins to
                # differ from the old one,
                for i in range(ndim - prev_dim - 2, agree_pos - 1, -1):
                    j = class_map[prev[i]]
                    counts[j] += 1
                    # tree_data[cur_dim, 1] = 0
                    # tree_data[cur_dim, 0] = 0
                    cur_dim += 1
                # I'm not sure why this didn't break earlier without this...
                tree_data[cur_dim, 0] = tree_data[cur_dim + 1, 0]

                state = state[agree_pos:]

        # we loop through the elements in the permutation and
        # add up number of elements in the subtree that would precede
        # the state in reverse-lexicographic order
        for i, el in enumerate(state):
            for j in range(nterms):
                if counts[j] == 0:
                    continue
                subtotal = cls._subtree_counts(tree_data[cur_dim, 1], cur_dim + 1, counts, j)
                if classes[j] == el:
                    cur_dim -= 1
                    counts[j] -= 1
                    tree_data[cur_dim, 1] = subtotal
                    tree_data[cur_dim, 0] = tree_data[cur_dim + 1, 0]
                    break
                else:
                    tree_data[cur_dim, 0] += subtotal

            # short circuit if we've gotten down to a terminal node where
            # there is just one unique element
            if tree_data[cur_dim, 1] == 1:
                break

        inds[sn] = tree_data[cur_dim, 0]

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _fill_permutation_indices(
            inds: np.ndarray,
            perms: np.ndarray,
            classes: np.ndarray,
            counts: np.ndarray,
            dim: int,
            num_permutations: int,
            block_size:int
    ):
        """
        JIT compiled
        :param inds:
        :type inds:
        :param perms:
        :type perms:
        :param diffs:
        :type diffs:
        :param counts:
        :type counts:
        :param counts_mask:
        :type counts_mask:
        :param init_counts:
        :type init_counts:
        :param tree_data:
        :type tree_data:
        :param num_permutations:
        :type num_permutations:
        :param ndim:
        :type ndim:
        :return:
        :rtype:
        """

        class_map = {}
        for i,v in enumerate(classes):
            class_map[v] = i

        init_counts = counts
        nterms = len(counts)

        ndim = dim

        n_steps = int(np.ceil(len(perms)/block_size))
        block_counts = np.zeros((n_steps, len(counts)), dtype=counts.dtype)
        for i in range(n_steps):
            block_counts[i] = counts

        block_tree_datas = np.zeros((n_steps, dim, 2), dtype=counts.dtype)
        cur_dims = np.full(n_steps, dim - 1)
        block_tree_datas[:, cur_dims[0], 1] = num_permutations

        half_dim = ndim // 2
        for _ in prange(n_steps):
            idx_start = block_size * _

            cur_dim = cur_dims[_]
            tree_data = block_tree_datas[_]
            cts = block_counts[_]

            rem_els = block_size if idx_start + block_size < len(perms) else len(perms) - idx_start
            for sn in range(rem_els):
                idx = idx_start + sn
                state = perms[idx]

                if sn > 0:
                    # we reuse as much work as we can by only backtracking where we need to
                    agree_pos = 0
                    for i in range(ndim):
                        if perms[idx, i] != perms[idx-1, i]:
                            break
                        agree_pos += 1

                    start_pos = ndim - cur_dim - 2
                    reverts = start_pos - agree_pos
                    num_diff = ndim - agree_pos  # number of differing states
                    if num_diff == 0:  # same state so just reuse the previous value
                        inds[idx] = inds[idx - 1]
                    elif agree_pos == 0 or reverts >= half_dim:
                        # faster _not_ to backtrack if we have to revert most of the way
                        cur_dim = ndim - 1
                        tree_data[cur_dim, 1] = num_permutations
                        tree_data[cur_dim, 0] = 0
                        for i in range(len(init_counts)):
                           cts[i] = init_counts[i]
                        # counts_mask[:] = True
                    else:
                        prev = perms[idx-1]
                        # at this point cur_dim gives us the number of trailing
                        # digits that are equivalent in the previous permutation
                        # so we only need to back-track to where the new state begins to
                        # differ from the old one,
                        for i in range(start_pos, agree_pos - 1, -1):
                            j = class_map[prev[i]]
                            cts[j] += 1
                            cur_dim += 1
                        tree_data[cur_dim, 0] = tree_data[cur_dim + 1, 0]

                        state = state[agree_pos:]
                    # cur_dim, state, num_diff = backtrack(sn, cur_dim, state)
                    if num_diff == 0:
                        continue
                # print(":", state)

                # we loop through the elements in the permutation and
                # add up number of elements in the subtree that would precede
                # the state in reverse-lexicographic order
                for i, el in enumerate(state):
                    for j in range(nterms):
                        if cts[j] == 0:
                            continue
                        mprod = tree_data[cur_dim, 1] * cts[j]
                        subtotal = mprod // (cur_dim + 1)  # , dtype=int)
                        if classes[j] == el:
                            cur_dim -= 1
                            cts[j] -= 1
                            tree_data[cur_dim, 1] = subtotal
                            tree_data[cur_dim, 0] = tree_data[cur_dim + 1, 0]
                            break
                        else:
                            tree_data[cur_dim, 0] += subtotal
                    # cur_dim = find_state(cur_dim, el)

                    # short circuit if we've gotten down to a terminal node where
                    # there is just one unique element
                    if tree_data[cur_dim, 1] == 1:
                        break

                inds[idx] = tree_data[cur_dim, 0]

                # print([_, cur_dim, tree_data[cur_dim, 0]])
        return inds#, sn

    @classmethod
    def get_permutation_indices(cls, perms, classes=None, counts=None, assume_sorted=False,
                                preserve_ordering=True, dim=None, num_permutations=None, dtype=None,
                                block_size=100
                                ):
        """
        Classmethod interface to get indices for permutations
        :param perms:
        :type perms:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

        perms = np.asanyarray(perms)

        if classes is None or counts is None:
            classes, counts = cls.get_permutation_class_counts(perms[0])
        classes = classes.astype(perms.dtype)

        smol = perms.ndim == 1
        if smol:
            assume_sorted = True
            perms = np.reshape(perms, (1, len(perms)))

        if not assume_sorted:
            sorting = np.lexsort(-np.flip(perms, axis=1).T)
            perms = perms[sorting,]
        else:
            sorting = None
        # tracks the number of prior nodes in the tree (first column)
        # and the number of total remaining permutations (used to calculate the first)
        if dim is None:
            dim = int(np.sum(counts))
        if num_permutations is None:
            num_permutations = cls.count_permutations(counts)
        if dtype is None:
            dtype = _infer_dtype(num_permutations)
        counts = counts.astype(dtype)

        # set up storage for indices
        inds = np.full((len(perms),), 0, dtype=dtype)

        inds = cls._fill_permutation_indices(
            inds,
            perms,
            classes,
            counts,
            dim,
            num_permutations,
            block_size
        )

        # if sn < len(perms)-1:
        #     raise ValueError("permutation at position {}, {}, tried to reused bad value from previous permutation {}".format(
        #         sn,
        #         perms[sn],
        #         perms[sn - 1]
        #     ))

        if preserve_ordering and sorting is not None:
            inds = inds[np.argsort(sorting)]
        elif smol:
            inds = inds[0]

        return inds

    @staticmethod
    @jit(nopython=True, cache=True)
    def _fill_permutations_from_indices(perms, indices, counts, classes, dim, num_permutations, block_size):

        # we make a constant-time lookup for what a value maps to in
        # terms of position in the counts array
        class_map = {}
        for i, v in enumerate(classes):
            class_map[v] = i
        init_counts = counts
        nterms = len(counts)

        n_steps = int(np.ceil(len(perms) / block_size))
        block_counts = np.zeros((n_steps, len(counts)), dtype=counts.dtype)
        for i in range(n_steps):
            block_counts[i] = counts

        for _ in prange(n_steps):
            start_idx = block_size * _

            tree_data = np.zeros((dim, 2), dtype=np.int64)
            depth = 0  # where we're currently writing
            tree_data[depth, 1] = num_permutations
            counts = block_counts[_]  # we're going to modify this in-place

            for sn, idx in enumerate(indices[start_idx:start_idx+block_size]):
                # we back track directly to where the sum of the subtotal and the current num_before
                # is greater than the target index
                if sn > 0:
                    sn += start_idx
                    if tree_data[depth, 0] == idx:
                        # we don't need to do any work
                        perms[sn] = perms[sn - 1]
                        continue
                    else:
                        tree_sums = tree_data[:depth, 0] + tree_data[1:depth+1, 1]
                        target_depth = np.where(tree_sums > idx)[0]
                        if len(target_depth) > 0:
                            target_depth = target_depth[-1]
                            prev = perms[sn - 1]
                            for d in range(depth-1, target_depth, -1):
                                inc_el = prev[d]
                                j = class_map[inc_el]
                                depth -= 1
                                counts[j] += 1
                            perms[sn, :depth] = prev[:depth]
                            tree_data[depth, 0] = tree_data[depth - 1, 0]
                        else:
                            # means we need to backtrack completely
                            # so why even walk?
                            depth = 0
                            counts = init_counts.copy()
                            tree_data[depth, 1] = num_permutations
                            tree_data[depth, 0] = 0
                else:
                    sn += start_idx

                done = False
                for i in range(depth, dim - 1):  # we only need to do at most cur_dim writes
                    # We'll get each element 1-by-1 in an O(d) fashion.
                    # This isn't blazingly fast but it'll work okay

                    # loop over the classes of elements and see at which point dropping an element exceeds the current index
                    # which tells is that the _previous_ term was the correct one
                    for j in range(nterms):
                        if counts[j] == 0:
                            continue

                        mprod = tree_data[depth, 1] * counts[j]
                        subtotal = mprod // (dim - depth)
                        test = tree_data[depth, 0] + subtotal
                        if test > idx:  # or j == nterms-1: there's got to be _some_ index at which we get past idx I think...
                            depth += 1
                            counts[j] -= 1
                            perms[sn, i] = classes[j]
                            tree_data[depth, 1] = subtotal
                            tree_data[depth, 0] = tree_data[depth - 1, 0]
                            if tree_data[depth, 0] == idx:
                                # we know that every next iteration will _also_ do an insertion
                                # so we can just do that all at once
                                remaining_counts = np.sum(counts)
                                insertion = np.zeros(remaining_counts, dtype=perms.dtype)
                                d = 0
                                for l in range(nterms):
                                    if counts[l] > 0:
                                        insertion[d:d + counts[l]] = np.full(counts[l], classes[l], dtype=perms.dtype)
                                        d += counts[l]
                                perms[sn, i + 1:] = insertion
                                done = True
                            break
                        else:
                            tree_data[depth, 0] = test

                    if done:
                        break


    @classmethod
    def get_permutations_from_indices(cls, classes, counts, indices, assume_sorted=False, preserve_ordering=True,
                                      dim=None, num_permutations=None, check_indices=True, no_backtracking=False,
                                      block_size=100
                                      ):
        """
        Classmethod interface to get permutations given a set of indices
        :param perms:
        :type perms:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

        smol = isinstance(indices, (int, np.integer))
        if smol:
            indices = [indices]
        indices = np.asanyarray(indices)

        if not assume_sorted:
            sorting = np.argsort(indices)
            indices = indices[sorting,]
        else:
            sorting = None

        # tracks the number of prior nodes in the tree (first column)
        # and the number of total remaining permutations (used to calculate the first)
        if dim is None:
            dim = int(np.sum(counts))
        if num_permutations is None:
            num_permutations = cls.count_permutations(counts)
        if check_indices:
            bad_spots = np.where(indices >= num_permutations)[0]
            if len(bad_spots) > 0:
                raise ValueError("Classes/counts {}/{} only supports {} permutations. Can't return permutations {}".format(
                    classes, counts, num_permutations,
                    indices[bad_spots]
                ))

        if np.any(classes < 0):
            max_term = np.max(np.abs(classes))
            perms = np.zeros((len(indices), dim), dtype=infer_int_dtype(max_term))
        else:
            max_term = np.max(np.abs(classes))
            perms = np.zeros((len(indices), dim), dtype=_infer_dtype(max_term))

        cls._fill_permutations_from_indices(perms, indices, counts, classes, dim, num_permutations, block_size)

        if preserve_ordering and sorting is not None:
            perms = perms[np.argsort(sorting)]
        elif smol:
            perms = perms[0]

        return perms

    def permutations_from_indices(self, indices, assume_sorted=False, preserve_ordering=True):
        """
        Gets permutations indices assuming all the data matches the held stuff
        :param perms:
        :type perms:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

        return self.get_permutations_from_indices(self.vals, self.counts, indices,
                                                  assume_sorted=assume_sorted, preserve_ordering=preserve_ordering,
                                                  dim=self.dim, num_permutations=self.num_permutations)

    @classmethod
    def get_standard_permutation(cls, counts, classes):
        return np.concatenate(
                                [np.full(counts[l], classes[l], dtype=classes.dtype) for l in range(len(counts)) if counts[l] > 0]
                            )

    # leaving this here so I remember how it works...
    @staticmethod
    @jit(nopython=True, cache=True)
    def _walk_perm_generator(counts, dim, num_permutations, indices, include_positions):

        tree_data = np.zeros((dim, 2), dtype=np.int64)
        depth = 0  # where we're currently writing
        tree_data[depth, 1] = num_permutations
        init_counts = counts
        classes = np.arange(len(counts))
        max_count = np.max(counts)
        pos_map = np.full((len(counts), max_count), -1, dtype=np.int64)
        counts = np.copy(counts)  # we're going to modify this in-place
        nterms = len(counts)
        perm = np.zeros(dim, dtype=np.int64)

        allow_bracktracking = False
        for idx in indices:
            # we back track directly to where the sum of the subtotal and the current num_before
            # is greater than the target index
            if allow_bracktracking:
                tree_sums = tree_data[:depth, 0] + tree_data[1:depth + 1, 1]
                target_depth = np.where(tree_sums >= idx)[0]
                if len(target_depth) > 0:
                    target_depth = target_depth[-1]
                    for d in range(depth - 1, target_depth, -1):
                        j = perm[d]
                        depth -= 1
                        counts[j] += 1
                    tree_data[depth, 0] = tree_data[depth - 1, 0]
                else:
                    # means we need to backtrack completely
                    # so why even walk?
                    depth = 0
                    counts = init_counts.copy()
                    tree_data[depth, 1] = num_permutations
                    tree_data[depth, 0] = 0
            else:
                allow_bracktracking = True

            done = False
            for i in range(depth, dim):  # we only need to do at most cur_dim writes
                # We'll get each element 1-by-1 in an O(d) fashion.
                # This isn't blazingly fast but it'll work okay

                # loop over the classes of elements and see at which point dropping an element exceeds the current index
                # which tells is that the _previous_ term was the correct one
                for j in range(nterms):
                    if counts[j] == 0:
                        continue

                    mprod = tree_data[depth, 1] * counts[j]
                    subtotal = mprod // (dim - depth)  # , dtype=int)
                    test = tree_data[depth, 0] + subtotal
                    if test > idx:  # or j == nterms-1: there's got to be _some_ index at which we get past idx I think...
                        if include_positions:
                            k = init_counts[j] - counts[j]
                            pos_map[j][k] = depth

                        depth += 1
                        counts[j] -= 1
                        perm[i] = classes[j]
                        tree_data[depth, 1] = subtotal
                        tree_data[depth, 0] = tree_data[depth - 1, 0]
                        if tree_data[depth, 0] == idx:
                            # we know that every next iteration will _also_ do an insertion
                            # so we can just do that all at once

                            remaining_counts = np.sum(counts)
                            insertion = np.zeros(remaining_counts, dtype=perm.dtype)
                            d = 0
                            for l in range(nterms):
                                if counts[l] > 0:
                                    insertion[d:d+counts[l]] = np.full(counts[l], classes[l], dtype=perm.dtype)
                                    d+=counts[l]

                            if include_positions:
                                d = depth
                                for l in range(nterms):
                                    if counts[l] > 0:  # we fill the rest of the pos_map block
                                        k = init_counts[l] - counts[l]  # how many have we already filled in
                                        pos_map[l][k:init_counts[l]] = d + np.arange(counts[l])  # and now fill in the rest
                                        d += counts[l]
                            perm[i + 1:] = insertion
                            done = True

                        break
                    else:
                        tree_data[depth, 0] = test

                if done:
                    with objmode():
                        UniquePermutations._tree_walk_callback(idx, perm, pos_map, counts, depth, tree_data)
                    break

    @classmethod
    def walk_permutation_tree(cls, counts, on_visit,
                              indices=None, dim=None, num_permutations=None,
                              include_positions=False
                              ):
        """
        Just a general purpose method that allows us to walk the permutation
        tree built from counts and apply a function every time a node is visited.
        This can be very powerful for building algorithms that need to consider every permutation of
        an object.

        :param perms:
        :type perms:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

        # We walk the counts tree.
        # This is adapted directly from the tree approach to
        # building permutations given a set of indices, except here we just loop over all
        # indices.

        # tracks the number of prior nodes in the tree (first column)
        # and the number of total remaining permutations (used to calculate the first)
        if dim is None:
            dim = int(np.sum(counts))
        tree_data = np.zeros((dim, 2), dtype=int)
        depth = 0  # where we're currently writing
        if num_permutations is None:
            num_permutations = cls.count_permutations(counts)
        tree_data[depth, 1] = num_permutations
        init_counts = counts
        classes = np.arange(len(counts))
        pos_map = [np.full(c, -1) for c in counts]
        counts = np.copy(counts)  # we're going to modify this in-place
        nterms = len(counts)

        perm = np.zeros(dim, dtype=infer_int_dtype(dim))

        if indices is None:
            indices = range(num_permutations)

        allow_bracktracking = False
        for idx in indices:
            # we back track directly to where the sum of the subtotal and the current num_before
            # is greater than the target index
            done = False
            if allow_bracktracking:
                if tree_data[depth, 0] == idx:
                    # we don't need to do any work
                    done = True
                else:
                    tree_sums = tree_data[:depth, 0] + tree_data[1:depth+1, 1]
                    target_depth = np.where(tree_sums > idx)[0]
                    if len(target_depth) > 0:
                        target_depth = np.max(target_depth)
                        for d in range(depth - 1, target_depth, -1):
                            j = perm[d]
                            depth -= 1
                            counts[j] += 1
                        tree_data[depth, 0] = tree_data[depth - 1, 0]
                    else:
                        # means we need to backtrack completely
                        # so why even walk?
                        depth = 0
                        counts = init_counts.copy()
                        tree_data[depth, 1] = num_permutations
                        tree_data[depth, 0] = 0

            else:
                allow_bracktracking = True
            if not done:
                for i in range(depth, dim-1):  # we only need to do at most cur_dim writes
                    # We'll get each element 1-by-1 in an O(d) fashion.
                    # This isn't blazingly fast but it'll work okay

                    # loop over the classes of elements and see at which point dropping an element exceeds the current index
                    # which tells is that the _previous_ term was the correct one
                    for j in range(nterms):
                        if counts[j] == 0:
                            continue

                        subtotal = cls._subtree_counts(tree_data[depth, 1], dim - depth, counts, j)
                        test = tree_data[depth, 0] + subtotal
                        if test > idx:  # or j == nterms-1: there's got to be _some_ index at which we get past idx I think...
                            if include_positions:
                                k = init_counts[j] - counts[j]
                                pos_map[j][k] = depth
                            depth += 1
                            counts[j] -= 1
                            perm[i] = classes[j]
                            tree_data[depth, 1] = subtotal
                            tree_data[depth, 0] = tree_data[depth - 1, 0]
                            if tree_data[depth, 0] == idx:
                                # we know that every next iteration will _also_ do an insertion
                                # so we can just do that all at once
                                insertion = np.concatenate(
                                    [np.full(counts[l], classes[l], dtype=perm.dtype) for l in range(nterms) if
                                     counts[l] > 0]
                                )
                                if include_positions:
                                    d = depth
                                    for l in range(nterms):
                                        if counts[l] > 0:  # we fill the rest of the pos_map block
                                            k = init_counts[l] - counts[l]  # how many have we already filled in
                                            pos_map[l][k:] = d + np.arange(counts[l])  # and now fill in the rest
                                            d += counts[l]
                                perm[i + 1:] = insertion
                                done = True
                            break
                        else:
                            tree_data[depth, 0] = test
                    if done:
                        on_visit(idx, perm, pos_map, counts, depth, tree_data)
                        break
            else:
                on_visit(idx, perm, pos_map, counts, depth, tree_data)

    @classmethod
    def descend_permutation_tree_indices(cls, perms, on_visit, classes=None, counts=None, dim=None, assume_sorted=False, num_permutations=None):
        """
        Not sure what to call this exactly, but given that `walk_permutation_tree` maps onto `permutations_from_indices`
        this is the counterpart that basically walks _down_ the way `permutation_indices` would.
        I guess this is basically a BFS type approach of something?

        :param perms:
        :type perms:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

        raise NotImplementedError("The idea is good but I don't need it yet")

        perms = np.asanyarray(perms)

        if classes is None or counts is None:
            classes, counts = cls.get_permutation_class_counts(perms[0])

        smol = perms.ndim == 1
        if smol:
            assume_sorted = True
            perms = np.reshape(perms, (1, len(perms)))

        if not assume_sorted:
            sorting = np.lexsort(-np.flip(perms, axis=1).T)
            perms = perms[sorting,]
        else:
            sorting = None

        # tracks the number of prior nodes in the tree (first column)
        # and the number of total remaining permutations (used to calculate the first)
        if dim is None:
            dim = int(np.sum(counts))
        tree_data = np.zeros((dim, 2), dtype=int)
        cur_dim = dim - 1
        if num_permutations is None:
            num_permutations = cls.count_permutations(counts)
        tree_data[cur_dim, 1] = num_permutations
        ndim = dim
        # we make a constant-time lookup for what a value maps to in
        # terms of position in the counts array
        class_map = {v: i for i, v in enumerate(classes)}
        init_counts = counts
        counts = np.copy(counts)  # we're going to modify this in-place
        nterms = len(counts)
        # stack = collections.deque()  # stack of current data to reuse

        # determine where each successive permutation differs so we can know how much to reuse
        permutation_indices = np.arange(ndim)
        diffs = np.not_equal(perms[:-1], perms[1:])
        for sn, state in enumerate(perms):
            if sn > 0:
                # we reuse as much work as we can by only backtracking where we need to
                agree_pos = np.where(diffs[sn - 1])[0][
                    0]  # where the first disagreement occurs...I'd like this to be less inefficient
                num_diff = ndim - agree_pos  # number of differing states
                if num_diff == 0:  # same state so just reuse the previous value
                    # if inds[sn - 1] == -1:
                    #     raise ValueError("permutation {} tried to reused bad value from permutation {}".format(
                    #         perms[sn], perms[sn - 1]
                    #     ))
                    # inds[sn] = inds[sn - 1]
                    continue
                elif agree_pos == 0:
                    # no need to actually backtrack when we know what we're gonna get
                    cur_dim = ndim - 1
                    tree_data[cur_dim, 1] = num_permutations
                    tree_data[cur_dim, 0] = 0
                    counts = init_counts.copy()
                else:
                    prev = perms[sn - 1]
                    # at this point cur_dim gives us the number of trailing
                    # digits that are equivalent in the previous permutation
                    # so we only need to back-track to where the new state begins to
                    # differ from the old one,
                    for i in range(ndim - cur_dim - 2, agree_pos - 1, -1):
                        j = class_map[prev[i]]
                        counts[j] += 1
                        cur_dim += 1
                    # I'm not sure why this didn't break earlier without this...
                    tree_data[cur_dim, 0] = tree_data[cur_dim + 1, 0]

                    state = state[agree_pos:]

            # we loop through the elements in the permutation and
            # add up number of elements in the subtree that would precede
            # the state in reverse-lexicographic order
            for i, el in enumerate(state):
                for j in range(nterms):
                    if counts[j] == 0:
                        continue
                    subtotal = cls._subtree_counts(tree_data[cur_dim, 1], cur_dim + 1, counts, j)
                    if classes[j] == el:
                        cur_dim -= 1
                        counts[j] -= 1
                        tree_data[cur_dim, 1] = subtotal
                        permutation_indices[cur_dim] = j
                        tree_data[cur_dim, 0] = tree_data[cur_dim + 1, 0]
                        break
                    else:
                        tree_data[cur_dim, 0] += subtotal

                # short circuit if we've gotten down to a terminal node where
                # there is just one unique element
                if tree_data[cur_dim, 1] == 1:
                    # now...we know what the permutation looks like in terms of the classes itself????
                    # except...I'm not totally sure how to put this
                    permutation_indices[cur_dim:] = np.arange(cur_dim, )
                    break

            on_visit(tree_data[cur_dim, 0], permutation_indices, tree_data)

class IntegerPartitionPermutations:
    """
    Provides tools for working with permutations of a given integer partition
    """
    def __init__(self, num, dim=None):
        self.int = num
        if dim is None:
            dim = num
            self.partitions = IntegerPartitioner.partitions(num, pad=True)
        else:
            if dim <= num:
                self.partitions = IntegerPartitioner.partitions(num, pad=True, max_len=dim)
            else:
                parts_basic = IntegerPartitioner.partitions(num, pad=True, max_len=num)
                self.partitions = np.concatenate(
                    [
                            parts_basic,
                            np.zeros((len(parts_basic), dim - num), dtype=parts_basic[0].dtype)
                        ],
                    axis=1
                )

        self.dim = dim

        self._class_counts = np.full(len(self.partitions), None, dtype=object)
        for i, x in enumerate(self.partitions):
            self._class_counts[i] = UniquePermutations.get_permutation_class_counts(x)
        self.partition_counts = np.array([UniquePermutations.count_permutations(x[1]) for x in self._class_counts])
        self._cumtotals = np.cumsum(np.concatenate([[0], self.partition_counts[:-1]]), axis=0)
        self._num_terms = np.sum(self.partition_counts)

    @property
    def num_elements(self):
        return self._num_terms

    def get_partition_permutations(self, return_indices=False, dtype=None, flatten=False):
        """


        :return:
        :rtype:
        """

        if dtype is not None:
            input_data = [
                (p.astype(dtype), (c[0].astype(dtype), c[1]))
                for p, c in zip(self.partitions, self._class_counts)
            ]
        else:
            input_data = zip(self.partitions, self._class_counts)

        basic = [UniquePermutations.get_subsequent_permutations(p,
                                                                return_indices=return_indices,
                                                                classes=c[0],
                                                                counts=c[1]
                                                               ) for p,c in input_data]
        if flatten:
            return np.concatenate(basic, axis=0)
        else:
            return basic

    def _get_partition_splits(self, perms,
                              assume_sorted=False,
                              assume_standard=False,
                              split_method='direct',
                              check_partition_counts=True
                              ):
        """

        :param perms:
        :type perms:
        :param split_method:
        :type split_method:
        :return:
        :rtype:
        """

        if assume_standard:
            partitions = perms
        else:
            partitions = np.flip(np.sort(perms, axis=1), axis=1)
        if len(perms) == 1: # special case
            inds = np.array([0])
            sorting = np.array([0])
            uinds = IntegerPartitioner.partition_indices(partitions, sums=np.full((1,), self.int), check=check_partition_counts)
            groups = [perms]
            splits = ((uinds, groups), sorting, inds)
        else:
            if split_method == '2D':
                if assume_sorted:
                    sorting = np.arange(len(partitions))
                else:
                    sorting = None
                udats, sorting, inds = group_by(perms, partitions, sorting=sorting, return_indices=True)
                subu, groups = udats
                uinds = IntegerPartitioner.partition_indices(subu, sums=np.full(len(subu), self.int), check=check_partition_counts)
                splits = ((uinds, groups), sorting, inds)
            else:
                if assume_sorted:
                    sorting = np.arange(len(partitions))
                else:
                    sorting = None
                partition_inds = IntegerPartitioner.partition_indices(partitions, sums=np.full(len(perms), self.int), check=check_partition_counts)
                splits = group_by(perms, partition_inds, sorting=sorting, return_indices=True)

        return splits

    def get_full_equivalence_class_data(self, perms, split_method='direct', assume_sorted=False, assume_standard=False,
                                        return_permutations=False,
                                        check_partition_counts=True
                                        ):
        """
        Returns the equivalence class data of the given permutations
        :param perms:
        :type perms:
        :param split_method:
        :type split_method:
        :return:
        :rtype:
        """

        # convert perms into their appropriate partitions
        # get the indices of those and then split
        splits, sorting, inds = self._get_partition_splits(perms,
                                                          assume_sorted=assume_sorted,
                                                          assume_standard=assume_standard,
                                                          split_method=split_method,
                                                          check_partition_counts=check_partition_counts)
        uinds, groups = splits
        partition_data = self._class_counts[uinds]

        if return_permutations:
            # these are the "permutations" in IntegerPartitionPermutations
            partition_sorting = np.argsort(np.argsort(-perms, axis=1), axis=1).astype(_infer_dtype(perms.shape[-1]))
            partition_groups = np.split(partition_sorting[sorting,], inds)[1:]
        else:
            partition_groups = None

        return uinds, partition_data, partition_groups, groups, sorting, self._cumtotals[uinds]

    def get_equivalence_classes(self, perms, split_method='direct',
                                assume_sorted=False,
                                return_permutations=True,
                                check_partition_counts=True
    ):
        """
        Returns the equivalence classes and permutations of the given permutations
        :param perms:
        :type perms:
        :param split_method:
        :type split_method:
        :return:
        :rtype:
        """

        _, partition_data, partition_groups, _, sorting, totals = self.get_full_equivalence_class_data(perms,
                                                                                                       assume_sorted=assume_sorted,
                                                                                                       split_method=split_method,
                                                                                                       return_permutations=return_permutations,
                                                                                                       check_partition_counts=check_partition_counts
                                                                                                       )

        return [(c[0], c[1], p) for c,p in zip(partition_data, partition_groups)], totals, sorting

    def get_partition_permutation_indices(self, perms,
                                          assume_sorted=False,
                                          preserve_ordering=True,
                                          assume_standard=False,
                                          check_partition_counts=True,
                                          dtype=None,
                                          split_method='direct'
                                          ):
        """
        Assumes the perms all add up to the stored int
        They're then grouped by partition index and finally
        Those are indexed

        :param perms:
        :type perms:
        :return:
        :rtype:
        """

        perms = np.asanyarray(perms)
        smol = perms.ndim == 1
        if smol:
            perms = perms[np.newaxis]

        uinds, partition_data, partition_groups, groups, sorting, totals = self.get_full_equivalence_class_data(perms,
                                                                                                                assume_sorted=assume_sorted,
                                                                                                                assume_standard=assume_standard,
                                                                                                                check_partition_counts=check_partition_counts,
                                                                                                                split_method=split_method)
        if assume_standard:
            subinds = [
                np.full(len(g), s, dtype=int)
                for d, g, s in zip(partition_data, groups, totals)
            ]
        else:
            ushifts = [
                UniquePermutations.get_permutation_indices(g, d[0], d[1], self.dim)
                for d, g in zip(partition_data, groups)
            ]
            if dtype is None:
                tets_val = np.max([np.max(s) for s in ushifts]) + np.max(totals) + 1
                dtype = _infer_dtype(tets_val)
            subinds = [
                (g.astype(dtype) + s) for g,s in zip(ushifts, totals)
            ]
            # for s in subinds:
            #     if np.any(s < 0):
            #         raise ValueError('overflow on {} from {}'.format(s, dtype))

        # raise Exception(groups, subinds)

        if smol:
            inds = subinds[0][0]
        else:
            subinds = np.concatenate(subinds, axis=0)
            if preserve_ordering and sorting is not None:
                inv = np.argsort(sorting)
                inds = subinds[inv]
            else:
                inds = subinds

        return inds

    def get_partition_permutations_from_indices(self, indices, assume_sorted=False, preserve_ordering=True):
        """
        Assumes the perms all add up to the stored int
        They're then grouped by partition index and finally
        Those are indexed

        :param perms:
        :type perms:
        :return:
        :rtype:
        """

        smol = isinstance(indices, (int, np.integer))
        if smol:
            assume_sorted = True
            indices = np.array([smol])

        # we sort the indices and then use that to split them into groups
        # by comparing them to the `_cumsums`

        if assume_sorted:
            sorting = np.arange(len(indices))
        else:
            sorting = None

        splits, sortings = split_by_regions(self._cumtotals - 1, indices, sortings=(sorting, np.arange(len(self._cumtotals))))
        sorting, _ = sortings
        uinds, groups = splits
        uinds = uinds - 1

        shifted_groups = [g - s for g,s in zip(groups, self._cumtotals[uinds])]

        partition_data = self._class_counts[uinds]
        perms = [
            UniquePermutations.get_permutations_from_indices(d[0], d[1], g, self.dim)
            for d, g in zip(partition_data, shifted_groups)
        ]

        cats = np.concatenate(perms, axis=0)
        if preserve_ordering and sorting is not None:
            inv = np.argsort(sorting)
            cats = cats[inv]

        return cats

    def __repr__(self):
        return "{}({}, dim={}, nels={})".format(type(self).__name__, self.int, self.dim, self.num_elements)

class EmptyIntegerPartitionPermutations(IntegerPartitionPermutations):

    def __init__(self, num, dim=None):
        self.int = num
        if dim is None:
            dim = 1

        self.dim = dim
        self._class_counts = np.full(1, None, dtype=object)
        self._class_counts[0] = [np.array([0], dtype='int8'), np.array([dim], dtype='int8')]
        self.partition_counts = np.array([], dtype='int8')
        self._cumtotals = np.array([0, 1])
        self._num_terms = 1

    def get_partition_permutations(self, return_indices=False, dtype=None):
        """


        :return:
        :rtype:
        """

        return [np.zeros((1, self.dim), dtype='int8' if dtype is None else dtype)]

    def get_partition_permutation_indices(self, perms,
                                          assume_sorted=None,
                                          preserve_ordering=None,
                                          assume_standard=None,
                                          check_partition_counts=None,
                                          dtype=None,
                                          split_method=None
                                          ):
        """

        :param perms:
        :type perms:
        :param split_method:
        :type split_method:
        :return:
        :rtype:
        """
        woof = np.asanyarray(perms)
        if woof.ndim == 1:
            return 0
        else:
            return np.zeros(len(woof), dtype='int8' if dtype is None else dtype)

    def get_partition_permutations_from_indices(self, indices,
                                                assume_sorted=None,
                                                preserve_ordering=None
                                                ):
        """

        :param indices:
        :type indices:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

        if isinstance(indices, (np.integer, )):
            return np.zeros(self.dim, dtype='int8')
        else:
            return np.zeros((len(indices), self.dim), dtype='int8')

    def _get_partition_splits(self, perms,
                              assume_sorted=None,
                              assume_standard=None,
                              check_partition_counts=None,
                              split_method=None
                              ):
        """

        :param perms:
        :type perms:
        :param split_method:
        :type split_method:
        :return:
        :rtype:
        """

        uinds = [0]
        inds = [0]
        sorting = np.arange(len(perms))
        groups = [perms]

        return (uinds, groups), sorting, inds

class SymmetricGroupGenerator:
    """
    I don't know what to call this.
    Manages elements of the symmetric group up to arbitrary size.
    Basically just exists to merge all of the prior integer partition/permutation stuff over many integers
    which makes it easier to calculate direct products of terms
    """

    def __init__(self, dim):
        """
        :param dim: the padding length of every term (needed for consistency reasons)
        :type dim: int
        """

        self.dim = dim
        self._partition_permutations = [EmptyIntegerPartitionPermutations(0, dim=self.dim)] #type: list[IntegerPartitionPermutations]
        self._counts = [1] #type: list[int]
        self._cumtotals = np.array([0])

    def __repr__(self):
        return "{}({})".format(type(self).__name__, self.dim)

    def _get_partition_perms(self, iterable, ignore_negatives=False):
        """
        returns IntegerPartitionPermutation objects and cum totals for the provided quanta

        :param iterable:
        :type iterable:
        :return:
        :rtype: Tuple[List[IntegerPartitionPermutations], List]
        """

        inds = list(iterable)
        if len(inds) == 0:
            return [], []
        max_n = max(inds)
        min_n = min(inds)
        if not ignore_negatives and min_n < 0:
            raise ValueError("can't deal with partitions for negative integers (got min {})".format(min_n))

        if max_n >= len(self._partition_permutations):
            new_stuff = [IntegerPartitionPermutations(n, dim=self.dim) for n in range(len(self._partition_permutations), max_n+1)]
            self._partition_permutations = self._partition_permutations + new_stuff #type: list[IntegerPartitionPermutations]
            new_counts = [x.num_elements for x in new_stuff]
            self._counts = self._counts + new_counts
            self._cumtotals = np.concatenate([[0], np.cumsum(self._counts)])

        if ignore_negatives:
            t1 = [self._partition_permutations[n] if n >= 0 else None for n in inds]  # type: list[IntegerPartitionPermutations]
            t2 = [self._cumtotals[n] if n >= 0 else None for n in inds]  # type: list[int]
        else:
            t1 = [self._partition_permutations[n] for n in inds] #type: list[IntegerPartitionPermutations]
            t2 = [self._cumtotals[n] for n in inds] #type: list[int]

        return t1, t2

    def load_to_size(self, size):
        while self._cumtotals[-1] < size:
            # fills stuff up until we have everything covered
            self._get_partition_perms([1+len(self._partition_permutations)*2])

    def get_terms(self, n, flatten=True):
        """
        Returns permutations of partitions
        :param n:
        :type n:
        :return:
        :rtype:
        """

        if isinstance(n, (int, np.integer)):
            n = [n]

        partitioners, counts = self._get_partition_perms(n)
        perms = [wat.get_partition_permutations() for wat in partitioners]
        if flatten:
            perms = np.concatenate([np.concatenate(x, axis=0) for x in perms], axis=0)
        return perms

    def to_indices(self, perms, sums=None, assume_sorted=False, assume_standard=False,
                   check_partition_counts=True,
                   preserve_ordering=True,
                   dtype=None):
        """
        Gets the indices for the given permutations.
        First splits by sum then allows the held integer partitioners to do the rest
        :param perms:
        :type perms:
        :return:
        :rtype:
        """

        if len(perms) == 0:
            return np.array([], dtype='int8')

        perms = np.asanyarray(perms)
        smol = perms.ndim == 1
        if smol:
            assume_sorted = True
            perms = perms[np.newaxis]

        big_shp = perms.shape[:-1]
        if perms.ndim > 2:
            perms = np.reshape(perms, (-1, perms.shape[-1]))

        if sums is None:
            sums = np.sum(perms, axis=1)

        if not assume_sorted and len(perms) > 1:
            sorting = np.argsort(sums)
            sums = sums[sorting]
            perms = perms[sorting]
        else:
            sorting = None

        usums, _, inds = unique(sums, sorting=np.arange(len(sums)), return_index=True)
        groups = np.split(perms, inds)[1:]

        partitioners, shifts = self._get_partition_perms(usums)
        perms_inds = [p.get_partition_permutation_indices(g,
                                                 assume_standard=assume_standard,
                                                 preserve_ordering=preserve_ordering,
                                                 check_partition_counts=check_partition_counts) for p, g in zip(partitioners, groups) ]
        if dtype is None:
        #     for i,p in enumerate(perms_inds):
        #         if np.any(p < 0):
        #             raise ValueError("dtype overflow on {}".format(groups[i]))
            dtype = _infer_dtype(np.max([np.max(p) for p in perms_inds]) + np.max(shifts))
        subinds = [ (g.astype(dtype) + s) for g,s in zip(perms_inds, shifts) ]
        indices = np.concatenate(subinds, axis=0)

        if preserve_ordering and sorting is not None:
            indices = indices[np.argsort(sorting)]

        if len(big_shp) > 1:
            indices = np.reshape(indices, big_shp)

        return indices

    def from_indices(self, indices, assume_sorted=False, preserve_ordering=True):
        """
        Gets the permutations for the given indices.
        First splits into by which integer partitioner is the generator and lets
        the partitioner do the rest

        :param perms:
        :type perms:
        :return:
        :rtype:
        """

        smol = isinstance(indices, (int, np.integer))
        if smol:
            indices = np.array([smol])
        else:
            indices = np.asanyarray(indices)

        big_shp = indices.shape
        if indices.ndim > 1:
            indices = np.reshape(indices, -1)

        if not assume_sorted:
            sorting = np.argsort(indices)
            indices = indices[sorting]
        else:
            sorting = None

        if len(indices) == 0:
            return np.array([], dtype='int8')

        max_ind = np.max(indices)
        while self._cumtotals[-1] < max_ind:
            # fills stuff up until we have everything covered
            self._get_partition_perms([1+len(self._partition_permutations)*2])

        insertion_spots = np.searchsorted(self._cumtotals - 1, indices, sorter=np.arange(len(self._cumtotals)))
        uinds, _, inds = unique(insertion_spots, sorting=np.arange(len(insertion_spots)), return_index=True)
        groups = np.split(indices, inds)[1:]
        uinds = uinds - 1

        partitioners, shifts = self._get_partition_perms(uinds)

        shifted_groups = [g - s for g,s in zip(groups, shifts)]
        # raise Exception(partitioners, shifted_groups, shifts)
        # wat = partitioners[0] # type: IntegerPartitionPermutations
        perms = np.concatenate([ p.get_partition_permutations_from_indices(g, preserve_ordering=preserve_ordering) for p,g in zip(partitioners, shifted_groups)], axis=0)

        if preserve_ordering and sorting is not None:
            perms = perms[np.argsort(sorting),]

        if len(big_shp) > 1:
            perms = perms.reshape(big_shp + (-1,))

        return perms

    class direct_sum_filter:
        def __init__(self, perms, inds):

            self.perms = perms
            self.inds = unique(inds)[0]
            self.ind_sort = np.arange(len(self.inds))

            if perms is not None:
                all_sums = np.sum(perms, axis=1)
                ind_grps, _ = group_by(inds, all_sums)
                self.sums, ind_grps = ind_grps
                self.ind_grps = {s:unique(i)[0] for s,i in zip(self.sums, ind_grps)}
            else:
                self.sums = None
                self.ind_grps = None

        @classmethod
        def from_perms(cls, parent, filter_perms):

            filter_perms = filter_perms
            filter_inds = parent.to_indices(filter_perms)

            return cls(filter_perms, filter_inds)

        @classmethod
        def from_inds(cls, inds):

            return cls(None, inds)

        @classmethod
        def from_data(cls, parent, filter_perms):

            if filter_perms is None or isinstance(filter_perms, cls):
                return filter_perms

            # filter_perms should be allowed to contain both the permutations
            # and the indices to filter on so that I can not only compute the
            # sums to filter against as a first step but also the counts to filter against
            # before finally dropping back on the indices themselves
            # I can also take the _sorted_ version of the filter perms so that I can
            # reuse it multiple times

            if not isinstance(filter_perms, np.ndarray):
                # we infer the structure so we can coerce it back to normal
                if isinstance(filter_perms[0], (int, np.integer)) or isinstance(filter_perms[0][0], (int, np.integer)):
                    # got indices
                    filter_perms = np.array(filter_perms)
                elif len(filter_perms) == 1:
                    # means we only got one of the perm elements
                    filter_perms = np.array(filter_perms[0])
                elif len(filter_perms) == 2:
                    filter_perms = tuple(np.asanyarray(o) if o is not None else o for o in filter_perms)
                else:
                    filter_perms = np.array(filter_perms)
                    if filter_perms.dtype == object:
                        filter_perms = tuple(np.asanyarray(o) if o is not None else o for o in filter_perms)

            if isinstance(filter_perms, np.ndarray):
                if filter_perms.ndim == 1:
                    return cls.from_inds(filter_perms)
                elif filter_perms.ndim == 3:
                    return cls.from_perms(parent, filter_perms[0])
                else:
                    return cls.from_perms(parent, filter_perms)
            elif len(filter_perms) == 1:
                if filter_perms[0].ndim == 1:
                    return cls.from_inds(filter_perms[0])
                else:
                    return cls.from_perms(parent, filter_perms[0])
            elif len(filter_perms) == 2:
                return cls(*filter_perms)
            else:
                raise NotImplementedError("Unsure how to use filter spec {}".format(filter_perms))

    # TODO: destructure _build_direct_sums into a series of jittable parts

    # from memory_profiler import profile
    # @profile
    @staticmethod
    @jit(nopython=True, cache=True)
    def _get_filter_mask(new_rep_perm, cls_inds, can_be_negative, class_negs):
        # if we run into negatives we need to mask them out
        not_negs = np.full(len(cls_inds), True)#, dtype=bool)
        for j in can_be_negative:
            all_clean = np.all(new_rep_perm[j][class_negs[j],] >= 0)
            not_negs[j] = all_clean
        return not_negs

    @staticmethod
    @jit(nopython=True, cache=True)
    def _filter_negs_by_comp(comp, not_negs, idx, idx_starts, mask, perm_counts, start, end):
        not_sel = np.where(np.logical_not(not_negs))[0]
        mask_inds = np.reshape(
            np.expand_dims(not_sel, 1) +
            np.expand_dims(idx_starts, 0),
            -1)
        for k in mask_inds:
            mask[k] = False
        perm_counts[start:end, idx] -= len(not_negs) - len(comp)

    @classmethod
    def _filter_negatives_perms(cls,
                                i, idx, idx_starts, perms, new_rep_perm,
                                storage,
                                ndim,
                                cls_inds, class_negs,
                                perm_counts, cum_counts,
                                mask, can_be_negative
                                ):
        # if we run into negatives we need to mask them out
        if len(can_be_negative[i]) == 0:
            not_negs = np.full(len(cls_inds[i]), True)
        else:
            not_negs = cls._get_filter_mask(new_rep_perm, cls_inds[i], can_be_negative[i], class_negs)

        # try:
        comp = cls_inds[i][not_negs]
        # except:
        #     raise Exception(cls_inds[i], not_negs)
        if len(comp) < len(cls_inds[i]):
            cls._filter_negs_by_comp(comp, not_negs, idx, idx_starts, mask, perm_counts, cum_counts[i],
                                      cum_counts[i + 1])
        if len(comp) > 0:
            sel = np.where(not_negs)[0]
            new_perms = new_rep_perm[sel[:, np.newaxis, np.newaxis], perms[np.newaxis, :, :]]
            stored_inds = np.reshape(sel[:, np.newaxis] + idx_starts[np.newaxis, :], -1)
            storage[stored_inds] = new_perms.reshape(-1, ndim)
        else:
            sel = []
            new_perms = None

        return comp, sel, new_perms

    # def filter_from_ind_spec(i, j, block_idx, block_sizes, insert_inds, full_inds_sorted, inv, *,
    #                          mask=mask, merged_sums=merged_sums, filter=filter):
    #

    @staticmethod
    def _get_standard_perms(perms):
        classes_count_data = [
            UniquePermutations.get_permutation_class_counts(p) for p in perms
        ]
        standard_rep_perms = np.array([
            UniquePermutations.get_standard_permutation(c[1], c[0]) for c in classes_count_data
        ], dtype=_infer_dtype(np.max(np.concatenate([x[0] for x in classes_count_data])))
        )
        return classes_count_data, standard_rep_perms

    @classmethod
    def _process_cached_index_blocks(cls, storage, cache, paritioners, indices,
                                    filter, mask, perm_counts, merged_sums,
                                    inds_dtype=None, full_basis=None):
        for k, block_dat in cache.items():
            classes_count_data = [np.array(x) for x in k]
            standard_rep_perms = block_dat['standards']
            i, j = block_dat['indices']
            perm_pos = np.concatenate(block_dat['storage_blocks'])
            new_perms = storage[perm_pos]

            # if full_basis is not None:
            #     full_inds = full_basis.find(new_perms)
            #     sorting = np.argsort(full_inds)
            #     inv = np.argsort(sorting)
            #     full_inds_sorted = full_inds[sorting]
            # else:
                # padding_1, padding_2 = get_standard_perm_offsets(i, j, standard_rep_perms, classes_count_data)
            padding_1 = paritioners[i][1][j]
            padding_2 = paritioners[i][0][j].get_partition_permutation_indices([standard_rep_perms],
                                                                               assume_standard=True
                                                                               , check_partition_counts=False
                                                                               , dtype=inds_dtype
                                                                               )
            sorting = np.lexsort(-new_perms.T)
            inv = np.argsort(sorting)
            sort_perms = new_perms[sorting,]
            new_inds = UniquePermutations.get_permutation_indices(sort_perms,
                                                         classes=classes_count_data[0],
                                                         counts=classes_count_data[1]
                                                         , assume_sorted=True
                                                         , dtype=inds_dtype
                                                         )

            full_inds_sorted = padding_1 + padding_2 + new_inds

            indices[perm_pos] = full_inds_sorted[inv]
            if filter is not None:
                sort_1 = np.arange(len(full_inds_sorted))  # assured sorting from before
                if filter.ind_grps is not None:
                    subinds = filter.ind_grps[merged_sums[i, j]]
                    sort_2 = np.arange(len(subinds))
                    submask, _, _ = contained(full_inds_sorted, subinds,
                                              assume_unique=(False, True),
                                              sortings=(sort_1, sort_2))
                else:
                    submask, _, _ = contained(full_inds_sorted, filter.inds,
                                              assume_unique=(False, True),
                                              sortings=(sort_1, filter.ind_sort))

                sort_mask = submask[inv]
                mask[perm_pos] = sort_mask
                # 'storage_blocks': [stored_inds],
                block_sizes = np.cumsum([len(x) for x in block_dat['storage_blocks']])
                for b, s in zip(block_dat['idx_blocs'], np.split(sort_mask, block_sizes)):
                    perm_counts[b[0]:b[1], b[2]] -= 1 - s

                # filter_from_ind_spec(i, j, block_idx, block_sizes, perm_pos, full_inds_sorted, inv)

    def _build_direct_sums(self, input_perm_classes, counts, classes,
                           return_indices=False,
                           return_excitations=True,
                           filter_negatives=True,
                           allow_widen_dtypes=True,
                           filter=None, inds_dtype=None,
                           full_basis=None
                           ):
        """
        Creates direct sums of `input_perm_classes` with the unique permutations of `classes` where
        each of the classes has the same counts (just so we don't have to walk the tree as much)
        The `input_perm_classes` are tuples like `(classes, counts, perms)` where the perms are sorted
        which ensures that every subsequent addition is _also_ sorted

        :param input_perm_classes:
        :type perms:
        :param counts:
        :type counts:
        :param classes:
        :type classes:
        :return:
        :rtype:
        """

        input_perm_classes = [(_as_pos_neg_dtype(x[0]), _as_pos_neg_dtype(x[1]), x[2],
                               UniquePermutations.get_standard_permutation(_as_pos_neg_dtype(x[1]), _as_pos_neg_dtype(x[0]))
                               ) for x in input_perm_classes]

        # set up storage
        num_perms = UniquePermutations.count_permutations(counts)

        if inds_dtype is None:
            inds_dtype = int

        perm_counts = [len(x[2]) for x in input_perm_classes]
        cum_counts = np.cumsum([0] + perm_counts)
        total_perm_count = np.sum(perm_counts)
        ndim = len(input_perm_classes[0][2][0])

        if return_indices and not filter_negatives:
            raise NotImplementedError("Can't return indices without filtering negatives... (as I've currently implemented things)")

        dropped_pairs = [] # set up storage for when things go negative

        # precompute a bunch of the partition data we'll need
        class_sums = np.array([np.sum(counts*c, dtype=int) for c in classes])
        input_sums = np.array([np.sum(x[0]*x[1], dtype=int) for x in input_perm_classes])
        merged_sums = input_sums[:, np.newaxis] + class_sums[np.newaxis, :]

        paritioners = [
            self._get_partition_perms(x, ignore_negatives=True) for x in merged_sums
        ]

        class_negatives = [np.where(c < 0) for c in classes]
        class_negatives = [w[0] if len(w) == 1 else w for w in class_negatives]
        can_be_negative = np.array([len(w) > 0 for w in class_negatives])[np.newaxis, :]
        can_be_negative = [can_be_negative.copy() for i in range(len(input_perm_classes))]
        # set up an initial prefilter to elminate any entirely impossible
        # permutations by our filters/filter_negatives
        cls_inds = [np.arange(len(classes))]*len(input_perm_classes)
        for i,class_data in enumerate(input_perm_classes):
            drops = []
            for j,v in enumerate(merged_sums[i]):
                if v is None or (filter_negatives and v < 0):
                    # we stick this onto the drops pile so we can eliminate it later
                    drops.append(j)

            cls_inds[i] = np.delete(cls_inds[i], drops)
            can_be_negative[i] = np.delete(can_be_negative[i], drops) # we drop this so the two always align
            if filter is not None and filter.sums is not None:
                test_sums = np.delete(merged_sums[i], drops)
                mask, _, _ = contained(test_sums, filter.sums, invert=True,
                                       sortings=(None, np.arange(len(filter.sums))))
                more_drops = np.where(mask)
                drops.extend(cls_inds[i][more_drops])
                cls_inds[i] = np.delete(cls_inds[i], more_drops)
                can_be_negative[i] = np.delete(can_be_negative[i], more_drops)

            if len(drops) > 0:
                dropped_pairs.append((i, slice(None, None, None), drops))

            w = np.where(can_be_negative[i])
            if len(w) > 0:
                w = w[0]
            can_be_negative[i] = w
        cls_inds = tuple(cls_inds)
        can_be_negative = tuple(can_be_negative)

        # now we do an initial filtering to hopefully keep the size of `storage` smaller
        perm_counts = np.full((total_perm_count, num_perms), len(classes))
        for pair in dropped_pairs:
            i, idx, negs = pair
            perm_counts[cum_counts[i]:cum_counts[i+1], idx] -= len(negs)
        # we use this to first get the total number of permutations
        total_perm_count = np.sum(perm_counts)
        # and then for each initial input permutation (i.e. the old `total_perm_count`)
        # we figure out how many permutations it has now
        # I might be able to take a shortcut here based on the fact that I'm only dropping the
        # totally impossible stuff
        subcounts = np.sum(perm_counts, axis=1, dtype=int)
        perm_subcounts = np.concatenate([
            np.zeros(len(perm_counts), dtype=int)[:, np.newaxis],
            np.cumsum(perm_counts, axis=1, dtype=int)
            ],
            axis=1
        )
        # and then we turn this into an index along the new `total_perm_count`
        input_class_counts = np.concatenate([[0], np.cumsum(subcounts)])

        # we widen the dtype if necessary
        if full_basis is not None:
            dtype = full_basis.permutation_dtype
        else:
            dtype = input_perm_classes[0][2][0].dtype
            if dtype is np.dtype(object):
                raise ValueError("Can't get permutations for object-dtype inputs")
            if allow_widen_dtypes:
                max_val = np.max([np.max(x[0]) for x in input_perm_classes])
                max_rule = np.max(classes)
                inferred = _infer_dtype(max_val + max_rule)
                if inferred > dtype:
                    dtype = inferred

        storage = np.zeros((total_perm_count, ndim), dtype=dtype)
        if return_indices:
            indices = np.zeros((total_perm_count,), dtype=inds_dtype)

        if filter is not None or (filter_negatives and any(len(x)>0 for x in can_be_negative)):
            mask = np.full((total_perm_count,), True)
        else:
            mask = None

        storage_indexing_dtype = _infer_dtype(total_perm_count)
        # We split the full algorithm into a bunch of smaller functions to
        # make it easier to determine where the total runtime is
        # del cls_inds
        cls_cache = {}
        def add_new_perms(idx, perm, cls_pos, cts, depth, tree_data,
                          classes=classes,
                          input_perm_classes=input_perm_classes,
                          class_negatives=class_negatives
                          ):
            """

            :param idx:
            :type idx:
            :param perm:
            :type perm:
            :param counts:
            :type counts:
            :param depth:
            :type depth:
            :param tree_data:
            :type tree_data:
            :return:
            :rtype:
            """

            class_perms = np.array([c[perm] for c in classes])
            class_neg_list = [
                np.concatenate([cls_pos[j] for j in neg]) if len(neg) > 0 else ()
                for neg in class_negatives
            ]
            for i,class_data in enumerate(input_perm_classes):
                if len(cls_inds[i]) == 0:
                    continue

                # shape of storage used to be `(total_perm_count, num_perms, len(classes), ndim)`
                # but we collapsed this down to a 2D shape so we could use less memory/calculate less
                # stuff/not need to reshape & copy
                # this means we need to figure out where the relevant block starts and where it ends
                idx_starts = input_class_counts[cum_counts[i]:cum_counts[i+1]] + perm_subcounts[cum_counts[i]:cum_counts[i+1], idx]

                cls, cts, perms, rep = class_data
                # we make use of the fact that first adding the `class_perms` and _then_
                # taking all the permutations specified in `perms` will get us to the same place
                # as taking all of the `perms` first and then adding the `class_perms` _if_ we
                # do this for all of the possible perms of `counts` (i.e. if we do this in full)
                # This gives us strict ordering relations that we can make use of and allows us to only calculate
                # the counts once
                new_rep_perm = rep[np.newaxis, :] + class_perms[cls_inds[i], :]
                if filter_negatives and mask is not None:
                    class_negs = tuple(np.array(class_neg_list[j], dtype=class_negatives[0].dtype) for j in cls_inds[i])
                    comp, sel, new_perms = self._filter_negatives_perms(
                                i, idx, idx_starts, perms, new_rep_perm,
                                storage,
                                ndim,
                                cls_inds, class_negs,
                                perm_counts, cum_counts,
                                mask, can_be_negative
                                )
                else:
                    comp = cls_inds[i]
                    sel = np.arange(len(cls_inds[i]))#np.where(not_negs)[0]
                    new_perms = new_rep_perm[sel[:, np.newaxis, np.newaxis], perms[np.newaxis, :, :]]
                    stored_inds = np.reshape(sel[:, np.newaxis] + idx_starts[np.newaxis, :], -1)
                    storage[stored_inds] = new_perms.reshape(-1, ndim)

                    # raise NotImplementedError("need to get storage right but never touch this code path anymore")
                    # comp = cls_inds[i]
                    # sel = np.arange(len(comp))
                    # new_perms = new_rep_perm[:, perms].transpose(1, 0, 2)
                    # # shape used to be `(total_perm_count, num_perms, len(classes), ndim)`
                    # # but we collapsed this down to a 2D shape so we could use less memory/calculate less
                    # # stuff/not need to reshape & copy
                    #
                    # storage[idx_s:idx_e] = new_perms

                if return_indices:
                    # since we're assured sorting we make use of that when getting indices
                    if len(comp) > 0:
                        classes_count_data, standard_rep_perms = self._get_standard_perms(new_rep_perm[sel])
                        for n,j in enumerate(comp): # we're iterating over classes (not input_classes) here
                            key = tuple(tuple(x) for x in classes_count_data[n])
                            stored_inds = (idx_starts + sel[n]).astype(storage_indexing_dtype)
                            # print(key, standard_rep_perms[n],
                            #       paritioners[i][1][j],
                            #       paritioners[i][0][j],
                            #       )

                            if key in cls_cache:
                                cls_cache[key]['storage_blocks'].append(stored_inds)
                                cls_cache[key]['idx_blocs'].append((cum_counts[i], cum_counts[i + 1], idx))
                            else:
                                cls_cache[key] = {
                                    'standards': standard_rep_perms[n],
                                    'indices': (i, j), # the same set should work for all of these
                                    'storage_blocks': [stored_inds],
                                    'idx_blocs': [(cum_counts[i], cum_counts[i + 1], idx)]
                                }

        UniquePermutations.walk_permutation_tree(counts, add_new_perms, include_positions=True)
        if return_indices:
            self._process_cached_index_blocks(storage, cls_cache, paritioners, indices,
                                         filter, mask, perm_counts, merged_sums,
                                         inds_dtype=inds_dtype, full_basis=full_basis)
            # process_cached_index_blocks(storage)

        if mask is not None:
            if return_excitations:
                storage = storage[mask]
            if return_indices:
                indices = indices[mask]
        perm_counts = np.sum(perm_counts, axis=1)

        if not return_excitations:
            storage = None

        if return_indices:
            return storage, perm_counts, indices
        else:
            return storage, perm_counts

    def _get_direct_sum_rule_groups(self, rules, dim, dtype):
        # first up we pad the rules
        rules = [
            np.concatenate([np.array(r, dtype=dtype), np.zeros(dim - len(r), dtype=dtype)]) if len(
                r) < dim else np.array(r, dtype=dtype)
            for r in rules
            if len(r) <= dim
        ]

        # raise Exception(rules[0].dtype)

        # get counts so we can split them up
        wat = [UniquePermutations.get_permutation_class_counts(rule, sort_by_counts=True) for rule in rules]
        rule_counts = np.empty(len(wat), dtype=object)
        for i in range(len(wat)):
            rule_counts[i] = wat[i]

        # first split by length
        count_lens = np.array([len(x[0]) for x in rule_counts])
        len_sort = np.argsort(count_lens)
        len_invs = np.argsort(len_sort)
        _, len_split = np.unique(count_lens[len_sort], return_index=True)
        rule_counts = rule_counts[len_sort]
        rule_count_splits = np.split(rule_counts, len_split)[1:]
        invs_splits = np.split(len_invs, len_split)[1:]
        # next sort and split the rules for real
        rule_groups = []  # no reason to be fancy here
        # rule_inv = []
        # raise Exception(rule_count_splits)
        for split, inv in zip(rule_count_splits, invs_splits):
            rule_counts = np.array([x[1] for x in split], dtype=_infer_dtype(dim))
            split_sort = np.lexsort(np.flip(rule_counts, axis=1).T)
            rule_counts = rule_counts[split_sort,]
            inv = inv[split_sort,]
            split = split[split_sort,]
            ucounts, sub_split = np.unique(rule_counts, axis=0, return_index=True)
            count_splits = np.split(split, sub_split)[1:]

            rule_groups.extend(count_splits)
            # rule_inv.append(inv)
        # rule_inv = np.concatenate(rule_inv)

        return rule_groups

    def get_equivalence_classes(self, perms, sums=None, assume_sorted=False):
        """
        Gets permutation equivalence classes
        :param perms:
        :type perms:
        :param sums:
        :type sums:
        :param assume_sorted:
        :type assume_sorted:
        :return:
        :rtype:
        """

        if sums is None:
            sums = np.sum(perms, axis=1)

        if not assume_sorted:
            sum_sorting = np.argsort(sums)
            sums = sums[sum_sorting]
            perms = perms[sum_sorting]
        else:
            sum_sorting = None

        usums, _, inds = unique(sums, sorting=np.arange(len(sums)), return_index=True)
        groups = np.split(perms, inds)[1:]

        partitioners, shifts = self._get_partition_perms(usums)
        class_data = [
            p.get_equivalence_classes(g, assume_sorted=assume_sorted, check_partition_counts=False) for p, g
            in zip(partitioners, groups)
        ]

        if assume_sorted:
            perm_classes = [c[0] for c in class_data]
            perm_subsortings = [None] * len(class_data)
        else:
            perm_classes = []
            perm_subsortings = []
            for c in class_data:
                substuff = []
                subsortstuff = []
                for s in c[0]:
                    subsort = np.lexsort(np.flip(s[2], axis=1).T)
                    new = (s[0], s[1], s[2][subsort])
                    substuff.append(new)
                    subsortstuff.append(subsort)
                perm_classes.append(substuff)
                perm_subsortings.append(subsortstuff)

        # perm_totals = [c[1] for c in class_data]
        perm_sorting = [c[2] for c in class_data]

        return sum_sorting, perm_sorting, usums, perm_classes, perm_subsortings

    def take_permutation_rule_direct_sum(self, perms, rules,
                                         sums=None,
                                         assume_sorted=False,
                                         return_indices=False,
                                         return_excitations=True,
                                         full_basis=None,
                                         split_results=False,
                                         filter_perms=None,
                                         return_filter=False,
                                         preserve_ordering=True,
                                         indexing_method='direct',
                                         logger=None
                                         ):
        """
        Applies `rules` to perms.
        Naively this is just taking every possible permutation of the rules padded to
        get to the appropriate length and then adding that to every element in perms
        and then taking the unique ones.
        We can be more intelligent about how we do this, though, first reducing perms to
        equivalence classes as integer partitions and then making use of that to
        minimize the number of operations we need to do while also ensuring sorting

        :param perms:
        :type perms:
        :param rules:
        :type rules:
        :return:
        :rtype:
        """

        if full_basis is not None:
            return_excitations = False

        if logger is None:
            logger = NullLogger()

        # if dim is None:
        dim = self.dim
        perms = np.asanyarray(perms)
        if isinstance(perms.dtype, np.unsignedinteger):
            perms = perms.astype(_infer_nearest_pos_neg_dtype(perms.dtype))
        if perms.dtype.names is not None:
            perms = unflatten_dtype(perms, (len(perms), self.dim), perms.dtype[0])

        if perms.ndim == 1:
            perms = perms[np.newaxis]
        # og_perms = perms # for debug
        # next we pad up the perms as needed
        if perms.shape[1] < dim:
            perms = np.concatenate([
                perms,
                np.zeros((perms.shape[0], dim - perms.shape[1]), dtype=perms.dtype)
            ],
                axis=1
            )
        elif perms.shape[1] > dim:
            raise ValueError("with dimension {} can't handle states of shape {}".format(dim, perms.shape))

        rules = [[0] if len(r) == 0 else r for r in rules]
        if self.dim == 1:
            # this becomes very easy and in fact all of the other code breaks...
            rules_1d = np.array([x[0] for x in rules if len(x) == 1]).reshape(-1, 1)
            new_perms = perms[:, :, np.newaxis] + rules_1d[np.newaxis, :, :]
            if split_results:
                new_perms = [x[x>=0].reshape(-1, 1) for x in new_perms]
                new_inds = [x.flatten() for x in new_perms]
            else:
                new_perms = np.concatenate(new_perms)
                new_perms = new_perms[new_perms>=0]
                new_inds = new_perms.flatten()

            if return_indices:
                if return_filter:
                    return new_perms, new_inds, filter_perms
                else:
                    return new_perms, new_inds
            else:
                return new_perms

        # fill counts arrays so we don't need to recalculate this a bunch
        if sums is None:
            sums = np.sum(perms, axis=1)
        max_rule = max(max(r) if len(r) > 0 else 0 for r in rules) if len(rules) > 0 else 0
        max_term = max(int(1 + max_rule + np.max(sums)), 0) # numpy dtype shit sometimes goes weird
        # raise Exception(max_rule, max_term, np.max(sums))
        logger.log_print('populating bases up to {nq} quanta...', nq=max_term, log_level=logger.LogLevel.Debug)
        IntegerPartitioner.fill_counts(max_term, max_term, self.dim)
        if full_basis is not None:
            full_basis.load_to_sum(max_term)
        _, total_possible_counts = self._get_partition_perms([max_term])
        inds_dtype = _infer_dtype(total_possible_counts[0])

        rule_groups = self._get_direct_sum_rule_groups(rules, dim, perms.dtype)

        # next split up the input permutations
        sum_sorting, perm_sorting, usums, perm_classes, perm_subsortings = self.get_equivalence_classes(perms, sums=sums, assume_sorted=assume_sorted)

        if return_indices and indexing_method == 'secondary':
            secondary_inds = True
            return_indices = False
        else:
            secondary_inds = False

        perms = []
        if return_indices:
            indices = []

        # we now set up filtering so that we can efficiently prune branches
        # as we calculate partition permutations
        filter = self.direct_sum_filter.from_data(self, filter_perms)
        if filter is not None and not return_indices:
            raise ValueError("if a filter is used indices must be requested since they are used in filtering")

        rule_counts = [group[0][1] for group in rule_groups]
        rule_classes = [[g[0] for g in group] for group in rule_groups]

        if not isinstance(logger, NullLogger): # can be slow to log prettily
            # input_classes_fmt = [
            #                 "class: {} counts: {} permutations: {}".format(y[0], y[1], len(y[2])) for x in perm_classes for y in x
            #             ]
            rule_class_fmt = [
                            "classes: {} counts: {}".format(", ".join(str(z) for z in y), x) for x, y in zip(rule_counts, rule_classes)
                        ]
        else:
            input_classes_fmt=[]
            rule_class_fmt=[]
        with logger.block(tag="taking direct product", log_level=logger.LogLevel.Debug):
            with logger.block(tag="selection rules:", log_level=logger.LogLevel.Debug):
                logger.log_print(rule_class_fmt, log_level=logger.LogLevel.Debug)
            start = time.time()
            for input_classes, nq, sorts in zip(perm_classes, usums, perm_subsortings):
                substart = time.time()
                with logger.block(tag='sum: {}'.format(nq), log_level=logger.LogLevel.Debug):
                    if not isinstance(logger, NullLogger):  # can be slow to log prettily
                        input_classes_fmt = ["Class/Counts/Permutations"]+ [
                             "{}/{}/{}".format(y[0], y[1], len(y[2])) for y in input_classes
                        ]
                        logger.log_print(input_classes_fmt, log_level=logger.LogLevel.Debug)
                    perm_block = []
                    if return_indices:
                        ind_block = []

                    for counts, classes in zip(rule_counts, rule_classes):
                        res = self._build_direct_sums(input_classes, counts, classes,
                                                      return_indices=return_indices,
                                                      return_excitations=return_excitations,
                                                      filter=filter, inds_dtype=inds_dtype,
                                                      full_basis=full_basis
                                                      )
                        # gc.collect()
                        if split_results or preserve_ordering:
                            split_blocks = np.cumsum(res[1][:-1])
                            if return_excitations:
                                res_perms = np.split(res[0], split_blocks)
                            if return_indices:
                                ind_block.append(np.split(res[2], split_blocks))
                        else:
                            if return_excitations:
                                res_perms = res[0]
                            if return_indices:
                                ind_block.append(res[2])
                        if return_excitations:
                            perm_block.append(res_perms)

                    # if nq == 5:# and len(counts) == 3 and counts[-1] == 9:
                    #     #     raise Exception(perm_pos.dtype)
                    #     raise Exception("oookay")

                    if split_results or preserve_ordering:
                        # zip to merge
                        if return_excitations:
                            new_perms = [
                                np.concatenate(blocks, axis=0)
                                for blocks in zip(*perm_block)
                            ]
                            if preserve_ordering and sorts is not None and len(new_perms) > 0:
                                cumlens = np.cumsum([0] + [len(x) for x in sorts[:-1]])
                                sorts = np.concatenate([x+s for x,s in zip(sorts, cumlens)])
                                argsorts = np.argsort(sorts)
                                new_perms = [new_perms[i] for i in argsorts]
                            # if not split_results:
                            #     if len(new_perms) == 0:
                            #         new_perms = np.array([], dtype='int8')
                            #     else:
                            #         new_perms = np.concatenate(new_perms, axis=0)
                            perms.append(new_perms)
                        if return_indices:
                            new_inds = [
                                np.concatenate(blocks)
                                for blocks in zip(*ind_block)
                            ]
                            if preserve_ordering and sorts is not None and len(new_inds) > 0:
                                if not return_excitations:
                                    cumlens = np.cumsum([0] + [len(x) for x in sorts[:-1]])
                                    sorts = np.concatenate([x + s for x, s in zip(sorts, cumlens)])
                                    argsorts = np.argsort(sorts)
                                new_inds = [new_inds[i] for i in argsorts]
                            indices.append(new_inds)
                    else:
                        if return_excitations:
                            new_perms = np.concatenate(perm_block, axis=0)
                            perms.append(new_perms)
                        if return_indices:
                            new_inds = np.concatenate(ind_block)
                            indices.append(new_inds)

                    if return_excitations:
                        subend = time.time()
                        logger.log_print([
                            'got {nt} partition-permutations{and_inds}',
                            'took {e:.3f}s...'
                        ],
                            nt=len(new_perms) if not (split_results or preserve_ordering) else sum(len(x) for x in new_perms),
                            and_inds=' and indices' if return_indices else '',
                            e=subend - substart,
                            log_level=logger.LogLevel.Debug
                        )
                    else:
                        subend = time.time()
                        logger.log_print([
                            'got {nt} partition-permutations indices',
                            'took {e:.3f}s...'
                        ],
                            nt=len(new_inds) if not (split_results or preserve_ordering) else sum(len(x) for x in new_inds),
                            e=subend - substart,
                            log_level=logger.LogLevel.Debug
                        )

            # now we need to also reshuffle the states so
            # that they come out in the input ordering
            if split_results or preserve_ordering:

                if return_excitations:
                    new_perms = []
                    if preserve_ordering:
                        for p,s in zip(perms, perm_sorting):
                            if len(p) > 0:
                                new_perms += [p[i] for i in np.argsort(s)]
                        perms = new_perms
                    else:
                        perms = sum(perms, [])
                    if preserve_ordering and sum_sorting is not None and len(perms) > 0:
                        inv = np.argsort(sum_sorting)
                        perms = [perms[i] for i in inv]
                if return_indices:
                    if preserve_ordering:
                        new_inds = []
                        for d,s in zip(indices, perm_sorting):
                            if len(d) > 0:
                                new_inds += [d[i] for i in np.argsort(s)]
                        indices = new_inds
                    else:
                        indices = sum(indices, [])
                    if preserve_ordering and sum_sorting is not None and len(indices) > 0:
                        if not return_excitations:
                            inv = np.argsort(sum_sorting)
                        indices = [indices[i] for i in inv]
                if not split_results:
                    if return_excitations:
                        if len(perms) == 0:
                            perms = np.array([], dtype='int8')
                        else:
                            perms = np.concatenate(perms, axis=0)
                    if return_indices:
                        if len(perms) == 0:
                            indices = np.array([], dtype='int8')
                        else:
                            indices = np.concatenate(indices, axis=0)
            else:
                if return_excitations:
                    perms = np.concatenate(perms, axis=0)
                if return_indices:
                    indices = np.concatenate(indices)

            end = time.time()
            if return_excitations:
                logger.log_print([
                    'in total got {nt} partition-permutations{and_inds}',
                    'took {e:.3f}s...'
                    ],
                    nt=len(perms) if not (split_results or preserve_ordering) else sum(len(x) for x in perms),
                    and_inds=' and indices' if return_indices else '',
                    e=end-start,
                    log_level=logger.LogLevel.Debug
                )
            else:
                logger.log_print([
                    'in total got {nt} partition-permutations indices',
                    'took {e:.3f}s...'
                    ],
                    nt=len(indices) if not (split_results or preserve_ordering) else sum(len(x) for x in indices),
                    e=end-start,
                    log_level=logger.LogLevel.Debug
                )

            if not return_excitations:
                perms = None
            if return_indices:
                if return_filter:
                    return perms, indices, filter
                else:
                    return perms, indices
            elif secondary_inds:
                if split_results:
                    full_perms = np.concatenate(perms, axis=0)
                    indices = self.to_indices(full_perms)
                    splits = np.cumsum([len(x) for x in perms])[:-1]
                    indices = np.split(indices, splits)
                else:
                    indices = self.to_indices(perms)
                if return_filter:
                    return perms, indices, filter
                else:
                    return perms, indices
            else:
                return perms

class CompleteSymmetricGroupSpace:
    """
    An object representing a full integer partition-permutation basis
    which will work nominally at any level of excitation
    """

    permutation_dtype = 'int8' # if we need to go up beyond dim 256 we're fucked anyway
    def __init__(self, dim, memory_constrained=False):
        self.generator = SymmetricGroupGenerator(dim)
        self._basis = None
        self._basis_sorting = None
        _, self._contracted_dtype, _, self._og_dtype = flatten_dtype(np.zeros((1, dim), dtype=self.permutation_dtype))
        self.memory_constrained = memory_constrained

    @property
    def dim(self):
        return self.generator.dim

    def __getstate__(self):
        return {'dim':self.dim}
    def __setstate__(self, state):
        self.__init__(state['dim'])
    def _contract_dtype(self, perms):
        if self._contracted_dtype is not None and perms.dtype == self._contracted_dtype:
            return perms
        else:
            if self._contracted_dtype is not None:
                return flatten_dtype(perms.astype(self.permutation_dtype), dtype=self._contracted_dtype)[0]
            else:
                new, self._contracted_dtype, _, _ = flatten_dtype(perms.astype(self.permutation_dtype))
                return new

    def load_to_size(self, size):
        if self.memory_constrained:
            return True
        cur_basis_size = -1 if self._basis is None else len(self._basis)
        if cur_basis_size < size:
            self.generator.load_to_size(size)
            need_to_load = np.where(self.generator._cumtotals >= cur_basis_size)
            if len(need_to_load) > 0:
                if not isinstance(need_to_load[0], (int, np.integer)):
                    need_to_load = need_to_load[0]
                partitioners = self.generator._get_partition_perms(need_to_load)[0] #type: list[IntegerPartitionPermutations]
                new_bases = [
                    c for p in partitioners for c in
                    p.get_partition_permutations(dtype=self.permutation_dtype)
                ]

                if self._basis is None:
                    self._basis = np.concatenate(new_bases, axis=0)
                    self._contracted_basis = self._contract_dtype(self._basis)
                else:
                    self._basis = np.concatenate([self._basis] + new_bases)
                    self._contracted_basis = self._contract_dtype(self._basis)

    def load_to_sum(self, max_sum):
        _, offset = self.generator._get_partition_perms([max_sum + 1])
        self.load_to_size(offset[0])

    def take(self, item, uncoerce=False, max_size=None):
        if self.memory_constrained:
            return self.generator.from_indices(item)
        if isinstance(item, (int, np.integer)):
            self.load_to_size(item+1)
            res = self._basis[item]
        # elif isinstance(item, slice):
        #     return self._basis[item]
        else:
            if max_size is None:
                max_size = np.max(item)
            max_size = max_size + 1
            self.load_to_size(max_size)
            res = self._basis[item]

        if uncoerce:
            # orig_shape, orig_dtype, axis
            if len(res) == 0:
                res = np.empty((0, self.dim), dtype=self.permutation_dtype)
            elif isinstance(item, (int, np.integer)):
                res = unflatten_dtype(res, (1, self.dim), self._og_dtype)
            else:
                res = unflatten_dtype(res, (len(res), self.dim), self._og_dtype)

        return res

    def __getitem__(self, item):
        return self.take(item)

    def find(self, perms,
             check_sums=True,
             max_sum=None,
             search_space_sorting=None
             ):
        if self.memory_constrained:
            return self.generator.to_indices(perms)

        p = np.asanyarray(perms)
        smol = p.ndim == 1
        if smol:
            p = p[np.newaxis]

        if check_sums:
            if max_sum is None:
                sums = np.sum(p, axis=1, dtype=int)
                max_sum = np.max(sums)
            self.load_to_sum(max_sum + 1)

        if self._basis_sorting is not None and len(self._basis_sorting) == len(self._basis):
            inds, self._basis_sorting = find(self._contracted_basis, self._contract_dtype(p), sorting=self._basis_sorting,
                                             search_space_sorting=search_space_sorting
                                             )
        else:
            inds, self._basis_sorting = find(self._contracted_basis, self._contract_dtype(p),
                                             search_space_sorting=search_space_sorting
                                             )

        return inds

class LatticePathGenerator:
    """
    An object to take direct products of lattice paths and
    filter them
    """

    def __init__(self, *steps, max_len=None):
        """
        :param steps: the steps to take a direct product of
        :type steps: Iterable[Iterable[int]]
        """
        if len(steps) == 1 and not isinstance(steps[0], (int, np.integer)):
            steps = steps[0]

        for x in steps:
            if len(x) > 0 and not (
                    isinstance(x[0], (int, np.integer))
                    or len(x[0]) == 0
                    or isinstance(x[0][0], (int, np.integer))
            ):
                raise TypeError("lattice path steps, {}, much be lists of ints or lists of lists of ints".format(steps))
        self.steps = [tuple(x) for x in steps]
        self.max_len = max_len
        self._subtrees = None
        self._rule_trees = None

    @property
    def subtrees(self):
        if self._subtrees is None:
            self._subtrees = self.generate_tree(self.steps, max_len=self.max_len)
        return self._subtrees

    @property
    def tree(self):
        if self._subtrees is None:
            self._subtrees = self.generate_tree(self.steps, max_len=self.max_len)
        return self._subtrees[-1]

    @property
    def subrules(self):
        if self._rule_trees is None:
            self._rule_trees = self.generate_tree(self.steps, track_positions=False, max_len=self.max_len)
        return self._rule_trees

    @property
    def rules(self):
        if self._rule_trees is None:
            self._rule_trees = self.generate_tree(self.steps, track_positions=False, max_len=self.max_len)
        return self._rule_trees[-1]

    @classmethod
    def generate_tree(self, rules,
                      max_len=None,
                      track_positions=True
                      ):
        """
        We take the combo of the specified rules, where we take successive products of 1D rules with the
        current set of rules following the pattern that
            1. a 1D change can apply to any index in an existing rule
            2. a 1D change can be appended to an existing rule

        We ensure at each step that the rules remain sorted & duplicates are removed so as to keep the rule sets compact.
        This is done in simple python loops, because doing it with arrayops seemed harder & not worth it for a relatively cheap operation.

        :param rules:
        :type rules:
        :return:
        :rtype:
        """

        rules = [
            np.sort(x)if len(x) > 0 and isinstance(x[0], (int, np.integer)) else
            tuple(np.sort(y) for y in x)
            for x in rules
        ]
        ndim = sum(
            0 if len(r) == 0 else
            1 if isinstance(r[0], (int, np.integer)) else
            max(len(x) for x in r) for r in rules
        )
        if max_len is None:
            max_len = ndim

        if track_positions:
            cur_rules = {((), (0,) * max_len)}
        else:
            cur_rules = {(0,) * max_len}
        subtrees = []
        for r in rules:
            if len(r) == 0:
                new_rules = cur_rules
            else:
                new_rules = set()
                for e in cur_rules:
                    if track_positions:
                        x, e = e
                    for j,s in enumerate(r):
                        if isinstance(s, (int, np.integer)):
                            for i in range(max_len):
                                shift = e[i] + s
                                new = e[:i] + (shift,) + e[i + 1:]
                                new = tuple(sorted(new, key=lambda l: -abs(l) * 10 - (1 if l > 0 else 0)))
                                if track_positions:
                                    new_x = x + (j,)
                                    new = (new_x, new)
                                new_rules.add(new)
                        else:
                            # means we were handed full-on selection rules
                            # and so we need to add the appropriate number of ints
                            # to the appropriate number of places
                            if len(s) == 0:
                                new = e
                                if track_positions:
                                    new_x = x + (j,)
                                    new = (new_x, new)
                                new_rules.add(new)
                            else:
                                for p in itertools.product(*(range(max_len) for _ in range(len(s)))):
                                    if len(np.unique(p)) == len(p): # filter out anything with dupe axes
                                        new = e
                                        for i,z in zip(p, s):
                                            shift = new[i] + z
                                            new = new[:i] + (shift,) + new[i + 1:]
                                        new = tuple(sorted(new, key=lambda l: -abs(l) * 10 - (1 if l > 0 else 0)))
                                        if track_positions:
                                            new_x = x + (j,)
                                            new = (new_x, new)
                                        new_rules.add(new)

            # print(cur_rules)
            subtrees.append(cur_rules)
            cur_rules = new_rules
        subtrees.append(new_rules)

        new_trees = []
        for cur_rules in subtrees:
            cur_rules = list(cur_rules)
            new_rules = []
            for r in cur_rules:
                if track_positions:
                    x, r = r
                if len(r) > 0:
                    for i,v in enumerate(r):
                        if v == 0: break
                    else:
                        i+=1
                    r = tuple(r[:i])
                    if track_positions:
                        r = (x, r)
                new_rules.append(r)


            if not track_positions:
                new_rules = list(sorted(new_rules, key=lambda l: len(l) * 100 + sum(l)))
            else:
                idx = np.arange(len(new_rules))
                idx_chunks, _ = group_by(idx, [k for k,v in new_rules])
                new_rules = [
                    (tuple(k), list(sorted([new_rules[i][1] for i in b], key=lambda l: len(l) * 100 + sum(l))) )
                    for k, b in zip(*idx_chunks)
                ]

            new_trees.append(new_rules)

        return new_trees

    def find_paths(self, end_spots):
        # to start we just populate the entire tree and find the steps that took us
        # to `end_spot`
        if len(end_spots) == 0 or isinstance(end_spots[0], (int, np.integer)):
            end_spots = [end_spots]
        res = set()
        for x,t in self.tree:
            if any(e in t for e in end_spots):
                res.add(x)
        return list(res)

    def get_path(self, path):
        """
        Pulls the places one can end up after applying the path

        :param other:
        :type other:
        :return:
        :rtype:
        """

        for t in self.tree:
            if t[0] == path:
                return t[1]
        else:
            raise ValueError("path {} not in tree".format(path))

    def find_intersections(self, other):
        """
        Finds the paths that will make self intersect with other

        :param other:
        :type other: LatticePathGenerator
        :return:
        :rtype:
        """

        return self.find_paths(other.rules)

class PermutationRelationGraph:
    """
    Takes permutations and a set of relations and builds a graph from
    them
    """

    def __init__(self, relations):
        """
        :param relations: sets of rules connecting permutations
        :type relations:
        """
        self.rels, self.indexer = self.make_relation_graph(relations)

    @classmethod
    def merge_groups(cls, groups):
        """
        This really needs to be cleaned up...

        :param groups:
        :type groups:
        :return:
        :rtype:
        """

        num_groups = np.inf
        while len(groups) < num_groups:
            num_groups = len(groups)
            new_groups = []
            for ind, grp in groups:
                ind, pos = np.unique(ind, return_index=True)
                grp = grp[pos,]
                for n, (ix, g) in enumerate(new_groups):
                    if np.any(np.isin(ind, ix)):
                        ix, pos = np.unique(np.concatenate([ix, ind], axis=0), return_index=True)
                        g = np.concatenate([g, grp], axis=0)[pos,]
                        new_groups[n] = (ix, g)
                        break
                else:
                    new_groups.append([ind, grp])

            groups = new_groups

        return groups


    @classmethod
    def make_relation_graph(cls, relations):
        """

        :param relations:
        :type relations: Iterable[Iterable[Iterable[int]]]
        :return:
        :rtype:
        """

        ndim = len(relations[0][0])
        indexer = SymmetricGroupGenerator(ndim)

        relations = [np.asanyarray(r) for r in relations]
        rel_inds = [indexer.to_indices(r) for r in relations]

        rel_groups = cls.merge_groups(list(zip(rel_inds, relations)))

        return rel_groups, indexer

    def apply_rels(self, states, max_sum=None):
        """
        For each state checks if it is divisible by one of the group rules and if so applies the
        relevant transformations to it

        :param states:
        :type states:
        :return:
        :rtype:
        """

        new_states = states
        changed = False
        for i,g in self.rels:
            for n,r in enumerate(g):
                # nzp = np.nonzero(r)
                # if len(nzp) > 0:
                #     nzp = nzp[0]
                # if len(nzp) > 0:
                #     test_states = states[:, nzp]
                #     print(">>>>", nzp)
                #     print("> ", test_states)
                #     print("> ", r[np.newaxis, nzp])
                #     divis_pos = np.where(np.all(test_states >= r[np.newaxis, nzp], axis=0))
                #     if len(divis_pos) > 0:
                #         divis_pos = divis_pos[0]
                # else:
                #     divis_pos = np.arange(len(states))

                # if len(divis_pos) > 0:
                #     substates = states[divis_pos,]

                rule_changes = np.delete(g - r[np.newaxis, :], n, axis=0)
                # print(">>>", rule_changes)
                gen_states = states[:, np.newaxis, :] + rule_changes[np.newaxis, :, :]
                gen_states = gen_states.reshape(-1, gen_states.shape[-1])
                # print("> ", gen_states)
                gen_states = gen_states[np.all(gen_states >= 0, axis=1),]
                if max_sum is not None:
                    gen_states = gen_states[np.sum(gen_states, axis=1) <= max_sum]
                changed = True
                new_states = np.concatenate([new_states, gen_states], axis=0)

        if changed:
            new_states = np.unique(new_states, axis=0)

        return new_states

    def build_state_graph(self, states, max_sum=None, extra_groups=None, max_iterations=10, raise_iteration_error=True):
        """

        :param states:
        :type states:
        :param max_iterations:
        :type max_iterations:
        :param raise_iteration_error:
        :type raise_iteration_error:
        :return:
        :rtype: Iterable[np.ndarray]
        """

        states = np.asanyarray(states)

        ix = self.indexer.to_indices(states)
        groups = [(np.array([i]), np.array([s])) for i,s in zip(ix, states)]
        for m in range(max_iterations):
            # at each pass we take the existing state groups (initally singletons) propagate rules
            # then check for intersections with the remaining groups and merge where possible/necessary
            changed_flag = False
            for n, (i,g) in enumerate(groups):
                g_new = self.apply_rels(g, max_sum=max_sum)
                # print(g)
                # print(g_new)
                if len(g_new) > len(g):
                    changed_flag = True
                    i_new = self.indexer.to_indices(g_new)
                    groups[n] = (i_new, g_new)

            if not changed_flag:
                break

            groups = self.merge_groups(groups)

        else:
            if raise_iteration_error:
                raise ValueError("relation graph from {} did not converge after {} iterations".format(self.rels, max_iterations))

        if extra_groups is not None:
            extra_groups = [np.asanyarray(g) for g in extra_groups]
            extra_groups = [(self.indexer.to_indices(g), g) for g in extra_groups]

            groups = self.merge_groups(groups + extra_groups)

        return [g[1] for g in groups]



