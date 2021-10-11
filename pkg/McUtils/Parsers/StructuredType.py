
import numpy as np
from collections import OrderedDict

__all__ = [
    "StructuredType",
    "StructuredTypeArray",
    "DisappearingType"
]

class StructuredType:
    """
    Represents a structured type with a defined calculus to simplify the construction of combined types when writing
    parsers that take multi-typed data

    Supports a compound StructuredType where the types are keyed
    """
    def __init__(self,
                 base_type,
                 shape = None,
                 is_alternative = False,
                 is_optional = False,
                 default_value = None
                 ):

        if shape is None and isinstance(base_type, tuple) and len(base_type) == 2 and (
                isinstance(base_type[1], int) or isinstance(base_type[1], tuple)
        ): # to make it possible to initialize the type like (str, 3) or (int, (3,))
            base_type, shape = base_type

        self.dtype = base_type
        self.shape = shape

        # there's some question as to how we should declare alternative types?
        # currently we have a flag in the type but we could also make 3 subtypes of StructuredType:
        #   AlternativeTypes(...), PrimitiveType(...), CompoundType(...) ?

        self.is_alternative = is_alternative
        self.is_optional = is_optional


        self.default = default_value if default_value is not None else self._infer_default_value() # for optional types

    def _infer_default_value(self):
        missing = None
        if isinstance(self.dtype, type):
            if self.dtype is str:
                missing = "[NaN]"
            elif self.dtype is int or self.dtype is np.int:
                missing = int(-121e12) # -(Na e N)
            elif self.dtype is float or self.dtype is np.floating:
                missing = np.nan
        # I guess we're gonna assume no other cases will realistically need padding/defaults?
        return missing


    @property
    def is_simple(self):
        return (
                self.dtype is None or (isinstance(self.dtype, type) and not self.is_optional and not self.is_alternative)
        )

    def add_types(self, other):
        """
        Constructs a new type by treating the two objects as siblings, that is if they can be merged due to type and
        shape similarity they will be, otherwise a non-nesting structure will be constructed from them

        We'll also want a nesting version of this I'm guessing, which probably we hook into __call__

        :param other:
        :type other:
        :return:
        :rtype:
        """

        if other is DisappearingType or isinstance(other, type(DisappearingType)):
            return self
        elif self is DisappearingType or isinstance(self, type(DisappearingType)):
            return other
        else:
            if not isinstance(other, StructuredType):
                other = StructuredType(other)

            # we define a very simple calculus on the structured types
            #   if both are the same we simply futz with the shapes if possible
            #   if they're different we construct a compound type from the two structured types
            # _unless_ we have an alternative type in which case...?

            if self.shape is None or other.shape is None:
                shape_mismatches = None
            else:
                shape_mismatches = np.array([(x is None) or (y is None) or int(x != y) for x,y in zip(self.shape, other.shape)])
            if (
                    self.is_simple and
                    self.dtype == other.dtype and (
                            shape_mismatches is None or (
                            len(self.shape) == len(other.shape) and np.sum(shape_mismatches) <= 1
                            )
                    )
                ):

                if self.shape is None and other.shape is None:
                    new_shape = (2,)
                elif self.shape is None:
                    new_shape = other.shape[:-1] + (other.shape[-1]+1, ) # this isn't quite right but will work for now
                elif other.shape is None:
                    new_shape = self.shape[:-1] + (self.shape[-1]+1, ) # this isn't quite right but will work for now
                else:
                    mismatch_pos = np.argwhere(shape_mismatches == 1)
                    if len(mismatch_pos) == 0:
                        mismatch_pos = -1
                    else:
                        mismatch_pos = mismatch_pos[-1]
                    new_shape = list(self.shape)
                    me = self.shape[mismatch_pos]
                    yu = other.shape[mismatch_pos]
                    if me is None or yu is None:
                        new_shape[mismatch_pos] = None
                    else:
                        new_shape[mismatch_pos] = me + yu

                import copy
                new = copy.copy(self)
                new.shape = tuple(new_shape)
                return new
            elif isinstance(self.dtype, type):
                # we do this because who knows what to do here otherwise...?
                return type(self)((self, other))
            else:
                import copy

                new = copy.copy(self)
                new.dtype += (other,)
                return new

    def __add__(self, other):
        return self.add_types(other)

    def compound_types(self, other):
        """Creates a structured type where rather than merging types they simply compound onto one another

        :param other:
        :type other:
        :return:
        :rtype:
        """
        raise NotImplemented
    def __call__(self, other):
        return self.compound_types(other)


    def repeat(self, n = None, m = None):
        """Returns a new version of the type, but with the appropriate shape for being repeated n-to-m times

        :param n:
        :type n:
        :param m:
        :type m:
        :return:
        :rtype:
        """

        import copy
        new = copy.copy(self)
        if new.shape is None:
            new.shape = (None,) if m is None else (m,)
        # elif n is not None and m is not None and n == m:
        #     new.shape = (n,) + new.shape
        elif m is not None: # if we have any kind of max number of elements, we know the size of the array we need to allocate
            new.shape = (m,) + new.shape
        else:
            new.shape = (None,) + new.shape # we won't handle things like between 10 and 20 elements for now
        return new

    def drop_axis(self, axis = 0):
        """Returns a new version of the type, but with the appropriate shape for dropping an axis

        :param axis:
        :type axis: int
        :return:
        :rtype:
        """

        import copy
        new = copy.copy(self)
        if new.shape is not None:
            new.shape = new.shape[:axis] + new.shape[axis+1:]
        elif not new.is_simple:
            # means we need to strip the axis from the subtypes
            dt = new.dtype
            if isinstance(dt, OrderedDict):
                new.dtype = OrderedDict( (k, d.drop_axis(axis=axis)) for k,d in dt.items())
            else:
                new.dtype = tuple(d.drop_axis(axis=axis) for d in dt)
        return new

    def extend_shape(self, base_shape):
        """Extends the shape of the type such that base_shape precedes the existing shape

        :param base_shape:
        :type base_shape:
        :return:
        :rtype:
        """
        shape = self.shape
        if self.shape is None:
            self.shape = base_shape
        else:
            self.shape = base_shape + shape

    def _condense_types(self, base_types, shape):
        # what do I do about this existing shape...?

        raw_types = [ b.dtype for b in base_types ]
        shapes = [ b.shape for b in base_types ]

        if shape is None and (
                all(r is raw_types[0] for r in raw_types) and
                (
                        all( s is None for s in shapes ) or
                        all( isinstance(s, tuple) and s == shapes[0] for s in shapes )
                )
        ):
            # means we can actually condense them but I might want to be a bit smarter with how I handle the tuple shape check...

            base_types = base_types[0] # no differences in type
            if shapes[0] is None:
                shape = (len(shapes),)
            else:
                shape += (len(shapes),)

        return base_types, shape


    def __repr__(self):
        return "{}({}, shape={})".format(
            type(self).__name__,
            self.dtype,
            self.shape
        )

class DisappearingTypeClass(StructuredType):
    """
    A special type that is entirely ignored in the structured type algebra
    """
    def __init__(self):
        self.is_disappearing = True
        super().__init__(None)
DisappearingType = DisappearingTypeClass() # redefinition but it should be a singleton anyway


"""
Something to think about w.r.t the correspondance between the stype shape and the stated shape of the 
array. 

There are two possible setups, really, in the first setup we have two cases:
    Simple stype => no shape for the stype
    Compound stype => shape of array is one _larger_ than shape of stype

In the second case:
    Simple dtype => no shape for the stype
    Compound dtype => shape of array is the shape of the stype

It's not clear that either is better than the other, but whichever one is chose, we need to be
consistent throughout the code in how we work with it.
"""

class StructuredTypeArray:
    """
    Represents an array of objects defined by the StructuredType spec provided
    mostly useful as it dispatches to NumPy where things are simple enough to do so

    It has a system to dispatch intelligently based on the type of array provided
    The kinds of structures supported are: OrderedDict, list, and np.ndarray

    A _simple_ StructuredTypeArray is one that can just be represented as a single np.ndarray
    A _compound_ StructuredTypeArray requires either a list or OrderedDict of StructuredTypeArray subarrays
    """

    # at some point this should make use of the more complex structured dtypes that NumPy provides...
    # for now we'll stick with this format, but using more NumPy will make stuff more efficient and easier to post-process
    def __init__(self, stype, num_elements = 50, padding_mode = 'fill', padding_value = None):
        """
        :param stype:
        :type stype: StructuredType
        :param num_elements: number of default elements in dynamically sized arrays
        :type num_elements: int
        """

        if not isinstance(stype, StructuredType):
            stype = StructuredType(stype)

        self._is_simple = stype.is_simple
        self._extend_along = None
        self._dtype = None
        self._stype = stype

        self._filled_to = None

        self._default_num_elements = num_elements
        self._array = None
        self._array = self.empty_array() # empty_array tries to use shape if possible
        self._append_depth = -1

        self.padding_mode = padding_mode
        self.padding_value = stype.default

    @property
    def is_simple(self):
        """Just returns wheter the core datatype is simple

        :return:
        :rtype:
        """
        return self._is_simple
    @property
    def dict_like(self):
        return isinstance(self._array, (dict, OrderedDict))
    @property
    def extension_axis(self):
        """Determines which axis to extend when adding more memory to the array
        :return:
        :rtype:
        """
        if self._extend_along is None:
            if self._stype.shape is None:
                self._extend_along = 0
            else:
                shape_nones = np.array([x is None for x in self._stype.shape], dtype=bool)
                if len(shape_nones) == 0:
                    self._extend_along = 0
                else:
                    self._extend_along = np.arange(len(shape_nones))
                    hmm = self._extend_along[shape_nones]
                    if len(hmm) == 0:
                        hmm = self._extend_along
                    self._extend_along = hmm[0]
        return self._extend_along
    @extension_axis.setter
    def extension_axis(self, ax):
        self._extend_along = ax
    @property
    def shape(self):
        if self._array is None:
            return None
        if self.is_simple:
            if isinstance(self._array, np.ndarray):
                # self._shape = list(self._array.shape)
                # self._shape[self.extension_axis] = self.filled_to[self.extension_axis] # this will mess up on things like (3,)...
                # self._shape = tuple(self._shape)
                self._shape = tuple(self.filled_to) if self.filled_to is not None else None
            else:
                self._shape = None
        else:
            if isinstance(self._array, OrderedDict):
                self._shape = [s.shape for s in self._array.values()]
            else:
                self._shape = [s.shape for s in self._array]
        return self._shape
    @shape.setter
    def shape(self, s):
        self._shape = s
    @property
    def block_size(self):
        if self.is_simple:
            s = list(self.shape)
            s[self.extension_axis] = 1 # this will mess up on things like (3,)...
            self._block_size = np.product(s)
        else:
            self._block_size = sum([s.block_size for s in self._array])
        return self._block_size
    @property
    def append_depth(self):
        return self._append_depth
    @append_depth.setter
    def append_depth(self, d):
        inc = d - self._append_depth
        self._append_depth = d
        if not self._is_simple:
            for a in (self._array.values() if isinstance(self._array, OrderedDict) else self._array):
                a.append_depth += inc

    def _invalidate_type_cache(self):
        """Whenever the dtype and stype are changed we can use this to invalidate the cached forms

        :return:
        :rtype:
        """
        self._dtype = None

    def _get_complex_dtype(self):
        # I think all the shape changes get adequately passed down...?
        # Like I think Repeating and friends manage stuff fine
        arr = self._array
        if arr is None:
            # Means we gotta do this recursively
            # The shape should be fed back down to the object's children at this point
            shape = self._stype.shape
            dt = self._stype.dtype
            if isinstance(dt, OrderedDict):
                dt = OrderedDict((k, StructuredType(s) if isinstance(s, type) else s) for k,s in dt.items())
            else:
                dt = tuple(
                    StructuredType(s) if isinstance(s, type) else s for s in self._stype.dtype
                )
            if shape is not None:
                # we have to take our shape and feed it back down to our children
                # the mechanism for this will have to be to take each type in dt and stick our shape onto the
                # front of the dtype's shape
                # one thing about this, is that once we've fed the shape down in
                import copy
                if isinstance(dt, OrderedDict):
                    dt = OrderedDict((k, copy.copy(d)) for k,d in dt.items())
                    for d in dt.values():
                        d.extend_shape(shape)
                else:
                    dt = tuple( copy.copy(d) for d in dt )
                    for d in dt:
                        d.extend_shape(shape)
                st = copy.copy(self._stype)
                st.shape = None
                st.dtype = dt
                self._stype = st
        elif isinstance(arr, OrderedDict):
            dt = OrderedDict((k, a.stype) for k,a in arr.items())
            self._stype = StructuredType(dt, shape=None)
        else:
            dt = tuple(a.stype for a in arr)
            self._stype = StructuredType(dt, shape=None)

        return dt
    @property
    def dtype(self):
        """Returns the core data type held by the StructuredType that represents the array

        :return:
        :rtype:
        """

        if self._dtype is None:
            if self.is_simple:
                self._dtype = self._stype.dtype
            else:
                self._dtype = self._get_complex_dtype()

        return self._dtype

    @property
    def stype(self):
        """Returns the StructuredType that the array holds data for

        :return:
        :rtype:
        """

        if self._dtype is None:
            # allows us to keep the dtype and stype aligned
            fix_me = self.dtype
        return self._stype

    @property
    def array(self):
        if self.is_simple:
            slices = tuple(slice(0, x) for x in self.filled_to if x > 0) # treating 0 as "all"
            try:
                return self._array[slices]
            except IndexError:
                raise ValueError("can't slice array of shape {} to `filled_to` spec {}".format(
                    self._array.shape,
                    self.filled_to
                ))
        else:
            return self._array
    @property
    def _subarrays(self):
        return self._array.values() if isinstance(self._array, OrderedDict) else self._array

    def axis_shape_indeterminate(self, axis):
        """Tries to determine if an axis has had any data placed into it or otherwise been given a determined shape

        :param axis:
        :type axis:
        :return:
        :rtype:
        """
        indet = self.filled_to[axis] == 0
        if indet and self.stype.shape is not None:
            indet = self.stype.shape[axis] is None
        return indet
    @property
    def has_indeterminate_shape(self):
        """Tries to determine if the entire array has a determined shape

        :param axis:
        :type axis:
        :return:
        :rtype:
        """
        if self.is_simple:
            indet = all(f == 0 for f in self.filled_to)
            if indet and self.stype.shape is not None:
                count_None = 0
                for c in self.stype.shape:
                    if c is None:
                        count_None += 1
                        if count_None > 1:
                            break
                indet = count_None == len(self.filled_to) or count_None > 1
            return indet # eh we'll call this enough for now
        else:
            return any( a.has_indeterminate_shape for a in self._subarrays )

    @property
    def filled_to(self):
        if self._is_simple:
            if self._filled_to is None:
                self._filled_to = [ 0 ] * len(self._array.shape)
            return self._filled_to
        else:
            return [ s.filled_to for s in self._subarrays ]
    @filled_to.setter
    def filled_to(self, filling):
        # print(">>>>", self._filled_to)
        if self._is_simple:
            if isinstance(filling, int):
                # print("  >>", 1, filling, self._filled_to)
                self.filled_to[0] = filling
            elif len(filling) < len(self.filled_to):
                # print("  >>", 2, filling, self._filled_to)
                self._filled_to = list(filling) + self._filled_to[len(filling):]
            else:
                # print("  >>", 3, filling, self._filled_to)
                self._filled_to = list(filling)
        else:
            # thread the setting
            raise NotImplementedError("can't apply filling {} to multidim array {} because I'm lazy".format(
                filling,
                self
            ))

        # print(self._filled_to, "<<<<")
    def set_filling(self, amt, axis = 0):
        if self._is_simple:
            if self._filled_to is None:
                _ = self.filled_to # populates it
            self._filled_to[axis] = amt
        else:
            # gotta propagate the filling down to the bottom axis
            for a in self._subarrays:
                a.set_filling(amt, axis = axis)
    def increment_filling(self, inc = 1, axis = 0):
        if self._is_simple:
            if self._filled_to is None:
                _ = self.filled_to # populates it
            self._filled_to[axis] += inc
        else:
            # gotta propagate the filling down to the bottom axis
            for a in self._subarrays:
                a.increment_filling(inc = inc, axis = axis)

    def __len__(self):
        return len(self.array)

    def empty_array(self, shape = None, num_elements = None):
        """Creates empty arrays with (potentially) default elements

        The shape handling rules operate like this:
            if shape is None, we assume we'll initialize this as an array with a single element to be filled out
            if shape is (None,) or (n,) we'll initialize this as an array with multiple elments to be filled out
            otherwise we'll just take the specified shape

        :param num_elements:
        :type num_elements:
        :return:
        :rtype:
        """

        dt = self.dtype
        stype = self.stype
        if shape is None:
            shape = stype.shape if self.shape is None else self.shape
        if num_elements is None:
            num_elements = self._default_num_elements

        if stype.is_simple:
            if shape is None:
                # means we expect to have single object, not a vector of them
                if stype.default is None:
                    arr = np.empty((1, ), dtype=dt)
                else:
                    arr = np.full((1, ), stype.default, dtype=dt)
            else:
                if any(x is None for x in shape):
                    # means we have an array of indeterminate size in that dimension
                    shape = tuple(num_elements if x is None else x for x in shape)
                # might want to add a check to see if dt is valid fo np.array
                if stype.default is None:
                    arr = np.empty(shape, dtype=dt)
                else:
                    arr = np.full(shape, stype.default, dtype=dt)
            return arr
        elif isinstance(dt, OrderedDict):
            return OrderedDict((k, StructuredTypeArray(s)) for k,s in dt.items())
        else:
            return tuple( StructuredTypeArray(s) for s in dt )

    def extend_array(self, axis = None):
        array = self._array
        if isinstance(array, np.ndarray):
            ax = self.extension_axis if axis is None else axis
            empty = self.empty_array(shape = self._array.shape) # should effectively double array size?
            self._array = np.concatenate(
                (
                    array,
                    empty
                ),
                axis=ax
            )
        elif isinstance(array, OrderedDict):
            for arr in array.items():
                arr.extend_array( axis = axis )
        else:
            for arr in array:
                arr.extend_array( axis = axis )

    def __setitem__(self, key, value):
        self.set_part(key, value)
    def set_part(self, key, value):
        """Recursively sets parts of an array if not simple, otherwise just delegates to NumPy

        :param key:
        :type key:
        :param value:
        :type value:
        :return:
        :rtype:
        """
        if self.is_simple:
            # this means that self._array is a plain numpy array

            append_chops = 1 if isinstance(key, int) else len(key) # how many dimensions in we dove for the append

            if isinstance(key, int) and self._array.shape[0] == key:
                self.extend_array(axis=0)
            else:
                for i, k in enumerate(key):
                    if k == self._array.shape[i]:
                        self.extend_array(axis=i)

            if value is not None: # we use None as a placeholder for the default value because we need it for Optional patterns
                if isinstance(value, StructuredTypeArray):
                    value = value.array
                residual_dims = len(self._array.shape) - append_chops
                if isinstance(value, np.ndarray) and residual_dims > 0:
                    if residual_dims == 0:
                        # nothing to do here
                        pass
                    elif residual_dims == 1:

                        value = value.flatten()
                        # we can actually manage to do some padding, so why not do so?
                        if self.axis_shape_indeterminate(append_chops):
                            curr_slices =  (slice(None, None), ) * append_chops
                            slices =  curr_slices + ( slice(0, value.shape[0]), )
                            self._array = self._array[slices]
                            # we do
                        else:
                            # we only allow 1D padding for now...
                            num_els = len(value)
                            # try:
                            num_needed = len(self._array[key])
                            # except:
                            #     print("Oh foook", key, self._array.shape)
                            #     raise
                            if num_els < num_needed:
                                if self.padding_mode == 'repeat':
                                    repeats = int(np.ceil(num_needed/num_els))
                                    value = np.tile(value, repeats)[:num_needed]
                                elif self.padding_mode == 'last':
                                    value = np.concatenate((value, np.full(num_needed-num_els, value[-1])))
                                elif self.padding_mode == 'fill':
                                    value = np.concatenate((value, np.full(num_needed-num_els, self.padding_value)))
                                else:
                                    raise StructuredTypeArrayException("unknown padding_mode '{}'".format(
                                        self.padding_mode
                                    ))

                    else:
                        # we can now determine our shape and so we will force the shape
                        take_all = slice(None, None)
                        curr_slices = (take_all, ) * append_chops
                        slices = curr_slices + tuple(
                            slice(0, x) if self.axis_shape_indeterminate(i) else take_all
                            for i, x in enumerate(value.shape)
                        )
                        self._array = self._array[slices]

                    value = value.astype(self.dtype)
                    # string check
                    if self.dtype is str and (self._array.dtype != value.dtype):
                        num_arr = int(str(self._array.dtype).strip("<US|"))
                        num_val = int(str(value.dtype).strip("<US|"))
                        if num_arr < num_val:
                            self._array = self._array.astype(value.dtype)
                    # print(key, value, self._array.shape)
                    self._array[key] = value
                    fill=self.filled_to

                    chopped_fill = fill[:append_chops]
                    max_chops = list(max(a, s) for a,s in zip(fill[append_chops:], value.shape))
                    self.filled_to =  chopped_fill + max_chops

            if isinstance(key, int) and key == self.filled_to[0]:
                self.filled_to[0] += 1
            elif key == tuple(self.filled_to[:append_chops]):
                self.filled_to[append_chops-1] += 1

        else:
            if isinstance(key, (int, str)):
                key = [ key ] * len(self._array)
            if value is None:
                value = [ None ] * len(self._array)

            if len(value) > len(self._array):
                # means we need to reshape our value arrays since it came from a groups call but groups just returns flat values,
                # not the actual grouping we might want
                shapes = self.shape
                # since we're assigning to a slice I guess this means we're assigning the shape of element 2?
                # so we can just split it by the number of elements we need from shape[1:] and then use np.reshape on that
                # ah but we have pre-populated a number of the np.array rows so really we need to go from shape[2:]
                old_value = value
                value = [ None ]*len(shapes)
                for i, s in enumerate(shapes):
                    sub_shape = s[1:]
                    s_num = np.product(sub_shape) # get the number of elements
                    if s_num == 1:
                        value[i] = old_value[0]
                        old_value = old_value[1:]
                    else:
                        v = old_value[:s_num]
                        old_value = old_value[s_num:]
                        value[i] = np.reshape(np.array(v, dtype=object), sub_shape)

            for a, k, v in zip(self._array, key, value):
                a[k] = v
    def __getitem__(self, item):
        return self.get_part(item, use_full_array=False)
    def get_part(self, item, use_full_array = True):
        """If simple, delegates to NumPy, otherwise tries to recursively get parts...?
        Unclear how slicing is best handled here.

        :param item:
        :type item:
        :return:
        :rtype:
        """
        if use_full_array:
            arr = self._array
        else:
            arr = self.array
        if self.is_simple:
            return arr[item]
        else:
            if isinstance(item, tuple):
                compound_index = True
                first_thingy = item[0]
            else:
                compound_index = False
                first_thingy = item
            if isinstance(first_thingy, slice) and compound_index:
                bits = arr[first_thingy]
                return [ b[item[1:]] for b in bits ]
            elif isinstance(first_thingy, slice):
                raise NotImplemented("Not sure how I want to slice StructuredType objects yet")
            elif compound_index:
                bit = arr[first_thingy]
                return bit[item[1:]]
            else:
                bit = arr[first_thingy]
                return bit

    def add_axis(self, which = 0, num_elements = None, change_shape = True):
        """Adds an axis to the array, generally used for expanding from singular or 1D data to higher dimensional
        This happens with parse_all and repeated things like that

        :param which:
        :type which:
        :param num_elements:
        :type num_elements:
        :return:
        :rtype:
        """

        # print(">>>>>", self)
        change_shape = True # just gonna see what effect this has...?
        if self.is_simple:
            if self.has_indeterminate_shape:
                import copy
                self._stype = copy.copy(self._stype)

                if self.stype.shape is None:
                    if change_shape:
                        self.stype.shape = (None,)
                        self.filled_to = [0] + self.filled_to
                else:
                    st = self.stype
                    s1 = st.shape
                    st.shape = (None,) + st.shape
                    self._array = None
                    self._array = self.empty_array(num_elements = self._default_num_elements if num_elements is None else num_elements)
                    if not change_shape:
                        self.stype.shape = s1
                    self.filled_to = [0] + self.filled_to

            elif self.stype.shape is not None:
                # if we've already got a flexible shape I think we actually don't need to add an axis...
                # otherwise we add an axis where specified, but in practice which will never not be 0

                shape = self._array.shape # just broadcast the numpy array
                if num_elements is None:
                    num_elements = self._default_num_elements

                new_shape_1 = shape[:which] + (1,) + shape[which:] # we gotta broadcast to the 1 first for some corner cases
                self._array = np.broadcast_to(self._array, new_shape_1)
                new_shape_2 = shape[:which] + (num_elements,) + shape[which:] # now we can fully broadcast
                self._array = np.broadcast_to(self._array, new_shape_2).copy()

                # now we change up the shape of our stype...?
                # maybe this should be possible to have it _not_ do so that we can put all shape
                # changes on the outer-most level...
                if change_shape:
                    import copy
                    # print(self.stype)
                    self._stype = copy.copy(self.stype)
                    if self._stype.shape is None:
                        self._stype.shape = (None,)
                    else:
                        self._stype.shape = self._stype.shape[:which] + (None,) + self._stype.shape[which:]
                    self._dtype = None # resets this flag so the caching is broken

                if self.filled_to[which] > 0:
                    self.filled_to = [1] + self.filled_to # we have one copy of our data already
                else:
                    self.filled_to = [0] + self.filled_to

        else:
            # we'll recursively add axes to the subarrays I think...
            # this might not be entirely in keeping with how a np.ndarray works but I think is acceptable
            arr = self._array
            if isinstance(arr, OrderedDict):
                aiter = arr.values()
            else:
                aiter = arr
            for a in aiter:
                a.add_axis(which, num_elements, change_shape = False)
                # eeeeh is this right?
                # for this to really make sense we'd need to both

            if isinstance(arr, OrderedDict):
                self._stype.dtype = OrderedDict((k, a.stype) for k, a in arr.items())# feels weird to have to correct this but oh well
            else:
                self._stype.dtype = tuple(a.stype for a in arr) # I think this is how the new thing will be typed...

            # now we change up the shape of our stype, hopefully only at the outer level...
            if change_shape:
                import copy
                self._stype = copy.copy(self._stype)
                if self._stype.shape is None:
                    self._stype.shape = (None,)
                else:
                    self._stype.shape = (None,) + self._stype.shape
                self._dtype = None # resets this flag so the caching is broken

        # print(self, "<<<<<")

    def can_cast(self, val):
        """Determines whether val can probably be cast to the right return type and shape without further processing or if that's definitely not possible

        :param val:
        :type val:
        :return:
        :rtype:
        """

        castable = self.is_simple
        if castable:
            try:
                val = np.asarray(val).astype(self.dtype) # make sure we can use .shape
            except ValueError:
                castable = False
            else:
                if not isinstance(val, np.ndarray) or val.shape == ():
                    # can't cast to some shape so we gotta have no shape and be a primitive type
                    castable = self.shape is None
                elif len(self.shape) == 1:
                    # gotta have a vector of values but we can fill without filling the entire thing I'd say
                    castable = len(val.shape) == 1
                else:
                    axis, block_size, remainder, shape = self._get_casting_shape(val)
                    castable = remainder == int(remainder)

        return castable

    def append(self, val, axis=0):
        """Puts val in the first empty slot in the array

        :param val:
        :type val:
        :return:
        :rtype:
        """
        axis = axis + max(self.append_depth, 0)
        # print("real_axis:", axis)
        if self.is_simple:
            pos = tuple(self.filled_to[:axis+1])
            # print(pos)
        else:
            pos = [tuple(f[:axis+1]) for f in self.filled_to]
        self[pos] = val
        # self._filled_to+=1 # handled automatically by a small bit of cleverness in the filling code

    def _get_casting_shape(self, val, axis = None):
        axis = self.extension_axis if axis is None else axis
        vs = val.shape
        ss = self.shape
        vs_a = vs[:axis] + vs[axis+1:]
        ss_a = ss[:axis] + ss[axis+1:]
        my_stuffs = np.product(ss_a)
        if vs_a != ss_a and my_stuffs > 0:
            total_stuffs = np.product(vs)
            remaining_stuff = total_stuffs/my_stuffs
            new_shape = ss[:axis] + (int(remaining_stuff),) + ss[axis+1:]
        else:
            # the my_stuffs == 0 is corner case for super indeterminate shapes...?
            # Or means I need to handle the shape more intelligently...
            # at this point what is there to do? Off of the extension axis none of the stuff has been filled...?
            # does this just mean we basically return the shape of the current thing? This will be used in two places:
            # SA.extend and SA.can_cast
            # in the first case this should basically tell us that we should just call SA.fill?
            # in the latter this should basically be saying "yes, can cast"?
            remaining_stuff = 0
            new_shape = vs
        return axis, my_stuffs, remaining_stuff, new_shape

    def extend(self, val, single = True, prepend = False, axis = None):
        """Adds the sequence val to the array

        :param val:
        :type val:
        :param single: a flag that indicates whether val can be treated as a single object or if it needs to be reshapen when handling in non-simple case
        :type single: bool
        :return:
        :rtype:
        """

        if isinstance(val, StructuredTypeArray):
            val = val.array

        if self.is_simple:
            # we have a minor issue here, where we might not actually have any data in our array and in that case we
            # won't know what 'extend' means
            # in that case I think we would be safe enough delegating to 'fill'
            # print(":::P", self.shape, self._array.shape, val.shape)
            if self.has_indeterminate_shape and len(self.shape) == len(val.shape if isinstance(val, np.ndarray) else val):
                return self.fill(val)


            # print("...?", axis)

            # now check for shape mismatches so they may be corrected _before_ the insertion
            ax, block_size, remainder, new_shape = self._get_casting_shape(val, axis = axis)
            # print(ax)
            malformed = remainder != int(remainder)
            if malformed:
                if int(remainder) == 0:
                    # there is a use case where we fill our array with a smaller, padded form of our data?
                    # we'll do this initially for only like 1D data...?
                    val = val.flatten()
                    num_els = len(val)
                    repeats = int(np.ceil(block_size/num_els))
                    val = np.tile(val, repeats)[:block_size]
                    ax, block_size, remainder, new_shape = self._get_casting_shape(val)
                    malformed = remainder != int(remainder)
                if malformed:
                    raise StructuredTypeArrayException("{}.{}: object with shape '{}' can't be used to extend array of shape '{}' along axis '{}'".format(
                        type(self).__name__,
                        'extend',
                        val.shape,
                        self.shape,
                        ax
                    ))
            val = val.reshape(new_shape).astype(self.dtype)

            filling = self.filled_to[ax] # I should update this so it works for axis != 0 too...
            # print(self._array.shape, val.shape, val)
            self._array = np.concatenate(
                (
                    self._array[:filling],
                    val
                ) if not prepend else (
                    val,
                    self._array[:filling]
                ),
                axis = ax
            )
            self.filled_to[ax] = self._array.shape[ax]
        else:
            if not single and isinstance(val, np.ndarray): # single alone might tell us we have an issue...
                if len(val[0]) == len(self._array):
                    # just need to transpose the groups, basically,
                    gg = val.T
                else:
                    blocks = [b.block_size for b in self._subarrays]
                    # we'll assume val is an np.ndarray for now since that's the most common case
                    # but this might not work in general...
                    gg = [ None ] * len(blocks)
                    sliced = 0
                    for i, b in enumerate(blocks):
                        if b == 1:
                            gg[i] = val[:, sliced]
                        else:
                            gg[i] = val[:, sliced:sliced+b]
                        sliced += b
                val = gg
            for a, v in zip(self._array, val):
                a.extend(v, axis = axis)

    # def fill_to(self, level, fill_value = None):
    #     """Sets the filled_to parameter level of the array to level, optionally filling with a value
    #
    #     :param level:
    #     :type level:
    #     :param fill_value:
    #     :type fill_value:
    #     :return:
    #     :rtype:
    #     """
    #
    #     base_level = self.filled_to[0]
    #     if level > base_level:
    #         if self.is_simple:
    #             self.filled_to[0] = level
    #         else:
    #             for a in self._subarrays:
    #                 a.fill_to(level, fill_value=fill_value)
    #     # uh... what do about the fill_value?
    #     if fill_value is not None:
    #         raise NotImplementedError("I still haven't implemented fill_value yet ;_;")

    def fill(self, array):
        """Sets the result array to be the passed array

        :param array:
        :type array: str | np.ndarray
        :return:
        :rtype:
        """

        # I do want to make it so that I don't lose my shape, though... or at least I don't lose the
        # entire structure of my shape. I can't try to preserve _too_ many axes, but I can at least preserve the number
        # of them I think...

        if isinstance(array, str):
            array = self.cast_to_array(array)
        elif isinstance(array, StructuredTypeArray):
            array = array.array

        if self._is_simple:
            # not sure why the array _wouldn't_ be an np.ndarray but there's a lot going on and I'm tired and don't
            # want to figure it out
            if isinstance(array, np.ndarray):
                # one big thing we have to watch out for is shrinking the array along some _determined_ axis
                # it doesn't matter what we do to indeterminate axes, but determined ones matter
                if self.dtype is not None:
                    array = array.astype(self.dtype)
                shp = self._stype.shape
                if shp is None or all(a is None for a in shp):
                    self._array = array
                else:
                    ashp = array.shape
                    if len(shp) > len(ashp):
                        # we'll force it to have the right number of axes?
                        # to be honest this is subtle and I'm not going to handle it all properly right now
                        # definitely a source of bugs in the future
                        # I would also probably need to handle the higher-dimensional case too

                        new_shp = [ 1 ] * len(shp)
                        indet_axis_count = len(shp) - len(ashp)
                        which_array_shp = 0
                        for i in range(len(shp)):
                            # we will either pull from shp if the thing is a determined axis
                            # or from ashp if not, but only after we've exhausted the index_axis_count
                            if shp[i] is not None:
                                new_shp[i] = shp[i]
                            elif indet_axis_count > 0:
                                indet_axis_count -= 1
                            else:
                                new_shp[i] = ashp[which_array_shp]
                                which_array_shp += 1
                        array = array.reshape(new_shp)
                        self._array = array
                    elif len(shp) == len(ashp):
                        # we should _probably_ check to make sure we're not breaking things but... meh
                        self._array = array
                    else:
                        raise NotImplementedError("Filling data of shape {} into {}, but I don't know how to coerce a higher dimensional shape into a lower dimensional one".format(
                            shp,
                            ashp
                        ))

                    # should we also force the values to be right...?

            else:
                # print(">>>>", array)
                self._array = np.array(array, dtype=str)

            # at this point we should deal with the filling level that we already had, I think...
            ft = self.filled_to[0]
            if ft > 0:
                # this isn't going to respect the axis quite right but I
                # don't want to deal with that yet...
                # basically just gonna add stuff to make it so we have the right
                # shape for the claimed filling level
                self._array = np.concatenate(
                    (
                        self._array[:ft],
                        self._array
                    )
                )
            self.filled_to = self._array.shape
        else:
            for arr, data in zip(self._array, array):
                arr.fill(data)
            self._filled_to = None

    def cast_to_array(self, txt):
        """Casts a string of things with a given data type to an array of that type and does some optional
        shape coercion

        :param txt:
        :type txt: str | iterable[str]
        :return:
        :rtype:
        """
        if self.is_simple:
            if len(txt.strip()) == 0:
                arr = np.array([], dtype=self._stype)
            else:
                try:
                    # we'll try the base conversion first just assuming we got a number or whatever
                    # and it managed to filter through the code to here
                    arr = np.array([txt], dtype=self._stype)
                except TypeError:
                    import io
                    arr = np.loadtxt(io.StringIO(txt), dtype=self._stype)
                    shape = np.array(self.shape)
                    axis = self.extension_axis
                    if shape is not None and shape[axis] > 0: # make sure arr needs to be reshaped...
                        arr = arr.flatten()
                        num_els = len(arr)
                        # the number of elements that the length of the parsed out array must be divisible by
                        num_not_along_axis = np.product(shape) / shape[axis]
                        if num_els % num_not_along_axis == 0:
                            # means we can cleanly reshape it once we know the target shape
                            shape[self.extension_axis] = num_els / num_not_along_axis
                            arr = arr.reshape(shape)
                        # should we raise an error if not?

        else: #we'll assume we got some block of strings since there's no reason to put a parser here...
            arr = [a.cast_to_array(t) for a, t in zip(self._array, txt)]

        return arr

    def __repr__(self):
        return "{}(shape={}, dtype={})".format(
            type(self).__name__,
            self.shape,
            self.dtype
        )


class StructuredTypeArrayException(Exception):
    pass