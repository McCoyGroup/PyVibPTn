"""
Simple utilities that support constructing Regex patterns
"""

import re
from collections import OrderedDict
from .StructuredType import StructuredType, DisappearingType
__reload_hook__ = [".StructuredType"]

__all__ = [
    "RegexPattern",
    "Capturing",
    "NonCapturing",
    "Optional",
    "Alternatives",
    "Longest",
    "Shortest",
    "Repeating",
    "Duplicated",
    "PatternClass",
    "Parenthesized",
    "Named",
    "Any",
    "Sign",
    "Number",
    "Integer",
    "PositiveInteger",
    "ASCIILetter",
    "AtomName",
    "WhitespaceCharacter",
    "Whitespace",
    "Word",
    "WordCharacter",
    "VariableName",
    "CartesianPoint",
    "IntXYZLine",
    "XYZLine",
    "Empty",
    "Newline",
    "ZMatPattern"
]

######################################################################################################################
#
#                                            RegexPattern
#
class RegexPattern:
    """
    Represents a combinator structure for building more complex regexes

    It might be worth working with this combinator structure in a _lazy_ fashion so that we can drill down
    into the expression structure... that way we can define a sort-of Regex calculus that we can use to build up higher
    order regexes but still be able to recursively inspect subparts?
    """
    def __init__(self,
                 pat,
                 name = None,
                 children = None,
                 parents = None,
                 dtype = None,
                 repetitions = None,
                 key = None,
                 joiner = "",
                 join_function = None,
                 wrapper_function = None,
                 suffix = None, # often we need a start or end RegexPattern for the match, but we don't actually care the data in it
                 prefix = None,
                 parser = None, # a parser function we can use instead of the default StringParser parser
                 handler = None, # a data handler for after capturing a block
                 default_value = None, # default value for Optional types
                 capturing = None,
                 allow_inner_captures = False # whether or not to multiple captures for a given pattern
                 ):
        """
        :param pat:
        :type pat: str | callable
        :param name:
        :type name: str
        :param dtype:
        :type dtype:
        :param repetitions:
        :type repetitions:
        :param key:
        :type key:
        :param joiner:
        :type joiner:
        :param children:
        :type children:
        :param parents:
        :type parents:
        :param wrapper_function:
        :type wrapper_function:
        :param suffix:
        :type suffix:
        :param prefix:
        :type prefix:
        :param parser:
        :type parser:
        :param handler:
        :type handler:
        :param capturing:
        :type capturing:
        :param allow_inner_captures:
        :type allow_inner_captures:
        """

        if isinstance(pat, (tuple, list)) and children is None:
            children = pat
            pat = lambda p, no_capture = False:p # basically represents a compound pattern...
        self._pat = pat
        self._cached = None
        self._comp = None

        self.name = name # not used for anything but can make the code clearer

        # will some day be useful in automatic type determination in a StringParser
        self._dtype = dtype
        self.repetitions = repetitions

        self.key = key # to be used when matching to find subcomponents

        # we'll build a regex tree structure
        # that we can use to walk the regex string structure
        if children is None:
            children = []
        elif not isinstance(children, list):
            if isinstance(children, str):
                children = [ children ]
            else:
                children = list(children)

        children = [ c if isinstance(c, RegexPattern) else NonCapturing(c) for c in children ]
        for c in children:
            c.add_parent(self)
        self._children = children
        self._child_map = None
        if parents is None:
            parents = []
        elif not isinstance(parents, list):
            parents = list(parents)
        self._parents = parents # we need this so we can propagate recompile calls all the way up
        self._joiner = joiner
        if join_function is None:
            def join_function(j, k, **ignore):
                try:
                    return j.join(k)
                except TypeError:
                    pass
                raise ValueError("failed to join {} with {}".format(k, j))
        elif join_function == "safe":
            join_function = self._join_kids
        self._join_function = join_function # sometimes we might want to join the kids in a slightly different way...
        self._wrapper_function = wrapper_function
        self._combine_args = ()
        self._combine_kwargs = {}

        self._prefix = prefix
        self._suffix = suffix
        self.parser = parser
        self.handler = handler
        self.default_value = default_value

        self._capturing = capturing
        self.allow_inner_captures = allow_inner_captures
        self.has_named_child = any((c.has_named_child or c.key is not None) for c in self._children)
        # since I chose to let named groups take precedence over regular groups I guess I need to be cognizant of this...
        self.has_capturing_child = any((c.has_capturing_child or c.capturing) for c in self._children)
        self._capturing_groups = None



    @property
    def pat(self):
        return self._pat
    @pat.setter
    def pat(self, pat):
        self._pat = pat
        self.invalidate_cache()
    @property
    def children(self):
        """

        :return:
        :rtype: tuple[RegexPattern]
        """
        return tuple(self._children)
    @property
    def child_count(self):
        """

        :return:
        :rtype: int
        """
        return len(self._children)
    @property
    def child_map(self):
        """Returns the map to subregexes for named regex components

        :return:
        :rtype: Dict[str, RegexPattern]
        """
        # we'll make a new OD every time to decrease the risk of mutability breakages
        # if self._child_map is None:
        #     self._child_map = OrderedDict( (r.key, r) for r in self._children if r.key is not None )
        return OrderedDict( (r.key, r) for r in self._children if r.key is not None )
    @property
    def parents(self):
        """

        :return:
        :rtype: tuple[RegexPattern]
        """
        return tuple(self._parents)
    @property
    def joiner(self):
        """

        :return:
        :rtype: str
        """
        return self._joiner
    @joiner.setter
    def joiner(self, j):
        self._joiner = j
        self._cached = None

    @property
    def join_function(self):
        """

        :return:
        :rtype: function
        """
        return self._join_function

    @join_function.setter
    def join_function(self, j):
        self._join_function = j
        self._cached = None
    @property
    def suffix(self):
        """

        :return:
        :rtype: str | RegexPattern
        """
        return self._suffix
    @suffix.setter
    def suffix(self, e):
        self._suffix = e
        self._cached = None
    @property
    def prefix(self):
        """

        :return:
        :rtype: str | RegexPattern
        """
        return self._prefix
    @prefix.setter
    def prefix(self, s):
        self._prefix = s
        self._cached = None
    @property
    def dtype(self):
        """Returns the StructuredType for the matched object

        The basic thing we do is build the type from the contained child dtypes
        The process effectively works like this:
            If there's a single object, we use its dtype no matter what
            Otherwise, we add together our type objects one by one, allowing the StructuredType to handle the calculus

        After we've built our raw types, we compute the shape on top of these, using the assigned repetitions object
        One thing I realize now I failed to do is to include the effects of sub-repetitions... only a single one will
        ever get called.

        :return:
        :rtype: None | StructuredType
        """
        dt = self._dtype

        # might be worth introducing some level of caching for this dtype so that we're not recomputing it
        # over and over for recursive processes
        if dt is None and self.child_count > 0:
            subdts = self.get_capturing_groups(allow_inners=False) # the recursive dtype computation will pick this up
            if len(subdts) == 0:
                subdts = self._children
            named = OrderedDict((g.key, g) for g in subdts if g.key is not None)
            if len(named) > 0:
                subdts = OrderedDict((k, g.dtype) for k, g in named.items())
            else:
                subdts = [ g.dtype for g in subdts ]

            if len(subdts) == 0:
                dt = StructuredType(str, default_value=self.default_value)
            elif isinstance(subdts, OrderedDict):
                dt = StructuredType(subdts, default_value=self.default_value)
            elif len(subdts) == 1:
                dt = subdts[0] # singular type
                if not isinstance(dt, StructuredType):
                    dt = StructuredType(dt, default_value=self.default_value)
            else:
                import functools as ft
                from operator import add

                if not isinstance(subdts[0], StructuredType):
                    subdts[0] = StructuredType(subdts[0], default_value=self.default_value)
                dt = ft.reduce(add, subdts) # build the compound type
            if self.repetitions is not None:
                reps = self.repetitions
                if isinstance(reps, int):
                    reps = (reps, None)
                dt = dt.repeat(*reps)
        elif dt is not None and (not isinstance(dt, StructuredType)):
            dt = StructuredType(dt, default_value=self.default_value)

        return dt

    @property
    def is_repeating(self):
        return isinstance(self.repetitions, tuple)
    @property
    def capturing(self):
        return self._capturing or (self._capturing is None and self.is_repeating and self.has_capturing_child)
    @capturing.setter
    def capturing(self, cap):
        self._capturing = cap

    def get_capturing_groups(self, allow_inners = None):
        """
        We walk down the tree to find the children with capturing groups in them and
        then find the outermost RegexPattern for those unless allow_inners is on in which case we pull them all
        """
        groups = []
        if allow_inners is None:
            allow_inners = self.allow_inner_captures
        for g in self._children:
            if g.capturing:
                groups.append(g)
                if allow_inners and g.has_capturing_child:
                    groups.extend(g.get_capturing_groups(allow_inners = True))
            elif g.has_capturing_child:
                groups.extend(g.get_capturing_groups(allow_inners = allow_inners))
        return groups

    @property
    def captures(self):
        """Subtly different from capturing n that it will tell us if we need to use the group in post-processing, essentially

        :return:
        :rtype:
        """

        return self.capturing or self.has_capturing_child or self.has_named_child
    @property
    def capturing_groups(self):
        """Returns the capturing children for the pattern

        :return:
        :rtype:
        """

        if not self.capturing and not self.has_capturing_child:
            return None
        elif not self.capturing:
            # we walk down the tree at this point, finding the outer-most capturing groups in a flat structure
            # I'd make it a tree but regex just returns captured stuff, it doesn't make a tree out of them
            if self._capturing_groups is None:
                self._capturing_groups = tuple(self.get_capturing_groups(allow_inners=self.allow_inner_captures))
            return self._capturing_groups
        else:
            return self

    @property
    def named_groups(self):
        """Returns the named children for the pattern

        :return:
        :rtype:
        """

        named = (self.key is not None)
        if not named and not self.has_named_child:
            return None
        elif not named:
            # we walk down the tree at this point, finding the outer-most named capturing groups in a flat structure
            # I'd make it a tree but regex just returns captured stuff, it doesn't make a tree out of them
            caps = self.capturing_groups
            if caps is None or caps is self:
                groups = None
            else:
                groups = OrderedDict(
                    (g.key, g) for g in caps if g.key is not None
                )
                if len(groups) == 0:
                    groups = None
            return groups
        else:
            return OrderedDict(((self.key, self), ) )

    # we want a Regex component to be reusable in lots of different Regexes without having to copy all of them
    # so we'll make it so that we can only walk _down_ a regex tree, and not worry about having to go back up
    # @property
    # def siblings(self):
    #     return

    def combine(self, other, *args, **kwargs):
        """Combines self and other

            :param other:
            :type other: RegexPattern | str
            :return:
            :rtype: str | callable
            """
        op = other.pat if isinstance(other, RegexPattern) else other
        if isinstance(self.pat, str) and isinstance(op, str):
            return self.pat + op
        elif isinstance(self.pat, str):
            def apply_op(p, *arg, wrap = op, prefix = self.pat, a = args, kw = kwargs, **kwarg):
                return prefix + wrap(p, *a, **kw)
            return apply_op
        elif isinstance(op, str):
            return self.pat(op, *args, **kwargs)
        else:
            def merge_ops(p, *arg, wrap = op, prewrap = self.pat, a = args, kw = kwargs, **kwarg):
                return prewrap(wrap(p, *a, **kw))
            return merge_ops

    def wrap(self, *args, **kwargs):
        """
        Applies wrapper function
        """
        self._combine_args = args
        self._combine_kwargs = kwargs
        if self._wrapper_function is not None:
            self._wrapper_function(self, *args, **kwargs)

    @staticmethod
    def _join_kids(joiner, kids, no_capture=True):
        if len(kids) > 1:
            kids = [ group(k, no_capture=no_capture) if not is_grouped(k) else k for k in kids ]
        return joiner.join(kids)
    def build(self,
              joiner = None,
              prefix = None,
              suffix = None,
              recompile = True,
              no_captures = False,
              verbose = False
              ):
        # might want to add a flag that checks if a block has already been wrapped in no-capture? That would cut down
        # on regex length...
        if recompile or self._cached is None:
            # this recompile rule might be excessive, but to be honest these compilations are so cheap it wouldn't even really
            # matter if recompile were always True
            recomp_captures = self.capturing and not self.allow_inner_captures
            no_caps = no_captures or recomp_captures
            kids = [
                c.build(
                    recompile = True if recompile else ('No Cache' if (c.has_capturing_child and recomp_captures) else False),
                    no_captures = no_caps
                ) for c in self._children
            ]

            if verbose:
                print("Compiling regex from: {}".format(kids))

            if joiner is None:
                joiner = self.joiner
            if prefix is None:
                prefix = self.prefix
            if suffix is None:
                suffix = self.suffix
            joiner = joiner.build( no_captures = no_caps ) if isinstance(joiner, RegexPattern) else joiner if joiner is not None else ""
            prefix = prefix.build( no_captures = no_caps ) if isinstance(prefix, RegexPattern) else prefix if prefix is not None else ""
            suffix = suffix.build( no_captures = no_caps ) if isinstance(suffix, RegexPattern) else suffix if suffix is not None else ""
            if verbose:
                print("Joiner:{}\n Prefix:{}\n Suffix:{}".format(joiner, prefix, suffix))

            # the big question now is how do I figure out if a Capturing was applied...?
            # I guess since I'm walking _down_ the tree I don't even need to check for that
            # I can just check self.capturing... huh wow
            if no_captures:
                # I guess we temporarily make our pattern a non-capturing one...?

                kwargs = self._combine_kwargs.copy()
                kwargs['no_capture'] = True

                comp = self.combine(
                    #unclear whether I should be putting the prefix/suffix inside or outside this ._.
                    prefix + self.join_function(joiner, kids, no_capture=True) + suffix,
                    *self._combine_args,
                    **kwargs
                )
            else:
                if 'no_capture' in self._combine_kwargs:
                    kwargs = self._combine_kwargs.copy()
                    kwargs['no_capture'] = (not self.capturing)
                else:
                    kwargs = self._combine_kwargs
                comp = self.combine(
                    #unclear whether I should be putting the prefix/suffix inside or outside this ._.
                    prefix + self.join_function(joiner, kids, no_capture=(not self.capturing)) + suffix,
                    *self._combine_args,
                    **kwargs
                )
            if isinstance(comp, RegexPattern): # to be honest I don't know how we got here...?
                comp = comp.build(
                    recompile= True if recompile else ('No Cache' if (comp.has_capturing_child and recomp_captures) else False),
                    no_captures = no_caps # is this right...?
                )

            if verbose:
                print("End Regex:", comp)

            if no_captures: # basically no_captures means operate in a not-quite-right world?
                return comp

            self._cached = comp

        return self._cached
    @property
    def compiled(self):
        if self._comp is None:
            self._comp = re.compile(self.build())
        return self._comp

    def add_parent(self, parent):
        self._parents.append(parent)
    def remove_parent(self, parent):
        self._parents.remove(parent)

    def add_child(self, child):
        self._children.append(child)
        self._child_map = None
        self.has_capturing_child = self.has_capturing_child or (child.capturing or child.has_capturing_child)
        self.has_named_child = self.has_named_child or (child.has_named_child or child.key is not None)
        self.invalidate_cache()
    def add_children(self, children):
        self._children.extend(children)
        self._child_map = None
        self.has_capturing_child = self.has_capturing_child or any((c.capturing or c.has_capturing_child) for c in children)
        self.has_named_child = self.has_named_child or any((c.has_named_child or c.key is not None) for c in children)
        self.invalidate_cache()
    def remove_child(self, child):
        self._children.remove(child)
        self._child_map = None
        self.has_capturing_child = self.has_capturing_child and any((c.capturing or c.has_capturing_child) for c in self._children)
        self.has_named_child = self.has_named_child and any((c.has_named_child or c.key is not None) for c in self._children)
        self.invalidate_cache()
    def insert_child(self, index, child):
        self._children.insert(index, child)
        self._child_map = None
        self.invalidate_cache()

    def invalidate_cache(self):
        # propagating this up the tree is the _only_ time I need _parents
        # which is a potential source of memory leaks... should I kill off the upwards propagation?
        self._cached = None
        self._comp = None
        self._capturing_groups = None
        for p in self._parents:
            p.invalidate_cache()

    # turns out python is clever enough to automatically handle cases like this?
    # def __del__(self):
    #     # we need this for cleanup to avoid memory leaks when people call
    #     # del ... on the Regex -- it won't work perfectly, but it'll be good enough
    #     for c in self._children:
    #         c.remove_parent(self)
    def __copy__(self):
        from copy import copy
        cls = type(self)
        new = cls.__new__(cls) # avoid init call
        new.__dict__.update(self.__dict__)
        new._children = copy(new._children) # copy the children list
        new._parents = [] # since this is a copy it has no parents
        new._cached = None # just safer...
        return new

    def __add__(self, other):
        """Combines self and other

        :param other:
        :type other: RegexPattern
        :return:
        :rtype:
        """
        if not isinstance(other, RegexPattern):
            other = RegexPattern(other)
        def null_pat(p):
            return p
        return type(self)(
            null_pat,
            children = [self, other]
        )
    def __radd__(self, other):
        """Combines self and other

        :param other:
        :type other: RegexPattern
        :return:
        :rtype:
        """
        if not isinstance(other, RegexPattern):
            other = RegexPattern(other)
        def null_pat(p):
            return p
        return type(self)(
            null_pat,
            children = [other, self]
        )
    def __call__(self, other,
                 *args,
                 # we'll treat this as an alternate constructor block
                 name = None,
                 dtype = None,
                 repetitions = None,
                 key = None,
                 joiner = None,
                 join_function = None,
                 wrap_function = None,
                 suffix = None,
                 prefix = None,
                 multiline = None,
                 parser = None,
                 handler = None,
                 capturing = None,
                 default=None,
                 allow_inner_captures = None,
                 **kwargs
                 ):
        """Wraps self around other

        :param other:
        :type other: RegexPattern
        :return:
        :rtype:
        """
        from copy import copy
        new = copy(self)

        if name is not None:
            new.name = name
        if key is not None:
            new.key = key
        if joiner is not None:
            new._joiner = joiner
        if join_function is not None:
            new._join_function = join_function
        if dtype is not None:
            new._dtype = dtype
        if repetitions is not None:
            new.repetitions = repetitions
        if wrap_function is not None:
            new._wrapper_function = wrap_function
        if suffix is not None:
            new._suffix = suffix
        if prefix is not None:
            new._prefix = prefix
        if parser is not None:
            new.parser = parser
        if handler is not None:
            new.handler = handler
        if default is not None:
            new.default_value = default
        if capturing is not None:
            new.capturing = capturing
        if allow_inner_captures is not None:
            new.allow_inner_captures = allow_inner_captures

        if isinstance(other, (tuple, list)):
            other = [ RegexPattern(o) if not isinstance(o, RegexPattern) else o for o in other ]
            new.add_children(other)
        else:
            if not isinstance(other, RegexPattern):
                other = RegexPattern(other)
            new.add_child(other)
        
        # print(self._wrapper_function, args, kwargs)
        new.wrap(*args, **kwargs)

        return new

    def __repr__(self):
        return "{}(key: {!r}, children : <{}>, pattern : {})".format(
            type(self).__name__,
            self.key,
            len(self._children),
            self._pat
        )
    def __str__(self):
        wat = self.build()
        return wat

    def __getitem__(self, item):
        return self.child_map[item]

    ### Supporting the re mechanisms directly for convenience
    def match(self, txt):
        return re.match(self.compiled, txt)
    def search(self, txt):
        return re.search(self.compiled, txt)
    def findall(self, txt):
        return re.findall(self.compiled, txt)
    def finditer(self, txt):
        return re.finditer(self.compiled, txt)

######################################################################################################################
#
#                                       Concrete RegexPatterns to combine and use
#
#
def is_grouped(p):
    """Takes a string pattern and tries to check if it's already in a singular construct (usually grouped...)

    :param p: pattern
    :type p: str
    :return:
    :rtype:
    """

    # first we'll check if it's syntactically _probably_ grouped
    all_is_well = starts_right = ends_right = p.startswith('(')
    if starts_right:
        ends_right = p.endswith(")")
        all_is_well = starts_right and ends_right
        if all_is_well:
            # we now want to check that we have balanced numbers of ( and ) ...
            count_l = p.count('(')
            count_r = p.count(')')
            all_is_well = count_l == count_r
        if all_is_well:
            # and I guess further check that we after splitting by the first ) we have an _unbalanced_ form
            right_hand = p.split('(', 1)[1]
            count_l = right_hand.count('(')
            count_r = right_hand.count(')')
            all_is_well = count_l == count_r

    if not all_is_well:
        # we'll then check if we have a simple character escape
        starts_right = all_is_well = p.startswith("\\")
        if starts_right:
            all_is_well = len(p) == 2 # indicates we

    if not all_is_well:
        # we'll try to check if we have a simple character class
        starts_right = p.startswith('[')
        if starts_right:
            ends_right = p.endswith("]")
            if starts_right and ends_right:
                all_is_well = p.count('[') == 1 and p.count(']') == 1

    return all_is_well
def group(p, no_capture = False):
    if no_capture:
        return non_capturing(p)
    else:
        return r"("+p+r")"
def non_capturing(p, *a, **kw):
    return r"(?:"+p+r")"
def optional(p, no_capture = False):
    if not is_grouped(p):
        return non_capturing(p) + "?"
    else:
        return p + "?"
def alternatives(p, no_capture = False):
    if is_grouped(p):
        return p
    else:
        return group(p, no_capture=no_capture)
def shortest(p, no_capture = False):
    if not p.endswith("*") | p.endswith("+"):
        if not is_grouped(p):
            return non_capturing(p) + "*?"
        else:
            return p + "*?"
    else:
        return p + "?"
def repeating(p, min = 1, max = None, no_capture = False):
    if not is_grouped(p):
        p = non_capturing(p)
    if max is None:
        if min is None:
            base_pattern = p+"*"
        elif min == 1:
            base_pattern = p+"+"
        else:
            base_pattern = p +"{" + str(min) + ",}"
    elif min == max:
        base_pattern = p +"{" + str(min) + "}"
    else:
        if min is None:
            min = 0
        base_pattern = p +"{" + str(min) + "," + str(max) + "}"
    if no_capture:
        return base_pattern
    else:
        return grp_p(base_pattern)
def duplicated(p, num, riffle="", no_capture = False):
    if isinstance(riffle, RegexPattern):
        riffle = str(riffle)
    return riffle.join([p]*num)
def named(p, n, no_capture=False):
    if no_capture:
        return non_cap_p(p)
    else:
        return "(?P<"+n+">"+p+")"

# wrapper patterns
grp_p = group # capturing group
Capturing = RegexPattern(grp_p, "Capturing", capturing=True)
Capturing.__name__ ="Capturing"
Capturing.__doc__ = """
    Represents a capturing group in a RegexPattern
    """

non_cap_p = non_capturing # non-capturing group
NonCapturing = RegexPattern(non_cap_p, "NonCapturing", dtype=DisappearingType)
NonCapturing.__name__ ="NonCapturing"
NonCapturing.__doc__ = """
    Represents something that should not be captured in a RegexPattern
    """

op_p = optional # optional group
def opnb_p(p, no_capture=False):
    return r"(?:"+p+r")?" # optional non-binding group
Optional = RegexPattern(optional,
                        "Optional"
                        )
Optional.__name__ ="Optional"
Optional.__doc__ = """
    Represents something that should be optional in a RegexPattern
    """

Alternatives = RegexPattern(alternatives, joiner="|")
Alternatives.__name__ ="Alternatives"
Alternatives.__doc__ = """
    Represents a set of alternatives in a RegexPattern
    """

lm_p = repeating
Longest = RegexPattern(lm_p, "Longest")
Longest.__name__ = "Longest"
Longest.__doc__ = """
    Represents that the longest match of the enclosed pattern should be searched for
    """

sm_p = shortest
Shortest = RegexPattern(sm_p, "Shortest")
Shortest.__name__ = "Shortest"
Shortest.__doc__ = """
    Represents that the shortest match of the enclosed pattern should be searched for
    """

def wrap_repeats(self, min = None, max = None, no_capture=None):
    self.repetitions = (min, max)
Repeating = RegexPattern(repeating,
                         "Repeating",
                         wrapper_function=wrap_repeats
                         )
Repeating.__name__ = "Repeating"
Repeating.__doc__ = """
    Represents that the patten can be repeated
    """

def wrap_name(self, n):
    self.key = n
    if self.name is None:
        self.name = n
Named = RegexPattern(
    named,
    "Named",
    wrapper_function = wrap_name,
    capturing=True
)
Named.__name__ = "Named"
Named.__doc__ = """
    Represents a named group. These are _always_ captured, to the exclusion of all else.
    """

def wrap_duplicate_type(self, n, riffle = ""):
    dt = self._dtype
    if isinstance(dt, tuple):
        dt, shape = dt
    else:
        shape = None
    if shape is None:
        shape = (n,)
    else:
        shape = (n,) + shape
    self._dtype = (dt, shape)
Duplicated = RegexPattern(duplicated,
                          "Duplicated",
                          wrapper_function = wrap_duplicate_type
                          )
Duplicated.__name__ = "Duplicated"
Duplicated.__doc__ = """
    Represents an explicitly duplicated pattern
    """

pc_p = lambda p, no_capture = False: r"["+p+r"]" # pattern class
PatternClass = RegexPattern(pc_p, "PatternClass")
PatternClass.__name__ = "PatternClass"
PatternClass.__doc__ = """
    Represents a pattern class, for wrapping other patterns
"""

parened_p = lambda p, no_capture = False: r"\("+p+"\)"
Parenthesized = RegexPattern(parened_p, "Parenthesized")
Parenthesized.__name__ = "Parenthesized"
Parenthesized.__doc__ = """
    Represents that something should be wrapped in parentheses, not treated as Capturing
    """

# raw declarative patters
any_p = "."
Any = RegexPattern(any_p, "Any")
Any.__name__ = "Any"
Any.__doc__ = """
    Represents any character
    """

sign_p = r"[\+\-]"
Sign = RegexPattern(sign_p, "Sign")
Sign.__name__ = "Sign"
Sign.__doc__ = """
    Represents a +/- sign
    """

paren_p = r"\("+".*?"+"\)"

num_p = opnb_p(sign_p)+r"\d*\.\d+" # real number
Number = RegexPattern(num_p, "Number", dtype=float)
Number.__name__ = "Number"
Number.__doc__ = """
    Represents a real number, like -1.23434; doesn't support "E" notation
    """

int_p = opnb_p(sign_p)+r"\d+" # integer
Integer = RegexPattern(int_p, "Integer", dtype=int)
Integer.__name__ = "Integer"
Integer.__doc__ = """
    Represents an integer
    """

posint_p = r"\d+" # only positive integer
PositiveInteger = RegexPattern(posint_p, "PositiveInteger", dtype=int)
PositiveInteger.__name__ = "PositiveInteger"
PositiveInteger.__doc__ = """
    Represents a positive integer (i.e. just a string of digits)
    """

ascii_p = "[a-zA-Z]"
ASCIILetter = RegexPattern(ascii_p, "ASCIILetter", dtype=str)
ASCIILetter.__name__ = "ASCIILetter"
ASCIILetter.__doc__ = """
    Represents a single ASCII letter
    """

name_p = ascii_p+"{1,2}" # atom name
AtomName = RegexPattern(name_p, "AtomName", dtype=str)
AtomName.__name__ = "AtomName"
AtomName.__doc__ = """
    Represents an atom symbol like Cl or O (this is misnamed, I know)
    """

ws_char_class = r"(?!\n)\s" # probably excessive... but w/e I'm not winning awards for speed here
WhitespaceCharacter = RegexPattern(ws_char_class, "WhitespaceCharacter", dtype=str)
WhitespaceCharacter.__name__ = "WhitespaceCharacter"
WhitespaceCharacter.__doc__ = """
    Represents a single whitespace character
    """

ws_p = non_capturing(ws_char_class)+"*" # whitespace
wsr_p = non_capturing(ws_char_class)+"+" # real whitespace
Whitespace = RegexPattern(ws_p, "Whitespace", dtype=str)
Whitespace.__name__ = "WhitespaceCharacter"
Whitespace.__doc__ = """
    Represents a block of whitespace
    """

WordCharacter = RegexPattern("\w", "WordCharacter", dtype=str)
WordCharacter.__name__ = "WordCharacter"
WordCharacter.__doc__ = """
    Represents a single number or letter (i.e. non-whitespace)
    """

Word = RegexPattern("\w+", "Word", dtype=str)
Word.__name__ = "Word"
Word.__doc__ = """
    Represents a block of WordCharacters
    """

VariableName = RegexPattern((ASCIILetter, Word), joiner="", dtype=str)
VariableName.__name__ = "VariableName"
VariableName.__doc__ = """
    Represents a possible variable name sans underscored, basically an ASCIILetter and then a word
    """

ascii_punc_char_class = r"\.,<>?/'\";:{}\[\]\+=\(\)\*&\^%$#@!~`"
ASCIIPunctuation = RegexPattern(ascii_punc_char_class, "ASCIIPunctuation", dtype=str)
ASCIIPunctuation.__name__ = "ASCIIPunctuation"
ASCIIPunctuation.__doc__ = """
    Represents a single piece of punctuation
    """

cart_p = ws_p.join([ grp_p(num_p) ]*3) # cartesian coordinate
CartesianPoint = RegexPattern(cart_p, "CartesianPoint", dtype=(float, (3,)))
CartesianPoint.__name__ = "CartesianPoint"
CartesianPoint.__doc__ = """
    Represents a 'point', i.e. 3 numbers separated by whitespace
    """

acart_p = "("+int_p+")"+ws_p+cart_p # atom coordinate as comes from a XYZ table
IntXYZLine = RegexPattern(acart_p, "IntXYZLine", dtype=(int, (float, (3,))))
IntXYZLine.__name__ = "IntXYZLine"
IntXYZLine.__doc__ = """
    Represents a line in an XYZ file that starts with an int, like
    ```
    1   -1.232323 2.23212421 43.44343434
    ```
    """

aNcart_p = "("+name_p+")"+ws_p+cart_p # atom coordinate as comes from a XYZ table
XYZLine = RegexPattern(aNcart_p, "XYZLine", dtype=(str, (float, (3,))))
XYZLine.__name__ = "XYZLine"
XYZLine.__doc__ = """
    Represents a line in an XYZ file that starts with an atom name, like
    ```
    Cl   -1.232323 2.23212421 43.44343434
    ```
    """

Empty = RegexPattern("", "Empty")
Empty.__name__ = "Empty"
Empty.__doc__ = """
    Represents an empty pattern...I can't remember why this is here
    """

Newline = RegexPattern(r"\n", "Newline", dtype=str)
Newline.__name__ = "Newline"
Newline.__doc__ = """
    Represents a newline character
    """

ZMatPattern = Capturing(AtomName)
for i in range(3):
    ZMatPattern += Optional(NonCapturing(
        Repeating(Whitespace, 1) + Capturing(PositiveInteger) + # ref int
        Repeating(Whitespace, 1) + Capturing(Number) # ref value
    ))
ZMatPattern.name = "ZMatPattern"
ZMatPattern.__name__ = "ZMatPattern"
ZMatPattern.__doc__ = """
    Represents Z-matrix block
    """
