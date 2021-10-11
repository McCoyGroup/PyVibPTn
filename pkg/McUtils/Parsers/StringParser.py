import numpy as np, re
from collections import OrderedDict
from .RegexPatterns import *
from .StructuredType import *

__all__ = [
    "StringParser",
    "StringParserException"
]

__reload_hook__ = ['.RegexPatterns']

########################################################################################################################
#
#                                           StringParser
#
class StringParserException(Exception):
    ...

# """
# One thing that's really tough with this stuff is figuring out how to manage shapes as we move across the different parsers and recursion level
#
# We can use the RegexPattern to manage the original StructuredType -> this seems to be pretty reliable
# We can use that to create an initial StructuredTypeArray -> this seems to be pretty reliable
# Then if we call parse_all, we need to reshape that StructuredTypeArray so that it can take a vector of return values
#     -> I _think_ this works as it should
# Next, no matter which parser we use, we need to get out our match or matches -> this is, again, quite reliable
# At that point we need to figure out how many groups each subarray should have -> this seems to be broken
#
#
# """
#
# """
# StringParser Design:
#
# input: RegexPattern
# output: StructuredTypeArray or list or OrderedDict thereof
#
# Basic Idea:
#
# RegexPattern can track which of its children are tagged as Capturing and which are tagged as Named. These children in turn
# can keep track of their data type (dtype) and also the shape of the data that should come out of their matched groups.
#
# So once we've compiled down RegexPattern--making sure that only groups at the outer-most level capture when compiling--
# what we can do is use the python re module to find the various matched components.
#
# At this point, we need to convert the matched strings into the appropriately formatted NumPy arrays. To handle this in
# recursive fashion I created a StructuredTypeArray class that has two flavors, simple and compound, which respectively
# handle the direct communication with NumPy and the delegation and distribution of higher-order data to simpler arrays.
#
# So in our parsing process we need to create a StructuredTypeArray for the found dtype from the RegexPattern. This is
# relatively straightforward. Next, we might be using parse_all which asks for a vector of results back. If this is the
# case we need to take our initial StructuredTypeArray and turn it into, in effect, a StructuredTypeArray vector. Rather
# than use an inefficient python list format, though, we call StructuredTypeArray.add_axis which broadcasts to a
# higher-dimensional data format. It also changes up the stored StructuredType such that a new axis has been prepended to
# the shapes of each.
#
# Next we have to work with our matched data somehow. There are a few issues here, but probably the biggest one is that the
# number of matched groups doesn't need to line up with the number of blocks we want returned. This means that we might need to take our
# matched groups and use some 'block_size' that each StructuredTypeArray tracks to figure out how to split these groups up.
#
# Assuming that works well, we now apply the block handler for each declared Capturing block to the matched data string.
# For very simple data, this will be effectively doing nothing, so that the StructuredTypeArray can handle the cast to an
# np.ndarray. For compound types or more complex data we need to basically go match-by-match and apply the block parser.
#
# This is where the recursion occurs and where things can get messy. The default block handler for an np.ndarray calls
# parse on a string. This means that parse will be called until the data can finally be put into a StructuredTypeArray
# format. On the other hand, the returned data needs to be consistent with the expected return type of the parent call.
# For that reason, the block handler passes in the return array, and the return array uses its stype when calling the
# subparser. This means, though, that there could be an apparent shape mismatch if the data itself is ragged. To resolve
# that, StructuredTypeArray will pad lower-dimensional data into a higher-dimensional format when possible.
#
# A major complicating factor in all of this is how one manages Repeating patterns. Every Repeating pattern actually needs
# to be managed using parse_all and its own Capturing groups, after first finding the matches for the repeats. Since the
# passed dtype for such a call is also a higher-dimensional object, the outer-most axis of the dtype needs to be dropped,
# then parse_all needs to be called on the child form. Managing this correctly inside a recursive process can be challenging.
#
# """
class StringParser:
    """
    A convenience class that makes it easy to pull blocks out of strings and whatnot
    """

    def __init__(self, regex):
        self.regex = regex

    def parse(self,
              txt,
              regex=None,
              block_handlers=None,
              dtypes=None,
              out=None
              ):
        """Finds a single match for the and applies parsers for the specified regex in txt

        :param txt: a chunk of text to be matched
        :type txt: str
        :param regex: the regex to match in _txt_
        :type regex: RegexPattern
        :param block_handlers: handlers for the matched blocks in _regex_ -- usually comes from _regex_
        :type block_handlers: iterable[callable] | OrderedDict[str: callable]
        :param dtypes: the types of the data that we expect to match -- usually comes from _regex_
        :type dtypes: iterable[type | StructuredType] | OrderedDict[str: type | StructuredType]
        :param out: where to place the parsed out data -- usually comes from _regex_
        :type out: None | StructuredTypeArray | iterable[StructuredTypeArray] | OrderedDict[str: StructuredTypeArray]
        :return:
        :rtype:
        """

        # we'll enforce that the return type be a StructuredTypeArray
        # the enforcement mechanism here will be basically to take the Regex dtype if provided and otherwise
        # to coerce to a StructuredType

        if regex is None:
            regex = self.regex

        if isinstance(regex, RegexPattern) and regex.parser is not None:
            return regex.parser(txt)

        # one lingering question is how to handle `Repeating` patterns.
        # Probably want to extract the child block and feed that into parse_all...
        # might need to figure out some alternate manner for handling dtypes and block_handlers
        # in this case since I think the delegation to the parse_all might fuck stuff up...
        if isinstance(regex, RegexPattern) and regex.is_repeating:
            import copy

            iterable_regex = copy.copy(regex)
            iterable_regex.pat = lambda p, *a, **kw: p # to unbind the Repeating capture screwups
            parser = type(self)(iterable_regex)

            # ah fack how do we handle the dtypes passing?
            # if dtypes was passed...I guess we gotta feed it through, right?
            # Or well maybe not...? If it was passed it should be a _vector_ stype, right?
            # So I guess we gotta strip the outer level of repeats...?
            if out is None:
                if isinstance(dtypes, StructuredType):
                    dtypes = dtypes.drop_axis()
                elif dtypes is not None:
                    dtypes = [d.drop_axis() for d in dtypes]
            else:
                dtypes = None
            if isinstance(out, (dict, OrderedDict)):
                out = out.copy()
                out['single'] = True
            else:
                out = {'array':out, 'single':True}

            res = parser.parse_all(txt, dtypes=dtypes, out=out)

        else:
            if block_handlers is None and isinstance(regex, RegexPattern):
                block_handlers = self.get_regex_block_handlers(regex)
            if dtypes is None and isinstance(regex, RegexPattern):
                dtypes = self.get_regex_dtypes(regex)

            if out is None:
                res = self._set_up_result_arrays(dtypes)
                append = False
                single = None
            elif isinstance(out, (dict, OrderedDict)):
                res = out['array']
                try:
                    append = out['append']
                except KeyError:
                    append = False
                try:
                    single = out['single']
                except KeyError:
                    single = None
            else:
                res = out
                append = False
                single = None

            # if we're given a non-string regex we compile it
            pattern = None
            if isinstance(regex, RegexPattern):
                pattern = regex
                regex = regex.compiled

            match = re.search(regex, txt)
            try:
                self._handle_parse_match(match, res, block_handlers, append=append, single=single)
            except:
                if append or (isinstance(append, int) and append == 0):
                    raise StringParserException('failed to append to results array {} for match {}'.format(
                        res,
                        match
                    ))
                else:
                    raise StringParserException('failed to insert into results array {} for match {}'.format(
                        res,
                        match
                    ))

        # one issue here is that by default this would return a list or some structured object like that
        # sometimes we don't _want_ that list
        # if we have something like Repeating(Capturing(Number), ...)
        # we don't really want [ StructuredTypeArray(...) ] returned
        # what we'd really like to get back is just the StructuredTypeArray
        # now here's the question, is there a case where if we have [ StructuredTypeArray ]
        # that we _wouldn't_ just want the array? My sense is no since StructuredType should simplify the
        # singular type by default...?
        # Or will this be unique to Repeating patterns?

        # Eh, we'll assume this is mis-structured since we've taken a basic RegexPattern and
        # then captured a repeating portion of it or some set of repeating parts of it but
        # didn't bother to simplify to just that Capturing part of the RegexPattern...?
        if not isinstance(res, (OrderedDict, StructuredTypeArray)) and len(res) == 1:
            res = res[0]

        return res

    # we need to add some stuff to generalize out the parser, add some useful general methods
    # like the handling of blocks of unknown size / unknown numbers of atoms / efficient casting from string
    # to NumPy types
    def parse_all(self,
                  txt,
                  regex = None,
                  num_results = None,
                  block_handlers = None,
                  dtypes = None,
                  out = None
                  ):

        if regex is None:
            regex = self.regex

        if block_handlers is None and isinstance(regex, RegexPattern):
            block_handlers = self.get_regex_block_handlers(regex)
        if dtypes is None and isinstance(regex, RegexPattern):
            dtypes = self.get_regex_dtypes(regex)

        if out is None:
            res = self._set_up_result_arrays(dtypes)
            if isinstance(res, StructuredTypeArray):
                res.add_axis()
            elif isinstance(res, (dict, OrderedDict)):
                for r in res.values():
                    r.add_axis()
            else:
                for r in res:
                    r.add_axis()
            append = False
            single = None
        elif isinstance(out, (dict, OrderedDict)):
            res = out['array']
            try:
                append = out['append']
            except KeyError:
                append = False
            try:
                single = out['single']
            except KeyError:
                single = None
        else:
            res = out
            append = False
            single = None

        # if we're given a non-string regex we compile it
        base_regex = regex
        # print(">>>>>>>>>", type(regex.compiled))
        if isinstance(regex, RegexPattern):
            # print("...wat?", type(regex.compiled))
            regex = regex.compiled

        try:
            match_iter = re.finditer(regex, txt)
        except:

            raise Exception(type(regex), type(regex.compiled))
        if num_results is not None:
            import itertools as it
            matches = list(it.islice(match_iter, None, num_results))
        else:
            matches = list(match_iter)

        if matches is None or len(matches)==0:
            raise StringParserException("Pattern {} not found in {}".format(regex, txt))

        self._handle_parse_match(matches, res, block_handlers, append = append, single = single)
        #  we don't need to append since we've pushed the filling iteration into the StructuredArray

        return res

    class MatchIterator:
        class Match:
            def __init__(self, parent, block):
                self.parent = parent
                self.block = block
            @property
            def value(self):
                res = self.parent.parser._set_up_result_arrays(self.parent.dtypes)
                self.parent.parser._handle_parse_match(self.block, res, block_handlers=self.parent.block_handlers)
                return res

        def __init__(self, parser, match_iter, num_results, dtypes, block_handlers):
            if num_results is not None:
                import itertools as it
                match_iter = it.islice(match_iter, None, num_results)

            self.parser = parser
            self.dtypes = dtypes
            self.block_handlers = block_handlers
            self.match_iter = match_iter

        def __iter__(self):
            for match in self.match_iter:
                yield self.Match(self, match)
        def __next__(self):
            match = next(self.match_iter)
            return self.Match(self, match)

    def parse_iter(self, txt,
                  regex = None,
                  num_results = None,
                  block_handlers = None,
                  dtypes = None
                  ):

        if regex is None:
            regex = self.regex

        if block_handlers is None and isinstance(regex, RegexPattern):
            block_handlers = self.get_regex_block_handlers(regex)
        if dtypes is None and isinstance(regex, RegexPattern):
            dtypes = self.get_regex_dtypes(regex)

        # if we're given a non-string regex we compile it
        if isinstance(regex, RegexPattern):
            regex = regex.compiled

        match_iter = re.finditer(regex, txt)
        return self.MatchIterator(self, match_iter, num_results, dtypes, block_handlers)


    #region RegexPattern data extractors
    @classmethod
    def get_regex_block_handlers(cls, regex):
        """Uses the uncompiled RegexPattern to determine what blocks exist and what handlers they should use

        :param regex:
        :type regex: RegexPattern
        :return:
        :rtype:
        """
        # I think it might make sense to allow a given regex to declare its _own_ parser that specifies how it
        # should be used in parse and parse_all. This can remain the default implementation, but that would allow
        # one to push the complexity of figuring out how to handle an object onto the object itself...

        # since we handle repeating patterns by delegating to parse_all by default, these handlers will need to be the
        # handlers for the child parse_all call instead of the basic parser from the first matched child, I think...
        # ah actually more generally we need to make sure that for each captured block (i.e. what powers parser)
        # we have a handler
        # the basic assumption has generally been that all children will be relevant, but as things
        # increase in complexity this assumption breaks down
        # therefore the regex needs to remember itself which of its children are capturing and which aren't

        handlers = regex.named_groups
        if handlers is not None:
            for k,r in tuple(handlers.items()): # we extract the full list first so we can go back and edit the handlers map
                handlers[k] = cls._get_regex_handler(r)
        else:
            # handlers = None
            # if we have no named groups to work with we'll create a block handler for the capturing groups?

            handlers = regex.capturing_groups
            if handlers is not None:
                if isinstance(handlers, RegexPattern):
                    handlers = [ handlers ]

                dtype = regex.dtype
                if dtype.shape is None:
                    handlers = None
                elif dtype.is_simple and len(handlers) == dtype.shape[-1]:
                    handlers = [ None ] # means we _wanted_ to have one block, but we had to declare it over multiple CapturingGroups
                else:
                    handlers = [ cls._get_regex_handler(r) for r in handlers if r.dtype is not DisappearingType ]


        return handlers
    @classmethod
    def get_regex_dtypes(cls, regex):
        """Uses the uncompiled RegexPattern to determine which StructuredTypes to return

        :param regex:
        :type regex: RegexPattern
        :return:
        :rtype:
        """

        dtypes = regex.dtype
        # print(">"*50, dtypes)
        # if dtypes is not None:
        #     for k, r in tuple(dtypes.items()):
        #         dtypes[k] = r.dtype
        # else:## there is some question as to whether I might actually want to make it so the _children_ get parsed out...
        #     dtypes = regex.capturing_groups
        #     if dtypes is not None:
        #         if isinstance(dtypes, RegexPattern):
        #             dtypes = dtypes.dtype
        #         else:
        #             dtypes = tuple(d.dtype for d in dtypes)
        #
        #     else:
        #         dtypes = regex.dtype
        # print(dtypes, "<"*50, sep="\n")
        return dtypes
    #endregion

    #region StructuredTypeArray interface

    def _set_up_result_arrays(self, dtypes):
        # we'll always force our dtypes and results to be a StructuredType and StructuredTypeArray
        # as these will force the results to be conformaing
        res = StructuredTypeArray(dtypes)
        return res

    #endregion

    #region Match Handler
    def _split_match_groups(self, groups, handlers, res):
        """This is _supposed_ to be splitting my groups so that the appropriate handler handles them, but it's missing a corner case
        Sometimes we have something like:
            RegexPattern((Capturing(Number), ..., Capturing(Number), ..., Capturing(Number))

        This happens to make 3 handlers even though it's really a single array of data...
        Then the grouping sees this and says "Aha! Must transpose"
        In reality it shouldn't even be bothering to touch it though...

        :param groups:
        :type groups:
        :param handlers:
        :type handlers:
        :param res:
        :type res:
        :return:
        :rtype:
        """

        if handlers is None:
            gg = groups
        elif len(groups[0]) == len(handlers): # the is None clause might be unnecessary...
            # just need to transpose the groups, basically,
            gg = groups.T
        elif len(groups[0]) < len(handlers):
            raise StringParserException(
                "{}.{}: got {} block handlers but only {} matched elements".format(
                    type(self).__name__,
                    '_split_match_groups',
                    len(handlers),
                    len(groups[0])
                )
            )
        elif all(isinstance(r, StructuredTypeArray) for r in res):
            # we have _more_ groups than handlers so we need to chunk them up
            blocks = [ r.block_size for r in res ]
            gg = [ None ] * len(blocks)
            sliced = 0
            for i, b in enumerate(blocks):
                if b == 1:
                    gg[i] = groups[:, sliced]
                else:
                    gg[i] = groups[:, sliced:sliced+b]
                sliced += b
        elif len(handlers) == 1:
            gg = [groups]
        else:
            gg = groups
        # print(">"*50)
        # print("Got in groups:", groups)
        # print("Got out:", gg)
        # print("<"*50)
        return gg # groups should now be chunked into groups if it wasn't before

    def _handle_insert_result(self, array, handler, data, single=True, append=False):
        """Handles the process of actually inserting the matches data/groups into array, applying any necessary
        handler in the process

        :param array: the array the data should be inserted into
        :type array: StructuredTypeArray
        :param handler: handler function for further processing of the matched data
        :type handler: None | callable
        :param data: matched data
        :type data: iterable[str]
        :param single: whether data should be treated as a single object (e.g. a subarray) or as multiple data objects
        :type single: bool
        :param append: whether the data should be appended or whether the array should entirely re-assign itself / which axes to assign to
        :type append: bool | iterable[int]
        """
        if handler is not None:
            data = handler(data, array=array, append=append)

        if data is not None:
            if array.has_indeterminate_shape:
                array.fill(data)
            elif single and (append or (append is not False and isinstance(append, int) and append == 0)):
                # print("pre-append:>", array)
                # print("axis:", append)
                # print("total axis:", append + array.append_depth)
                # print(data)
                if append is True  or not isinstance(append, int):
                    append = 0
                array.append(data, axis=append)
                # print("<:post-append", array)
            elif (append or (append is not False and  isinstance(append, int) and append == 0)):
                # print("pre-extend:>", array)
                if append is True or not isinstance(append, int):
                    append = 0
                array.extend(data, single=single, axis=append)
                # print("<:post-extend", array)
            else:
                # print("pre-fill:>", array)
                array.fill(data)
                # print("<:post-fill", array)
    def _handle_parse_match(self, match, res,
                            block_handlers,
                            append=False,
                            single=None,
                            default_value = ""
                            ):
        """Figures out how to handle the matched data from the parser
        In general the way we _want_ to do this is to basically do the most minimal amount of processing we can
        and then use the efficient methods inside StructuredTypeArray to cast this to properly typed arrays

        One issue is that we can end up with stuff that looks like:
            [
                [ group_a match_1, group_a match_2, group_b match_1, ... ]
                ...
            ]
        where we need to first chop the data into group_a, group_b, ...

        The type-calculus in RegexPattern and StructuredType can give us a "block size" for each subarray inside the
        StructuredTypeArray which indicates how many elements it expects to see from the group

        We use that to chop up the array as needed in these multi-match cases before trying to feed it into the
        StructuredTypeArray

        Complicating that, though, is that we sometimes need to do an extra layer of parsing on our matches

        Imagine that we found every instance of Repeating(Capturing(Number), suffix=Optional(Whitespace))

        This could give us something like
            [
                '-1232.123-234.345 3434.3434 10000000000',
                '5',
                '6'
            ]

        It would be hard to say what we should do with this, so we'll assume (possibly incorrectly) that we won't get a
        ragged array. If we need to introduce raggedness we'll do it later

        Even then, we could have
            [
                '-1232.123-234.345 3434.3434 10000000000',
                '5         5       5         5',
                '6         6       6         6'
            ]

        Here we'd need to go through line-by-line and apply the subparser of Number to each line to split it up

        On the other hand, what if we matched Duplicated(Capturing(Number), 4, riffle=Optional(Whitespace))?

        Then we'd have
            [
                ['-1232.123', '-234.345', '3434.3434', '10000000000'],
                ['5',         '5',        '5',         '5'],
                ['6',         '6',        '6',         '6']
            ]

        And here we wouldn't want to apply the subparser.

        The only way we could determine the difference is by comparing the shape of the StructuredTypeArray to the
        matched data. This means our block_handlers need to take the passed res array to figure their shit out too.


        :param append: whether to append the data or not
        :type append:
        :param single: whether the matched data should be treated like a singular object
        :type single:
        :param default_value:
        :type default_value:
        :param match:
        :type match: re.Match | iterable[re.Match]
        :param res:
        :type res:
        :param block_handlers:
        :type block_handlers:
        :return:
        :rtype:
        """

        if hasattr(match, 'groupdict'):
            single_match = True
            if single is None:
                single = single_match
            first_match = match # indicates we were fed a single match
        else:
            single_match = False
            if single is None:
                single = single_match
            if match is None or len(match) == 0:
                raise StringParserException("Can't handle series of matches if there are no matches")
            first_match = match[0] # indicates we were fed a series of matches
            # indicates we were fed a single group

        gd = first_match.groupdict(default=default_value) if res.dict_like else []
        # print("WTF", first_match, first_match.groups())
        if len(gd) == 0:
            # we have no _named_ groups so we just pull everything and run with it

            # one difficulty about this setup is that it makes vector data into a 2D array
            # on the other hand I don't think there's anything else to be done here?
            groups = match.groups(default=default_value) if (single_match) else np.array([ m.groups(default=default_value) for m in match ], dtype=str)

            # from here we might need to be a bit clever, actually...
            # the number of groups should align with the number of block_handlers, of course, which _should_ be connected
            # to what res looks like

            if not single_match:
                groups = self._split_match_groups(groups, block_handlers, res._array)

            if block_handlers is None:
                # print("-"*10 + "No Handler" + "-"*10)
                self._handle_insert_result(res, None, groups, single = single, append = append)
            elif len(block_handlers) == 1:
                # print("-"*10 + "Single Handler" + "-"*10)
                handler = block_handlers[0]
                self._handle_insert_result(res, handler, groups[0], single = single, append = append)
            # elif len(block_handlers) > :
            #     raise StringParserException("{}.{}: got multiple block_handlers {} but expected a singular dtype".format(
            #         type(self).__name__,
            #         '_handle_parse_match',
            #         block_handlers
            #     ))
            elif len(block_handlers) == 0:
                raise StringParserException("No block handlers available to parse data")
            else:
                # print("-"*10 + "Multiple Handlers" + "-"*10)
                # print(block_handlers, res, groups)
                for b,a,g in zip(block_handlers, res._array, groups):
                    self._handle_insert_result(a, b, g, single=single, append=append)

        else:
            # we have named groups so we only use the data associated with them
            if block_handlers is None:
                block_handlers = {}
            groups = gd if single_match else [ m.groupdict(default=default_value) for m in match ]
            # now we need to remerge the groupdicts by name if not single
            if not single_match:
                groups = OrderedDict(
                    (
                        k,
                        np.array([ g[k] for g in groups ], dtype=str)
                    ) for k in groups[0]
                )

            for k, v in groups.items():
                r = res[k]
                if k in block_handlers:
                    handler = block_handlers[k]
                else:
                    # unclear what we do in this case?
                    # Unless it breaks too much stuff, we'll throw an error...
                    # Originally, by default we just chose to not bother to handle it
                    # handler = None
                    raise StringParserException("{}: can't find a handler for group '{}' in block handlers ({})".format(
                        type(self).__name__,
                        k,
                        block_handlers
                    ))

                try:
                    self._handle_insert_result(r, handler, v, single=single, append=append)
                except:
                    if append or (isinstance(append, int) and append == 0):
                        raise StringParserException(
                            "failed to append to key '{}' for results array {}; block handler={}, value={}".format(
                                k,
                                res,
                                handler,
                                v[:50] + "...<{}>...".format(len(v) - 50) + v[-50:] if len(v) > 100 else v
                            ))
                    else:
                        raise StringParserException("failed to insert into key '{}' for results array {}; block handler={}, value={}".format(
                            k,
                            res,
                            handler,
                            v[:50] + "...<{}>...".format(len(v) - 50) + v[-50:] if len(v) > 100 else v
                        ))


    @classmethod
    def _get_regex_handler(cls, r):
        # if we have a simple dtype we just parse out the whole thing, no need for parse_all
        # otherwise if we have non-None repetitions we need parse_all

        if r.handler is not None:
            return r.handler

        if r.dtype.is_simple and r.dtype.shape is None:
            handler = None
        else:
            # if not r.dtype.is_simple:

            # we should only ever end up here on Named and Capturing objects
            # these will cause infinite recursion if we don't strip their Capturing status somehow
            import copy
            r = copy.copy(r)
            r.capturing = False
            r.key = None
            r.invalidate_cache()

            # kids = r.named_groups
            # if kids is None:
            #     kids = r.capturing_groups
            # else:
            #     kids = list(kids.values())
            #     if kids[0] is r:
            #         kids = r.capturing_groups
            #
            # if len(kids) == 0:
            #     raise StringParserException("Can't create a block handler if there won't be a match...?")
            # elif len(kids) == 1:
            #     r = kids[0]

            def handler(t, r=r, cls = cls, array = None, append = False):
                """A default handler for a regex pattern, constructed to be flexible and to recurse cleanly
                If you lots of data, you might want to hook in your own handler

                :param t: text to parse
                :type t: str | iterable[str] | iterable[iterable[str]]
                :param r: the regex pattern we're subparsing with with
                :type r: RegexPattern
                :param cls: the StringParser class this comes from
                :type cls: type
                :param array: the results array passed through for shape determination and stuff
                :type array: StructuredTypeArray
                :return:
                :rtype:
                """

                if array is not None and array.can_cast(t):
                    rr = t
                elif isinstance(t, str):
                    rr = None
                    if len(t) > 0:
                        s = cls(r)
                        out_array = {'array':array, 'append':append}
                        s.parse(t, out = out_array)
                else:
                    # we assume we got a list of strings...?
                    rr = None
                    skipped_rows = 0 # needed to handle Optional patterns...
                    s = cls(r)
                    # at some point this will be pushed into a parse_vector
                    # function or something where we're more efficient
                    # about how we do the parsing and data caching

                    if array is not None:
                        # One thing that needs to happen here is that the code needs some indicator that the
                        # append depth level of the array needs to be incremented
                        # Unfortunately we also need some way to manage this such that the changes won't lead to
                        # depth blow-up
                        # Like I think it'd be good for people to be able to specify the append level to the array
                        # but we don't want the incremented append level to propagate back to here somehow...
                        array.append_depth += 1
                        append_depth = array.append_depth
                        if append is True or append is False:
                            append = 0
                        out_array = {
                            'array':array,
                            'append':append
                        }
                        # print(append_depth, append + append_depth, array)
                        app_depth = append + append_depth
                        array.set_filling(0, axis = app_depth)
                    # print("????", t)
                    for i, l in enumerate(t):
                        # we gotta basically go line-by-line and add our stuff in
                        # the way we do this is by applying parse to each one of these things
                        # then calling append to add all the data in
                        if array is not None:
                            if len(l) == 0:
                                array.append(None, axis = append)
                            else:
                                # print(l)
                                s.parse(l, dtypes=array.stype, out = out_array)
                            array.set_filling(i+1, axis = app_depth)
                            # if array.is_simple:
                            #     array.filled_to[app_depth] = i+1
                            # else:
                            #     print(l)

                        elif rr is None:

                            if len(l) == 0:
                                # nothing to do here, basically a 'None' return
                                skipped_rows += 1
                                continue

                            rr = s.parse(l)
                            if skipped_rows > 0:
                                def add_skips(r, skipped = skipped_rows):
                                    a = r.array
                                    if isinstance(a, np.ndarray):
                                        r.extend(a[:skipped], prepend = True)
                                    else:
                                        for a2 in a:
                                            add_skips(a2)
                            else:
                                add_skips = lambda p:None

                            # now that we have this object we need to basically add an axis to it
                            # the raw thing will be lower-dimensional that we want I think
                            # the way we handle that is then...? I guess hitting it all with an add_axis?
                            if isinstance(rr, StructuredTypeArray):
                                rr.add_axis()
                                add_skips(rr)
                            elif isinstance(rr, OrderedDict):
                                for k in rr:
                                    rr[k].add_axis()
                                    add_skips(rr[k])
                            else:
                                for r in rr:
                                    r.add_axis()
                                    add_skips(r)

                        else:
                            if len(l) == 0:
                                r2 = None
                            else:
                                r2 = s.parse(l)
                            # we might have an OrderedDict or we might have a list or we might have a StructuredArray
                            # gotta make sure all of these cases are clean
                            if isinstance(rr, StructuredTypeArray):
                                # print("y tho :>>", rr)
                                # print(r2)
                                rr.append(r2)
                                # print(rr)
                                # print("<<: y tho")
                            elif isinstance(rr, OrderedDict):
                                for k in rr:
                                    rr[k].append(r2[k])
                            else:
                                for r1, r2 in zip(rr, r2):
                                    r1.append(r2) # should I be using extend...?

                    # gotta do some shape management because otherwise the array never knows how many times we iterated
                    # through here...
                    if array is not None:
                        if append is True or append is False:
                            pass
                        else:
                            array.append_depth -= 1
                            app_depth = append + append_depth
                            if app_depth > 0:
                                array.increment_filling(axis=app_depth - 1)

                return rr

        return handler

    #endregion

    #region HandlerMethods
    @classmethod
    def handler_method(cls, method):
        """Turns a regular function into a handler method by adding in (and ignoring) the array argument

        :param method:
        :type method:
        :return:
        :rtype:
        """

        def handler(data, method = method, array = None, append = False):
            return method(data)

        return handler

    @staticmethod
    def load_array(data, dtype = 'float'):
        import io
        return np.loadtxt(io.StringIO(data), dtype=dtype)
    @classmethod
    def to_array(cls,
                 data,
                 array = None,
                 append = False,
                 dtype = 'float',
                 shape = None,
                 pre=None
                 ):
        """A method to take a string or iterable of strings and quickly dump it to a NumPy array of the right dtype (if it can be cast as one)

        :param data:
        :type data:
        :param dtype:
        :type dtype:
        :return:
        :rtype:
        """
        import io

        if not isinstance(data, str):
            if isinstance(data, np.ndarray) and data.ndim > 1:
                # we need to load each chunk in turn...
                def hold_me_closer_tiny_dancer(data, dtype = dtype, cls = cls):
                    if data.ndim == 1:
                        bleh = '\n'.join((d.strip() for d in data))
                        if pre is not None:
                            bleh = pre(bleh)
                        return cls.load_array(bleh, dtype = dtype)
                    else:
                        return [ hold_me_closer_tiny_dancer(d) for d in data ]
                arr = np.array(hold_me_closer_tiny_dancer(data))
            else:
                bleh = '\n'.join((d.strip() for d in data))
                if pre is not None:
                    bleh = pre(bleh)
                arr = cls.load_array(bleh, dtype=dtype)
        else:
            if pre is not None:
                data = pre(data)
            arr = cls.load_array(data, dtype=dtype)
        if shape is not None:
            fudge_axis = [ i for i, s in enumerate(shape) if s is None]
            if len(fudge_axis) == 0:
                arr = arr.reshape(shape)
            elif len(fudge_axis) == 1:
                fudge_axis = fudge_axis[0]
                num_els = np.product(arr.shape)
                shoop = list(shape)
                shoop[fudge_axis] = 1
                num_not_on_axis = np.product(shoop)
                if num_els % num_not_on_axis == 0:
                    num_left = num_els // num_not_on_axis
                    shoop[fudge_axis] = num_left
                    arr = arr.reshape(shoop)
                else:
                    raise StringParserException("{}.{}: can't reshape array with shape '{}' to shape '{}' as number of elements is not divisible by block size".format(
                        cls.__name__,
                        'to_array',
                        arr.shape,
                        shape
                    ))
            else:
                raise StringParserException("{}.{}: can't reshape to shape '{}' with more than one indeterminate slot".format(
                    cls.__name__,
                    'to_array',
                    shape
                ))

        return arr
    @classmethod
    def array_handler(cls,
                      array = None,
                      append = False,
                      dtype = 'float',
                      shape = None,
                      pre = None
                      ):
        """Returns a handler that uses to_array

        :param dtype:
        :type dtype:
        :param array:
        :type array:
        :param shape:
        :type shape:
        :return:
        :rtype:
        """
        def handler(data, array = array, append = append, dtype = dtype, shape = shape, cls=cls):
            return cls.to_array(data, array = array, append = append, dtype = dtype, shape = shape, pre=pre)
        return handler
    #endregion
