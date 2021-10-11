import re, os

__all__ = [
    "StringMatcher",
    "MatchList",
    "FileMatcher"
]

class StringMatcher:
    """
    Defines a simple filter that applies to a file and determines whether or not it matches the pattern
    """

    def __init__(self, match_patterns, negative_match = False):
        if isinstance(match_patterns, str):
            pattern = re.compile(match_patterns)
            self.matcher = lambda f, p = pattern: re.match(p, f)
        elif hasattr(re, "Pattern") and isinstance(match_patterns, re.Pattern): # re.Pattern is new as of python3.7...
            self.matcher = lambda f, p = match_patterns: re.match(p, f)
        elif type(match_patterns).__name__=="SRE_Pattern": # pre 3.7
            self.matcher = lambda f, p = match_patterns: re.match(p, f)
        elif isinstance(match_patterns, StringMatcher):
            self.matcher = match_patterns.matches
        elif callable(match_patterns):
            self.matcher = match_patterns
        else:
            ff = type(self)
            match_patterns = tuple(ff(m) if not isinstance(m, StringMatcher) else m for m in match_patterns)
            self.matcher = lambda f, p = match_patterns: all(m.matches(f) for m in p)

        self.negate = negative_match

    def matches(self, f):
        m = self.matcher(f)
        if self.negate:
            m = not m
        return m

class MatchList(StringMatcher):
    """
    Defines a set of matches that must be matched directly (uses `set` to make this basically a constant time check)
    """

    def __init__(self, *matches, negative_match = False):
        self.match_list = set(matches)
        super().__init__(lambda f, m=self.test_match: m(f), negative_match = negative_match)
    def test_match(self, f):
        return f in self.match_list

class FileMatcher(StringMatcher):
    """
    Defines a filter that uses StringMatcher to specifically match files
    """

    def __init__(self, match_patterns, negative_match = False, use_basename = False):
        super().__init__(match_patterns, negative_match = negative_match)
        self.use_basename = use_basename

    def matches(self, f):
        f_name = f if not self.use_basename else os.path.basename(f)
        return super().matches(f_name)