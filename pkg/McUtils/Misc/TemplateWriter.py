import os, shutil
from .FileMatcher import *

__all__ = [
    "TemplateWriter"
]

class TemplateWriter:
    """
    A general class that can take a directory layout and apply template parameters to it
    Very unsophisticated but workable
    """

    ignored_files = [".DS_Store"]

    def __init__(self, template_dir, replacements = None, file_filter = None, **opts):
        if replacements is not None:
            opts = replacements
        self._rep_dict = opts
        self._reps = None

        if file_filter is None:
            file_filter = FileMatcher(MatchList(*self.ignored_files, negative_match=True), use_basename=True)
        self.filter = FileMatcher(file_filter) if not isinstance(file_filter, StringMatcher) else file_filter
        self.template_dir = os.path.abspath(template_dir)

    @property
    def replacements(self):
        if self._reps is None:
            self._reps =  tuple(("`"+k+"`", str(v)) for k,v in self._rep_dict.items())
        return self._reps

    def apply_replacements(self, string):
        """Applies the defined replacements to the

        :param string:
        :type string:
        :return:
        :rtype:
        """
        import functools

        return functools.reduce(lambda s, r: s.replace(*r), self.replacements, string)

    def write_file(self, template_file, out_dir, apply_template = True, template_dir = None):
        """writes a single _file_ to _dir_ and fills the template from the parameters passed when intializing the class

        :param template_file: the file to load and write into
        :type template_file: str
        :param out_dir: the directory to write the file into
        :type out_dir: str
        :param apply_template: whether to apply the template parameters to the file content or not
        :type apply_template: bool
        :return:
        :rtype:
        """

        if template_dir is None:
            template_dir = self.template_dir

        stripped_loc = template_file.split(template_dir, 1)[-1]
        rep_loc = self.apply_replacements(stripped_loc).strip(os.sep)
        new_file = os.path.join(out_dir, rep_loc)

        os.makedirs(os.path.dirname(new_file), exist_ok = True)

        if apply_template:
            with open(template_file) as tf:
                try:
                    content = tf.read()
                except:
                    raise IOError("Couldn't read content from {}".format(template_file))
                with open(new_file, "w+") as out:
                    out.write(self.apply_replacements(content))
        else:
            shutil.copy(template_file, new_file)

        return new_file

    def iterate_write(self, out_dir, apply_template = True, src_dir = None, template_dir = None):
        """Iterates through the files in the template_dir and writes them out to dir

        :return:
        :rtype:
        """

        if template_dir is None:
            template_dir = self.template_dir
        if src_dir is None:
            src_dir = template_dir

        files = os.listdir(src_dir)
        for f in files:
            f = os.path.join(src_dir, f)
            if self.filter.matches(f.split(template_dir, 1)[-1]):
                if not os.path.isdir(f):
                    self.write_file(f, out_dir, apply_template = apply_template, template_dir = template_dir)
                else:
                    self.iterate_write(out_dir, apply_template = apply_template, src_dir = f, template_dir = template_dir)
