"""
Simple package for easily creating command line interfaces in a
nestable way with automatic argument dispatch
"""

import inspect, typing, os, sys, argparse, collections

__all__ = [
    "CLI",
    "CommandGroup",
    "Command"
]

class Command:
    """
    A holder for a command that just automates type handling &
    that sort of thing
    """
    def __init__(self, name, method):
        self.name = name
        self.meth = method
        self.hints = typing.get_type_hints(method)
        self.pos_only_args = {
            arg.name for arg in inspect.signature(method).parameters.values()
            if arg.default is inspect.Parameter.empty
        }

    def get_help(self):
        """
        Gets single method help string
        :return:
        :rtype:
        """
        base_str = "{} - ".format(self.name)
        padding = " " * len(base_str)
        doc_str = self.meth.__doc__
        if doc_str is None:
            doc_str = ""
        doc_str = inspect.cleandoc(doc_str)
        doc_str += "\n" + "ARGS {}".format(
            " ".join(
                "{}{}:{}".format(
                    "*" if k in self.pos_only_args else "",
                    k, self._hint_str(v)
                ) for k, v in self.hints.items()
            )
        )
        doc_str = doc_str.strip()

        return base_str + doc_str.replace("\n", "\n" + padding)

    @classmethod
    def _hint_types(cls, hint):
        """
        :param hint:
        :type hint: type | str | typing.GenericAlias
        :return:
        :rtype:
        """
        if isinstance(hint, type):
            return (hint,)
        elif hasattr(hint, '__origin__') and hint.__origin__ is typing.Union:
            return tuple(a for a in hint.__args__ if a is not type(None))
        else:
            return None
    @classmethod
    def _hint_str(cls, hint):
        """
        :param hint:
        :type hint: type | str | typing.GenericAlias
        :return:
        :rtype:
        """
        if hint is type(None):
            hint = "None"
        elif isinstance(hint, type):
            hint = hint.__name__
        elif isinstance(hint, tuple):
            hint = " | ".join(cls._hint_str(x) for x in hint)
        elif isinstance(hint, str):
            hint = hint
        else:
            types = cls._hint_types(hint)
            if types is None:
                hint = str(hint).replace("typing.", "")
            else:
                return cls._hint_str(types)

        return hint

    @staticmethod
    def get_parse_dict(*spec):
        """
        Builds a parse spec to feed into an ArgumentParser later
        :param spec:
        :type spec:
        :return:
        :rtype:
        """
        argv_0 = sys.argv[0]
        try:
            sys.argv[0] = "parsing_dict"  # self.group + " " + self.cmd
            parser = argparse.ArgumentParser(add_help=False)
            keys = []
            for arg in spec:
                if len(arg) > 1:
                    arg_name, arg_dict = arg
                else:
                    arg_name = arg[0]
                    arg_dict = {}
                if 'dest' in arg_dict:
                    keys.append(arg_dict['dest'])
                else:
                    keys.append(arg_name)
                parser.add_argument(arg_name, **arg_dict)
            args = parser.parse_args()
            opts = {k: getattr(args, k) for k in keys}
        finally:
            sys.argv[0] = argv_0
        return {k: o for k, o in opts.items() if not (isinstance(o, str) and o == "")}

    @staticmethod
    def _get_typed(val:str, converter:callable):
        if val == "":
            return None
        else:
            return converter(val)
    def get_parse_spec(self):
        """
        Gets a parse spec that can be fed to ArgumentParser

        :return:
        :rtype:
        """
        args = []
        for k, v in self.hints.items():
            spec = {'default':""}
            if k not in self.pos_only_args:
                spec['dest'] = k
                key = "--" + k
            else:
                key = k
            types = self._hint_types(v)
            if types is None or len(types) == 0:
                spec['type'] = lambda v:v
            elif int in types:
                spec['type'] = lambda v: self._get_typed(v, int)
            elif float in types:
                spec['type'] = lambda v: self._get_typed(v, float)
            elif bool in types:
                spec['type'] = lambda v: self._get_typed(v, bool)
            elif str in types:
                spec['type'] = lambda v: v
            else:
                raise ValueError("don't know what to do with type hint {}".format(types))

            args.append((key, spec))
        return args

    def parse(self):
        """
        Generates a parse spec, builds an ArgumentParser, and parses the arguments

        :return:
        :rtype:
        """
        spec = self.get_parse_spec()
        return self.get_parse_dict(*spec)

    def __call__(self):
        """
        Parse argv and call bound method
        :return:
        :rtype:
        """
        parse = self.parse()
        args = []
        kwargs = {}
        for k in self.hints.keys():
            if k in self.pos_only_args:
                # if k not in parse:
                #     raise ValueError("positional only argument {} missing")
                args.append(parse[k])
            else:
                if k in parse:
                    kwargs[k] = parse[k]
        return self.meth(*args, **kwargs)

class CommandGroup:
    """
    Generic interface that defines an available set of commands
    as class methods.
    Basically just exists to be ingested by a CLI.
    """
    _name=""
    _description=""
    _tag=None # the group tag that the CLI will ingest

    @classmethod
    def _get_tag(cls):
        if cls._tag is None:
            return cls._name.lower()
        else:
            return cls._tag

    @classmethod
    def _get_commands(cls):
        """
        Returns the commands defined on the class as well
        as their arguments and allowed types
        :return:
        :rtype:
        """

        methods = {}
        for name, attr in cls.__dict__.items():
            if isinstance(name, str) and not name.startswith("_"):
                # print("...", name, inspect.ismethoddescriptor(attr))
                if inspect.ismethoddescriptor(attr):
                    methods[name] = Command(name, getattr(cls, name))
        sort_keys = sorted(methods.keys())
        sort_methods = collections.OrderedDict(
            (k, methods[k]) for k in sort_keys
        )

        return sort_methods

    @classmethod
    def _get_command(cls, k):
        """
        Returns the commands defined on the class as well
        as their arguments and allowed types
        :return:
        :rtype:
        """

        methods = cls._get_commands()
        if k in methods:
            return methods[k]
        else:
            raise KeyError("No command '{}' for group '{}'. Available commands: {}".format(k, cls._get_tag(), list(methods.keys())))

    @classmethod
    def _get_help(cls):
        """
        Gets the help string for the CLI
        :return:
        :rtype:
        """

        help_str = "\n  ".join(
            ["{}: {}".format(cls._get_tag(), cls._description)]
            + [c.get_help().replace("\n", "\n  ") for c in cls._get_commands().values()]
        )

        return help_str


class CLI:
    """
    A representation of a command line interface
    which layers simple command dispatching on the basic
    ArgParse interface
    """
    def __init__(self, name, description, *groups, cmd_name=None):
        """
        :param name:
        :type name: str
        :param name:
        :type description: str
        :param cmd_name:
        :type cmd_name: str | None
        :param groups:
        :type groups: Type[CommandGroup]
        """
        self.name = name
        self.description=description
        self.cmd_name = name.lower() if cmd_name is None else cmd_name
        self.groups = collections.OrderedDict(
            (g._get_tag(), g) for g in groups
        )
        self.argv = sys.argv

    def parse_group_command(self):
        """
        Parses a group and command argument (if possible) and prunes `sys.argv`

        :param group:
        :type group:
        :param command:
        :type command:
        :return:
        :rtype:
        """
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("group", type=str)
        parser.add_argument("command", type=str, default='', nargs="?")
        parse, unknown = parser.parse_known_args()
        group = parse.group
        cmd = parse.command
        sys.argv = [sys.argv[0]] + unknown

        return (group, cmd)

    def get_command(self):
        group, cmd = self.parse_group_command()
        if cmd == "": # either means we asked for --help or we've got a default command group
            if '--help' == sys.argv[0]: # quick hack but effective
                return self.get_group(cmd)._get_help()
            else:
                cmd, group = group, cmd
        return self.get_group(group)._get_command(cmd)

    def get_group(self, grp):
        if grp in self.groups:
            return self.groups[grp]
        elif grp == "":
            raise KeyError("No default command group (i.e. with key ''). Available groups: {}".format(list(self.groups.keys())))
        else:
            raise KeyError("No command group '{}'. Available groups: {}".format(grp, list(self.groups.keys())))

    def run_command(self):
        res = self.get_command()
        if not isinstance(res, str):
            # parse and call semantics are
            # handled by the stored Command object
            res = res()
        else:
            print(res)
        return res

    def get_help(self):
        """
        Gets the help string for the CLI
        :return:
        :rtype:
        """

        title_tag = "{} [{}]: ".format(self.cmd_name, "|".join(self.extra_commands.keys()))
        title_tag += self.description.replace("\n", "\n" + " " * len(title_tag))
        help_str = "\n".join([
            title_tag,
            *(g._get_help() for g in self.groups.values())
        ])

        return help_str

    def help(self, print_help=True):
        sys.argv.pop(1)
        res = self.get_help()
        if print_help:
            print(res)
        return res

    def run_parse(self, parse, unknown):
        """
        Provides a standard entry point to running stuff using the default CLI

        :param parse:
        :type parse:
        :param unknown:
        :type unknown:
        :return:
        :rtype:
        """

        # detect whether interactive run or not
        # interact = parse.interact or (
        #         len(sys.argv) == 1
        #         and not parse.help
        #         and not parse.script
        # )
        interact = parse.interact

        # in interactive/script envs we expose stuff
        if parse.script or interact:
            sys.path.insert(0, os.getcwd())
            interactive_env = {
                "__name__": "McEnv.script"
            }

        # In a script environment we just read in the script and run it
        # this is usefule because often these kinds of CLIs are set up
        # for things like containerized runtimes.
        # In that case it's nice to have a direct route into the loaded code
        # without having to set up some kind of backdoor
        if parse.script:
            script = sys.argv[1]
            sys.argv.pop(0)
            interactive_env["__name__"] = self.name + ".scripts." + os.path.splitext(os.path.basename(script))[0]
            with open(script) as scr:
                src = scr.read()
                src = compile(src, script, 'exec')
            interactive_env["__file__"] = script
            exec(src, interactive_env, interactive_env)
        elif parse.help:
            self.help(print_help=True)
        elif len(unknown) > 0:
            sys.argv = sys.argv + unknown
            self.run_command()

        # Start an interactive python...basically just because it can be convenient
        if interact:
            import code
            code.interact(banner=self.name + " Interactive Session", readfunc=None, local=interactive_env, exitmsg=None)

    extra_commands = {
        '--interact': dict(
            default=False, action='store_const', const=True, dest="interact",
            help='start an interactive session after running'
        ),
        '--script': dict(
            default=False, action='store_const', const=True, dest="script",
            help='run a script'
        ),
        '--help': dict(
            default=False, action='store_const', const=True, dest="help",
            help='print help string'
        ),
        '--fulltb': dict(
            default=False, action='store_const', const=True, dest="full_traceback",
            help='print full traceback'
        )
    }
    def parse_toplevel_args(self):
        """
        Parses out the top level flags that the program supports

        :return:
        :rtype:
        """

        parser = argparse.ArgumentParser(add_help=False)
        for k,v in self.extra_commands.items():
            parser.add_argument(k, **v)

        new_argv = []
        for k in sys.argv[1:]:
            if not k.startswith("--"):
                break
            new_argv.append(k)
        unknown = sys.argv[1 + len(new_argv):]
        sys.argv = [sys.argv[0]] + new_argv
        parse = parser.parse_args()

        return parse, unknown

    def run(self):
        """
        Parses the arguments in `sys.argv` and dispatches to the approriate action.
        By default supports interactive sessions, running scripts, and abbreviated tracebacks.

        :return:
        :rtype:
        """

        parse, unknown = self.parse_toplevel_args()
        if parse.full_traceback:
            self.run_parse(parse, unknown)
        else:
            error = None
            try:
                self.run_parse(parse, unknown)
            except Exception as e:
                error = e
            if error is not None:
                print(error)

