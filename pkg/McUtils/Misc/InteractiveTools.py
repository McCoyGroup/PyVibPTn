"""
Miscellaneous tools for interactive messing around in Jupyter environments
"""
import sys, os, types, importlib, inspect

__all__ = [
    "ModuleReloader",
    "MoleculeGraphics"
]

class ModuleReloader:
    """
    Reloads a module & recursively descends its 'all' tree
    to make sure that all submodules are also reloaded
    """

    def __init__(self, modspec):
        """
        :param modspec:
        :type modspec: str | types.ModuleType
        """
        if isinstance(modspec, str):
            modspec=sys.modules[modspec]
        self.mod = modspec

    def get_parents(self):
        """
        Returns module parents
        :return:
        :rtype:
        """
        split = self.mod.__name__.split(".")
        return [".".join(split[:i]) for i in range(len(split)-1, 0, -1)]

    def get_members(self):
        """
        Returns module members
        :return:
        :rtype:
        """

        base = self.mod.__all__ if hasattr(self.mod, '__all__') else dir(self.mod)
        if hasattr(self.mod, '__reload_hook__'):
            try:
                others = list(self.mod.__reload_hook__)
            except TypeError:
                pass
            else:
                base = [list(base), others]
        return base

    def reload_member(self, member,
        stack=None,
        reloaded=None, blacklist=None, reload_parents=True,
        verbose=False,
        print_indent=""
        ):
        # print(print_indent + " member:", member)
        if member.startswith('.'):
            how_many = 0
            while member[how_many] == ".":
                how_many += 1
                if how_many == len(member):
                    break
            main_name = self.mod.__name__.rsplit(".", how_many)[0]
            test_key = main_name + "." + member[how_many:]
        else:
            test_key = self.mod.__name__ + "." + member
        if test_key in sys.modules:
            type(self)(test_key).reload(
                reloaded=reloaded, blacklist=blacklist,
                verbose=verbose,
                reload_parents=reload_parents, print_indent=print_indent
            )
        else:
            obj = getattr(self.mod, member)
            if isinstance(obj, types.ModuleType):
                type(self)(obj).reload(
                    reloaded=reloaded, blacklist=blacklist,
                    verbose=verbose,
                    reload_parents=reload_parents, print_indent=print_indent
                )
            elif isinstance(obj, type):
                type(self)(obj.__module__).reload(
                    reloaded=reloaded, blacklist=blacklist,
                    verbose=verbose,
                    reload_parents=reload_parents, print_indent=print_indent
                )
            else:
                obj = type(obj)
                type(self)(obj.__module__).reload(
                    reloaded=reloaded, blacklist=blacklist,
                    verbose=verbose,
                    reload_parents=reload_parents, print_indent=print_indent
                )

    blacklist_keys = ['site-packages', os.path.abspath(os.path.dirname(inspect.getfile(os)))]
    def reload(self, 
        stack=None,
        reloaded=None, blacklist=None, reload_parents=True, 
        verbose=False,
        print_indent=""
        ):
        """
        Recursively searches for modules to reload and then reloads them.
        Uses a cache to break cyclic dependencies of any sort.
        This turns out to also be a challenging problem, since we need to basically
        load depth-first, while never jumping too far back...


        :return:
        :rtype:
        """

        if reloaded is None:
            reloaded = set()

        if blacklist is None:
            blacklist = set()
        blacklist.update(sys.builtin_module_names)

        key = self.mod.__name__
        if (
                key not in reloaded
                and key not in blacklist
                and all(k not in inspect.getfile(self.mod) for k in self.blacklist_keys)
        ):
            if verbose:
                print(print_indent + "Reloading:", self.mod.__name__)
            reloaded.add(self.mod.__name__)

            print_indent += "  "

            mems = self.get_members()

            if isinstance(mems[0], list):
                req, opts = mems
            else:
                req = mems
                opts = []

            for member in req:
                self.reload_member(member,
                                   stack=stack,
                                   reloaded=reloaded,
                                   blacklist=blacklist,
                                   reload_parents=reload_parents,
                                   verbose=verbose,
                                   print_indent=print_indent
                                   )
            for member in opts:
                try:
                    self.reload_member(member,
                                       stack=stack,
                                       reloaded=reloaded,
                                       blacklist=blacklist,
                                       reload_parents=reload_parents,
                                       verbose=verbose,
                                       print_indent=print_indent
                                       )
                except:
                    pass

           
            
            if hasattr(self.mod, '__reload_hook__'):
                try:
                    self.mod.__reload_hook__()
                except TypeError:
                    pass
            if verbose:
                print(print_indent + "loading:", self.mod.__name__)
            importlib.reload(self.mod)


            load_parents = []
            if reload_parents:
                # make sure parents get loaded in the appropriate order...
                for parent in self.get_parents():
                    if parent in reloaded:
                        # prevent us from jumping back too far...
                        break
                    # print(" storing", parent)
                    load_parents.append(parent)
                    type(self)(parent).reload(
                        reloaded=reloaded, blacklist=blacklist, 
                        reload_parents=reload_parents, verbose=verbose,
                        print_indent=print_indent
                        )

try:
    import nglview
except ImportError:
    class MoleculeGraphics:
        def __init__(self, *args, **kwargs):
            raise ImportError("{} requires `nglview`".format(type(self).__name__))
else:
    import numpy as np
    class MoleculeGraphics(nglview.Structure, nglview.Trajectory):
        misc_useless_structural_data_header = " 0     0  0  0  0  0  0999 V2000"
        def __init__(self,
                     atoms,
                     coords,
                     bonds=None,

                     displacements=None,
                     displacement_range=(-1, 1),
                     displacement_steps=5,

                     name='Molecule',  # 'My special little molecule I love it and do not hate secretely hate it',
                     program='Python',  # 'WTF-who-keeps-this-info-these-days',
                     comment="",  # "I have nothing to say to you",
                     metadata=None,
                     **params
                     ):
            super().__init__()

            self.ext = "sdf"
            self.params = params
            self.atoms = atoms
            self.coords = np.asanyarray(coords)
            self.bonds = [] if bonds is None else bonds
            self.name = name
            self.program = program
            self.comment = comment
            self.meta = metadata

            self.dips = displacements
            if displacements is None:
                self.scales = [0.]
            else:
                base_scales = np.linspace(*displacement_range, displacement_steps)
                self.scales = np.concatenate([
                    base_scales,
                    base_scales[-2:0:-1]
                ])

        def convert_header(self, comment=None):
            return "\n".join([
                self.name,
                "  " + self.program,
                " " + self.comment + ("" if comment is None else comment)
            ])

        def convert_counts_line(self):
            return "{:>3.0f}{:>3.0f} {}".format(len(self.atoms), len(self.bonds),
                                                self.misc_useless_structural_data_header)

        def convert_coordinate_block(self, coords):
            return "\n".join(
                " {0[0]:>9.5f} {0[1]:>9.5f} {0[2]:>9.5f} {1:<3} 0  0  0  0  0  0  0  0  0  0  0  0".format(
                    crd,
                    at
                ) for crd, at in zip(coords, self.atoms)
            )

        def convert_bond_block(self):
            return "\n".join(
                "{:>3.0f}{:>3.0f}{:>3.0f}  0  0  0  0".format(
                    b[0] + 1,
                    b[1] + 1,
                    b[2] if len(b) > 2 else 1
                ) for b in self.bonds
            )

        def get_single_structure_string(self, coords, comment=None):
            return """
{header}
{counts}
{atoms}
{bonds}
M  END
{meta}
$$$$
            """.format(
                header=self.convert_header(comment=comment),
                counts=self.convert_counts_line(),
                atoms=self.convert_coordinate_block(coords),
                bonds=self.convert_bond_block(),
                meta="" if self.meta is None else self.meta
            ).strip()

        def get_coordinates(self, index):
            if self.dips is None and not isinstance(index, (int, np.integer)) and not index == 0:
                raise ValueError("no Cartesian displacements passed...")
            scales = self.scales[index]
            arr = self.coords + self.dips * scales
            return arr

        def get_substructure(self, idx):
            import copy

            new = copy.copy(self)
            new.coords = self.get_coordinates(idx)
            new.dips = None
            new.scales = [0.]
            new.comment += "<Structure {}>".format(idx)
            return new

        def __getitem__(self, idx):
            return self.get_substructure(idx)

        def __iter__(self):
            for i in range(self.n_frames):
                yield self.get_substructure(i)

        def get_structure_string(self):
            if self.dips is None:
                return self.get_single_structure_string(self.coords)
            else:
                return "\n".join(
                    self.get_single_structure_string(self.get_coordinates(i)) for i in range(self.n_frames)
                )

        @property
        def n_frames(self):
            return len(self.scales)

        # basically arguments to "add representation"
        default_theme = [
            ['licorice'],
            ['ball+stick', dict(selection='_H', aspect_ratio=2.5)],
            ['ball+stick', dict(selection='not _H', aspect_ratio=3)]
        ]

        def show(self,
                 themes=None,
                 frame_size=('100%', 500),
                 scale=1.2,
                 **opts):
            """Basically a hack tower to make NGLView actually visualize molecules well"""
            viewer = nglview.NGLWidget(self)
            if themes is None:
                themes = self.default_theme
            if len(themes) > 0:
                viewer.clear()
                for t in themes:
                    if len(t) > 1:
                        viewer.add_representation(t[0], **t[1])
                    else:
                        viewer.add_representation(t[0])
            viewer.display(**opts)
            # this is a temporary hack to get a better window size
            if frame_size is not None:
                arg_str = [s if isinstance(s, str) else '{}px'.format(s) for s in frame_size]
                viewer._remote_call(
                    "setSize", target="Widget",
                    args=arg_str
                )
            if scale is not None:
                viewer[0].set_scale(scale)
            return viewer
