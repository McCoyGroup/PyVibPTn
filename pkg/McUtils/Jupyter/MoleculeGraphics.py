
import numpy as np

__all__ = [
    "MoleculeGraphics"
]


class MoleculeGraphics:
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
                 **params):

        self.obj = self._load_nglview()(
            atoms,
            coords,
            bonds=bonds,
            displacements=displacements,
            displacement_range=displacement_range,
            displacement_steps=displacement_steps,

            name=name,  # 'My special little molecule I love it and do not hate secretely hate it',
            program=program,  # 'WTF-who-keeps-this-info-these-days',
            comment=comment,  # "I have nothing to say to you",
            metadata=metadata,
            **params
        )
        self._widg = None

    def to_widget(self):
        if self._widg is None:
            self._widg = self.obj.show()
        return self._widg
    def show(self):
        return self.to_widget()
    def _ipython_display_(self):
        from .JHTML import JupyterAPIs
        JupyterAPIs().display_api.display(self.show())

    @classmethod
    def _load_nglview(cls):
        try:
            import nglview
        except ImportError:
            class MoleculeGraphics:
                def __init__(self, *args, **kwargs):
                    raise ImportError("{} requires `nglview`".format(type(self).__name__))
        else:
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
                    if self.coords.ndim == 3:
                        return self.coords[index]
                    else:
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
                    if self.dips is None and self.coords.ndim == 2:
                        return self.get_single_structure_string(self.coords)
                    else:
                        return "\n".join(
                            self.get_single_structure_string(self.get_coordinates(i)) for i in range(self.n_frames)
                        )

                @property
                def n_frames(self):
                    if self.coords.ndim == 2:
                        return len(self.scales)
                    else:
                        return len(self.coords)

                # basically arguments to "add representation"
                default_theme = [
                    ['licorice'],
                    ['ball+stick', dict(selection='_H', aspect_ratio=2.5)],
                    ['ball+stick', dict(selection='not _H', aspect_ratio=3)]
                ]

                def show(self,
                         themes=None,
                         frame_size=('100%', 'vh'),
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
                    # try:
                    #     player = viewer.player.widget_player
                    # except AttributeError:
                    #     pass
                    # else:
                    #     if player is not None:
                    #         # player = player.children[0]
                    #         player._playing = playing
                    #         player._repeat = repeat
                    viewer.display(**opts)

                    # this is a temporary hack to get a better window size
                    if frame_size is not None:
                        arg_str = [s if isinstance(s, str) else '{}px'.format(s) for s in frame_size]
                        viewer._remote_call(
                            "setSize",
                            target="Widget",
                            args=arg_str
                        )
                    if scale is not None:
                        viewer[0].set_scale(scale)
                    return viewer

        return MoleculeGraphics