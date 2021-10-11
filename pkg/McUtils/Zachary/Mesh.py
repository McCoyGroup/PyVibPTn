"""
Represents an n-dimensional grid, used by Interpolator and (eventually) FiniteDifferenceFunction to automatically
know what kind of data is fed in
"""

import numpy as np, enum

__all__ = [ "Mesh", "MeshType" ]

# we define a bunch of MeshType attributes so that the string names can change up or whatever but the interface
# can remain consistent
class MeshType(enum.Enum):
    Regular = "regular" # direct product with consistent mesh spacings
    Structured = "structured" # direct product grids
    Unstructured = "unstructured" # totally unstructured data points
    SemiStructured = "semistructured" # almost full direct products
    Indeterminate = "indeterminate" # couldn't assess

class MeshError(Exception):
    ...

class Mesh(np.ndarray):
    """
    A general Mesh class representing data points in n-dimensions
    in either a structured, unstructured, or semi-structured manner.
    Exists mostly to provides a unified interface to difference FD and Surface methods.
    """

    # just to make it accessible through Mesh
    MeshError = MeshError
    MeshType = MeshType

    _allow_indeterminate = False
    # subclassing np.ndarray is weird... maybe I don't even want to do it... but I _think_ I do
    def __new__(cls,
                data,
                mesh_type=None,
                allow_indeterminate=None
                ):
        """
        :param griddata: the raw grid-point data the mesh uses
        :type griddata: np.ndarray
        :param mesh_type: the type of mesh we have
        :type mesh_type: None | str
        :param opts:
        :type opts:
        """
        data = np.asarray(data)
        if cls._is_meshgrid(data):
            # we turn mesh grids into grid point grids...
            data = np.moveaxis(data, 0, data.ndim-1)
        aid = cls._allow_indeterminate
        try:
            cls._allow_indeterminate = allow_indeterminate # gotta temp disable this...
            self = data.view(cls)
        finally:
            cls._allow_indeterminate = aid
        self.allow_indeterminate = allow_indeterminate
        self.mesh_type = self.get_mesh_type(self) if mesh_type is None else mesh_type
        self._spacings = None
        return self

    def __init__(self, *args, **kwargs):
        # empty init call because numpy is weird
        ...

    def __array_finalize__(self, mesh):
        # basically just a validator...
        if mesh is None:
            return None

        self.mesh_type = getattr(mesh, "mesh_type", None)
        if self.mesh_type is None:
            self.mesh_type = self.get_mesh_type(self)

        self.allow_indeterminate = getattr(mesh, "allow_indeterminate", None)
        aid = self.allow_indeterminate
        if aid is None:
            aid = self._allow_indeterminate
        if not aid and self.mesh_type is MeshType.Indeterminate:
            raise MeshError("indeterminate MeshType, but `allow_indeterminate` turned off")

    @property
    def mesh_spacings(self):
        if self._spacings is None:
            self._spacings = self.get_mesh_spacings(self)
        return self._spacings
    @property
    def subgrids(self):
        if (
                self.mesh_type == MeshType.Regular
                or self.mesh_type == MeshType.Structured
                # or self.mesh_type == MeshType.SemiStructured
        ):
            return self.get_mesh_subgrids(self)
        else:
            return None
    @property
    def dimension(self):
        """Returns the dimension of the grid (not necessarily ndim)

        :return:
        :rtype: int
        """
        return self.shape[-1]
    @property
    def npoints(self):
        """Returns the number of gridpoints in the mesh

        :return:
        :rtype: int
        """
        return self.get_npoints(self)

    @property
    def gridpoints(self):
        """Returns the flattened set of gridpoints for a structured tensor grid and otherwise just returns the gridpoints

        :return:
        :rtype:
        """
        return self.get_gridpoints(self)

    @classmethod
    def get_npoints(cls, g):
        """Returns the number of gridpoints in the grid

        :param g:
        :type g: np.ndarray
        :return:
        :rtype: int
        """
        return np.product(g.shape[:-1])
    @classmethod
    def get_gridpoints(cls, g):
        """Returns the gridpoints in the grid

        :param g:
        :type g: np.ndarray
        :return:
        :rtype: int
        """
        return np.reshape(g, (cls.get_npoints(g), g.shape[-1]))

    @classmethod
    def get_mesh_subgrids(cls, grid, tol=8):
        """
        Returns the subgrids for a mesh
        :param grid:
        :type grid:
        :param tol:
        :type tol:
        :return:
        :rtype:
        """

        if isinstance(grid, Mesh):
            grid = grid.view(np.ndarray)

        if grid.ndim == 1:
            _, idx = np.unique(grid, return_index=True)
            return [grid[np.sort(idx),]]
        else:
            meshes = np.moveaxis(grid, len(grid.shape)-1, 0)
            def _pull_u(g):
                flat = g.flatten()
                _, idx = np.unique(g.flatten(), return_index=True)
                return flat[np.sort(idx),]
            return [_pull_u(np.round(m, tol)) for m in meshes]

    @classmethod
    def get_mesh_spacings(cls, grid, tol=8):
        subgrids = cls.get_mesh_subgrids(grid, tol=tol)
        if subgrids is None:
            return None

        mesh_spacings = [np.unique(np.diff(g)) for g in subgrids]
        if len(subgrids) == 1:
            mesh_spacings = mesh_spacings[0]
        return mesh_spacings

    @staticmethod
    def _is_meshgrid(grid):
        """
        Just checks if the shape of the grid
        is consistent with being a meshgrid
        :param grid:
        :type grid:
        :return:
        :rtype:
        """

        shape = grid.shape
        return (
            len(shape) > 1
            and any(shape[0] != s for s in shape[1:])
            and (shape[0] == len(shape) - 1)
        ) # might introduce edge-case...? Hard to know

    @classmethod
    def get_mesh_type(cls, grid, check_product_grid=True, check_regular_grid=True, tol=8):
        """
        Determines what kind of grid we're working with

        :param grid:
        :type grid: np.ndarray
        :return: mesh_type
        :rtype: MeshType
        """

        grid = np.asanyarray(grid).view(np.ndarray)

        if grid.dtype == np.dtype(object):
            return MeshType.Indeterminate

        ndim = grid.ndim
        shape = grid.shape
        cdim = ndim-1

        # we either need grid points or a full mesh
        if ndim > 2 and shape[-1] != cdim and shape[0] != cdim:
            raise ValueError("don't know how to interpret mesh with shape {}".format(shape))

        if ndim > 1 and shape[-1] != cdim and shape[0] == cdim:
            grid = np.moveaxis(grid, 0, ndim-1) # make full mesh

        if ndim == 1:
            mesh_spacings = cls.get_mesh_spacings(grid, tol=tol)
            if len(mesh_spacings) == 1:
                return MeshType.Regular
            else:
                return MeshType.Structured # all grids are structured in 1D
        elif ndim == 2 and shape[-1] != 1: # this means we got gridpoints
            subgrids = cls.get_mesh_subgrids(grid)  # check if can be read as a structured grid
            if grid.shape[0] == np.prod([len(x) for x in subgrids]):

                # check if it really is a product grid
                if check_product_grid:
                    true_mesh = np.moveaxis(np.array(np.meshgrid(*subgrids, indexing='ij')), 0, ndim-1)
                    test_grid = np.reshape(grid, true_mesh.shape)
                    product_grid = np.allclose(true_mesh, test_grid, atol=10 ** (-tol))
                else:
                    product_grid = True

                if product_grid and check_regular_grid:
                    spacings = np.unique([np.diff(x) for x in subgrids])
                    regular_grid = all(len(x) == 1 for x in spacings)
                else:
                    regular_grid = product_grid

                if regular_grid:
                    return MeshType.Regular
                elif product_grid:
                    return MeshType.Structured
                else:
                    return MeshType.Unstructured

            else:
                # should try to check for semistructured grids...
                return MeshType.Unstructured
        else:
            # means we _probably_ have a structured grid
            subgrids = cls.get_mesh_subgrids(grid, tol=tol)
            # raise Exception(
            #     np.prod(grid.shape[:-1]), grid.shape,
            #     np.prod([len(x) for x in subgrids]), subgrids
            # )
            if np.prod(grid.shape[:-1]) == np.prod([len(x) for x in subgrids]): # check if it could be a product grid
                # check if it really is a product grid
                if check_product_grid:
                    true_mesh = np.moveaxis(np.array(np.meshgrid(*subgrids, indexing='ij')), 0, ndim-1)
                    product_grid = np.allclose(true_mesh, grid, atol=10**(-tol))
                else:
                    product_grid = True

                if product_grid and check_regular_grid:
                    spacings = [np.unique(np.diff(x)) for x in subgrids]
                    regular_grid = all(len(x) == 1 for x in spacings)
                else:
                    regular_grid = product_grid

                if regular_grid:
                    return MeshType.Regular
                elif product_grid:
                    return MeshType.Structured
                else:
                    return MeshType.Unstructured
            else:
                # should try to check for semistructured grids...
                return MeshType.Unstructured

    @classmethod
    def RegularMesh(cls, *mesh_specs):
        """
        Builds a grid from multiple linspace arguments,
        basically insuring it's structured (if non-Empty)
        :param mesh_specs:
        :type mesh_specs:
        :return:
        :rtype:
        """
        # should probably handle the empty mesh subcase...
        coords = [ np.linspace(*m) for m in mesh_specs ]

        if any(len(x) == 0 for x in coords):
            raise ValueError("can't handle empty linspaces")
        cg = np.array(np.meshgrid(*coords, indexing='ij'))
        wat = np.moveaxis(cg, 0, len(mesh_specs))
        return cls(wat, mesh_type=MeshType.Regular)
