from .CoordinateSystemConverter import CoordinateSystemConverter
from .CommonCoordinateSystems import CartesianCoordinates3D, ZMatrixCoordinates
from ...Numputils import *
import numpy as np

class ZMatrixToCartesianConverter(CoordinateSystemConverter):
    """
    A converter class for going from ZMatrix coordinates to Cartesian coordinates
    """

    @property
    def types(self):
        return (ZMatrixCoordinates, CartesianCoordinates3D)

    def default_ordering(self, coordlist):
        if coordlist.shape[-1] == 6:
            ordering = coordlist[:, :, (0, 2, 4)]
            coordlist = coordlist[:, :, (1, 3, 5)]
        else:
            r = np.arange(len(coordlist[0]))
            ordering = np.broadcast_to(
                np.array([r, np.roll(r, 1), np.roll(r, 2)]).T[np.newaxis],
                coordlist.shape[:2] + (3,)
            )
        return ordering, coordlist

    def convert_many(self,
                     coordlist,
                     ordering=None, origins=None, axes=None, use_rad=True,
                     return_derivs=False,
                     **kw
                     ):
        """Expects to get a list of configurations
        These will look like:
            [
                [dist, angle, dihedral]
                ...
            ]
        and ordering will be
            [
                [pos, point, line, plane]
                ...
            ]
        **For efficiency it is assumed that all configurations have the same length**

        :param coordlist:
        :type coordlist:
        :param origins:
        :type origins:
        :param axes:
        :type axes:
        :param use_rad:
        :type use_rad:
        :param kw:
        :type kw:
        :param ordering:
        :type ordering:
        :param return_derivs:
        :type return_derivs:
        :return:
        :rtype:
        """

        # make sure we have the ordering stuff in hand
        if ordering is None:
           ordering, coordlist = self.default_ordering(coordlist)
        else:
            ordering = np.array(ordering)
        coordlist = np.asarray(coordlist)

        if np.min(ordering) > 0:
            ordering = ordering - 1
        dim_diff = coordlist.ndim - ordering.ndim
        if dim_diff > 0:
            missing = coordlist.shape[:dim_diff]
            ordering = np.broadcast_to(ordering, missing + ordering.shape )

        if ordering.shape[-1] > 3:
            atom_ordering = ordering[:, :, 0]
            ordering = ordering[:, 1:, 1:]
        else:
            atom_ordering = None

        sysnum = len(coordlist)
        coordnum = len(coordlist[0])
        total_points = np.empty((sysnum, coordnum+1, 3))
        if return_derivs is not True and return_derivs is not False and isinstance(return_derivs, int):
            return_derivs = True
            return_deriv_order = return_derivs
        elif return_derivs:
            return_deriv_order = 2
        if return_derivs:
            derivs = [
                None, # no need to stoare a copy of total_points here...
                np.zeros((sysnum, coordnum, 3, coordnum + 1, 3)),
                np.zeros((sysnum, coordnum, 3, coordnum, 3, coordnum + 1, 3))
            ]


        # first we put the origin whereever the origins are specified
        if origins is None:
            origins = [0, 0, 0]
        origins = np.asarray(origins)
        if len(origins.shape) < 2:
            origins = np.broadcast_to(origins, (sysnum, 3))
        total_points[:, 0] = origins

        # set up the next points by just setting them along the x-axis by default
        if axes is None:
            axes = [1, 0, 0]
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = np.array([
                axes,
                [0, 1, 0]
            ])  # np.concatenate((np.random.uniform(low=.5, high=1, size=(2,)), np.zeros((1,)) ))])
        if axes.ndim == 2:
            axes = np.broadcast_to(axes[np.newaxis], (sysnum, 2, 3))
        x_pts = origins + vec_normalize(axes[:, 0])
        y_pts = origins + vec_normalize(axes[:, 1])

        dists = coordlist[:, 0, 0]
        if return_derivs:
            der_stuff = cartesian_from_rad_derivatives(origins,
                                                       x_pts, y_pts, dists,
                                                       None, None,
                                                       0,
                                                       np.full((len(dists),), -1, dtype=int),
                                                       np.full((len(dists),), -1, dtype=int),
                                                       np.full((len(dists),), -1, dtype=int),
                                                       derivs,
                                                       order=return_deriv_order
                                                       )
            total_points[:, 1] = der_stuff[0]
            if return_deriv_order > 0:
                derivs[1][np.arange(sysnum), :1, :, 1, :] = der_stuff[1]
            if return_deriv_order > 1:
                derivs[2][np.arange(sysnum), :1, :, :1, :, 1, :] = der_stuff[2]

        else:
            ref_points_1, _ = cartesian_from_rad(origins, x_pts, y_pts, dists, None, None)
            total_points[:, 1] = ref_points_1

        # print(">> z2c >> ordering", ordering[0])

         # iteratively build the rest of the coords with one special cases for n=2
        for i in range(1, coordnum):
            # Get the distances away

            ref_coords1 = ordering[:, i, 0] # reference atom numbers for first coordinate
            refs1 = total_points[np.arange(sysnum), ref_coords1.astype(int)] # get the actual reference coordinates
            dists = np.reshape(coordlist[:, i, 0], (sysnum, 1)) # pull the requisite distances

            ref_coords2 = ordering[:, i, 1] # reference atom numbers for second coordinate
            refs2 = total_points[np.arange(sysnum), ref_coords2.astype(int)] # get the actual reference coordinates for the angle
            angle = coordlist[:, i, 1] # pull the requisite angle values
            if not use_rad:
                angle = np.deg2rad(angle)

            if i == 1:
                refs3 = y_pts
                dihed = None
                ref_coords3 = np.full((len(dists),), -1, dtype=int)
                psi_flag = False
            else:
                ref_coords3 = ordering[:, i, 2] # reference atom numbers for dihedral ref coordinate
                refs3 = total_points[np.arange(sysnum), ref_coords3.astype(int)] # get the actual reference coordinates for the dihed
                dihed = coordlist[:, i, 2] # pull proper dihedral values
                if not use_rad:
                    dihed = np.deg2rad(dihed)
                if ordering.shape[-1] == 4:
                    raise ValueError("Unclear if there is a difference between tau and psi")
                    psi_flag = ordering[:, i, 3] == 1
                    # dihed[psi_flag] = -dihed[psi_flag]
                else:
                    psi_flag = False

            if return_derivs:
                if ordering.shape[-1] == 4:
                    raise NotImplementedError("don't have derivatives for case with psi angles")
                der_stuff = cartesian_from_rad_derivatives(
                    refs1, refs2, refs3,
                    dists, angle, dihed,
                    i,
                    ref_coords1,
                    ref_coords2,
                    ref_coords3,
                    derivs,
                    order=return_deriv_order
                )
                # crd, d1, d2 = stuff

                total_points[:, i+1] = der_stuff[0]
                if return_deriv_order > 0:
                    derivs[1][np.arange(sysnum), :i+1, :, i+1, :] = der_stuff[1]
                if return_deriv_order > 1:
                    derivs[2][np.arange(sysnum), :i+1, :, :i+1, :, i+1, :] = der_stuff[2]
            else:
                ref_points_1, _ = cartesian_from_rad(refs1, refs2, refs3, dists, angle, dihed, psi=psi_flag)
                total_points[:, i+1] = ref_points_1

        if atom_ordering is not None:
            rev_ord = atom_ordering#np.argsort(atom_ordering, axis=1)
            total_points = total_points[np.arange(len(atom_ordering))[:, np.newaxis], rev_ord] #wat?

        converter_opts = dict(use_rad=use_rad, ordering=ordering)
        if return_derivs:
            if return_deriv_order > 0:
                converter_opts['derivs'] = derivs[1:][:return_deriv_order]

        return total_points, converter_opts

    def convert(self, coords, **kw):
        """dipatches to convert_many but only pulls the first"""
        total_points, opts = self.convert_many(coords[np.newaxis], **kw)
        return total_points[0], opts

__converters__ = [ ZMatrixToCartesianConverter() ]