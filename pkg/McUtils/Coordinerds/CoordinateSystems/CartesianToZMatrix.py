from .CoordinateSystemConverter import CoordinateSystemConverter
from .CommonCoordinateSystems import CartesianCoordinateSystem, ZMatrixCoordinateSystem
from ...Numputils import vec_norms, vec_angles, pts_dihedrals, dist_deriv, angle_deriv, dihed_deriv
import numpy as np
# this import gets bound at load time, so unfortunately PyCharm can't know just yet
# what properties its class will have and will try to claim that the files don't exist

class CartesianToZMatrixConverter(CoordinateSystemConverter):
    """A converter class for going from Cartesian coordinates to ZMatrix coordinates

    """

    @property
    def types(self):
        return (CartesianCoordinateSystem, ZMatrixCoordinateSystem)

    def canonicalize_order_list(self, ncoords, order_list):
        """Normalizes the way the ZMatrix coordinates are built out

        :param ncoords:
        :type ncoords:
        :param order_list: the basic ordering to apply for the
        :type order_list: iterable or None
        :return:
        :rtype: iterator of int triples
        """
        if order_list is None:
            normalized_list = np.array( (
                   np.arange(ncoords),
                   np.arange(-1, ncoords-1),
                   np.arange(-2, ncoords-2),
                   np.arange(-3, ncoords-3),
                ) ).T
        else:
            normalized_list = [None] * len(order_list)
            for i, el in enumerate(order_list):
                if isinstance(el, int):
                    spec = (
                        el,
                        normalized_list[i-1][0] if i > 0 else -1,
                        normalized_list[i-2][0] if i > 1 else -1,
                        normalized_list[i-3][0] if i > 2 else -1
                    )
                else:
                    spec = tuple(el)
                    # if len(spec) < 4:
                    #     spec = (i,) + spec + (
                    #     normalized_list[i-1][0] if i > 0 else -1,
                    #     normalized_list[i-2][0] if i > 1 else -1,
                    #     normalized_list[i-3][0] if i > 2 else -1
                    # )
                    # spec = spec[:4]
                    if len(spec) < 4:
                        raise ValueError(
                            "Z-matrix conversion spec {} not long enough. Expected ({}, {}, {}, {})".format(
                                el,
                                "atomNum", "distAtomNum", "angleAtomNum", "dihedAtomNum"
                            ))

                normalized_list[i] = spec
        return np.asarray(normalized_list, dtype=np.int8)

    @staticmethod
    def get_dists(points, centers):
        return vec_norms(centers-points)
    @staticmethod
    def get_angles(lefts, centers, rights):
        # need to look up again what the convention is for which atom is the central one...
        v1s = centers-lefts
        v2s = centers-rights
        return vec_angles(v1s, v2s)[0]
    @staticmethod
    def get_diheds(points, centers, seconds, thirds):
        return pts_dihedrals(points, centers, seconds, thirds)

    def convert_many(self, coords, ordering=None, use_rad=True, return_derivs=False, **kw):
        """
        We'll implement this by having the ordering arg wrap around in coords?
        """
        if ordering is None:
            ordering = range(len(coords[0]))
        base_shape = coords.shape
        new_coords = np.reshape(coords, (np.product(base_shape[:-1]),) + base_shape[-1:])
        new_coords, ops = self.convert(new_coords, ordering=ordering, use_rad=use_rad, return_derivs=return_derivs)
        single_coord_shape = (base_shape[-2]-1, new_coords.shape[-1])
        new_shape = base_shape[:-2] + single_coord_shape
        new_coords = np.reshape(new_coords, new_shape)
        if return_derivs:
            ders = ops['derivs']
            # we assume we get a list of derivatives?
            reshaped_ders = [None]*len(ders)
            for i, v in enumerate(ders):
                single_base_shape = (base_shape[-2], new_coords.shape[-1])
                ders_shape = coords.shape + single_base_shape*i + single_coord_shape
                v = v.reshape(ders_shape)
                reshaped_ders[i] = v
                ops['derivs'] = reshaped_ders
        return new_coords, ops

    def convert(self, coords, ordering=None, use_rad=True, return_derivs=False, **kw):
        """The ordering should be specified like:

        [
            [n1],
            [n2, n1]
            [n3, n1/n2, n1/n2]
            [n4, n1/n2/n3, n1/n2/n3, n1/n2/n3]
            [n5, ...]
            ...
        ]

        :param coords:    array of cartesian coordinates
        :type coords:     np.ndarray
        :param use_rad:   whether to user radians or not
        :type use_rad:    bool
        :param ordering:  optional ordering parameter for the z-matrix
        :type ordering:   None or tuple of ints or tuple of tuple of ints
        :param kw:        ignored key-word arguments
        :type kw:
        :return: z-matrix coords
        :rtype: np.ndarray
        """
        ncoords = len(coords)
        orig_ol = self.canonicalize_order_list(ncoords, ordering)
        ol = orig_ol
        nol = len(ol)

        multiconfig = nol < ncoords
        # print(coords)
        if multiconfig:
            fsteps = ncoords / nol
            steps = int(fsteps)
            if steps != fsteps:
                raise ValueError(
                    "{}: Number of coordinates {} and number of specifed elements {} misaligned".format(
                        type(self),
                        ncoords,
                        nol
                    )
                )
            # broadcasts a single order spec to be a multiple order spec
            ol = np.reshape(
                np.broadcast_to(ol, (steps, nol, 4)) +
                np.reshape(np.arange(0, ncoords, nol), (steps, 1, 1)),
                (ncoords, 4)
            )
            mc_ol = ol.copy()

        # we define an order map that we'll index into to get the new indices for a
        # given coordinate
        om = 1+np.argsort(ol[:, 0])

        # need to check against the cases of like 1, 2, 3 atom molecules
        # annoying but not hard
        if return_derivs:
            derivs = [
                np.zeros(coords.shape + (nol-1, 3)),
                np.zeros(coords.shape + (nol, 3) + (nol - 1, 3))
            ]
        if not multiconfig:
            ix = ol[1:, 0]
            jx = ol[1:, 1]
            dists = self.get_dists(coords[ix], coords[jx])
            if return_derivs:
                _dists, dist_derivs, dist_derivs_2 = dist_deriv(coords, ix, jx, order=2)
                drang = np.arange(len(ix))
                derivs[0][ix, :, drang, 0] = dist_derivs[0]
                derivs[0][jx, :, drang, 0] = dist_derivs[1]

                for i, x1 in enumerate([ix, jx]):
                    for j, x2 in enumerate([ix, jx]):
                        # print(i, j, x1, x2,
                        #       # dist_derivs_2[i, j][0, 0],
                        #       drang
                        #       )
                        derivs[1][x1, :, x2, :, drang, 0] = dist_derivs_2[i, j]

            if len(ol) > 2:
                ix = ol[2:, 0]
                jx = ol[2:, 1]
                kx = ol[2:, 2]
                angles = np.concatenate( (
                    [0], self.get_angles(coords[ix], coords[jx], coords[kx])
                ) )
                if not use_rad:
                    angles = np.rad2deg(angles)
                if return_derivs:
                    _angles, angle_derivs, angle_derivs_2 = angle_deriv(coords, jx, ix, kx, order=2)
                    drang = 1+np.arange(len(ix))
                    # print(">>>>", np.max(np.abs(angle_derivs)))
                    derivs[0][jx, :, drang, 1] = angle_derivs[0]
                    derivs[0][ix, :, drang, 1] = angle_derivs[1]
                    derivs[0][kx, :, drang, 1] = angle_derivs[2]

                    for i, x1 in enumerate([jx, ix, kx]):
                        for j, x2 in enumerate([jx, ix, kx]):
                            derivs[1][x1, :, x2, :, drang, 1] = angle_derivs_2[i, j]
            else:
                angles = np.array([0.])
            if len(ol) > 3:
                ix = ol[3:, 0]
                jx = ol[3:, 1]
                kx = ol[3:, 2]
                lx = ol[3:, 3]
                diheds = np.concatenate( (
                    [0, 0],
                    self.get_diheds(coords[ix], coords[jx], coords[kx], coords[lx])
                ) )
                if not use_rad:
                    diheds = np.rad2deg(diheds)
                if return_derivs:
                    _diheds, dihed_derivs, dihed_derivs_2 = dihed_deriv(coords, ix, jx, kx, lx, order=2)
                    drang = 2+np.arange(len(ix))
                    derivs[0][ix, :, drang, 2] = dihed_derivs[0]
                    derivs[0][jx, :, drang, 2] = dihed_derivs[1]
                    derivs[0][kx, :, drang, 2] = dihed_derivs[2]
                    derivs[0][lx, :, drang, 2] = dihed_derivs[3]

                    for i, x1 in enumerate([ix, jx, kx, lx]):
                        for j, x2 in enumerate([ix, jx, kx, lx]):
                            derivs[1][x1, :, x2, :, drang, 2] = dihed_derivs_2[i, j]
            else:
                diheds = np.array([0, 0])
            ol = ol[1:]

        else: # multiconfig

            # we do all of this stuff with masking operations in the multiconfiguration cases
            mask = np.repeat(True, ncoords)
            mask[np.arange(0, ncoords, nol)] = False
            ix = ol[mask, 0]
            jx = ol[mask, 1]
            dists = self.get_dists(coords[ix], coords[jx])
            if return_derivs:
                _, dist_derivs, dist_derivs_2 = dist_deriv(coords, ix, jx, order=2)
                drang = np.arange(nol-1)
                nreps = int(len(ix)/(nol-1))
                drang = np.broadcast_to(drang[np.newaxis], (nreps,) + drang.shape).flatten()
                derivs[0][ix, :, drang, 0] = dist_derivs[0]
                derivs[0][jx, :, drang, 0] = dist_derivs[1]

                for i, x1 in enumerate([ix, jx]):
                    for j, x2 in enumerate([ix, jx]):
                        derivs[1][x1, :, x2 % nol, :, drang, 0] = dist_derivs_2[i, j]

            if nol>2:
                # set up the mask to drop all of the first bits
                mask[np.arange(1, ncoords, nol)] = False
                ix = ol[mask, 0]
                jx = ol[mask, 1]
                kx = ol[mask, 2]
                angles = self.get_angles(coords[ix], coords[jx], coords[kx])
                angles = np.append(angles, np.zeros(steps))
                insert_pos = np.arange(0, ncoords-1*steps-1, nol-2)
                angles = np.insert(angles, insert_pos, 0)
                angles = angles[:ncoords-steps]
                if not use_rad:
                    angles = np.rad2deg(angles)
                if return_derivs:
                    # we might need to mess with the masks akin to the insert call...
                    _, angle_derivs, angle_derivs_2 = angle_deriv(coords, jx, ix, kx, order=2)
                    drang = 1+np.arange(nol-2)
                    nreps = int(len(ix)/(nol-2))
                    drang = np.broadcast_to(drang[np.newaxis], (nreps,) + drang.shape).flatten()
                    derivs[0][jx, :, drang, 1] = angle_derivs[0]
                    derivs[0][ix, :, drang, 1] = angle_derivs[1]
                    derivs[0][kx, :, drang, 1] = angle_derivs[2]

                    for i, x1 in enumerate([ix, jx, kx]):
                        for j, x2 in enumerate([ix, jx, kx]):
                            derivs[1][x1, :, x2 % nol, :, drang, 0] = angle_derivs_2[i, j]
            else:
                angles = np.zeros(ncoords-steps)

            if nol > 3:
                # set up mask to drop all of the second atom bits (wtf it means 'second')
                mask[np.arange(2, ncoords, nol)] = False
                ix = ol[mask, 0]
                jx = ol[mask, 1]
                kx = ol[mask, 2]
                lx = ol[mask, 3]
                diheds = self.get_diheds(coords[ix], coords[jx], coords[kx], coords[lx])
                # pad diheds to be the size of ncoords
                diheds = np.append(diheds, np.zeros(2*steps))

                # insert zeros where undefined
                diheds = np.insert(diheds, np.repeat(np.arange(0, ncoords-2*steps-1, nol-3), 2), 0)
                # take only as many as actually used
                diheds = diheds[:ncoords-steps]
                if not use_rad:
                    diheds = np.rad2deg(diheds)
                if return_derivs:
                    # Negative sign because my dihed_deriv code is for slightly different
                    # ordering than expected
                    _, dihed_derivs, dihed_derivs_2 = dihed_deriv(coords, ix, jx, kx, lx, order=2)
                    drang = 2+np.arange(nol-3)
                    nreps = int(len(ix)/(nol-3))
                    drang = np.broadcast_to(drang[np.newaxis], (nreps,) + drang.shape).flatten()
                    derivs[0][ix, :, drang, 2] = dihed_derivs[0]
                    derivs[0][jx, :, drang, 2] = dihed_derivs[1]
                    derivs[0][kx, :, drang, 2] = dihed_derivs[2]
                    derivs[0][lx, :, drang, 2] = dihed_derivs[3]

                    for i, x1 in enumerate([ix, jx, kx, lx]):
                        for j, x2 in enumerate([ix, jx, kx, lx]):
                            derivs[1][x1, :, x2 % nol, :, drang, 0] = dihed_derivs_2[i, j]

            else:
                diheds = np.zeros(ncoords-steps)

            # after the np.insert calls we have the right number of final elements, but too many
            # ol and om elements and they're generally too large
            # so we need to shift them down and mask out the elements we don't want
            mask = np.repeat(True, ncoords)
            mask[np.arange(0, ncoords, nol)] = False
            ol = np.reshape(ol[mask], (steps, nol-1, 4))-np.reshape(np.arange(steps), (steps, 1, 1))
            ol = np.reshape(ol, (ncoords-steps, 4))
            om = np.reshape(om[mask], (steps, nol-1))-nol*np.reshape(np.arange(steps), (steps, 1))-1
            om = np.reshape(om, (ncoords-steps,))

        final_coords = np.array(
            [
                dists, angles, diheds
            ]
        ).T

        if multiconfig:
            # figure out what to use for the axes
            origins = coords[mc_ol[1::nol,  1]]
            x_axes  = coords[mc_ol[1::nol,  0]] - origins # the first displacement vector
            y_axes  = coords[mc_ol[2::nol,  0]] - origins # the second displacement vector (just defines the x-y plane, not the real y-axis)
            axes = np.array([x_axes, y_axes]).transpose((1, 0, 2))

        else:
            origins = coords[ol[0, 1]]
            axes = np.array([coords[ol[0, 0]] - origins, coords[ol[1, 0]] - origins])

        ol = orig_ol
        om = om - 1
        ordering = np.array(
                [
                    np.argsort(ol[:, 0]), om[ol[:, 1]], om[ol[:, 2]], om[ol[:, 3]]
                ]
            ).T
        opts = dict(use_rad=use_rad, ordering=ordering, origins=origins, axes=axes)

        # if we're returning derivs, we also need to make sure that they're ordered the same way the other data is...
        if return_derivs:
            opts['derivs'] = derivs#[:1]

        return final_coords, opts

__converters__ = [ CartesianToZMatrixConverter() ]