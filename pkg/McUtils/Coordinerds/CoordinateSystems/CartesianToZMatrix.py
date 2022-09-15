from .CoordinateSystemConverter import CoordinateSystemConverter
from .CommonCoordinateSystems import CartesianCoordinates3D, ZMatrixCoordinates
from ...Numputils import vec_norms, vec_angles, pts_dihedrals, dist_deriv, angle_deriv, dihed_deriv
import numpy as np
# this import gets bound at load time, so unfortunately PyCharm can't know just yet
# what properties its class will have and will try to claim that the files don't exist

class CartesianToZMatrixConverter(CoordinateSystemConverter):
    """
    A converter class for going from Cartesian coordinates to ZMatrix coordinates
    """

    @property
    def types(self):
        return (CartesianCoordinates3D, ZMatrixCoordinates)

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
        orig_ol = ZMatrixCoordinates.canonicalize_order_list(ncoords, ordering)
        ol = orig_ol
        nol = len(ol)
        ncol = len(ol[0])
        fsteps = ncoords / nol
        steps = int(fsteps)

        # print(">> c2z >> ordering:", ol)

        multiconfig = nol < ncoords
        if multiconfig:
            ol = ZMatrixCoordinates.tile_order_list(ol, ncoords)
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
                if ol.shape[1] == 5:
                    raise NotImplementedError("psi angles might be unnecessary")
                    ix = ix.copy()
                    jx = jx.copy()
                    kx = kx.copy()
                    lx = lx.copy()
                    fx = ol[3:, 4]
                    swap_pos = np.where(fx == 1)
                    swap_i = ix[swap_pos]
                    swap_j = jx[swap_pos]
                    swap_k = kx[swap_pos]
                    swap_l = lx[swap_pos]
                    ix[swap_pos] = swap_l
                    jx[swap_pos] = swap_i
                    kx[swap_pos] = swap_j
                    lx[swap_pos] = swap_k

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
                if ol.shape[1] == 5:
                    raise ValueError("Unclear if there is a difference between tau and psi")
                    ix = ix.copy()
                    jx = jx.copy()
                    kx = kx.copy()
                    lx = lx.copy()
                    fx = ol[mask, 4]
                    swap_pos = np.where(fx == 1)
                    swap_i = ix[swap_pos]
                    swap_j = jx[swap_pos]
                    swap_k = kx[swap_pos]
                    swap_l = lx[swap_pos]
                    ix[swap_pos] = swap_l
                    jx[swap_pos] = swap_i
                    kx[swap_pos] = swap_j
                    lx[swap_pos] = swap_k
                # print(ol)

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
            ol = np.reshape(ol[mask], (steps, nol-1, ncol))-np.reshape(np.arange(steps), (steps, 1, 1))
            ol = np.reshape(ol, (ncoords-steps, ncol))
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
        if ncol == 5:
            ordering = np.array([
                        np.argsort(ol[:, 0]), om[ol[:, 1]], om[ol[:, 2]], om[ol[:, 3]], ol[:, 4]
                    ]).T
        else:
            ordering = np.array([
                    np.argsort(ol[:, 0]), om[ol[:, 1]], om[ol[:, 2]], om[ol[:, 3]]
                ]).T
        opts = dict(use_rad=use_rad, ordering=ordering, origins=origins, axes=axes)

        # if we're returning derivs, we also need to make sure that they're ordered the same way the other data is...
        if return_derivs:
            opts['derivs'] = derivs#[:1]

        return final_coords, opts

__converters__ = [ CartesianToZMatrixConverter() ]