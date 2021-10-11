"""
Provides constants data and conversions between units and unit systems
"""
from .CommonData import DataHandler
from collections import OrderedDict, deque

__all__ = [ "UnitsData", "UnitsDataHandler" ]

class ConversionError(Exception):
    pass

class UnitGraph:
    def __init__(self, stuff_to_update = ()):
        self._graph = {}
        self.update(stuff_to_update)

    def __contains__(self, item):
        return item in self._graph
    def add(self, node, connection):
        if node in self._graph:
            self._graph[node].add(connection)
        else:
            self._graph[node]={connection}
        if not connection in self._graph:
            self._graph[connection] = set()
    def update(self, iterable):
        for connection in iterable:
            self.add(*connection)
    def find_path_bfs(self, start, end):
        # we use a little poor-man's Dijkstra to find the shortest unit conversion path

        if not start in self._graph or not end in self._graph:
            return None

        q = deque() # deque as a FIFO queue
        q.append(start)
        parents = { start:None }
        steps = 0
        while len(q)>0:
            cur = q.pop()
            steps += 1
            for k in self._graph[cur]:
                if k is end:
                    parents[k] = cur
                    break
                elif k not in parents:
                    q.append(k)
                    parents[k] = cur
        if end not in parents:
            return None
        else:
            path = []*steps
            cur = end
            while cur != start:
                path.append(cur)
                cur = parents[cur]
            path.append(start)
            return list(reversed(path))

class UnitsDataHandler(DataHandler):
    """
    A DataHandler that's built for use with the units data we've collected.
    Usually used through the `UnitsData` object.
    """
    prefix_map=OrderedDict((
        # most common
        ('Kilo', 1E3), ('Milli', 1E-3), ('Centi', 1E-2),
        # second most common ones
        ('Giga', 1E9), ('Mega', 1E6), ('Micro', 1E-6),
        ('Nano', 1E-9), ('Pico', 1E-12), ('Femto', 1E-15),
        ('Atto', 1E-18), ('Tera', 1E12), ('Deci', 1E-1),
        # least common ones
        ('Hecto', 1E2), ('Deka', 1E1), ('Yotta', 1E24),
        ('Zetta', 1E21), ('Exa', 1E18), ('Peta', 1E15),
        ('Zepto', 1E-21), ('Yocto', 1E-24)
    ))
    postfix_map=OrderedDict((
        ("Squared", 2),
        ("Cubed", 3),
        ("Fourthed", 4),
        ("Fifthed", 5)
        # need a better way to do this... but that's for a later date
    ))
    def __init__(self):
        super().__init__("ConstantsData")
        self._unit_graph=UnitGraph()

    def load(self):
        super().load()
        self._load_unit_graph()

    #region Unit Conversions
    def _load_unit_graph(self):
        """Builds a graph of units to traverse when finding conversions"""
        extras = {}
        for u, v in self._data.items():
            if isinstance(u, tuple) and "Conversion" in v: # trying to tell if something is a conversion or not...
                self._unit_graph.add(*u)
                symm = (u[1], u[0])
                if symm not in self._data:
                    extras[symm] = {"Value":1/v["Value"]}
                    self._unit_graph.add(*symm)

                # add inverses to maps too
                if isinstance(u[0], str):
                    uinv1 = u[0]
                else:
                    uinv1 = u[0][0]
                if uinv1.startswith("Inverse"):
                    uinv1 = uinv1.split("Inverse", 2)[0]
                else:
                    uinv1 = "Inverse"+uinv1
                if not isinstance(u[0], str):
                    uinv1 = (uinv1, u[0][1])

                if isinstance(u[1], str):
                    uinv2 = u[1]
                else:
                    uinv2 = u[1][0]
                if uinv2.startswith("Inverse"):
                    uinv2 = uinv2.split("Inverse", 2)[0]
                else:
                    uinv2 = "Inverse"+uinv2
                if not isinstance(u[1], str):
                    uinv2 = (uinv2, u[1][1])

                invu = (uinv1, uinv2)
                if invu not in self._data:
                    extras[invu] = {"Value":1/v["Value"]}
                    self._unit_graph.add(*invu)
                invsu = (uinv2, uinv1)
                if invsu not in self._data:
                    extras[invsu] = {"Value":v["Value"]}
                    self._unit_graph.add(*invsu)

        self._data.update(extras)

    def _get_unit_modifiers(self, unit):
        """Pulls modifiers off strings like InverseDecimeters

        :param unit:
        :type unit: str
        :return: scaling, inverted, base_unit, power
        :rtype:
        """
        already_there = unit in self._unit_graph
        if already_there:
            scaling = 1
            inverted = False
            if unit.startswith("Inverse"):
                inverted = True
                unit = unit.split("Inverse", 2)[1]
            base_unit = unit
            power = 1
        else:
            power = 1
            for postfix in self.postfix_map:
                if unit.endswith(postfix):
                    power = self.postfix_map[postfix]
                    unit = postfix.join(unit.split(postfix)[:-1])
                    break
            inverted = False
            if unit.startswith("Inverse"):
                inverted = True
                unit = unit.split("Inverse", 2)[1]
            scaling = 1
            for prefix in self.prefix_map:
                if unit.startswith(prefix):
                    scaling = self.prefix_map[prefix]
                    unit = unit.split(prefix, 2)[1]
                    unit = unit[0].upper() + unit[1:]
                    break

            # if unit not in self._unit_graph:
            #     base_unit = None
            #     # raise KeyError("{}: base unit {} not in graph".format(type(self).__name__, unit))
            # else:
            #     base_unit = unit
            base_unit = unit # we'll handle all the in or out stuff later

        return base_unit, inverted, scaling, power

    def _canonicalize_unit(self, unit):
        if isinstance(unit, str):
            bits = list(map(self._get_unit_modifiers, unit.split()))
        else:
            bits = []
            for u in unit:
                if isinstance(u, tuple): # things can be fed in like (unit, power) for generality
                    base_unit, inverted, scaling, power = self._get_unit_modifiers(u[0])
                    power = u[1]
                    bits.append((base_unit, inverted, scaling, power))

        return bits

    def _find_direct_conversion(self, src, targ):
        # basically this tries the like 4 common types of conversions for units with a direct conversion rule
        # for things without one already built in it returns None

        src_base, src_inv, src_scale, src_pow = src
        targ_base, targ_inv, targ_scale, targ_pow = targ
        # At this point we actually add the inverse flags back on to start
        src = src_base
        if src_inv:
            src = "Inverse" + src
        targ = targ_base
        if targ_inv:
            targ = "Inverse" + targ

        # now we have to exhaust all of our direct conversion options:
        #   not inverted / no power
        #   inverted / no power
        #   not inverted / power
        #   inverted / power

        hooray_it_worked = False
        conv = None
        if not hooray_it_worked: # not inverted / no power
            try:
                conv = self.data[(src, targ)]
            except KeyError:
                pass
            else:
                hooray_it_worked = True
                conv = ( (src_scale / targ_scale) * conv["Value"] ) ** ( src_pow / targ_pow )

        if not hooray_it_worked: # inverted / no power
            if src_inv:
                src_2 = src_base
            else:
                src_2 = src
            if targ_inv:
                targ_2 = targ_base
            else:
                targ_2 = targ
            try:
                conv = self.data[(src_2, targ_2)]
            except KeyError:
                pass
            else:
                hooray_it_worked = True
                conv = ( 1 / ((src_scale / targ_scale) * conv["Value"]) ) ** ( targ_pow / src_pow )

        if (src_pow > 1 or targ_pow > 1):
            if not hooray_it_worked: # not inverted / power
                if src_pow > 1:
                    src_2 = (src, src_pow)
                else:
                    src_2 = src
                if targ_pow > 1:
                    targ_2 = (targ, targ_pow)
                else:
                    targ_2 = targ
                try:
                    conv = self.data[(src_2, targ_2)]
                except KeyError:
                    pass
                else:
                    hooray_it_worked = True
                    conv = ( (src_scale / targ_scale) * conv["Value"] )
            if not hooray_it_worked: # inverted / power
                if src_inv:
                    src_2 = src_base
                else:
                    src_2 = src
                if targ_inv:
                    targ_2 = targ_base
                else:
                    targ_2 = targ
                if src_pow > 1:
                    src_2 = (src_2, src_pow)
                if targ_pow > 1:
                    targ_2 = (targ_2, targ_pow)
                try:
                    conv = self.data[(src_2, targ_2)]
                except KeyError:
                    pass
                else:
                    hooray_it_worked = True
                    conv = 1 / ( (src_scale / targ_scale) * conv["Value"] )

        return conv

    def _get_pathy_conversion(self, src, targ):
        conv_path = self._unit_graph.find_path_bfs(src, targ)
        if conv_path is None:
            return conv_path
        cval = 1
        for me, you in zip(conv_path, conv_path[1:]):
            cval *= self.data[(me, you)]["Value"]
        return cval
    def _find_path_conversion(self, src, targ):
        # here we have to try the 4 or so types of path conversions

        src_base, src_inv, src_scale, src_pow = src
        targ_base, targ_inv, targ_scale, targ_pow = targ
        # At this point we actually add the inverse flags back on to start
        src = src_base
        if src_inv:
            src = "Inverse" + src
        targ = targ_base
        if targ_inv:
            targ = "Inverse" + targ

        # now we have to exhaust all of our direct conversion options:
        #   not inverted / no power
        #   inverted / no power
        #   not inverted / power
        #   inverted / power

        hooray_it_worked = False
        conv = None
        if not hooray_it_worked: # not inverted / no power
            conv = self._get_pathy_conversion(src, targ)
            if conv is not None:
                hooray_it_worked = True
                conv = ( (src_scale / targ_scale) * conv) ** ( src_pow / targ_pow )

        if not hooray_it_worked: # inverted / no power
            if src_inv:
                src_2 = src_base
            else:
                src_2 = "Inverse" + src
            if targ_inv:
                targ_2 = targ_base
            else:
                targ_2 = "Inverse" + targ

            conv = self._get_pathy_conversion(src_2, targ_2)
            if conv is not None:
                hooray_it_worked = True
                conv = ( 1 / ((src_scale / targ_scale) * conv) ) ** ( targ_pow / src_pow )

        if (src_pow > 1 or targ_pow > 1):
            if not hooray_it_worked: # not inverted / power
                if src_pow > 1:
                    src_2 = (src, src_pow)
                else:
                    src_2 = src
                if targ_pow > 1:
                    targ_2 = (targ, targ_pow)
                else:
                    targ_2 = targ
                conv = self._get_pathy_conversion(src_2, targ_2)
                if conv is not None:
                    hooray_it_worked = True
                    conv = ( (src_scale / targ_scale) * conv )
            if not hooray_it_worked: # inverted / power
                if src_inv:
                    src_2 = src_base
                else:
                    src_2 = src
                if targ_inv:
                    targ_2 = targ_base
                else:
                    targ_2 = targ
                if src_pow > 1:
                    src_2 = (src_2, src_pow)
                if targ_pow > 1:
                    targ_2 = (targ_2, targ_pow)
                conv = self._get_pathy_conversion(src_2, targ_2)
                if conv is not None:
                    hooray_it_worked = True
                    conv = 1 / ( (src_scale / targ_scale) * conv )

        return conv

    def find_conversion(self, unit, target):
        """Attempts to find a conversion between two sets of units. Currently only implemented for "plain" units.

        :param unit:
        :type unit:
        :param target:
        :type target:
        :return:
        :rtype:
        """
        unit_stuff_1 = self._canonicalize_unit(unit)
        unit_stuff_2 = self._canonicalize_unit(target)
        convo = 1
        for src, targ in zip(unit_stuff_1, unit_stuff_2):
            conv = self._find_direct_conversion(src, targ)
            if conv is None:
                conv = self._find_path_conversion(src, targ)
            if conv is None:
                raise ConversionError("Couldn't find conversion factor between units '{0[0]}' and '{1[0]}'".format(src, targ))
            convo *= conv

        return convo

    def add_conversion(self, unit, target, value):
        self._unit_graph.add(unit, target)
        self._data[(unit, target)] = {"Value":value}

    def convert(self, unit, target):
        """Converts base unit into target using the scraped NIST data

        :param unit:
        :type unit:
        :param target:
        :type target:
        :return:
        :rtype:
        """
        try:
            conv = self.data[(unit, target)]
        except KeyError:
            try:
                conv = self.data[(target, unit)]
            except KeyError:
                conv = self.find_conversion(unit, target)
                self.add_conversion(unit, target, conv)
            else:
                conv = 1/conv["Value"]
        else:
            conv = conv["Value"]

        return conv


    #endregion


UnitsData = UnitsDataHandler()
UnitsData.__doc__ = """An instance of UnitsDataHandler that can be used for unit conversion and fundamental constant lookups"""
UnitsData.__name__ = "UnitsData"