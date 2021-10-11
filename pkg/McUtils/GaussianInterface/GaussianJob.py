"""
Defines a symbolic representation of a Gaussian job for batch generation
"""

import os, numpy as np, inspect
from collections import OrderedDict, namedtuple
from ..Data import AtomData

__all__ = [
    "GaussianJob",
    "GaussianJobArray"
]

####################################################################################################################
#
#                                               GaussianJobException
#
class GaussianJobException(Exception):
    pass

####################################################################################################################
#
#                                               GJFOptFormatter
#
class GJFOptFormatter:
    @classmethod
    def format_base_opt(self, opt):
        return (str(opt), True)
    @classmethod
    def format_compound_opt(self, opt):
        return ("({})".format(", ".join(*(self.format_opt(o)[0] for o in opt))), True)
    @classmethod
    def format_bool_opt(self, opt):
        return (opt, False) if opt else ("", "ignore")
    @classmethod
    def format_none_opt(self, opt):
        return ("", "ignore")
    @classmethod
    def format_dict_opt(self, opt):
        chunks=[]
        for k, i in opt.items():
            opt_val, opt_tag = self.format_opt(i)
            if opt_tag is True:
                chunks.append("{}={}".format(k, opt_val))
            elif not isinstance(opt_tag, str) or opt_tag != "ignore":
                chunks.append(opt_val)
        return ("({})".format(", ".join(*chunks)), True)
    @classmethod
    def format_opt(self, opt):
        router = {
            str: self.format_base_opt,
            int: self.format_base_opt,
            tuple: self.format_compound_opt,
            list: self.format_compound_opt,
            bool: self.format_bool_opt,
            dict: self.format_dict_opt,
            OrderedDict: self.format_dict_opt,
            type(None): self.format_none_opt
        }
        meth = router[type(opt)]
        return meth(opt)
    @classmethod
    def format(self, opt_dict, tag):
        chunks=[]
        for k, i in opt_dict.items():
            opt_val, opt_tag = self.format_opt(i)
            if opt_tag is True:
                chunks.append(tag+"{}={}".format(k, opt_val))
            elif not isinstance(opt_tag, str) or opt_tag != "ignore":
                chunks.append(tag+k)
        return "\n".join(chunks)

####################################################################################################################
#
#                                               GaussianJob
#

class GaussianJob:
    """A class that writes Gaussian .gjf files given a system and config/template options"""

    job_template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Templates")

    def __init__(
            self,
            name,
            *args,
            description = None,
            system = None,
            job = None,
            config = None,
            template = "TemplateTerse.gjf",#"Template.gjf",
            footer=None,
            file = None
            ):

        if not isinstance(name, str):
            raise GaussianJobException("{}: name argument must be string instead of {}".format(
                type(self).__name__,
                name
            ))
        self.name = name

        def throw_bad_field(self, field, must, val):
            raise GaussianJobException("{}: '{}' field must be of type {} (got {})".format(
                type(self).__name__, field, must, val
            ))
        if not isinstance(description, str):
            throw_bad_field(self, 'description', str, description)
        self.desc = description
        if not isinstance(system, self.System):
            throw_bad_field(self, 'system', self.System, system)
        self.system = system
        if not isinstance(job, self.Job):
            throw_bad_field(self, 'job', self.Job, job)
        self.job = job
        if not isinstance(config, self.Config):
            throw_bad_field(self, 'config', self.Config, config)
        self.config = config

        if file is None:
            import tempfile as tf
            f = tf.NamedTemporaryFile()
            dd = os.path.dirname(f.name)
            f.close()
            file = os.path.join(dd, name.replace(" ", "_") + ".gjf")
        if not "Chk" in config:
            config["Chk"] = os.path.splitext(file)[0] + ".chk"
        self.file = file
        self.temp = template

        if footer is None:
            footer = ""
        self.footer = footer

    def format(self):
        """
        Formats the job string

        :return:
        :rtype:
        """
        temp = self.temp
        if not ('{header}' in temp and '{job}' in temp):
            if not os.path.isfile(temp):
                temp = os.path.join(self.job_template_dir, temp)
            with open(temp) as t:
                temp = t.read()
        return temp.format(
            header = self.config.format(),
            description = self.desc,
            job = self.job.format(),
            system = self.system.format(job_type = self.job.job_type),
            footer = inspect.cleandoc(self.footer)
        )

    def write(self, file=None):
        """
        Writes the job to a file

        :param file:
        :type file:
        :return:
        :rtype:
        """
        import tempfile as tf

        if file is None:
            file = self.file
        if file is None:
            tmp = tf.NamedTemporaryFile()
            file = tmp.name + '.gjf'
            tmp.close()

        with open(file, 'w+') as gjf:
            gjf.write(self.format())

        return file

    def start(self, *cmd, binary = 'g09', **kwargs):
        """Starts a Gaussian job

        :param cmd:
        :type cmd:
        :param binary:
        :type binary:
        :param kwargs:
        :type kwargs:
        :return: started process
        :rtype:
        """
        import subprocess

        args = []
        args.extend(cmd)
        if len(args) == 0:
            args.append(binary)

        for k,i in kwargs.items():
            if i is True:
                args.append("-"+k)
            elif i is not None:
                args.append("-{}={:!r}".format(k, i))

        return subprocess.Popen(args)

    def run(self, *args, **kwargs):
        job = self.start(*args, **kwargs)
        job.wait()

    def __str__(self):
        return "{}({},\n\tfile = {} )".format(
            type(self).__name__,
            self.format(),
            self.file if not isinstance(self.file, str) else "'{}'".format(self.file)
        )

    class Job(OrderedDict):
        """Inner class for handling the main job options"""
        job_types = dict(
            {
                "SinglePoint": "SP",
                "PartialOptimization" : "POpt",
                "Optimization": "Opt",
                "Frequency": "Freq",

                "Optimization+Frequency": "POpt Freq",

                "ReactionPath": "IRC",
                "TransitionState": "IRCMax",
                "WavefunctionStability": "Stable",
                "BornOppenheimerDynamics": "BOMD",
                "DensityMatrixDynamics": "AOMD"
            },

            **{k:k for k in ["SP", "POpt", "Opt", "Freq", "POpt Freq", "IRC", "IRCMax", "Scan",
                             "Polar", "ADMP", "BOMD", "Force", "Stable", "Volume"]}

        )
        level_of_theory_keys = ['mp2', 'ccsd', 'b2plypd3', 'b3lyp'] # some level of theory stuff...lots of DFT functionals
        basis_set_keys = ["cc", "aug", "sto"] # common level-of-theory/basis set specs...


        def __init__(self, job_type=None, basis_set=None, **kw):
            for k in kw:
                if job_type is None and k in self.job_types:
                    job_type = k
                if basis_set is None:
                    for b in self.basis_set_keys:
                        if k.startswith(b):
                            basis_set = k
                            break
            if job_type is None:
                job_type = "SinglePoint"
            if basis_set is None:
                basis_set = "MP2/aug-cc-pvdz"
            self.job_type = self.job_types[job_type]

            if self.job_type != job_type and job_type in kw:  # this is what I get for allowing things like ReactionPath -_-
                jtv = kw[job_type]
                del kw[job_type]
                kw[self.job_type] = jtv
            elif self.job_type not in kw:
                kw[self.job_type] = True

            self.basis_set = basis_set
            if basis_set not in kw:
                kw[basis_set] = True

            super().__init__(kw)

        def format(self):
            """Returns a formatted version of the job setup

            :return:
            :rtype:
            """
            return GJFOptFormatter.format(self, "#")

        def __str__(self):
            return self.format()

    class Config(OrderedDict):
        """Inner class for handling the main config options"""
        def __init__(self,
                     *args,
                     Mem=None,
                     NProc=None,
                     Chk=None,
                     **kwargs
                     ):
            """
            Transparent wrapper to super().__init__ that
            just provides the ability to get autocompletion
            """
            if len(args) > 0:
                raise GaussianJobException("{} doesn't support positional args".format(
                    type(self).__name__
                ))
            for k,v in (
                    ('Mem', Mem),
                    ('NProc', NProc),
                    ('Chk', Chk),
            ):
                if v is not None:
                    kwargs[k] = v
            super().__init__(kwargs)

        def format(self):
            """Returns a formatted version of the job setup

            :return:
            :rtype:
            """
            return GJFOptFormatter.format(self, "%")

        def __str__(self):
            return self.format()

    class System:
        def __init__(self,
                     charge = 0,
                     molecule = None,
                     vars = None,
                     bonds = None
                     ):
            self.charge = charge
            self.mol = molecule
            self.bonds = bonds
            self.vars = vars

        def format(self, job_type="SP"):
            return {
                "molecule":self.format_molecule(job_type),
                "charge": self.format_charges(),
                "bonds": self.format_bonds()
            }
        def format_molecule(self, job_type):
            blocks, vars = self.prep_mol(self.mol)

            if job_type == "Scan":
                blocks.extend(self.format_scan_vars(vars))
            elif job_type in {"Opt", "POpt"}:
                blocks.extend(self.format_opt_vars(vars))

            elif job_type == "POpt Freq":
                blocks.extend(self.format_opt_vars(vars))

            else:
                blocks.extend(self.format_const_vars(vars))
            return "\n".join(blocks)

        charge_templates = {
            'integer': '{:<3.0f}',
            'charge_template': '{integer} {integer}'
        }
        def format_charges(self, templates=None):
            """
            Formats charge block

            :return:
            :rtype:
            """

            try:
                c, mult = self.charge
            except TypeError:
                c = self.charge
                mult = 1

            if templates is None:
                templates = {}
            templates = dict(self.charge_templates, **templates)
            charge_template = templates['charge_template'].format(**templates)

            return charge_template.format(c, mult)

        bond_templates = {
            'integer':'{:<5.0f}',
            'bond_template': ' {integer} {integer} {integer}'
        }
        def format_bonds(self, templates=None):
            """
            Formats the bonds held in the job spec
            so that they are appropriate for the bottom of the Gaussian
            job spec

            :param templates:
            :type templates:
            :return:
            :rtype:
            """
            b = self.bonds
            if b is None:
                return ""

            if templates is None:
                templates = {}
            templates = dict(self.bond_templates, **templates)
            bond_template = templates['bond_template'].format(**templates)

            bond_block = [ '' ]*len(b)
            for i,bond in enumerate(b):
                bond = bond if len(bond) > 2 else tuple(bond) + (1,)
                bond_block[i] = bond_template.format(*bond)
            return "\n".join(bond_block)

        opt_variable_templates = {
            'name':'{:>8}',
            'real':'{:>12.8f}',
            'integer':'{:<5.0f}',
            'value_template':'  {name} = {real}',
            'scan_template':'  {name} = {real} s {integer} {real}',
            'constant_template':'  {name} = {real} f'
        }
        @classmethod
        def format_opt_vars(cls, vars, templates=None):
            """
            Formats variable definitions for Gaussian optimization jobs

            :param vars:
            :type vars:
            :return:
            :rtype:
            """
            if templates is None:
                templates = {}
            templates = dict(cls.opt_variable_templates, **templates)
            val_temp = templates['value_template'].format(**templates)
            scan_temp = templates['scan_template'].format(**templates)
            const_temp = templates['constant_template'].format(**templates)

            variables = vars["vars"]
            if len(variables) > 0:
                variables_blocks = [" Variables:"]
            else:
                variables_blocks = []
            for k,c in variables.items():
                if c[1] is None:
                    variables_blocks.append(val_temp.format(k, c[0]))
                else:
                    variables_blocks.append(scan_temp.format(k, *c))

            consts = vars["consts"]
            for k,c in consts.items():
                variables_blocks.append(const_temp.format(k, c))

            return variables_blocks

        scan_variable_templates = {
            'name': '{:>8}',
            'real': '{:>12.8f}',
            'integer': '{:<5.0f}',
            'scan_template': '  {name} = {real} {integer} {real}',
            'constant_template': '  {name} = {real} {integer} {real}'
        }
        @classmethod
        def format_scan_vars(cls, vars, templates=None):
            """
            Formats variable definitions for Gaussian scan jobs

            :param vars:
            :type vars:
            :return:
            :rtype:
            """
            if templates is None:
                templates = {}
            templates = dict(cls.scan_variable_templates, **templates)
            scan_temp = templates['scan_template'].format(**templates)
            const_temp = templates['constant_template'].format(**templates)

            variables = vars["vars"]
            if len(variables) > 0:
                variables_blocks = [" Variables:"]
            else:
                variables_blocks = []
            for k,c in variables.items():
                variables_blocks.append(scan_temp.format(k, *c))

            consts = vars["consts"]
            if len(consts) > 0:
                constants_blocks = []#[" Constants:"]
            else:
                constants_blocks = []
            for k,c in consts.items():
                constants_blocks.append(const_temp.format(k, c, 0, 0))

            return variables_blocks + constants_blocks

        sp_variable_templates = {
            'name': '{:>8}',
            'real': '{:>12.8f}',
            'constant_template': '  {name} = {real}'
        }
        @classmethod
        def format_const_vars(cls, vars, templates=None):
            """
            Formats constant block for Gaussian single point jobs

            :param vars:
            :type vars:
            :return:
            :rtype:
            """

            if templates is None:
                templates = {}
            templates = dict(cls.sp_variable_templates, **templates)
            const_temp = templates['constant_template'].format(**templates)

            variables = vars["vars"]
            consts = vars["consts"]
            if len(consts) + len(variables) > 0:
                constants_blocks = [" Variables:"]
            else:
                constants_blocks = []
            for k,c in variables.items():
                constants_blocks.append(const_temp.format(k, *c))
            for k,c in consts.items():
                constants_blocks.append(const_temp.format(k, c))
            return constants_blocks

        @staticmethod
        def get_coord_type(crds):
            """
            Tries to infer coordinates type

            :param crds:
            :type crds:
            :return:
            :rtype:
            """
            crd_type = None
            try:
                csys = crds.system # try to pull the coordinate system directly off the coordinates
            except AttributeError:
                try:
                    spec, zms = crds # if we can't find it try to get the spec and coordinates for a zmat
                    # if not isinstance(spec[0], str):
                    #     raise ValueError("Oops")
                except ValueError:
                    dim0 = len(crds[0]) # if that doesn't work, then we just have a block of coordinates to work with
                    dimn1 = len(crds[-1]) # in that case we simply check lengths to see what we have
                    if dim0 == 1 or dimn1 == 6:
                        crd_type = "zmat"
                    elif dim0 == 3 and dimn1 == 3:
                        crd_type = "cart"
                else:
                    crd_type = "zmatspec"
            else:
                try:
                    from ..Coordinerds import ZMatrixCoordinateSystem, CartesianCoordinateSystem
                    if isinstance(csys, ZMatrixCoordinateSystem):
                        crd_type = "zmat"
                    elif isinstance(csys, CartesianCoordinateSystem):
                        crd_type = "cart"
                except ImportError:
                    name = crd_type.__name__
                    if "ZMatrix" in name:
                        crd_type = "zmat"
                    elif "Cartesian" in name:
                        crd_type = "cart"

            return crd_type

        def prep_vars(self, *vars):
            """
            Prepares variable specifications so that they
            can be nicely fed into the various formatters

            :param vars:
            :type vars:
            :return:
            :rtype:
            """
            var_map = { # to map variables to coordinates
                "vars"  : OrderedDict(),
                "consts": OrderedDict()
            }
            for var_spec in vars:
                if var_spec is not None:
                    for var in var_spec:
                        if isinstance(var, self.Variable) or len(var) == 4:
                            if not isinstance(var, self.Variable):
                                var = self.Variable(*var)
                            start = var.start
                            num   = var.num
                            step  = var.step
                            if isinstance(num, float):
                                num = np.ceil((num - start)/step)
                            var_map["vars"][var.name] = (start, num, step)
                            try:
                                del var_map["consts"][var.name]
                            except KeyError:
                                pass
                        elif isinstance(var, self.Constant):
                            var_map["consts"][var.name] = var.value
                            try:
                                del var_map["vars"][var.name]
                            except KeyError:
                                pass
                        else:
                            if var[0] not in var_map["consts"]:
                                var_map["consts"][var[0]] = var[1]
                                try:
                                    del var_map['vars'][var[0]]
                                except KeyError:
                                    pass
            return var_map

        def prep_mol(self, mol):
            """
            Prepares the atom for formatting, in particular
            making sure isotopic stuff is clean

            :param mol:
            :type mol:
            :return:
            :rtype:
            """
            try:
                ats = mol["atoms"],
                crds = mol["coords"]
            except TypeError:
                try:
                    ats, crds = mol
                except ValueError:
                    ats = mol.atoms
                    crds = mol.crds

            atoms = ['']*len(ats)
            for i, a in enumerate(ats):
                main = AtomData[a, "PrimaryIsotope"]
                if main:
                    atoms[i] = AtomData[a]["Symbol"]
                else:
                    atoms[i] = "{}(Iso={})".format(AtomData[a]["ElementSymbol"], AtomData[a]["MassNumber"])

            # get molspec blocks based on type of coordinates that were fed in
            crd_type = self.get_coord_type(crds)

            var_list = []
            blocks = [None]*len(ats)
            if crd_type == "zmat":
                if len(crds[0]) > 0:
                    crds = [ [] ] + list(crds)
                for zz, a in zip(enumerate(crds), atoms):
                    subblock = [""]*7 # need a new list per line...
                    i,l = zz
                    subblock[0] = a
                    for j,el in enumerate(l):
                        if (j % 2) == 0:
                            subblock[j+1] = int(el)
                        elif isinstance(el, str):
                            # strings get treated as literals
                            subblock[j+1] = el
                        else:
                            var_type = "r" if j == 1 else "a" if j == 3 else "d" if j == 5 else "??"
                            var = var_type + str(i+1)
                            subblock[j+1] = var
                            var_list.append(self.Constant(var, el))

                    blocks[i] = "{:<20} {:<5} {:<12} {:<5} {:<12} {:<5} {:<12}".format(*subblock)

            elif crd_type == "zmatspec":
                # basically like zmat coords but we loop through two things at once
                spec, crds = crds
                if len(crds[0]) > 0:
                    crds = [ [] ] + list(crds)
                    spec = [ [] ] + list(spec)

                for i, csa in enumerate(zip(crds, spec, atoms)):
                    l, s, a = csa
                    subblock = [""]*7 # need a new list per line...
                    subblock[0] = a
                    for j,e in enumerate(s):
                        if j == i:
                            break
                        subblock[1+2*j] = int(e)
                    for j,el in enumerate(l):
                        if j == i:
                            break
                        if isinstance(el, str):
                            # strings get treated as literals
                            subblock[2+2*j] = el
                        else:
                            var_type = "dist" if j == 0 else "angle" if j == 1 else "dihed" if j == 2 else "??"
                            var = var_type + str(i)
                            subblock[2+2*j] = var
                            var_list.append(self.Constant(var, el))
                    blocks[i] = "{:<20} {:<5} {:<12} {:<5} {:<12} {:<5} {:<12}".format(*subblock)
            elif crd_type == "cart":
                for i, ca in enumerate(zip(crds, atoms)):
                    c, a = ca
                    subblock = [None]*4
                    subblock[0] = a
                    for j,el in enumerate(c):
                        if isinstance(el, str):
                            # strings get treated as literals
                            subblock[1+j] = el
                        else:
                            var_type = "x" if j == 0 else "y" if j == 1 else "z" if j == 2 else "??"
                            var = var_type + str(i+1)
                            subblock[1+j] = var
                            var_list.append(self.Constant(var, el))
                    blocks[i] = "{:<20} 0 {:<12} {:<12} {:<12}".format(*subblock)

            var_map = self.prep_vars(var_list, self.vars)
            return blocks, var_map

        Variable = namedtuple("Variable", ["name", "start", "num", "step"])
        Variable.__new__.__defaults__ = (None,) * 4 # start and name really shouldn't take defaults but w/e
        Constant = namedtuple("Constant", ["name", "value"])

class GaussianJobArray:
    """
    Represents a linked set of Gaussian jobs
    """
    def __init__(self, jobs, link="--Link1--"):
        """
        :param jobs:
        :type jobs: Iterable[GaussianJob]
        :param link: link command (defaults to just `--Link1--`)
        :type link: str
        """
        self.jobs = tuple(jobs)
        self.link_cmd = link

    def format(self):
        """
        Formats a linked Gaussian job

        :return:
        :rtype:
        """
        linker="\n{}\n".format(self.link_cmd)
        return linker.join(
            j.format() for j in self.jobs
        )

    def write(self, file):
        """
        Writes a linked Gaussian job to file

        :param file:
        :type file:
        :return:
        :rtype:
        """

        with open(file, 'w+') as gjf:
            gjf.write(self.format())

        return file