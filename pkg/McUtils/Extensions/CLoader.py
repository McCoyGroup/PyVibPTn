import shutil, os, sys, subprocess, importlib, platform
from distutils.core import setup, Extension

__all__ = [
    "CLoader"
]

class CLoader:
    """
    A general loader for C++ extensions to python, based off of the kind of thing that I have had to do multiple times
    """

    def __init__(self,
                 lib_name,
                 lib_dir,
                 load_path=None,
                 src_ext = 'src',
                 description = "An extension module",
                 version = "1.0.0",
                 include_dirs = None,
                 runtime_dirs = None,
                 linked_libs = None,
                 macros = None,
                 extra_link_args=None,
                 extra_compile_args=None,
                 extra_objects=None,
                 source_files = None,
                 build_script = None,
                 requires_make = False,
                 out_dir = None,
                 cleanup_build = True
                 ):
        self.lib_name = lib_name
        self.lib_dir = lib_dir
        self.lib_description = description
        self.lib_version = version

        self.load_path = [self.lib_dir] if load_path is None else load_path
        self.include_dirs = () if include_dirs is None else tuple(include_dirs)
        self.runtime_dirs = runtime_dirs
        self.linked_libs = () if linked_libs is None else tuple(linked_libs)
        self.extra_link_args = () if extra_link_args is None else tuple(extra_link_args)
        self.extra_compile_args = () if extra_compile_args is None else tuple(extra_compile_args)
        self.extra_objects = () if extra_objects is None else tuple(extra_objects)
        self.src_ext = src_ext
        self.macros = () if macros is None else tuple(macros)
        self.source_files = (lib_name+'.cpp',) if source_files is None else source_files
        self.build_script = build_script
        self.requires_make = requires_make

        self.out_dir = out_dir
        self.cleanup_build = cleanup_build

        self._lib = None

    def load(self):
        if self._lib is None:
            ext = self.find_extension()
            if ext is None:
                ext = self.compile_extension()
            if ext is None:
                raise ImportError("Couldn't load/compile extension {} at {}".format(
                    self.lib_name,
                    self.lib_dir
                ))
            try:
                sys.path.insert(0, os.path.dirname(ext))
                module = os.path.splitext(os.path.basename(ext))[0]
                self._lib = importlib.import_module(module, self.lib_name)#+"Lib")
            finally:
                sys.path.pop(0)

        return self._lib

    def find_extension(self):
        """
        Tries to find the extension in the top-level directory

        :return:
        :rtype:
        """

        for l in self.load_path:
            r = self.locate_lib(l)[0]
            if r is not None:
                return r

    def compile_extension(self):
        """
        Compiles and loads a C++ extension

        :return:
        :rtype:
        """

        self.make_required_libs()
        self.build_lib()
        ext = self.cleanup()
        return ext

    @property
    def src_dir(self):
        return os.path.join(self.lib_dir, self.src_ext)
    @property
    def lib_lib_dir(self):
        return os.path.join(self.lib_dir, "libs")

    def get_extension(self):
        """
        Gets the Extension module to be compiled

        :return:
        :rtype:
        """
        lib_lib_dir = os.path.abspath(self.lib_lib_dir)

        lib_dirs = [os.path.abspath(d) for d in self.include_dirs + (lib_lib_dir,)]
        runtime_dirs = lib_dirs if self.runtime_dirs is None else self.runtime_dirs
        libbies = self.linked_libs
        mroos = self.macros
        sources = self.source_files

        extra_link_args = list(self.extra_link_args)
        extra_compile_args = list(self.extra_compile_args)
        if platform.system() == 'Darwin':
            extra_link_args.append('-Wl,-rpath,' + ":".join(lib_dirs))

        module = Extension(
            self.lib_name,
            sources=list(sources),
            library_dirs=list(lib_dirs),
            runtime_library_dirs=list(runtime_dirs),
            include_dirs=[os.path.abspath(d) for d in self.include_dirs],
            libraries=list(libbies),
            define_macros=list(mroos),
            extra_objects=list(self.extra_objects),
            extra_link_args=extra_link_args,
            extra_compile_args=extra_compile_args,
            language="c++"
        )

        return module

    def custom_make(self, make_file, make_dir):
        """
        A way to call a custom make file either for building the helper lib or for building the proper lib

        :param make_file:
        :type make_file:
        :param make_dir:
        :type make_dir:
        :return:
        :rtype:
        """

        curdir = os.getcwd()

        if isinstance(make_file, str) and os.path.isfile(make_file):
            make_dir = os.path.dirname(make_file)
            make_file = os.path.basename(make_file)
            if os.path.splitext(make_file)[1] == ".sh":
                make_cmd = ["bash", make_file]
            else:
                make_cmd = ["make", "-f", make_file]
        elif isinstance(make_file, dict):
            # little way to call the make process without a build.sh file

            python_dir = make_file["python_dir"]
            compiler = make_file["compiler"]
            sources = make_file["sources"] if 'sources' in make_file else self.source_files
            compiler_flags = make_file["compiler_flags"] if 'compiler_flags' in make_file else self.extra_compile_args
            include_dirs = make_file["include_directories"] if 'include_directories' in make_file else self.include_dirs
            runtime_dirs = make_file["runtime_directories"] if 'runtime_directories' in make_file else self.runtime_dirs
            if runtime_dirs is None:
                runtime_dirs = include_dirs

            linker = make_file["linker"]
            link_flags = make_file["link_flags"] if 'link_flags' in make_file else self.extra_link_args
            linked_libs = make_file["linked_libs"] if 'linked_libs' in make_file else self.linked_libs
            macros = make_file["macros"] if 'macros' in make_file else self.macros

            shared_lib = make_file["library_name"] if 'library_name' in make_file else self.lib_name

            if 'build_dir' not in make_file:
                source_dir = os.path.dirname(sources[0])
                build_dir = os.path.join(source_dir, "build")
            else:
                build_dir = make_file['build_dir']
            os.makedirs(build_dir)

            objects = [os.path.join(build_dir, os.path.splitext(s)[0]+".o") for s in sources]

            compiler_flags = ["-f"+flag for flag in compiler_flags]
            include_dirs = ["-I"+i for i in include_dirs]

            if compiler == "CC":
                rpath = "-Wl,-R,"
            else:
                rpath = "-Wl,-rpath,"
            link_flags = ["-Wl,"+f for f in link_flags]
            if len(include_dirs) > 0:
                runtime_dirs = [rpath+":".join(include_dirs)]
            else:
                runtime_dirs = []
            linked_libs = ["-l"+l for l in linked_libs]
            link_dirs = ["-L"+i for i in include_dirs]
            "$link_flags $ryn_lib_src/build/RynLib.o $link_libs -o $ryn_lib_src/RynLib.so"

            shared_lib = shared_lib + (".pydll" if sys.platform() == "win32" else ".so")

            make_cmd = [
               [compiler] + compiler_flags + macros + include_dirs + ["-c", src] + ["-o", obj] for src,obj in zip(sources, objects)
            ] + [
                [linker] + link_flags + runtime_dirs + objects + link_dirs + linked_libs + ["-o", shared_lib]
            ]

        else:

            if os.path.exists(os.path.join(make_dir, "Makefile")):
                make_cmd = ["make"]
            elif os.path.exists(os.path.join(make_dir, "build.sh")):
                make_cmd = ["bash", "build.sh"]
            else:
                raise IOError(
                    "Can't figure out which file in {} should be the makefile. Expected either Makefile or build.sh".format(
                        make_dir
                    )
                )

        try:
            os.chdir(make_dir)
            if isinstance(make_cmd[0], str):
                out = subprocess.check_output(make_cmd)
                if len(out) > 0:
                    print(out.decode())
            else:
                for cmd in make_cmd:
                    out = subprocess.check_output(cmd)
                    if len(out) > 0:
                        print(out.decode())
        except subprocess.CalledProcessError as e:
            print(e.output.decode())
            raise
        finally:
            os.chdir(curdir)

    def make_required_libs(self):
        """
        Makes any libs required by the current one

        :return:
        :rtype:
        """
        if self.requires_make:
            lib_d = os.path.abspath(self.lib_lib_dir)
            for lib in os.listdir(lib_d):
                lib = os.path.join(lib_d, lib)
                if os.path.isdir(lib):
                    self.custom_make(self.requires_make, lib)

    def build_lib(self):

        curdir = os.getcwd()

        src_dir = os.path.abspath(self.src_dir)
        module = self.get_extension()

        sysargv1 = sys.argv
        custom_build = self.build_script
        if custom_build:
            self.custom_make(custom_build, src_dir)
        else:
            try:
                sys.argv = ['build', 'build_ext', '--inplace']
                os.chdir(src_dir)

                setup(
                    name=self.lib_description,
                    version=self.lib_version,
                    description=self.lib_description,
                    ext_modules=[module]
                )
            finally:
                sys.argv = sysargv1
                os.chdir(curdir)

    def locate_lib(self, root = None):
        """
        Tries to locate the build library file (if it exists)

        :return:
        :rtype:
        """

        libname = self.lib_name
        lib_dir = os.path.abspath(self.lib_dir)
        if root is None:
            root = os.path.join(lib_dir, "src")
        target = libname
        built = None
        ext = ""

        for f in os.listdir(root):
            if f.startswith(libname) and f.endswith(".so"):
                ext = ".so"
                built = os.path.join(root, f)
                target += ext
                break
            elif f.startswith(libname) and f.endswith(".pyd"):
                ext = ".pyd"
                built = os.path.join(root, f)
                target += ext
                break

        if built is None:
            target = None

        return built, target, ext

    def cleanup(self):
        # Locate the library and copy it out (if it exists)

        built, target, ext = self.locate_lib()

        if built is not None:
            target_dir = self.out_dir
            if target_dir is None:
                target_dir = self.lib_dir
            target = os.path.join(target_dir, target)
            try:
                os.remove(target)
            except:
                pass
            os.rename(built, target)

            if self.cleanup_build:
                build_dir = os.path.join(self.src_dir, "build")
                if os.path.isdir:
                    shutil.rmtree(build_dir)

        return target