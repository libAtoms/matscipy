#
# Copyright 2014-2017, 2020-2021 Lars Pastewka (U. Freiburg)
#           2018, 2020-2021 Jan Griesser (U. Freiburg)
#           2020 Jonas Oldenstaedt (U. Freiburg)
#           2014-2015, 2017, 2019-2020 James Kermode (Warwick U.)
#           2019-2020 Johannes Hoermann (U. Freiburg)
#           2018 libAtomsBuildSystem@users.noreply.github.com
#           2018 Jacek Golebiowski (Imperial College London)
#           2016 Adrien Gola (KIT)
#
# matscipy - Materials science with Python at the atomic-scale
# https://github.com/libAtoms/matscipy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import setuptools
import os
import glob
import sys
import re

from numpy.distutils.core import setup, Extension

from numpy.distutils.command.build_ext import build_ext
from numpy.distutils import log
from distutils.dep_util import newer_group
from numpy.distutils.misc_util import filter_sources, has_f_sources, \
     has_cxx_sources, get_ext_source_files, \
     get_numpy_include_dirs, is_sequence, get_build_architecture, \
     msvc_version

import versioneer

# custom compilation options for various compilers
# entry with key None gives default vaalues always used
copt =  {
          None: []
        }

cxxopt = {
          None: ['-std=c++0x']
         }

lopt =  {
        }

version = versioneer.get_version()
_version_short = re.findall('\d+\.\d+\.\d+', version)
if len(_version_short) > 0:
    version_short = _version_short[0]
else:
    version_short = 'master'
download_url = "https://github.com/libAtoms/matscipy/archive/%s.tar.gz" % version_short

scripts = []

# subclass build_ext so that we can override build_extension()
# to supply different flags to the C and C++ compiler
class build_ext_subclass(build_ext):
    def build_extension(self, ext):
        sources = ext.sources
        if sources is None or not is_sequence(sources):
            raise DistutilsSetupError(
                ("in 'ext_modules' option (extension '%s'), " +
                 "'sources' must be present and must be " +
                 "a list of source filenames") % ext.name)
        sources = list(sources)

        if not sources:
            return

        fullname = self.get_ext_fullname(ext.name)
        if self.inplace:
            modpath = fullname.split('.')
            package = '.'.join(modpath[0:-1])
            base = modpath[-1]
            build_py = self.get_finalized_command('build_py')
            package_dir = build_py.get_package_dir(package)
            ext_filename = os.path.join(package_dir,
                                        self.get_ext_filename(base))
        else:
            ext_filename = os.path.join(self.build_lib,
                                        self.get_ext_filename(fullname))
        depends = sources + ext.depends

        if not (self.force or newer_group(depends, ext_filename, 'newer')):
            log.debug("skipping '%s' extension (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' extension", ext.name)

        extra_args = ext.extra_compile_args or []
        cxx_extra_args = ext.extra_compile_args or []
        extra_link_args = ext.extra_link_args or []

        c = os.path.basename(self.compiler.compiler[0])
        cxx = os.path.basename(self.compiler.compiler_cxx[0])
        if None in copt:
            extra_args += copt[None]
        if c in copt:
            extra_args += copt[c]
        if None in cxxopt:
            cxx_extra_args += cxxopt[None]
        if cxx in cxxopt:
            cxx_extra_args += cxxopt[cxx]
        if None in lopt:
            extra_link_args += lopt[None]
        if c in lopt:
            extra_link_args += lopt[c]
        if cxx in lopt:
            extra_link_args += lopt[cxx]

        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))

        c_sources, cxx_sources, f_sources, fmodule_sources = \
                   filter_sources(ext.sources)

        if self.compiler.compiler_type=='msvc':
            if cxx_sources:
                # Needed to compile kiva.agg._agg extension.
                cxx_extra_args.append('/Zm1000')
            # this hack works around the msvc compiler attributes
            # problem, msvc uses its own convention :(
            c_sources += cxx_sources
            cxx_sources = []

        # Set Fortran/C++ compilers for compilation and linking.
        if ext.language=='f90':
            fcompiler = self._f90_compiler
        elif ext.language=='f77':
            fcompiler = self._f77_compiler
        else: # in case ext.language is c++, for instance
            fcompiler = self._f90_compiler or self._f77_compiler
        if fcompiler is not None:
            fcompiler.extra_f77_compile_args = (ext.extra_f77_compile_args or []) if hasattr(ext, 'extra_f77_compile_args') else []
            fcompiler.extra_f90_compile_args = (ext.extra_f90_compile_args or []) if hasattr(ext, 'extra_f90_compile_args') else []
        cxx_compiler = self._cxx_compiler

        # check for the availability of required compilers
        if cxx_sources and cxx_compiler is None:
            raise DistutilsError("extension %r has C++ sources" \
                  "but no C++ compiler found" % (ext.name))
        if (f_sources or fmodule_sources) and fcompiler is None:
            raise DistutilsError("extension %r has Fortran sources " \
                  "but no Fortran compiler found" % (ext.name))
        if ext.language in ['f77', 'f90'] and fcompiler is None:
            self.warn("extension %r has Fortran libraries " \
                  "but no Fortran linker found, using default linker" % (ext.name))
        if ext.language=='c++' and cxx_compiler is None:
            self.warn("extension %r has C++ libraries " \
                  "but no C++ linker found, using default linker" % (ext.name))

        kws = {'depends':ext.depends}
        output_dir = self.build_temp

        include_dirs = ext.include_dirs + get_numpy_include_dirs()

        c_objects = []
        if c_sources:
            log.info("compiling C sources with arguments %r", extra_args)
            c_objects = self.compiler.compile(c_sources,
                                              output_dir=output_dir,
                                              macros=macros,
                                              include_dirs=include_dirs,
                                              debug=self.debug,
                                              extra_postargs=extra_args,
                                              **kws)

        if cxx_sources:
            log.info("compiling C++ sources with arguments %r", cxx_extra_args)
            c_objects += cxx_compiler.compile(cxx_sources,
                                              output_dir=output_dir,
                                              macros=macros,
                                              include_dirs=include_dirs,
                                              debug=self.debug,
                                              extra_postargs=cxx_extra_args,
                                              **kws)

        extra_postargs = []
        f_objects = []
        if fmodule_sources:
            log.info("compiling Fortran 90 module sources")
            module_dirs = ext.module_dirs[:]
            module_build_dir = os.path.join(
                self.build_temp, os.path.dirname(
                    self.get_ext_filename(fullname)))

            self.mkpath(module_build_dir)
            if fcompiler.module_dir_switch is None:
                existing_modules = glob('*.mod')
            extra_postargs += fcompiler.module_options(
                module_dirs, module_build_dir)
            f_objects += fcompiler.compile(fmodule_sources,
                                           output_dir=self.build_temp,
                                           macros=macros,
                                           include_dirs=include_dirs,
                                           debug=self.debug,
                                           extra_postargs=extra_postargs,
                                           depends=ext.depends)

            if fcompiler.module_dir_switch is None:
                for f in glob('*.mod'):
                    if f in existing_modules:
                        continue
                    t = os.path.join(module_build_dir, f)
                    if os.path.abspath(f)==os.path.abspath(t):
                        continue
                    if os.path.isfile(t):
                        os.remove(t)
                    try:
                        self.move_file(f, module_build_dir)
                    except DistutilsFileError:
                        log.warn('failed to move %r to %r' %
                                 (f, module_build_dir))
        if f_sources:
            log.info("compiling Fortran sources")
            f_objects += fcompiler.compile(f_sources,
                                           output_dir=self.build_temp,
                                           macros=macros,
                                           include_dirs=include_dirs,
                                           debug=self.debug,
                                           extra_postargs=extra_postargs,
                                           depends=ext.depends)

        objects = c_objects + f_objects

        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        libraries = self.get_libraries(ext)[:]
        library_dirs = ext.library_dirs[:]

        linker = self.compiler.link_shared_object
        # Always use system linker when using MSVC compiler.
        if self.compiler.compiler_type=='msvc':
            # expand libraries with fcompiler libraries as we are
            # not using fcompiler linker
            self._libs_with_msvc_and_fortran(fcompiler, libraries, library_dirs)

        elif ext.language in ['f77', 'f90'] and fcompiler is not None:
            linker = fcompiler.link_shared_object
        if ext.language=='c++' and cxx_compiler is not None:
            linker = cxx_compiler.link_shared_object

        if sys.version[:3]>='2.3':
            kws = {'target_lang':ext.language}
        else:
            kws = {}

        linker(objects, ext_filename,
               libraries=libraries,
               library_dirs=library_dirs,
               runtime_library_dirs=ext.runtime_library_dirs,
               extra_postargs=extra_link_args,
               export_symbols=self.get_export_symbols(ext),
               debug=self.debug,
               build_temp=self.build_temp, **kws)


cmdclass = versioneer.get_cmdclass()
cmdclass.update({'build_ext': build_ext_subclass})
setup(name='matscipy',
      version=version,
      cmdclass=cmdclass,
      description='Generic Python Materials Science tools',
      maintainer='James Kermode & Lars Pastewka',
      maintainer_email='james.kermode@gmail.com',
      license='LGPLv2.1+',
      package_dir={'matscipy': 'matscipy'},
      packages=['matscipy',
                'matscipy.io',
                'matscipy.tool',
                'matscipy.fracture_mechanics',
                'matscipy.contact_mechanics',
                'matscipy.electrochemistry',
                'matscipy.calculators',
                'matscipy.calculators.eam',
                'matscipy.calculators.pair_potential',
                'matscipy.calculators.polydisperse',
                'matscipy.calculators.mcfm',
                'matscipy.calculators.mcfm.mcfm_parallel',
                'matscipy.calculators.mcfm.neighbour_list_mcfm',
                'matscipy.calculators.mcfm.qm_cluster_tools',
                'matscipy.calculators.manybody',
                'matscipy.calculators.ewald',
                'matscipy.calculators.manybody.explicit_forms',
                'matscipy.cli.electrochemistry',
            ],
      scripts=scripts,
      extras_require={
            'cli': [],
            'fenics': [
                'fenics-dijitso',
                'fenics-dolfin',
                'fenics-ffc',
                'fenics-fiat',
                'fenics-ufl',
                'mshr',
            ]
      },
      entry_points={
            'console_scripts': [
                'c2d = matscipy.cli.electrochemistry.c2d:main [cli]',
                'pnp = matscipy.cli.electrochemistry.pnp:main [cli]',
                'stericify = matscipy.cli.electrochemistry.stericify:main [cli]'
            ],
        },
      ext_modules=[
        Extension(
            '_matscipy',
            ['c/tools.c',
             'c/angle_distribution.c',
             'c/neighbours.c',
             'c/islands.cpp',
             'c/ring_statistics.cpp',
             'c/matscipymodule.c'],
            )
        ],
      download_url=download_url,
      url="https://github.com/libAtoms/matscipy",
      setup_requires=['pytest-runner'],
      test_require=['pytest']
      )
