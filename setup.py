import os
import glob
from numpy.distutils.core import setup, Extension

version = (os.popen('git config --get remote.origin.url').read() + ',' +
           os.popen('git describe --always --tags --dirty').read())

scripts = []


extra_compile_args = []
if 'CC' not in os.environ or not os.environ['CC'].endswith('clang'):
    extra_compile_args.append('-std=c++0x')

setup(name='matscipy',
      version=version,
      description='Generic Python Materials Science tools',
      maintainer='James Kermode & Lars Pastewka',
      maintainer_email='james.kermode@gmail.com',
      license='LGPLv2.1+',
      package_dir={'matscipy': 'matscipy'},
      packages=['matscipy', 'matscipy.fracture_mechanics',
                'matscipy.contact_mechanics', 'matscipy.calculators',
                'matscipy.calculators.eam'],
      scripts=scripts,
      ext_modules=[
        Extension(
            '_matscipy',
            ['c/tools.c',
             'c/angle_distribution.c',
             'c/islands.cpp',
             'c/neighbours.c',
             'c/ring_statistics.cpp',
             'c/matscipymodule.c'],
            extra_compile_args=extra_compile_args
            )
        ]
      )
