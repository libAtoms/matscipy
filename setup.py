import os
import glob
from numpy.distutils.core import setup, Extension

version = (os.popen('git config --get remote.origin.url').read() + ',' +
           os.popen('git describe --always --tags --dirty').read())

scripts = []

setup(name='matscipy',
      version=version,
      description='Generic Python Materials Science tools',
      maintainer='James Kermode & Lars Pastewka',
      maintainer_email='james.kermode@gmail.com',
      license='LGPLv2.1+',
      package_dir={'matscipy': 'matscipy'},
      packages=['matscipy', 'matscipy.fracture_mechanics', 'matscipy.contact_mechanics'],
      scripts=scripts,
      ext_modules=[
        Extension(
            '_matscipy',
            [ 'c/tools.c',
              'c/neighbours.c',
              'c/matscipymodule.c' ],
            )
        ]
      )
