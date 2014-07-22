import os
import glob
from distutils.core import setup

version = (os.popen('git config --get remote.origin.url').read() + ',' +
           os.popen('git describe --always --tags --dirty').read())

scripts = ['scripts/qcount']

setup(name='matscipy',
      version=version,
      description='Generic Python Materials Science tools',
      maintainer='James Kermode & Lars Pastewka',
      maintainer_email='james.kermode@gmail.com',
      license='LGPLv2.1+',
      package_dir={'': 'matscipy'},
      packages=[''],
      scripts=scripts
      )
