# https://mesonbuild.com/Python-module.html

# Pure Python sources
python_sources = [
    '__init__.py',
    'calculator.py',
    'fitting.py',
    'supercell_calculator.py'
]

# Install pure Python
python.install_sources(
    python_sources,
    subdir: 'matscipy/calculators'
)

subdir('eam')
subdir('ewald')
subdir('committee')
subdir('manybody')
subdir('mcfm')
subdir('pair_potential')
subdir('polydisperse')
