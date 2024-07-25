# https://mesonbuild.com/Python-module.html

conf_data = configuration_data()
conf_data.set('VERSION', meson.project_version())
version_file = configure_file(
    input: '_version.py.in',
    output: '_version.py',
    configuration: conf_data
)

# Pure Python sources
python_sources = [
    '__init__.py',
    version_file,
    'angle_distribution.py',
    'atomic_strain.py',
    'compat.py',
    'deformation.py',
    'dislocation.py',
    'drift.py',
    'elasticity.py',
    'ffi.py',
    'gamma_surface.py',
    'hydrogenate.py',
    'logger.py',
    'molecules.py',
    'neighbours.py',
    'numerical.py',
    'numpy_tricks.py',
    'opls.py',
    'optimize.py',
    'precon.py',
    'pressurecoupling.py',
    'rings.py',
    'socketcalc.py',
    'spatial_correlation_function.py',
    'surface.py',
    'visualise.py',
    'cauchy_born.py',
    'surface_reconstruction.py',
    'utils.py'
]

# Install pure Python
python.install_sources(
    python_sources,
    subdir: 'matscipy'
)

subdir('calculators')
subdir('cli')
subdir('electrochemistry')
subdir('fracture_mechanics')
subdir('io')
