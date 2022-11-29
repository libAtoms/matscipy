# https://mesonbuild.com/Python-module.html

# Pure Python sources
python_sources = [
    '__init__.py',
    '_version.py',
    'angle_distribution.py',
    'atomic_strain.py',
    'deformation.py',
    'dislocation.py',
    'distributed_computation.py',
    'drift.py',
    'elasticity.py',
    'hessian_finite_differences.py',
    'hydrogenate.py',
    'logger.py',
    'neighbours.py',
    'opls.py',
    'pressurecoupling.py',
    'rings.py',
    'socketcalc.py',
    'spatial_correlation_function.py',
    'surface.py',
    'visualise.py'
]

# Install pure Python
python.install_sources(
    python_sources,
    subdir: 'matscipy'
)

subdir('calculators')
subdir('contact_mechanics')
subdir('electrochemistry')
subdir('fracture_mechanics')
subdir('io')
subdir('tool')

