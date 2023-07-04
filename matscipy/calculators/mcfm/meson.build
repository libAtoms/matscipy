# https://mesonbuild.com/Python-module.html

# Pure Python sources
python_sources = [
    '__init__.py',
    'calculator.py',
    'cluster_data.py',
    'qm_cluster.py'
]

# Install pure Python
python.install_sources(
    python_sources,
    subdir: 'matscipy/calculators/mcfm'
)

subdir('mcfm_parallel')
subdir('neighbour_list_mcfm')
subdir('qm_cluster_tools')