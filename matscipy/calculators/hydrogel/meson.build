# https://mesonbuild.com/Python-module.html

# Pure Python sources
python_sources = [
    '__init__.py',
    'calculator.py',
    'embedding.py',
    'embedding_constant_coordination.py',
    'network.py', 
    'weight_functions.py',
]

# Install pure Python
python.install_sources(
    python_sources,
    subdir: 'matscipy/calculators/hydrogel'
)

subdir('reference_solutions')
subdir('snippets')