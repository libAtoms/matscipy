# https://mesonbuild.com/
project(
    'matscipy', # Project name
    'c', 'cpp', # Project type. We need a C and C++ compiler.
    default_options : ['cpp_std=c++11'],
    version: run_command('python3', 'discover_version.py', check: true).stdout().strip(), # Project version
)

# https://mesonbuild.com/Python-module.html
pymod = import('python')
python = pymod.find_installation('python3',
    required: true,
    pure: false
)

subdir('c')
subdir('matscipy')