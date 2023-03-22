# Find numpy include directory
include_numpy = run_command(python,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

link_args = []

# Statically linking gcc components if on windows
if build_machine.system() == 'windows'
    link_args = ['-static-libstdc++', '-static-libgcc', '-static']
endif


# Build and install the extension module
module = python.extension_module(
    '_matscipy', # Name of the module
    [  # List of sources
        'angle_distribution.cpp',
        'islands.cpp',
        'neighbours.c',
        'ring_statistics.cpp',
        'tools.c',
        'matscipymodule.c'
    ],
    install: true,  # Install it
    subdir: 'matscipy',
    include_directories: include_directories(include_numpy),
    dependencies: [ # List of dependencies
        python.dependency(), # Add Python.h as dependency
    ],
    link_args: link_args,
)
