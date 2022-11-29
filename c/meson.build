# Find numpy include directory
include_numpy = run_command(python,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

# Build and install the extension module
module = python.extension_module(
    '_SurfaceTopography', # Name of the module
    [  # List of sources
        'angle_distribution.c',
        'islands.cpp',
        'neighbours.c',
        'ring_statistics.cpp',
        'tools.c',
        'matscipymodule.c'
    ],
    install: true,  # Install it
    include_directories: include_directories(include_numpy),
    dependencies: [ # List of dependencies
        python.dependency(), # Add Python.h as dependency
    ]
)
