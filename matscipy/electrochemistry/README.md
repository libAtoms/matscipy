# Introduction

Samples discrete coordinate sets from arbitrary continuous distributions

# Background

In order to investigate the electrochemical double layer at interfaces, these
tool samples discrete coordinate sets from classical continuum solutions to
Poisson-Nernst-Planck systems. This reduces to the Poisson-Boltzmann
distribution as an analyitc solution for the special case of a binary
electrolyte half space. Coordinate sets are stored as .xyz or LAMMPS data files.

![pic](poisson-bolzmann-sketch.png)

# Content
* `continuous2discrete.py`: sampling and plotting
* `poisson_boltzmann_distribution.py`: generate potential and density
  distributions by solving full Poisson-Nernst-Planck systems.

# FEniCS

matscipy.electrochemistry relies on the FEniCS framework (https://fenicsproject.org/).
On Ubuntu, follow https://fenics.readthedocs.io/en/latest/installation.html#ubuntu-ppa
to install pre-compiled binaries, i.e. on Ubuntu 20.04

    sudo apt install --no-install-recommends software-properties-common
    sudo add-apt-repository ppa:fenics-packages/fenics
    sudo apt update
    sudo apt install fenics=1:2019.1.0.3

If working in a virtual environment, i.e. 

    mkdir -p $HOME/venv
    python3 -m venv $HOME/venv/matscipy-env
    source $HOME/venv/matscipy-env/bin/activate
    pip install --upgrade pip

additional steps are necessary. First, pay attention to install python packages of 
a version compatible with the system's FEniCS binaries, exemplary on Ubuntu 20.04

```console
$ apt show fenics
Package: fenics
Version: 1:2019.1.0.3
...

$ pip install fenics==
ERROR: Could not find a version that satisfies the requirement fenics== (from versions: 2017.1.0, 2017.1.0.post0, 2017.2.0, 2018.1.0, 2019.1.0)    

$ pip install fenics==2019.1.0
```

Before continuing, make sure to have `pybind11` available, i.e. via

    pip install pybind11

or
    
    PYBIND11_VERSION=2.2.4
    tar -xf v${PYBIND11_VERSION}.tar.gz && cd pybind11-${PYBIND11_VERSION}
    wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz    
    mkdir build && cd build && cmake -DPYBIND11_TEST=off -DCMAKE_INSTALL_PREFIX=${VIRTUAL_ENV} -DPYTHON_EXECUTABLE=$(which python) .. && make install

In some cases, pybind selects the wrong Python interpreter.
This issue can be resolved with the `DPYTHON_EXECUTABLE` option.

Following steps for building python bindings are found on 
https://fenics.readthedocs.io/en/latest/installation.html#stable-version. 
Adapted for a virtual environment, these may look like

```bash
FENICS_VERSION=$(python -c"import ffc; print(ffc.__version__)")
MSHR_VERSION=$(echo $FENICS_VERSION | sed 's/.post[0-9]$//')
git clone --branch=${FENICS_VERSION} https://bitbucket.org/fenics-project/dolfin
git clone --branch=${MSHR_VERSION} https://bitbucket.org/fenics-project/mshr
mkdir dolfin/build && cd dolfin/build && cmake -DCMAKE_INSTALL_PREFIX=${VIRTUAL_ENV} .. && make install && cd ../..

mkdir mshr/build   && cd mshr/build   && cmake -DCMAKE_INSTALL_PREFIX=${VIRTUAL_ENV} -DPYTHON_EXECUTABLE=$(which python) .. && make install && cd ../..

cd dolfin/python && pip3 install . && cd ../..
cd mshr/python && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV -DPYTHON_EXECUTABLE=$(which python) .. && cd .. && pip3 install . && cd ../..
```

executed from within an activated virtual environment (with `$VIRTUAL_ENV`) variable available) under 
some clean directory (`dolfin` and `mshr` are downloaded into the current directory).

If built successfully, `dolfin` will suggest to source a file `dolfin.conf` 
to set some environment variables, i.e. via

    source ${VIRTUAL_ENV}/share/dolfin/dolfin.conf

After doing so, FEniCS should be available within the virtual environment.

# Usage
See `../cli/electrochemistry/README.md`.
