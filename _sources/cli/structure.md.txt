# Structure generation

## `matscipy-quench`

This script carries out a liquid quench procedure. The structure is equilibrated at temperature `T1`, followed by an
exponential quench to temperature `T2`. The script uses a Langevin thermostat for equilibration and quench.

`matscipy-quench` takes no command line parameters but expects a `params.py` file to be present in the current
directory. An example `params.py` for liquid quench of an amorphous Carbon structure could look like this:

```Python
from ase.units import fs, kB
from atomistica import Tersoff, TersoffScr

stoichiometry = '4096C'
densities = [3.5, 3.3, 2.3, 2.5, 2.7, 2.9, 3.1, 2.0]  # Densites in g/cm^3

dt1 = 0.2*fs  # Time step for equilibration
dt2 = 0.2*fs  # Time step for quench
tau1 = 5e3*fs # Relaxation time constant for equilibration
tau2 = 0.5e3*fs # Relaxation time constant for quench
T1 = 10000*kB  # Equilibration temperature
T2 = 300*kB  # Quench down to this temperature
teq = 50e3*fs  # Equlibrate initial structure at T1 for this duration
tqu = 20e3*fs  # Total time of quench

quick_calc = Tersoff()
calc = TersoffScr()
```

The initial structure is just random positioning of the atoms. The calculator `quick_calc` is used for an initial
optimization of the structure and can differ from the actual calculator. In the above example, a computationally less
expensive potential is used for this initial equilibration. Equilibration at `T1` and quench to `T2` is then carried
out with `calc`.