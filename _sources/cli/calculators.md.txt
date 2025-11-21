# Interatomic potentials

## `matscipy-average-eam-potential`

This command generates an averaged EAM potential for an alloy. The basic theory behind averaged alloy EAM potentials is
described in
[Varvenne, Luque, Nöhring, Curtin, Phys. Rev. B 93, 104201 (2016)](https://doi.org/10.1103/PhysRevB.93.104201). The
basic usage of the command is

```bash
 matscipy-average-eam-potential INPUT_TABLE OUTPUT_TABLE [CONCENTRATIONS]
```

The command reads an EAM potential from `INPUT_TABLE`, and creates the average-atom potential for the random alloy with
composition specified by `CONCENTRATIONS`. A new table with both the original and the A-atom potential functions is 
written to the file specified by `OUTPUT_TABLE`. `CONCENTRATIONS` is a whitespace-separated list of the concentration of
the elements, in the order in which the appear in the input table.
