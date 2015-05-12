/* ======================================================================
   matscipy - Python materials science tools
   https://github.com/pastewka/atomistica

   https://github.com/libAtoms/matscipy

   Copyright (2014) James Kermode, King's College London
                    Lars Pastewka, Karlsruhe Institute of Technology

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 2 of the License, or
   (at your option) any later version.
  
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
  
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
   ====================================================================== */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL MATSCIPY_ARRAY_API
#include <numpy/arrayobject.h>

#include <stdbool.h>
#include <stddef.h>

#include "angle_distribution.h"
#include "neighbours.h"
#include "ring_statistics.h"

#include "matscipymodule.h"

/*
 * Method declaration
 */

static PyMethodDef module_methods[] = {
    { "angle_distribution", (PyCFunction) py_angle_distribution, METH_VARARGS,
      "Compute a bond angle distribution from a neighbour list." },
    { "distance_map", (PyCFunction) py_distance_map, METH_VARARGS,
      "Compute a map of distance on a graph." },
    { "find_sp_rings", (PyCFunction) py_find_sp_rings, METH_VARARGS,
      "Identify shortest-path rings on a graph." },
    { "neighbour_list", (PyCFunction) py_neighbour_list, METH_VARARGS,
      "Compute a neighbour list for an atomic configuration." },
    { NULL, NULL, 0, NULL }  /* Sentinel */
};


/*
 * Module initialization
 */

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC
init_matscipy(void)
{
    PyObject* m;

    import_array();

    m = Py_InitModule3("_matscipy", module_methods,
                       "C support functions for matscipy.");
    if (m == NULL)
        return;
}
