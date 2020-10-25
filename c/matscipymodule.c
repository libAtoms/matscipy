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
#define NPY_NO_DEPRECATED_API NPY_1_5_API_VERSION
#include <numpy/arrayobject.h>

#include <stdbool.h>
#include <stddef.h>

#include "angle_distribution.h"
#include "islands.h"
#include "neighbours.h"
#include "ring_statistics.h"

#include "matscipymodule.h"

/*
 * Method declaration
 */

static PyMethodDef module_methods[] = {
    { "angle_distribution", (PyCFunction) py_angle_distribution, METH_VARARGS,
      "Compute a bond angle distribution from a neighbour list." },
    { "distances_on_graph", (PyCFunction) py_distances_on_graph, METH_VARARGS,
      "Compute a map of distances on a graph." },
    { "find_sp_rings", (PyCFunction) py_find_sp_rings, METH_VARARGS,
      "Identify shortest-path rings on a graph." },
    { "neighbour_list", (PyCFunction) py_neighbour_list, METH_VARARGS,
      "Compute a neighbour list for an atomic configuration." },
    { "first_neighbours", (PyCFunction) py_first_neighbours, METH_VARARGS,
      "Compute indices of first neighbours in neighbour list array." },
    { "triplet_list", (PyCFunction) py_triplet_list, METH_VARARGS,
      "Compute a triplet list for a first_neighbour list." },
    { "get_jump_indicies", (PyCFunction) py_get_jump_indicies, METH_VARARGS,
      "Get jump indicies of an ordered list. Does not need list's length \
       as an argument - only the ordered list." },
    { "count_islands", (PyCFunction) py_count_islands, METH_VARARGS,
      "N/A" },
    { "count_segments", (PyCFunction) py_count_segments, METH_VARARGS,
      "N/A" },
    { "correlation_function", (PyCFunction) py_correlation_function, METH_VARARGS,
      "N/A" },
    { "distance_map", (PyCFunction) py_distance_map, METH_VARARGS,
      "N/A" },
    { "perimeter_length", (PyCFunction) py_perimeter_length, METH_VARARGS,
      "N/A" },
    { "shortest_distance", (PyCFunction) py_shortest_distance, METH_VARARGS,
      "N/A" },
    { NULL, NULL, 0, NULL }  /* Sentinel */
};

/*
 * Module initialization
 */

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

/*
 * Module declaration
 */

#if PY_MAJOR_VERSION >= 3
    #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
    #define MOD_DEF(ob, name, methods, doc) \
        static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
        ob = PyModule_Create(&moduledef);
#else
    #define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
    #define MOD_DEF(ob, name, methods, doc) \
        ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(_matscipy)
{
    PyObject* m;

    import_array();

    MOD_DEF(m, "_matscipy", module_methods,
            "C support functions for matscipy.");

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
