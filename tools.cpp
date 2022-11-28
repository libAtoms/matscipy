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
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "tools.h"

/*
 * Some basic linear algebra
 */

void
cross_product(double *a, double *b, double *c)
{
    c[0] = a[1]*b[2]-a[2]*b[1];
    c[1] = a[2]*b[0]-a[0]*b[2];
    c[2] = a[0]*b[1]-a[1]*b[0];
}

void
mat_mul_vec(double *mat, double *vin, double *vout)
{
    int i, j;
    for (i = 0; i < 3; i++, vout++){
        *vout = 0.0;
        for (j = 0; j < 3; j++, mat++) {
            *vout += (*mat)*vin[j];
        }
    }
}

double
normsq(double *a)
{
    return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

/*
 * Helper functions
 */

void *
resize_array(PyObject *py_arr, npy_intp newsize)
{
    if (!py_arr) {
        PyErr_SetString(PyExc_RuntimeError,
                        "NULL pointer passed to resize_array.");
        return NULL;
    }

    int i, ndim = PyArray_NDIM((PyArrayObject *) py_arr);
    npy_intp *dims = malloc(ndim*sizeof(npy_intp));
    for (i = 0; i < ndim; i++)  dims[i] = PyArray_DIM((PyArrayObject *) py_arr,
                                                      i);
    dims[0] = newsize;

    PyArray_Dims newshape;
    newshape.ptr = dims;
    newshape.len = ndim;
        
    PyObject *retval;
    retval = PyArray_Resize((PyArrayObject *) py_arr, &newshape, 1, NPY_CORDER);
    if (!retval)  return NULL;
    Py_DECREF(retval);

    return PyArray_DATA((PyArrayObject *) py_arr);
}

