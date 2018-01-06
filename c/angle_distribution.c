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
#define NPY_NO_DEPRECATED_API NPY_1_5_API_VERSION
#include <numpy/arrayobject.h>

#include "angle_distribution.h"

/*
 * Compute bond angle distribution
 */

PyObject *
py_angle_distribution(PyObject *self, PyObject *args)
{
  PyObject *i_arr, *j_arr, *r_arr;
  int nbins;
  double cutoff = -1.0;

  if (!PyArg_ParseTuple(args, "O!O!O!i|d", &PyArray_Type, &i_arr, &PyArray_Type,
                        &j_arr, &PyArray_Type, &r_arr, &nbins, &cutoff))
    return NULL;

  if (PyArray_NDIM(i_arr) != 1 || PyArray_TYPE(i_arr) != NPY_INT) {
    PyErr_SetString(PyExc_TypeError, "First argument needs to be one-dimensional "
                    "integer array.");
    return NULL;
  }
  if (PyArray_NDIM(j_arr) != 1 || PyArray_TYPE(j_arr) != NPY_INT) {
    PyErr_SetString(PyExc_TypeError, "Second argument needs to be one-dimensional "
                    "integer array.");
    return NULL;
  }
  if (PyArray_NDIM(r_arr) != 2 || PyArray_DIM(r_arr, 1) != 3 ||
      PyArray_TYPE(r_arr) != NPY_DOUBLE) {
    PyErr_SetString(PyExc_TypeError, "Third argument needs to be two-dimensional "
                    "double array.");
    return NULL;
  }

  npy_intp npairs = PyArray_DIM(i_arr, 0);
  if (PyArray_DIM(j_arr, 0) != npairs || PyArray_DIM(r_arr, 0) != npairs) {
    PyErr_SetString(PyExc_RuntimeError, "First three arguments need to be arrays of "
                    "identical length.");
    return NULL;
  }

  npy_intp dim = nbins;
  PyObject *h_arr = PyArray_ZEROS(1, &dim, NPY_INT, 1);
  PyObject *tmp_arr = PyArray_ZEROS(1, &dim, NPY_INT, 1);

  npy_int *i = PyArray_DATA(i_arr);
  npy_int *j = PyArray_DATA(j_arr);
  double *r = PyArray_DATA(r_arr);
  npy_int *h = PyArray_DATA(h_arr);
  npy_int *tmp = PyArray_DATA(tmp_arr);

  npy_int last_i = i[0], i_start = 0;
  memset(tmp, 0, nbins*sizeof(npy_int));
  int nangle = 1, p;
  double cutoff_sq = cutoff*cutoff;
  for (p = 0; p < npairs; p++) {
    /* Avoid double counting */
    if (last_i != i[p]) {
      int bin;
      for (bin = 0; bin < nbins; bin++) {
        h[bin] += tmp[bin];
      }
      memset(tmp, 0, nbins*sizeof(npy_int));
      last_i = i[p];
      i_start = p;
    }

    double n = r[3*p]*r[3*p] + r[3*p+1]*r[3*p+1] + r[3*p+2]*r[3*p+2];

    if (cutoff < 0.0 || n < cutoff_sq) {
      int p2;
      for (p2 = i_start; i[p2] == last_i; p2++) {
        if (p2 != p) {
          double n2 = r[3*p2]*r[3*p2] + r[3*p2+1]*r[3*p2+1] + r[3*p2+2]*r[3*p2+2];
          if (cutoff < 0.0 || n2 < cutoff_sq) {
            double angle = r[3*p]*r[3*p2] + r[3*p+1]*r[3*p2+1] + r[3*p+2]*r[3*p2+2];
            angle = acos(angle/sqrt(n*n2));
            int bin = (int) (nbins*angle/M_PI);
            while (bin < 0)  bin += nbins;
            while (bin >= nbins)  bin -= nbins;
            tmp[bin]++;
            nangle++;
          } /* n2 < cutoff_sq */
        } /* p!= p */
      }
    } /* n < cutoff_sq */
  }
  /* add angles of last element */
  int bin;
  for (bin = 0; bin < nbins; bin++) {
    h[bin] += tmp[bin];
  }
  
  Py_DECREF(tmp_arr);

  return h_arr;
}
