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

#include <math.h>

#include "islands.h"
#include "stack.h"

/* This is sufficient for typically 2048x2048 */
#define DEFAULT_STACK_SIZE 16*1024*1024

#define MIN(a,b) ((a)<(b)?(a):(b))

/*
 * Find continous 2d patches
 */

#define DEFAULT_SX 8
const npy_intp default_sx = DEFAULT_SX;
static npy_long default_stencil[2*DEFAULT_SX] = {
    1,1, 1,0, 1,-1, 0,-1, -1,-1, -1,0, -1,1, 0,1
};



void fill_patch(npy_intp nx, npy_intp ny, npy_bool *map, int i0, int j0,
                npy_int p, npy_int sx, npy_long *stencil, npy_int *id)
{
  Stack stack(DEFAULT_STACK_SIZE);

  stack.push(i0, j0);
  while (!stack.is_empty()) {
    int i, j;

    stack.pop(i, j);

    //id[i+nx*j] = p;
    id[i*ny+j] = p;

    int s;

    for (s = 0; s < 2*sx; s+=2) {
      int di, dj;

      di = stencil[s];
      dj = stencil[s+1];

      /* Periodic boundary conditions */
      int jj = j+dj;
      if (jj < 0)     jj += ny;
      if (jj > ny-1)  jj -= ny;

      /* Periodic boundary conditions */
      int ii = i+di;
      if (ii < 0)     ii += nx;
      if (ii > nx-1)  ii -= nx;

      //int k = ii+nx*jj;
      int k = ii*ny+jj;
      if (map[k] && id[k] == 0) {
        stack.push(ii, jj);
      }
    }
  }
}


extern "C" PyObject *
py_count_islands(PyObject *self, PyObject *args)
{
  PyObject *py_map = NULL, *py_stencil = NULL;

  if (!PyArg_ParseTuple(args, "O|O",
                        &py_map, &py_stencil))
    return NULL;
  if (!py_map)
    return NULL;

  npy_intp sx;
  npy_long *stencil;

  PyObject *py_bool_map = NULL;
  PyObject *py_long_stencil = NULL;

  if (py_stencil) {
    py_long_stencil =  PyArray_FROMANY(py_stencil, NPY_LONG, 2, 2,
                                       NPY_C_CONTIGUOUS);
    if (!py_long_stencil)
      return NULL;

    sx = PyArray_DIM(py_long_stencil, 0);
    npy_intp sy = PyArray_DIM(py_long_stencil, 1);

    stencil = (npy_long*) PyArray_DATA(py_long_stencil);

    if (sy != 2) {
      PyErr_SetString(PyExc_TypeError, "Stencil must have dimension 2 in the "
                      "second axis.");
    }
  }
  else {
    sx = default_sx;
    stencil = default_stencil;
  }

  py_bool_map = PyArray_FROMANY(py_map, NPY_BOOL, 2, 2, NPY_C_CONTIGUOUS);
  if (!py_bool_map)
    return NULL;

  npy_intp nx = PyArray_DIM(py_bool_map, 0);
  npy_intp ny = PyArray_DIM(py_bool_map, 1);

  npy_bool *map = (npy_bool*) PyArray_DATA(py_bool_map);

  npy_intp dims[2] = { nx, ny };

  PyObject *py_id = PyArray_ZEROS(2, dims, NPY_INT, 0);
  if (!py_id)
    return NULL;
  npy_int *id = (npy_int *) PyArray_DATA(py_id);

  int i, j, k = 0;
  npy_int p = 0;

  for (i = 0; i < nx; i++) {
    for (j = 0; j < ny; j++) {

      if (map[k] && id[k] == 0) {
        p++;
        fill_patch(nx, ny, map, i, j, p, sx, stencil, id);
      }

      k++;
    }
  }

  PyObject *r = Py_BuildValue("iO", p, py_id);
  Py_DECREF(py_id);
  Py_DECREF(py_bool_map);
  if (py_long_stencil)
    Py_DECREF(py_long_stencil);
  return r;
}


/*
 * Find continuous 1d segments
 */

void fill_segment(npy_intp nx, npy_bool *map, int i, npy_int p, npy_int *id)
{
  id[i] = p;

  int ii = i+1;
  /* Periodic boundary conditions */
  if (ii > nx-1)  ii -= nx;

  while (map[ii] && id[ii] == 0) {
    id[ii] = p;

    ii++;
    if (ii > nx-1)  ii -= nx;
  }

  ii = i-1;
  if (ii < 0)  ii += nx;

  while (map[ii] && id[ii] == 0) {
    id[ii] = p;

    ii--;
    /* Periodic boundary conditions */
    if (ii < 0)  ii += nx;
  }
}


/*
 * Assign a unique number to each segment on the map
 */

extern "C" PyObject *
py_count_segments(PyObject *self, PyObject *args)
{
  PyObject *py_map;

  if (!PyArg_ParseTuple(args, "O", &py_map))
    return NULL;
  if (!py_map)
    return NULL;

  PyObject *py_bool_map = NULL;
  py_bool_map = PyArray_FROMANY(py_map, NPY_BOOL, 2, 2, NPY_C_CONTIGUOUS);
  if (!py_bool_map)
    return NULL;

  npy_intp nx = PyArray_DIM(py_bool_map, 0);
  npy_intp ny = PyArray_DIM(py_bool_map, 1);

  npy_bool *map = (npy_bool*) PyArray_DATA(py_bool_map);

  npy_intp dims[2] = { nx, ny };

  PyObject *py_id = PyArray_ZEROS(2, dims, NPY_INT, 0);
  if (!py_id)
    return NULL;
  npy_int *id = (npy_int *) PyArray_DATA(py_id);

  int i, j, k = 0;
  npy_int p = 0;

  for (i = 0; i < nx; i++) {
    for (j = 0; j < ny; j++) {
      if (map[k] && id[k] == 0) {
        p++;
        fill_segment(nx, &map[ny*i], j, p, &id[ny*i]);
        //fill_segment(ny, &map[ny*i], i, p, &id[ny*i]);
      }

      k++;
    }
  }

  PyObject *r = Py_BuildValue("iO", p, py_id);
  Py_DECREF(py_id);
  Py_DECREF(py_bool_map);
  return r;
}


extern "C" PyObject *
py_shortest_distance(PyObject *self, PyObject *args)
{
  PyObject *py_fromc = NULL, *py_fromp = NULL, *py_to = NULL;
  int maxd = -1;

  if (!PyArg_ParseTuple(args, "OOO|i", &py_fromc, &py_fromp, &py_to, &maxd))
    return NULL;

  PyObject *py_bool_fromc = NULL, *py_bool_fromp = NULL;
  PyObject *py_bool_to = NULL;

  py_bool_fromc = PyArray_FROMANY(py_fromc, NPY_BOOL, 2, 2, NPY_C_CONTIGUOUS);
  if (!py_bool_fromc)
    return NULL;
  py_bool_fromp = PyArray_FROMANY(py_fromp, NPY_BOOL, 2, 2, NPY_C_CONTIGUOUS);
  if (!py_bool_fromp)
    return NULL;
  py_bool_to = PyArray_FROMANY(py_to, NPY_BOOL, 2, 2, NPY_C_CONTIGUOUS);
  if (!py_bool_to)
    return NULL;

  npy_intp nx = PyArray_DIM(py_bool_fromc, 0);
  npy_intp ny = PyArray_DIM(py_bool_fromc, 1);

  if (PyArray_DIM(py_bool_fromp, 0) != nx || 
      PyArray_DIM(py_bool_fromp, 1) != ny) {
    PyErr_SetString(PyExc_TypeError,
                    "All three maps need to have identical dimensions.");
    return NULL;
  }
  if (PyArray_DIM(py_bool_to, 0) != nx || PyArray_DIM(py_bool_to, 1) != ny) {
    PyErr_SetString(PyExc_TypeError,
                    "All three maps need to have identical dimensions.");
    return NULL;
  }

  npy_bool *fromc = (npy_bool*) PyArray_DATA(py_bool_fromc);
  npy_bool *fromp = (npy_bool*) PyArray_DATA(py_bool_fromp);
  npy_bool *to = (npy_bool*) PyArray_DATA(py_bool_to);

  npy_intp dims[2] = { nx, ny };
  PyObject *py_dist = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
  if (!py_dist)
    return NULL;
  npy_double *dist = (npy_double *) PyArray_DATA(py_dist);

  /*
   * Make sure there is something to find
   */
  int k;
  int found = 0;
  for (k = 0; k < nx*ny; k++ && !found) {
    if (to[k])
      found = 1;
  }
  if (!found) {
    PyErr_SetString(PyExc_RuntimeError, "No patches found in second map.");
    return NULL;
  }

  /*
   * Find distance to patches in *to*
   */
  if (maxd < 0) maxd = 2*nx;
  double sqrt2 = sqrt(2.0);
  int i, j;
  k = 0;
  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {

      if (fromc[k]) {

        double d = maxd+1.0;

        if (to[k]) {
          /*
           * This is also the edge, set distance to 0
           */
          d = 0.0;
        }
        else {
          int ter = maxd+1;
          int n;
          int on_some_patch = 1;
          for (n = 1; n <= ter && on_some_patch; n++) {

            //printf("%i\n", n);
            int jj;
            on_some_patch = 0;
            
            for (jj = -n; jj <= n; jj++) {

              int jjj = j+jj;
              while (jjj < 0)   jjj += ny;
              while (jjj >= ny) jjj -= ny;

              int ii;
              for (ii = -n; ii <= n; ii++) {

                if (abs(ii) == n || abs(jj) == n) {

                  int iii = i+ii;
                  while (iii < 0)   iii += nx;
                  while (iii >= nx) iii -= nx;

                  int m = jjj*nx+iii;
                  
                  if (fromp[m])
                    on_some_patch = 1;

                  if (to[m]) {
                    double curd = sqrt(ii*ii + jj*jj);
                    if (curd < d) {
                      d = curd;
                      /*
                       * this could be at distance sqrt(2)*n, hence we need to 
                       * go to rectangles with side length >sqrt(2)*n
                       */
                      int newter = (int) (n*(sqrt2+1));
                      if (newter < ter)
                        ter = newter;
                    }
                  } // if (to[m])
                  
                } // if (abs(ii) ...

              } // for ii
              
            } // for jj
            
          } // for n
          
        } // if (to[k])

        if (d < maxd)
          dist[k] = d;
      } // if fromc

      k++;
    } // for i
  } // for j

  PyObject *r = Py_BuildValue("O", py_dist);
  Py_DECREF(py_dist);
  Py_DECREF(py_bool_fromc);
  Py_DECREF(py_bool_fromp);
  Py_DECREF(py_bool_to);
  return r;
}


/*!
 * Given *map* with dimensions *nx*,*ny*, compute the minimal distance from
 * each of the points and store into *dist*. *next* with contain the index of
 * the closest point. Distance from a point with itself is 0.
 */
void track_distance(int nx, int ny, npy_bool *map, npy_double *dist,
                    npy_int *next)
{
    Stack stack(DEFAULT_STACK_SIZE);

    /* Fill stack with all possible map points */
    int k = 0;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            if (map[k]) {
                /* Start tracking here with zero distance */
                stack.push(i, j, i, j);
            }
    
            k++;
        }
    }

    /* While there is something to look for */
    while (!stack.is_empty()) {

        int i, j, i0, j0;

        stack.pop_bottom(i, j, i0, j0);

        int di = i-i0;
        int dj = j-j0;

        if (di > nx/2)  di = nx-di;
        if (dj > ny/2)  dj = ny-dj;

        double d = sqrt(di*di+dj*dj);

        int k = i*ny+j;

        /* Is i0, j0 closer than what is currently stored for k? */
        if (d < dist[k]) {
            dist[k] = d;
            next[k] = i0*ny+j0;

            /* Loop over all neighbors */
            int jj;
            for (jj = -1; jj <= 1; jj++) {

                /* Periodic boundary conditions */
                int jjj = j+jj;
                while (jjj < 0)   jjj += ny;
                while (jjj >= ny) jjj -= ny;

                int ii;
                for (ii = -1; ii <= 1; ii++) {

                    /* Exclude middle */
                    if (ii != 0 || jj != 0) {

                        /* Periodic boundary conditions */
                        int iii = i+ii;
                        while (iii < 0)   iii += nx;
                        while (iii >= nx) iii -= nx;
            
                        /* Push to stack if not on map */
                        int kkk = iii*ny+jjj;
                        if (!map[kkk]) {
                            stack.push(iii, jjj, i0, j0);
                        } /* if (!map[kkk]) */
      
                    } /* if (abs(ii) == ... */

                } /* for (ii = ... */

            } /* for (jj = ... */

        } /* if (d < dist[k]) */

    } /* while (!stack.is_empty()) */

}


/*!
 * Given a bool map, compute the minimal distance from each of the points
 * marked on the map. Distance from a point which is marked is 0.
 */
extern "C" PyObject *
py_distance_map(PyObject *self, PyObject *args)
{
  PyObject *py_map_xy = NULL;

  if (!PyArg_ParseTuple(args, "O", &py_map_xy))
    return NULL;

  PyObject *py_bool_map_xy = NULL;

  py_bool_map_xy = PyArray_FROMANY(py_map_xy, NPY_BOOL, 2, 2,
                                   NPY_C_CONTIGUOUS);
  if (!py_bool_map_xy)
    return NULL;

  npy_intp nx = PyArray_DIM(py_bool_map_xy, 0);
  npy_intp ny = PyArray_DIM(py_bool_map_xy, 1);

  npy_bool *map_xy = (npy_bool*) PyArray_DATA(py_bool_map_xy);

  /* This stores the distance to the closest point on the contour */
  npy_intp dims[2] = { nx, ny };
  PyObject *py_dist_xy = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
  if (!py_dist_xy)
    return NULL;
  npy_double *dist_xy = (npy_double *) PyArray_DATA(py_dist_xy);

  /* This stores the index of the closest point */
  PyObject *py_next_xy = PyArray_ZEROS(2, dims, NPY_INT, 0);
  if (!py_next_xy)
    return NULL;
  npy_int *next_xy = (npy_int *) PyArray_DATA(py_next_xy);

  /*
   * Fill map with maximum distance
   */
  int k;
  for (k = 0; k < nx*ny; k++) {
    dist_xy[k] = nx*ny;
    next_xy[k] = nx*ny;
  }

  /*
   * Track distances from contact edge
   */
  track_distance(nx, ny, map_xy, dist_xy, next_xy);

  PyObject *r = Py_BuildValue("O", py_dist_xy);
  Py_DECREF(py_dist_xy);
  Py_DECREF(py_next_xy);
  Py_DECREF(py_bool_map_xy);
  return r;
}


/*
 * Compute real-space correlation function between two maps
 */

void fill_correlation_function(int, int, double, int, int,
                               int, int, npy_double *,
                               npy_double *, npy_int *);

extern "C" PyObject *
py_correlation_function(PyObject *self, PyObject *args)
{
  PyObject *py_map1 = NULL, *py_map2 = NULL;
  int max_dist, max_dist_sq;

  if (!PyArg_ParseTuple(args, "OOi", &py_map1, &py_map2, &max_dist))
    return NULL;
  max_dist_sq = max_dist*max_dist;

  PyObject *py_double_map1 = NULL, *py_double_map2 = NULL;

  py_double_map1 = PyArray_FROMANY(py_map1, NPY_DOUBLE, 2, 2,
                                   NPY_C_CONTIGUOUS);
  if (!py_double_map1)
    return NULL;
  py_double_map2 = PyArray_FROMANY(py_map2, NPY_DOUBLE, 2, 2,
                                   NPY_C_CONTIGUOUS);
  if (!py_double_map2)
    return NULL;

  npy_intp nx = PyArray_DIM(py_double_map1, 0);
  npy_intp ny = PyArray_DIM(py_double_map1, 1);

  if (PyArray_DIM(py_double_map2, 0) != nx ||
      PyArray_DIM(py_double_map2, 1) != ny) {
    PyErr_SetString(PyExc_TypeError,
                    "Both maps need to have the identical dimensions.");
  }

  npy_double *map1 = (npy_double*) PyArray_DATA(py_double_map1);
  npy_double *map2 = (npy_double*) PyArray_DATA(py_double_map2);

  /*
   * Correlation function
   */
  npy_intp dims[1] = { max_dist_sq };
  PyObject *py_c = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
  if (!py_c)
    return NULL;
  npy_double *c = (npy_double*) PyArray_DATA(py_c);

  /*
   * Number of points found at a certain distance
   */
  PyObject *py_n = PyArray_ZEROS(1, dims, NPY_INT, 0);
  if (!py_n)
    return NULL;
  npy_int *n = (npy_int*) PyArray_DATA(py_n);

  /*
   * Fill with zeros
   */
  int k;
  for (k = 0; k < max_dist_sq; k++) {
    c[k] = 0.0;
    n[k] = 0;
  }

  /*
   * Maximum search distance in x and y directions
   */
  int max_lindist = ((int) floor(sqrt(max_dist_sq))) + 1;

  /*
   * Tracking algorithm
   */
  int si, sj;
  k = 0;
  for (sj = 0; sj < ny; sj++) {
    for (si = 0; si < nx; si++) {

      /*
       * Start tracking here with zero distance
       */
      fill_correlation_function(max_lindist, max_dist_sq, map1[k], si, sj, nx,
                                ny, map2, c, n);

      k++;
    }
  }

  /*
   * Release maps
   */
  Py_DECREF(py_double_map1);
  Py_DECREF(py_double_map2);

  /*
   * Find nonzero entries
   */
  int nz = 0;
  for (k = 0; k < max_dist_sq; k++) {
    if (n[k] > 0)
      nz++;
  }

  /*
   * Allocate correlation function of proper length
   */
  dims[0] = nz;
  /* Distance */
  PyObject *py_r = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
  if (!py_r)
    return NULL;
  npy_double *r = (npy_double*) PyArray_DATA(py_r);
  /* Correlation function */
  PyObject *py_cc = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
  if (!py_cc)
    return NULL;
  npy_double *cc = (npy_double*) PyArray_DATA(py_cc);
  /* Integrated correlation function */
  PyObject *py_Icc = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
  if (!py_Icc)
    return NULL;
  npy_double *Icc = (npy_double*) PyArray_DATA(py_Icc);

  /*
   * Normalize and integrate
   */
  for (k = 0; k < nz; k++) {
    Icc[k] = 0.0;
  }
  nz = 0;
  for (k = 0; k < max_dist_sq; k++) {
    if (n[k] > 0) {
      double sqrtk = sqrt(1+k);

      /* Integrate */
      double ival = c[k]/sqrtk;
      int l, inz = nz;
      for (l = k; l < max_dist_sq; l++) {
        if (n[l] > 0) {
          Icc[inz] += ival;
          inz++;
        }
      }

      /* Normalize */
      r[nz] = sqrtk;
      cc[nz] = c[k]/n[k];
      nz++;
    }
  }

  /*
   * Release c and n
   */
  Py_DECREF(py_c);
  Py_DECREF(py_n);

  PyObject *ret = Py_BuildValue("OOO", py_r, py_cc, py_Icc);

  /*
   * Release cc and r
   */
  Py_DECREF(py_r);
  Py_DECREF(py_cc);
  Py_DECREF(py_Icc);

  return ret;
}



void fill_correlation_function(int max_lin_dist, int max_dist_sq,
                               double map1val, int i, int j,
                               int nx, int ny, npy_double *map2,
                               npy_double *c, npy_int *n)
{
  /*
   * Loop over all neighbors
   */

  int jj;
  for (jj = -max_lin_dist; jj <= max_lin_dist; jj++) {

    int jjj = j+jj;
    while (jjj < 0)   jjj += ny;
    while (jjj >= ny) jjj -= ny;

    int ii;
    for (ii = -max_lin_dist; ii <= max_lin_dist; ii++) {

      /* Exclude middle */
      if (ii != 0 || jj != 0) {

        int iii = i+ii;
        while (iii < 0)   iii += nx;
        while (iii >= nx) iii -= nx;

        int kkk = jjj*nx+iii;

        int dist_sq = ii*ii + jj*jj;
        if (dist_sq <= max_dist_sq) {
          c[dist_sq-1] += map1val*map2[kkk];
          n[dist_sq-1] += 1;
        }

      }

    }
  }
}


/*
 * Compute the total length of the perimeter
 */

extern "C" PyObject *
py_perimeter_length(PyObject *self, PyObject *args)
{
  double sqrt2 = sqrt(2.0);

  PyObject *py_map;

  if (!PyArg_ParseTuple(args, "O", &py_map))
    return NULL;
  if (!py_map)
    return NULL;

  PyObject *py_bool_map = NULL;

  py_bool_map = PyArray_FROMANY(py_map, NPY_BOOL, 2, 2, NPY_C_CONTIGUOUS);
  if (!py_bool_map)
    return NULL;

  npy_intp nx = PyArray_DIM(py_bool_map, 0);
  npy_intp ny = PyArray_DIM(py_bool_map, 1);

  npy_bool *map = (npy_bool*) PyArray_DATA(py_bool_map);

  double length = 0.0;

  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      if (map[j*nx+i]) {
        int ii, jj, n1 = 0, nsqrt2 = 0;

        /* Top right */
        ii = i+1;
        if (ii > nx-1)  ii -= nx;
        jj = j+1;
        if (jj > ny-1)  ii -= ny;

        /* Right */
        if (map[j*nx+ii])  n1++;
        /* Top */
        if (map[jj*nx+i])  n1++;

        /* Top right */
        if (map[jj*nx+ii])  nsqrt2++;

        /* Top left */
        ii = i-1;
        if (ii < 0)  ii += nx;

        /* Left */
        if (map[j*nx+ii])  n1++;

        /* Top left */
        if (map[jj*nx+ii])  nsqrt2++;

        /* Bottom left */
        jj = j-1;
        if (jj < 0)  jj += ny;

        /* Bottom */
        if (map[jj*nx+i])  n1++;

        /* Bottom left */
        if (map[jj*nx+ii])  nsqrt2++;

        /* Bottom right */
        ii = i+1;
        if (ii > nx-1)  ii -= nx;

        /* Bottom right */
        if (map[jj*nx+ii])  nsqrt2++;

        if (n1 >= 2) {
          length += 1.0;
        }
        else if (n1 == 1) {
          if (nsqrt2 >= 1) {
            length += 0.5*(1.0+sqrt2);
          }
        }
        else {
          length += 0.5*sqrt2;
        }
      }
    }
  }

  Py_DECREF(py_bool_map);

  return Py_BuildValue("d", length);
}
