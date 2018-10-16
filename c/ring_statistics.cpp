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

#include <algorithm>
#include <array>
#include <vector>

#include "neighbours.h"
#include "tools.h"

const int MAX_WALKER = 4096;

/*
 * Tolerance for determining if a ring has closed (i.e. if the sum of all bond
 * vectors is zero).
 */
const double TOL = 0.0001;


/*
 * Look for the shortest distance starting at atom *root*,
 */
bool
find_shortest_distances(int *seed, int *neighbours, int root, int *dist)
{
    int n_walker, walker[MAX_WALKER];

    n_walker = 1;
    walker[0] = root;

    while (n_walker > 0) {
        int n_new_walker = 0;
        int new_walker[MAX_WALKER];

        for (int k = 0; k < n_walker; k++) {
            int i = walker[k];

            for (int ni = seed[i]; ni < seed[i+1]; ni++) {
                int j = neighbours[ni];
                if (dist[j] == 0) {
                    if (n_new_walker > MAX_WALKER) {
                        PyErr_SetString(PyExc_RuntimeError,
                                        "MAX_WALKER exceeded");
                        return false;
                    }

                    new_walker[n_new_walker] = j;
                    dist[j] = dist[i]+1;

                    n_new_walker++;
                }
            }
        }
 
        n_walker = n_new_walker;
        std::copy(new_walker, new_walker+n_new_walker, walker);
    }

    dist[root] = 0;

    return true;
}


/*
 * Look for the shortest distance starting at atom *root*,
 * look only for elements *f*
 */
bool
distances_on_graph(int nat, int *seed, int *neighbours, int *dist, int *diameter)
{
    if (diameter) *diameter = 0;
    std::fill(dist, dist+nat*nat, 0);

    for (int i = 0; i < nat; i++) {
        if (!find_shortest_distances(seed, neighbours, i, &dist[i*nat]))
            return false;
        if (diameter) {
            *diameter = std::max(*diameter,
                                 *std::max_element(&dist[i*nat],
                                                   &dist[i*nat]+nat));
        }
    }

    return true;
}


/*
 * Python wrapper
 */
extern "C" PyObject *
py_distances_on_graph(PyObject *self, PyObject *args)
{
    PyObject *py_i, *py_j;

    if (!PyArg_ParseTuple(args, "OO", &py_i, &py_j))
        return NULL;

    /* Make sure our arrays are contiguous */
    py_i = PyArray_FROMANY(py_i, NPY_INT, 1, 1, NPY_C_CONTIGUOUS);
    if (!py_i) return NULL;
    py_j = PyArray_FROMANY(py_j, NPY_INT, 1, 1, NPY_C_CONTIGUOUS);
    if (!py_j) return NULL;

    /* Check array shapes. */
    npy_intp nneigh = PyArray_DIM((PyArrayObject *) py_i, 0);
    if (PyArray_DIM((PyArrayObject *) py_j, 0) != nneigh) {
        PyErr_SetString(PyExc_ValueError, "Arrays must have same length.");
        return NULL;
    }

    /* Get total number of atoms */
    npy_int *i = (npy_int *) PyArray_DATA(py_i);
    int nat = *std::max_element(i, i+nneigh)+1;

    /* Construct seed array */
    int seed[nat+1];
    first_neighbours(nat, nneigh, i, seed);

    npy_intp dims[2];
    dims[0] = nat;
    dims[1] = nat;
    PyObject *py_dist = PyArray_ZEROS(2, dims, NPY_INT, 0);

    if (!distances_on_graph(nat, seed, (npy_int *) PyArray_DATA(py_j),
                            (npy_int *) PyArray_DATA(py_dist), NULL)) {
        Py_DECREF(py_dist);
        return NULL;
    }

    return py_dist;
}

class vec3 : public std::array<double, 3> {
public:
    vec3(const double *v) {
        (*this)[0] = v[0];
        (*this)[1] = v[1];
        (*this)[2] = v[2];
    }
};

class Walker {
public:
    int vertex, previous_vertex;
    std::vector<int> ring_vertices;
    std::vector<vec3> distances_to_root_vertex;

    Walker() : vertex(0), previous_vertex(0) { }

    Walker(int _vertex, int _previous_vertex, vec3 &_dist) :
        vertex(_vertex), previous_vertex(_previous_vertex),
        ring_vertices(1, _vertex),
        distances_to_root_vertex(1, _dist) { }

    Walker(int _vertex, int _previous_vertex, const double *_dist) :
        vertex(_vertex), previous_vertex(_previous_vertex),
        ring_vertices(1, _vertex),
        distances_to_root_vertex(1, vec3(_dist)) { }

    Walker(Walker &from, int new_vertex, vec3 &step_dist) : 
        vertex(new_vertex), previous_vertex(from.vertex),
        ring_vertices(from.ring_vertices),
        distances_to_root_vertex(from.distances_to_root_vertex)
    {
        add_vertex(new_vertex, step_dist);
    }

    Walker(Walker &from, int new_vertex, const double *step_dist) : 
        vertex(new_vertex), previous_vertex(from.vertex),
        ring_vertices(from.ring_vertices),
        distances_to_root_vertex(from.distances_to_root_vertex)
    {
        add_vertex(new_vertex, vec3(step_dist));
    }

    void add_vertex(int new_vertex, vec3 step_dist) {
        ring_vertices.push_back(new_vertex);

        vec3 new_dist = distances_to_root_vertex.back();
        new_dist[0] += step_dist[0];
        new_dist[1] += step_dist[1];
        new_dist[2] += step_dist[2];
        distances_to_root_vertex.push_back(new_dist);       
    }

    std::vector<int>::size_type ring_size() { return ring_vertices.size(); }
};


bool
step_away(std::vector<Walker> &new_walkers, Walker &walker,
          int root, /* root vertex */
          int nat, int *seed, int *neighbours, double *r, /* neighbour list */
          int *dist, /* distance map */
          std::vector<bool> &done, npy_intp maxlength)
{
    /* Loop over neighbours of walker atom */
    int i = walker.vertex;
    for (int ni = seed[i]; ni < seed[i+1]; ni++) {
        int j = neighbours[ni];
        /* Check if edge has already been visited or if
           vertex is identical to previous_vertices vertex of
           walker k. (This would be a reverse jump.) */
        if (!done[ni] && j != walker.previous_vertex) {
            /* Did we jump farther away from the root
               vertex? */
            if (dist[nat*root+j] == dist[nat*root+i]+1) {
                /* Don't continue stepping further if we are already at half the
                   maximum ring length */
                if (maxlength < 0) {
                    new_walkers.push_back(Walker(walker, j, &r[3*ni]));
                }
                else if (walker.ring_vertices.size() <
                         (std::vector<int>::size_type(maxlength)-1)/2) {
                    new_walkers.push_back(Walker(walker, j, &r[3*ni]));
                }
            }
            /* Did we either not change distance from root vertex or moved
               closer to root vertex? */
            else if (dist[nat*root+j] == dist[nat*root+i] ||
                     dist[nat*root+j] == dist[nat*root+i]-1) {
                /* This is a jump back towards the root */
                new_walkers.push_back(Walker(walker, -j, &r[3*ni]));
            }
            else {
                PyErr_SetString(PyExc_RuntimeError, "Distance map and "
                                "graph do not match.");
                return false;
            }
        }
    }

    return true;
}


bool
step_closer(std::vector<Walker> &new_walkers, Walker &walker,
            int root, /* root vertex */
            int nat, int *seed, int *neighbours, double *r, /* neighbour list */
            int *dist, /* distance map */
            std::vector<bool> &done,
            std::vector<npy_int> &ringstat)
{
    /* Loop over neighbours of walker vertex */
    int i = -walker.vertex;
    for (int ni = seed[i]; ni < seed[i+1]; ni++) {
        int j = neighbours[ni];
        /* Check if edge has already been visited or if
           vertex is identical to previous_vertices vertex of
           walker k. (This would be a reverse jump.) */
        if (!done[ni] && j != walker.previous_vertex) {
            /* Are we back to the root vertex? */
            if (j == root) {
                auto droot = walker.distances_to_root_vertex.back();
                double d[3];
                d[0] = r[3*ni+0] + droot[0];
                d[1] = r[3*ni+1] + droot[1];
                d[2] = r[3*ni+2] + droot[2];

                /* Check if the sum of vertex vectors is zero. This is to
                   exclude chains that cross the periodic boundaries from
                   counting as rings. */
                if (normsq(d) < TOL) {
                    /* Now we need to check whether this ring is SP */
                    bool is_sp = true;

                    /* Add the root vertex */
                    walker.ring_vertices.push_back(root);

                    /* Check if the length along the ring agrees with the map
                       distance. Otherwise, there is a short ring that cuts
                       this one. */
                    int ring_size = walker.ring_size();
                    for (int m = 0; m < ring_size; m++) {
                        for (int n = m+1; n < ring_size; n++) {
                            int dn = n-m;
                            if (dn > ring_size/2) dn = ring_size-dn;
                            if (dist[nat*abs(walker.ring_vertices[n])+
                                     abs(walker.ring_vertices[m])] != dn)
                                is_sp = false;
                        }
                    }

                    if (is_sp) {
                        if (ringstat.size() < walker.ring_size()+1)
                            ringstat.resize(walker.ring_size()+1);
                        ringstat[walker.ring_size()]++;
                    }
                }
            }
            /* Did we jump closer to the root vertex? */
            else if (dist[nat*root+j] == dist[nat*root+i]-1) {
                new_walkers.push_back(Walker(walker, -j, &r[3*ni]));
            }
            /* We discard this path if we jump away again. */
        }
    }

    return true;
}


/*
 * Look for the "shortest path" ring starting at atom *at*,
 * look only for elements *f*
 */
bool
find_sp_ring_vertices(int nat, int *seed, int neighbours_size, int *neighbours,
                      double *r, int *dist, int maxlength,
                      std::vector<int> &ringstat)
{
    std::vector<bool> done(neighbours_size, false);

    /* Loop over all vertices */
    for (int a = 0; a < nat; a++) {
        /* Loop over neighbours of vertex a, i.e. walk on graph */
        for (int na = seed[a]; na < seed[a+1]; na++) {
            int b = neighbours[na];

            /* Only walk in one direction */
            if (a < b) {
                /* We have visited this site. Loop over neighbours of b and
                   mark reverse jump as visited. */
                done[na] = true;
                for (int ni = seed[b]; ni < seed[b+1]; ni++) {
                    if (neighbours[ni] == a) done[ni] = true;
                }

                /* Initialize Single walker on atom b coming from atom a. */
                std::vector<Walker> *walkers = 
                    new std::vector<Walker>(1, Walker(b, a, &r[3*na]));

                /* Continue loop while there are walkers active. */
                while (walkers->size() > 0) {
                    std::vector<Walker> *new_walkers = 
                        new std::vector<Walker>(0);

                    /* Loop over all walkers and advance them. */
                    for (auto walker: *walkers) {
                        /* Walker walks away from root */
                        if (walker.vertex > 0) {
                            if (!step_away(*new_walkers, walker, a,
                                           nat, seed, neighbours, r,
                                           dist, done,
                                           maxlength))
                                return false;
                        }
                        /* Walker walks towards root */
                        else {
                            if (!step_closer(*new_walkers, walker, a,
                                             nat, seed, neighbours, r,
                                             dist, done,
                                             ringstat))
                                return false;
                        }
                    }

                    /* Copy new walker list to old walker list */
                    delete walkers;
                    walkers = new_walkers;
                }
            }
        }
    }

    return true;
}


/*
 * Python wrapper
 */
extern "C" PyObject *
py_find_sp_rings(PyObject *self, PyObject *args)
{
    PyObject *py_i, *py_j, *py_r, *py_dist;
    npy_int maxlength = -1;

    if (!PyArg_ParseTuple(args, "OOOO|i", &py_i, &py_j, &py_r, &py_dist,
                          &maxlength))
        return NULL;

    /* Make sure our arrays are contiguous */
    py_i = PyArray_FROMANY(py_i, NPY_INT, 1, 1, NPY_C_CONTIGUOUS);
    if (!py_i) return NULL;
    py_j = PyArray_FROMANY(py_j, NPY_INT, 1, 1, NPY_C_CONTIGUOUS);
    if (!py_j) return NULL;
    py_r = PyArray_FROMANY(py_r, NPY_DOUBLE, 2, 2, NPY_C_CONTIGUOUS);
    if (!py_r) return NULL;
    py_dist = PyArray_FROMANY(py_dist, NPY_INT, 2, 2, NPY_C_CONTIGUOUS);
    if (!py_dist) return NULL;

    /* Check array shapes. */
    npy_intp nneigh = PyArray_DIM((PyArrayObject *) py_i, 0);
    if (PyArray_DIM((PyArrayObject *) py_j, 0) != nneigh) {
        PyErr_SetString(PyExc_ValueError, "Array must have same length.");
        return NULL;
    }
    if (PyArray_DIM((PyArrayObject *) py_r, 0) != nneigh) {
        PyErr_SetString(PyExc_ValueError, "Array must have same length.");
        return NULL;
    }
    if (PyArray_DIM((PyArrayObject *) py_r, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, "Distance array must have second "
                                          "dimension of length 3.");
        return NULL;
    }

    /* Get total number of atoms */
    npy_int *i = (npy_int *) PyArray_DATA(py_i);
    int nat = *std::max_element(i, i+nneigh)+1;

    /* Check shape of distance map */
    if (PyArray_DIM((PyArrayObject *) py_dist, 0) != nat ||
        PyArray_DIM((PyArrayObject *) py_dist, 1) != nat) {
        /*
        char errstr[1024];
        sprintf(errstr, "Distance map has shape %" NPY_INTP_FMT " x %" 
                NPY_INTP_FMT " while number of atoms is %i.",
                PyArray_DIM((PyArrayObject *) py_dist, 0),
                PyArray_DIM((PyArrayObject *) py_dist, 1),
                nat);
        PyErr_SetString(PyExc_ValueError, errstr);
        */
        PyErr_SetString(PyExc_ValueError, "Distance map has wrong shape.");
        return NULL;
    }

    /* Construct seed array */
    int seed[nat+1];
    first_neighbours(nat, nneigh, i, seed);

    std::vector<npy_int> ringstat;
    if (!find_sp_ring_vertices(nat, seed, nneigh, (int *) PyArray_DATA(py_j),
                               (npy_double *) PyArray_DATA(py_r),
                               (npy_int *) PyArray_DATA(py_dist),
                               maxlength, ringstat)) {
        return NULL;
    }

    npy_intp ringstat_size = ringstat.size();
    PyObject *py_ringstat = PyArray_ZEROS(1, &ringstat_size, NPY_INT, 0);
    std::copy(ringstat.begin(), ringstat.end(),
              (npy_int *) PyArray_DATA(py_ringstat));

    return py_ringstat;
}
