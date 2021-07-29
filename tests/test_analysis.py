#
# Copyright 2015, 2020 Lars Pastewka (U. Freiburg)
#
# matscipy - Materials science with Python at the atomic-scale
# https://github.com/libAtoms/matscipy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from math import sqrt
import unittest

import numpy as np

from matscipy.contact_mechanics.analysis import (count_islands,
                                                 count_segments,
                                                 distance_map,
                                                 inner_perimeter,
                                                 outer_perimeter)

###

class TestAnalysis(unittest.TestCase):

    def test_count_islands(self):
        m_xy = np.zeros([3,3], dtype=bool)
        m_xy[1,1] = True

        nump, p_xy = count_islands(m_xy)
       
        self.assertEqual(nump, 1)
        self.assertTrue(np.all(
            p_xy == np.array([[0,0,0],
                              [0,1,0],
                              [0,0,0]])))
                              
        m_xy = np.zeros([5,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[3,1] = True

        nump, p_xy = count_islands(m_xy)

        self.assertEqual(nump, 2)
        self.assertTrue(np.all(
            p_xy == np.array([[0,0,0],
                              [0,1,0],
                              [0,0,0],
                              [0,2,0],
                              [0,0,0]])))

        m_xy = np.zeros([6,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[3,1] = True
        m_xy[3,2] = True

        nump, p_xy = count_islands(m_xy)

        self.assertEqual(nump, 2)
        self.assertTrue(np.all(
            p_xy == np.array([[0,0,0],
                              [0,1,0],
                              [0,0,0],
                              [0,2,2],
                              [0,0,0],
                              [0,0,0]])))

        m_xy = np.zeros([6,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[2,1] = True
        m_xy[3,1] = True
        m_xy[3,2] = True

        nump, p_xy = count_islands(m_xy)

        self.assertEqual(nump, 1)
        self.assertTrue(np.all(
            p_xy == np.array([[0,0,0],
                              [0,1,0],
                              [0,1,0],
                              [0,1,1],
                              [0,0,0],
                              [0,0,0]])))


    def test_count_segments(self):
        m_xy = np.zeros([3,3], dtype=bool)
        m_xy[1,1] = True
        
        nump, p_xy = count_segments(m_xy)
       
        self.assertEqual(nump, 1)
        self.assertTrue(np.all(
            p_xy == np.array([[0,0,0],
                              [0,1,0],
                              [0,0,0]])))
                              
        m_xy = np.zeros([5,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[3,1] = True

        nump, p_xy = count_segments(m_xy)

        self.assertEqual(nump, 2)
        self.assertTrue(np.all(
            p_xy == np.array([[0,0,0],
                              [0,1,0],
                              [0,0,0],
                              [0,2,0],
                              [0,0,0]])))

        m_xy = np.zeros([6,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[2,1] = True
        m_xy[3,1] = True
        m_xy[3,2] = True

        nump, p_xy = count_segments(m_xy)

        self.assertEqual(nump, 3)
        self.assertTrue(np.all(
            p_xy == np.array([[0,0,0],
                              [0,1,0],
                              [0,2,0],
                              [0,3,3],
                              [0,0,0],
                              [0,0,0]])))


    def test_distance_map(self):
        m_xy = np.zeros([3,3], dtype=bool)
        m_xy[1,1] = True
        
        d_xy = distance_map(m_xy)
        
        sqrt_2 = sqrt(2.0)
        self.assertTrue(np.all(np.abs(
            d_xy-np.array([[sqrt_2,1.0,sqrt_2],[1.0,0.0,1.0],[sqrt_2,1.0,sqrt_2]])
            <1e-12)))
            
        m_xy = np.zeros([5,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[3,1] = True
        
        d_xy = distance_map(m_xy)
       
        self.assertTrue(np.all(np.abs(
            d_xy-np.array([[sqrt_2,1.0,sqrt_2],[1.0,0.0,1.0],
                           [sqrt_2,1.0,sqrt_2],[1.0,0.0,1.0],
                           [sqrt_2,1.0,sqrt_2]])
            <1e-12)))

        m_xy = np.zeros([6,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[3,1] = True
        
        d_xy = distance_map(m_xy)
        
        sqrt_5 = sqrt(5.0)
        self.assertTrue(np.all(np.abs(
            d_xy-np.array([[sqrt_2,1.0,sqrt_2],[1.0,0.0,1.0],
                           [sqrt_2,1.0,sqrt_2],[1.0,0.0,1.0],
                           [sqrt_2,1.0,sqrt_2],[sqrt_5,2.0,sqrt_5]])
            <1e-12)))
            
            
    def test_perimeter(self):
        m_xy = np.zeros([3,3], dtype=bool)
        m_xy[1,1] = True
        
        i_xy = inner_perimeter(m_xy)
        o_xy = outer_perimeter(m_xy)
        
        self.assertTrue(np.array_equal(i_xy, m_xy))
        self.assertTrue(np.array_equal(o_xy, np.array([[False,True, False],
                                                       [True, False,True ],
                                                       [False,True, False]])))
            
        m_xy = np.zeros([5,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[3,1] = True
        
        i_xy = inner_perimeter(m_xy)
        o_xy = outer_perimeter(m_xy)
        
        self.assertTrue(np.array_equal(i_xy, np.array([[False,False,False],
                                                       [False,True, False],
                                                       [False,False,False],
                                                       [False,True, False],
                                                       [False,False,False]])))
        self.assertTrue(np.array_equal(o_xy, np.array([[False,True, False],
                                                       [True, False,True ],
                                                       [False,True, False],
                                                       [True, False,True ],
                                                       [False,True, False]])))

###

if __name__ == '__main__':
    unittest.main()
