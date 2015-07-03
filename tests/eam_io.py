#! /usr/bin/env python

# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
#                  Adrien Gola, Karlsruhe Institute of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ======================================================================

from __future__ import division
import numpy as np
from scipy import interpolate

import os
from matscipy.eam.io import (read_eam,read_eam_alloy,write_eam,write_eam_alloy)

def test_eam_io():
    source,parameters,F,f,rep = read_eam("Au_u3.eam")
    write_eam(source,parameters,F,f,rep,"Au_u3_copy.eam")
    source1,parameters1,F1,f1,rep1 = read_eam("Au_u3_copy.eam")
    fail = 0
    for i,p in enumerate(parameters):
	if p != parameters1[i] :
	    fail+=1
    if F.any() != F1.any() :
	fail+=1
    if f.any() != f1.any() :
	fail+=1
    if rep.any() != rep1.any() :
	fail+=1
    if fail:
	print "test fail eam_in != eam_out"
    os.remove("Au_u3_copy.eam")
    
    source,parameters,F,f,rep = read_eam_alloy("CuAgNi_Zhou.eam.alloy")
    write_eam_alloy(source,parameters,F,f,rep,"CuAgNi_Zhou.eam_copy.alloy")
    source1,parameters1,F1,f1,rep1 = read_eam_alloy("CuAgNi_Zhou.eam_copy.alloy")
    fail = 0
    for i,p in enumerate(parameters):
	try :
	  for j,d in enumerate(p):
	      if d != parameters[i][j]:
		  fail+=1
	except:
	    if p != parameters[i]:
		fail +=1
    if F.any() != F1.any() :
      fail+=1
    if f.any() != f1.any() :
      fail+=1
    if rep.any() != rep1.any() :
      fail+=1
    if fail:
      print "test fail eam_alloy_in != eam_alloy_out"
    os.remove("CuAgNi_Zhou.eam_copy.alloy")

    
test_eam_io()
