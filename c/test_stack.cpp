/* ======================================================================
   matscipy - Python materials science tools
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

#include <stdio.h>

#include "stack.h"

int main(int argc, char *argv[])
{
  Stack s(8);

  int i;
  double d;

  s.push((int) 1);
  s.push((double) 3.4);
  s.push((int) 2);
  s.pop_bottom(i);
  printf("%i\n", i);
  s.pop(i);
  printf("%i\n", i);
  s.push((double) 4.8);
  s.pop_bottom(d);
  printf("%f\n", d);
  s.pop_bottom(d);
  printf("%f\n", d);
  printf("size = %i\n", s.get_size());
}
