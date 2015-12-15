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

#ifndef __STACK_H
#define __STACK_H

#include <cassert>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

class Stack {
 public:
  Stack(size_t size) {
    size_ = size;
    top_ = size;
    tp_ = 0;
    bp_ = 0;
    is_empty_ = true;
    data_ = malloc(size_);
  }
  ~Stack() {
    free(data_);
  }

  bool is_empty() {
    return is_empty_;
  }

  size_t get_size() {
    if (is_empty_)
      return 0;
    if (tp_ >= bp_)
      return tp_-bp_;
    return top_-bp_+tp_;
  }

  template<typename T> void push(T value) {
    if (tp_+sizeof(T) > size_) {
      if (bp_ < sizeof(T)) {
	expand(2*size_);
      }
      else {
	top_ = tp_;
	tp_ = 0;
      }
    }
    else if (bp_ == tp_ && !is_empty_) {
      expand(2*size_);
    }

    *((T*) ((uint8_t*) data_+tp_)) = value;
    tp_ += sizeof(T);

    is_empty_ = false;
  }

  template<typename T> void pop(T &value) {
    if (tp_ == 0) {
      tp_ = top_-sizeof(T);
    }
    else {
      assert(tp_ >= sizeof(T));
      tp_ -= sizeof(T);
    }
    value = *((T*) ((uint8_t*) data_+tp_));

    is_empty_ = bp_ == tp_;
  }

  template<typename T> void pop_bottom(T &value) {
    if (bp_+sizeof(T) > size_) {
      assert(bp_ == top_);
      bp_ = 0;
    }
    value = *((T*) ((uint8_t*) data_+bp_));
    bp_ += sizeof(T);

    is_empty_ = bp_ == tp_;
  }

  template<typename T1, typename T2> void push(T1 value1, T2 value2) {
    push(value1);
    push(value2);
  }
  template<typename T1, typename T2> void pop(T1 &value1, T2 &value2) {
    pop(value2);
    pop(value1);
  }
  template<typename T1, typename T2> void pop_bottom(T1 &value1, T2 &value2) {
    pop_bottom(value1);
    pop_bottom(value2);
  }

  template<typename T1, typename T2, typename T3>
    void push(T1 value1, T2 value2, T3 value3) {
    push(value1);
    push(value2);
    push(value3);
  }
  template<typename T1, typename T2, typename T3>
    void pop(T1 &value1, T2 &value2, T3 &value3) {
    pop(value3);
    pop(value2);
    pop(value1);
  }
  template<typename T1, typename T2, typename T3>
    void pop_bottom(T1 &value1, T2 &value2, T3 &value3) {
    pop_bottom(value1);
    pop_bottom(value2);
    pop_bottom(value3);
  }

  template<typename T1, typename T2, typename T3, typename T4>
    void push(T1 value1, T2 value2, T3 value3, T4 value4) {
    push(value1);
    push(value2);
    push(value3);
    push(value4);
  }
  template<typename T1, typename T2, typename T3, typename T4>
    void pop(T1 &value1, T2 &value2, T3 &value3, T4 &value4) {
    pop(value4);
    pop(value3);
    pop(value2);
    pop(value1);
  }
  template<typename T1, typename T2, typename T3, typename T4>
    void pop_bottom(T1 &value1, T2 &value2, T3 &value3, T4 &value4) {
    pop_bottom(value1);
    pop_bottom(value2);
    pop_bottom(value3);
    pop_bottom(value4);
  }

 private:
  size_t size_;       /* Total size of the stack */
  size_t top_;        /* Where does the data end after wrapping tp_? */
  size_t tp_, bp_;    /* Top pointer, bottom pointer, end markers of stack */

  bool is_empty_;

  void *data_;

  void expand(size_t new_size) {
    printf("Expanding stack size to %3.2f MB.\n",
	   ((double) new_size)/(1024*1024));
    void *new_data = malloc(new_size);
    if (!new_data) {
      printf("Failed to allocate new stack!\n");
    }
    if (tp_ > bp_) {
      memcpy(new_data, ((uint8_t *) data_+bp_), tp_-bp_);
      tp_ -= bp_;
    }
    else {
      memcpy(new_data, ((uint8_t *) data_+bp_), top_-bp_);
      memcpy(((uint8_t *) new_data+top_-bp_), data_, tp_);
      tp_ = top_-bp_+tp_;
    }
    free(data_);
    size_ = new_size;
    data_ = new_data;
    bp_ = 0;
  }
};

#endif
