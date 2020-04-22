#!/bin/bash
./run_std.sh   | tee run_std.log
./run_robin.sh | tee run_robin.log
./eval.sh      | tee eval.log