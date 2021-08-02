#!/bin/bash
./run_robin.sh | tee run_robin.log
./eval.sh      | tee eval.log