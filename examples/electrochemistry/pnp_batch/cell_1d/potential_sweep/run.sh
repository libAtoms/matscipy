#!/bin/bash
bash ./run_std.sh   | tee run_std.log
bash ./run_robin.sh | tee run_robin.log
bash ./eval.sh      | tee eval.log
