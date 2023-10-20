#!/bin/bash
bash ./run_std.sh   | tee run_std.log
bash ./eval.sh      | tee eval.log
