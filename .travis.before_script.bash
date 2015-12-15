#!/usr/bin/env bash

if [ "$TRAVIS_OS_NAME" == "linux" ]; then
  #sudo apt-get install python-setuptools
  pip install setuptools
elif [ "$TRAVIS_OS_NAME" == "osx" ]; then
  # noop?
  pip install setuptools
fi
