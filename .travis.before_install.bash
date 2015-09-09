#!/usr/bin/env bash

if [ "$TRAVIS_OS_NAME" == "linux" ]; then
  echo "OK"
elif [ "$TRAVIS_OS_NAME" == "osx" ]; then
  brew install gcc
  brew link --overwrite gcc
  if [ "$PYTHON" == "/usr/local/bin/python3" ]; then
    brew install python3
    brew link --overwrite python
  elif [ "$PYTHON" == "/usr/local/bin/python" ]; then
    brew install python
    brew link --overwrite python
  fi
  sudo pip install virtualenv
  virtualenv -p $PYTHON venv
  source venv/bin/activate
fi
