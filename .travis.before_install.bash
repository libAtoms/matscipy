#!/usr/bin/env bash

if [ "$TRAVIS_OS_NAME" == "linux" ]; then
  echo "AOK"
  #if [ "$PYTHON" == "/usr/bin/python3.4" ]; then
    #sudo add-apt-repository ppa:fkrull/deadsnakes -y
    #sudo apt-get update
    #sudo apt-get install python3.4 python3-dev
  #fi
elif [ "$TRAVIS_OS_NAME" == "osx" ]; then
  if [ "$PYTHON" == "/usr/local/bin/python3" ]; then
    brew install python3
  fi
  sudo pip install virtualenv
  virtualenv -p $PYTHON venv
  source venv/bin/activate
fi