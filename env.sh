#! /bin/sh

PYTHON="$1"
if [ -z "$PYTHON" ]; then
  PYTHON="python"
fi

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PLATFORM=`$PYTHON -c "from distutils.util import get_platform ; from distutils.sysconfig import get_python_version ; print('{0}-{1}'.format(get_platform(), get_python_version()))"`

echo "Setting Python environment"
echo "--------------------------"
echo "Python executable: $PYTHON"
echo "Root directory:    $ROOT"
echo "Platform:          $PLATFORM"

export PYTHONPATH="$ROOT:$ROOT/build/lib.$PLATFORM:$PYTHONPATH"
