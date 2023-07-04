#
# Copyright 2022 Lars Pastewka
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

#
# This is the most minimal-idiotic way of discovering the version that I
# could come up with. It deals with the following issues:
# * If we are installed, we can get the version from package metadata,
#   either via importlib.metadata or from pkg_resources. This also holds for
#   wheels that contain the metadata. We are good! Yay!
# * If we are not installed, there are two options:
#   - We are working within the source git repository. Then
#        git describe --tags --always
#     yields a reasonable version descriptor, but that is unfortunately not
#     PEP 440 compliant (see https://peps.python.org/pep-0440/). We need to
#     mangle the version string to yield something compatible.
# - If we install from a source tarball, we need to parse PKG-INFO manually.
#

import re
import subprocess

class CannotDiscoverVersion(Exception):
    pass


def get_version_from_pkg_info():
    """
    Discover version from PKG-INFO file.
    """
    f = open('PKG-INFO', 'r')
    l = f.readline()
    while l:
        if l.startswith('Version:'):
            return l[8:].strip()
        l = f.readline()
    raise CannotDiscoverVersion("No line starting with 'Version:' in 'PKG-INFO'.")


def get_version_from_git():
    """
    Discover version from git repository.
    """
    git_describe = subprocess.run(
        ['git', 'describe', '--tags', '--dirty', '--always'],
        stdout=subprocess.PIPE)
    if git_describe.returncode != 0:
        raise CannotDiscoverVersion('git execution failed.')
    version = git_describe.stdout.decode('latin-1').strip()

    dirty = version.endswith('-dirty')

    # Make version PEP 440 compliant
    if dirty:
        version = version.replace('-dirty', '')
    version = version.strip('v')  # Remove leading 'v' if it exists
    version = version.replace('-', '.dev', 1)
    version = version.replace('-', '+', 1)
    if dirty:
        version += '.dirty'

    return version


try:
    version = get_version_from_git()
except CannotDiscoverVersion:
    version = get_version_from_pkg_info()

#
# Print version to screen
#

print(version)
