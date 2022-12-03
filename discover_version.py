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
# - If we install from a source tarball, all version information is lost.
#   Fortunately, Meson uses git archive to create the source tarball, which
#   replaces certain tags with commit information. Unfortunately, what this
#   yields is different from `git describe` - in particular, it only yields the
#   tag (which contains the version information) if we are *exactly* on the
#   tag commit. (`git describe` tells us the distance from the latest tag.) We
#   need to extract the version information from the string provided, but if
#   we are not on the tag we can only return a bogus version (here 0.0.0.0).
#   It works for releases, but not for a tarball generated from a random
#   commit. I am not happy and open for suggestions.
#

import re
import subprocess

# As examples, format %D yields the following:
# Release tag
#     HEAD, tag: 1.2.3, origin/master, origin/HEAD, master
# Random branch
#     HEAD -> 22_version_discovery_second_attempt, origin/22_version_discovery_second_attempt
#
# To check this from the command line, run
#     git show -s --format="%D"

archived_version = '$Format:%D$'
archived_hash = '$Format:%H$'


class CannotDiscoverVersion(Exception):
    pass


def get_archived_version():
    """
    Discover version from substitutions within this file during during git
    archive.
    """
    # We have to deal with some git idiocy. Git cannot substitute the tag, but
    # only substitutes a string that contains the tag. This is some heuristics
    # to figure out what is going on.

    if archived_version.startswith('$Format:'):
        # The tag has not been replaced. This is not an archive.
        raise CannotDiscoverVersion
    if archived_version.startswith('HEAD ->'):
        # This is an archive, but of some branch without a tag, we don't know
        # what the version is.
        raise CannotDiscoverVersion
    else:
        version = re.search('tag: ([0-9\.]*),', s)
        if version:
            # This cannot be dirty! Return version and hash
            return False, version, archived_hash


def get_version_from_git():
    """
    Discover version from git repository.
    """
    git_describe = subprocess.run(
        ['git', 'describe', '--tags', '--dirty', '--always'],
        stdout=subprocess.PIPE)
    if git_describe.returncode != 0:
        raise CannotDiscoverVersion('git execution failed')
    version = git_describe.stdout.decode('latin-1').strip()
    git_hash = subprocess.run(
        ['git', 'show', '-s', '--format=%H'],
        stdout=subprocess.PIPE)
    if git_hash.returncode != 0:
        raise CannotDiscoverVersion('git execution failed')
    hash = git_hash.stdout.decode('latin-1').strip()

    dirty = version.endswith('-dirty')

    # Make version PEP 440 compliant
    if dirty:
        version = version.replace('-dirty', '')
    version = version.strip('v')  # Remove leading 'v' if it exists
    version = version.replace('-', '.dev', 1)
    version = version.replace('-', '+', 1)
    if dirty:
        version += '-dirty'

    return dirty, version, hash


try:
    dirty, version, hash = get_archived_version()
except CannotDiscoverVersion:
    try:
        dirty, version, hash = get_version_from_git()
    except CannotDiscoverVersion:
        # We return version 0.0.0.0 if version discovery fails
        version = '0.0.0.0'

#
# Print version to screen
#

print(version)
