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

import subprocess


def get_version_from_git():
    """
    Discover muSpectre version from git repository.
    """
    git_describe = subprocess.run(
        ['git', 'describe', '--tags', '--dirty', '--always'],
        stdout=subprocess.PIPE)
    if git_describe.returncode != 0:
        raise RuntimeError('git execution failed')
    version = git_describe.stdout.decode('latin-1').strip()
    git_hash = subprocess.run(
        ['git', 'show', '-s', '--format=%H'],
        stdout=subprocess.PIPE)
    if git_hash.returncode != 0:
        raise RuntimeError('git execution failed')
    hash = git_hash.stdout.decode('latin-1').strip()

    dirty = version.endswith('-dirty')

    # Make version PEP 440 compliant
    if dirty:
        version = version.replace('-dirty', '')
    version = version.strip('v')  # Remove leading 'v' if it exists
    if 'rc' in version:
        # If version has a .rc1, .rc2, we cannot attach another .dev
        version = version.replace('-', '+dev', 1)
    else:
        version = version.replace('-', '.dev', 1)
    version = version.replace('-', '+', 1)
    if dirty:
        version += '-dirty'

    return dirty, version, hash


dirty, version, hash = get_version_from_git()

#
# Print version to screen
#

print(version)
