#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
# convert .ipynb to .html
import argparse
import os

import nbformat as nbf

from traitlets.config import Config
from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor


def main():
    """Convert .ipynb to .html."""

    class ArgumentDefaultsAndRawDescriptionHelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsAndRawDescriptionHelpFormatter)

    parser.add_argument('infile', metavar='IN',
                        help='.py input file')
    parser.add_argument('outfile', metavar='OUT', nargs='?',
                        help='.html output file', default=None)

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
        # This supports bash autocompletion. To enable this, pip install
        # argcomplete, activate global completion, or add
        #      eval "$(register-python-argcomplete lpad)"
        # into your .bash_profile or .bashrc
    except ImportError:
        pass


    args = parser.parse_args()

    if args.outfile is None:
        prefix, _ = os.path.splitext(args.infile)
        args.outfile = prefix + '.html'

    c = Config()
    # Configure tag removal
    c.TagRemovePreprocessor.remove_cell_tags = ('remove_cell',)
    c.TagRemovePreprocessor.remove_all_outputs_tags = ('remove_output',)
    c.TagRemovePreprocessor.remove_input_tags = ('remove_input',)
    c.TagRemovePreprocessor.enabled = True
    c.ExecutePreprocessor
    c.HTMLExporter.preprocessors = ['nbconvert.preprocessors.TagRemovePreprocessor']
    (body, resources) = HTMLExporter(config=c).from_filename(args.infile)
    with open(args.outfile, 'w') as f:
        f.write(body)


if __name__ == '__main__':
    main()
