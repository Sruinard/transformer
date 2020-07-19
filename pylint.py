"""
Pylint script to run pylint from the root of a module. This ensures that
all paths and imports are resolved correctly.
"""

import os
import glob
import sys
import logging
import argparse
import contextlib
import subprocess
from typing import Generator, Any

# pylint: disable=invalid-name
parser = argparse.ArgumentParser()
parser.add_argument('--module',
                    dest='module',
                    help='Module to pylint',
                    default=None,
                    required=False,
                    type=str)


@contextlib.contextmanager
def working_directory(path: str) -> Generator:
    """A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.
    """
    current_cwd = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(current_cwd)


def main(args: Any) -> None:
    """Run pylint on provided modules."""
    if args.module is None:
        # Read in all modules to be pylinted
        with open('./pylint-modules', 'r') as f:
            modules = [l.strip('\n') for l in f.readlines()]
    else:
        modules = [args.module]

    logging.warning('Running pylint for %s', modules)

    # Run pylint for each module from the module directory
    failures = []
    cwd = os.getcwd()
    for module in modules:
        with working_directory(module):
            # Only run if directory contains Python files (also in subdirectory)
            pyfiles = glob.glob('**/*.py', recursive=True)
            if any(pyfiles):
                process = subprocess.run(['pylint'] + pyfiles +
                                         ['--rcfile={}/.pylintrc'.format(cwd)],
                                         check=False)
                if process.returncode != 0:
                    failures.append(module)

    GREEN, RED, NC = '\033[0;32m', '\033[0;31m', '\033[0m'

    # Print some encouraging messages with info
    if failures:
        print(RED + "Failure :'(" + NC)
        print("Following modules failed: {}".format(failures))
        sys.exit(1)
    else:
        print(GREEN + 'Succes!' + NC)


if __name__ == '__main__':
    main(parser.parse_args())
