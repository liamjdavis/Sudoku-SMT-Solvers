import os
import setuptools
import re

NAME = "sudoku_smt_solvers"
AUTHOR = "Liam Davis"
AUTHOR_EMAIL = "ljdavis27@amherst.edu"
DESCRIPTION = "A collection of SAT and SMT solvers for solving Sudoku puzzles"
LICENSE = "MIT"
URL = "https://liamjdavis.github.io/sudoku-smt-solvers"
README = "README.md"
CLASSIFIERS = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
INSTALL_REQUIRES = [
    "cvc5",
    "python-sat",
    "z3-solver",
]
ENTRY_POINTS = {}
SCRIPTS = []

HERE = os.path.dirname(__file__)


def read(file):
    with open(os.path.join(HERE, file), "r") as fh:
        return fh.read()


VERSION = re.search(
    r'__version__ = [\'"]([^\'"]*)[\'"]', read(NAME.replace("-", "_") + "/__init__.py")
).group(1)
LONG_DESCRIPTION = read(README)


if __name__ == "__main__":
    setuptools.setup(
        name=NAME,
        version=VERSION,
        packages=setuptools.find_packages(),
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        license=LICENSE,
        url=URL,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        scripts=SCRIPTS,
        include_package_data=True,
        python_requires=">=3.10",
    )
