"""Run the test suite.

This used to drive `unittest` discovery, which only collects TestCase
subclasses. Most of this suite is written in pytest style -- plain classes,
fixtures, parametrised cases -- so discovery silently skipped 12 of the 17
files and reported `Ran 51 tests ... OK` while pytest ran 336. A runner that
passes by not looking is worse than no runner, so this one delegates.
"""
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    return subprocess.call(
        [sys.executable, '-m', 'pytest', str(repo_root / 'utils' / 'tests'),
         *sys.argv[1:]],
        cwd=str(repo_root),
    )


if __name__ == '__main__':
    sys.exit(main())
