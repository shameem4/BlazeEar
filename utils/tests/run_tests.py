import sys
import unittest
from pathlib import Path


def main() -> int:
    """Discover and run all unit tests under tests/tests."""
    script_dir = Path(__file__).parent
    # Go up two levels: tests -> utils -> repo_root
    repo_root = script_dir.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    tests_dir = script_dir / "tests"
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(tests_dir), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
