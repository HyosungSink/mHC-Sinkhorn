"""Run the five deterministic local Mock workloads with MSPTI kernel timing."""
import sys
from tests.st.scripts.run_tests import main

if __name__ == '__main__':
    if '--suite' in sys.argv[1:] or any(arg.startswith('--suite=') for arg in sys.argv[1:]):
        raise SystemExit('This entry point always selects the mock suite')
    raise SystemExit(main(['--suite', 'mock', *sys.argv[1:]]))
