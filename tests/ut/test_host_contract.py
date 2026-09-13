"""Check public Host behavior through the operator definition's callbacks."""
import subprocess
from pathlib import Path

import pytest

from tests.st.scripts.build_ops import MockFailure, toolkit_root


def test_real_host_context(tmp_path):
    try:
        toolkit = toolkit_root()
    except MockFailure as error:
        pytest.skip(str(error))
    if not (toolkit / 'include/base/context_builder/op_tiling_context_builder.h').is_file():
        pytest.skip('CANN context builders are not installed')
    binary = tmp_path / 'host_contract'
    libraries = ['nnopbase', 'exe_graph', 'register', 'tiling_api', 'platform',
                 'rt2_registry', 'unified_dlog', 'graph_base', 'c_sec', 'metadef', 'dl', 'pthread']
    build = subprocess.run([
        'c++', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0', '-DOP_TILING_LIB',
        '-I' + str(toolkit / 'include'), str(Path(__file__).with_name('host_contract.cpp')),
        '-L' + str(toolkit / 'lib64'), '-Wl,-rpath,' + str(toolkit / 'lib64'),
        '-Wl,--copy-dt-needed-entries', *('-l' + name for name in libraries),
        '-o', str(binary),
    ], capture_output=True, text=True, timeout=120)
    if build.returncode:
        pytest.fail('Host contract harness compilation failed:\n' + build.stdout + build.stderr)
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    if result.returncode:
        pytest.fail('Host interface contract failed:\n' + result.stdout + result.stderr)
