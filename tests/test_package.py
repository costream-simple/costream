from importlib.metadata import version

import costream


def test_version_matches_distribution():
    assert costream.__version__ == version('costream') == '0.2.0'
