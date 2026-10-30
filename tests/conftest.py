import pytest

from tests.tiny_lm import save_tiny_lm


@pytest.fixture(scope="session")
def tiny_lm_dir(tmp_path_factory):
    return save_tiny_lm(tmp_path_factory.mktemp("tiny_lm"))
