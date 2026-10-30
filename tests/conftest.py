import pytest
import torch

from tests.tiny_lm import save_tiny_lm

# The tiny test model is fastest on one thread, and this keeps test times
# steady when other jobs are using the CPU.
torch.set_num_threads(1)


@pytest.fixture(scope="session")
def tiny_lm_dir(tmp_path_factory):
    return save_tiny_lm(tmp_path_factory.mktemp("tiny_lm"))
