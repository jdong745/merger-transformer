from pathlib import Path

import pytest
import torch

from merger_transformer import WaveformModel


def pytest_addoption(parser):
    parser.addoption("--weights", default=str(Path(__file__).resolve().parents[2] / "weights" / "weights.pt"))


@pytest.fixture(scope="session")
def weights_path(request):
    path = Path(request.config.getoption("--weights"))
    if not path.is_file():
        pytest.fail("Supply the separately distributed weights with --weights /path/to/weights.pt")
    return path


@pytest.fixture(scope="session")
def model(weights_path):
    torch.set_num_threads(1)
    return WaveformModel.from_pretrained(weights_path)


@pytest.fixture
def parameters():
    return torch.tensor([2., .2, -.1, .3, -.05, .1, -.2], dtype=torch.float64)
