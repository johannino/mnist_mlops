# tests/test_data.py
import os
import pytest
from torchvision.datasets import MNIST

# Adjust these paths according to your project structure
_TEST_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")

N_train = 60000  # or 50000, depending on your dataset
N_test = 10000

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_loading():
    train_dataset = MNIST(root=_PATH_DATA, train=True, download=True)
    test_dataset = MNIST(root=_PATH_DATA, train=False, download=True)

    assert len(train_dataset) == N_train, "Incorrect number of samples in training set"
    assert len(test_dataset) == N_test, "Incorrect number of samples in test set"
    # Add more assertions as needed
