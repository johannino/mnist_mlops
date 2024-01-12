# tests/test_data.py
import pytest
from torchvision.datasets import MNIST
from tempfile import TemporaryDirectory

N_train = 60000  # or 50000, depending on your dataset
N_test = 10000

def test_data_loading():
    with TemporaryDirectory() as tmp_dir:
        train_dataset = MNIST(root=tmp_dir, train=True, download=True)
        test_dataset = MNIST(root=tmp_dir, train=False, download=True)

        assert len(train_dataset) == N_train, "Incorrect number of samples in training set"
        assert len(test_dataset) == N_test, "Incorrect number of samples in test set"
        # Add more assertions as needed
