import pytest
import torch
from mnist_mlops.models.model import myawesomemodel  # Adjust the import path as needed

def test_model_input_output_shape():
    """
    Test if the model accepts the correct input shape and produces the expected output shape.
    """
    # Create a dummy input tensor of the correct shape
    # Assuming the input is a 1-channel image of size 28x28
    dummy_input = torch.randn(1, 1, 28, 28)  # Batch size of 1

    # Forward pass through the model
    output = myawesomemodel(dummy_input)

    # Check if the output shape is as expected
    # Assuming the output is a tensor corresponding to 10 classes
    assert output.shape == (1, 10), f"Output shape is incorrect: {output.shape} != (1, 10)"

def test_error_on_wrong_input_shape():
    """
    Test if the model raises an error (or misbehaves) with incorrect input shape.
    """
    # Create a dummy input tensor of incorrect shape
    wrong_input = torch.randn(1, 1, 32, 32)  # Incorrect image size

    # Check if the model handles wrong input shape gracefully
    # This test depends on how you want to handle wrong input shapes
    try:
        _ = myawesomemodel(wrong_input)
        # You can assert False here if you expect an error to be raised
        # assert False, "Model did not raise error on wrong input shape"
    except Exception as e:
        # You can check for specific error types and messages if needed
        pass
