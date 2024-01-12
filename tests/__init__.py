# tests/__init__.py
import os

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_TEST_ROOT = os.path.join(_TEST_ROOT, "mnist_mlops")  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "Data")  # root of data
