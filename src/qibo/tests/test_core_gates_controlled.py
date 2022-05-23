"""Test execution of `controlled_by` gates."""
import pytest
import numpy as np
from qibo import gates, K
from qibo.models import Circuit
from qibo.tests.utils import random_state


