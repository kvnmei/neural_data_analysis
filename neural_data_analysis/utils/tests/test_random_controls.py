import unittest
import numpy as np
from neural_data_analysis.utils import (
    randomize_binary_array_by_group,
)


class TestRandomizeBinaryArray(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        rng = np.random.default_rng(seed=self.seed)
        self.binary_array = rng.choice([0, 1], size=20)

    def test_randomize_binary_array(self):
        # Test the basic functionality
        randomized_array = randomize_binary_array_by_group(
            self.binary_array, self.seed, return_seed=False
        )
        # Check if the array is shuffled
        self.assertFalse(np.array_equal(self.binary_array, randomized_array))
        assert sum(self.binary_array) == sum(randomized_array)

        # counting number of groups doesn't work because certain groups of 1 might be consecutive

        # n_groups_original = len(
        #     find_consecutive_sequences_in_binary_array(self.binary_array)
        # )
        # n_groups_randomized = len(
        #     find_consecutive_sequences_in_binary_array(randomized_array)
        # )
        # assert n_groups_original == n_groups_randomized
