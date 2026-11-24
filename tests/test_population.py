"""Seeded initialization and population memory/shape checks."""

import copy
import unittest

import numpy as np

from ga import initialize_population


class PopulationInitializationTests(unittest.TestCase):
    def test_population_shape_dtype_and_binary_values(self):
        for size, length in ((2, 1), (5, 7), (100, 32)):
            with self.subTest(size=size, length=length):
                population = initialize_population(size, length, np.random.default_rng(42))
                self.assertEqual(population.shape, (size, length))
                self.assertEqual(population.dtype, np.dtype(np.uint8))
                self.assertTrue(np.all((population == 0) | (population == 1)))

    def test_recreating_a_seed_reproduces_the_population(self):
        first = initialize_population(100, 32, np.random.default_rng(42))
        second = initialize_population(100, 32, np.random.default_rng(42))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.shares_memory(first, second))

    def test_uses_supplied_generator_and_leaves_global_rng_untouched(self):
        rng = np.random.default_rng(42)
        before = copy.deepcopy(rng.bit_generator.state)
        global_before = np.random.get_state()
        initialize_population(100, 32, rng)
        self.assertNotEqual(before, rng.bit_generator.state)
        global_after = np.random.get_state()
        for expected, actual in zip(global_before, global_after):
            np.testing.assert_equal(actual, expected)

    def test_rows_and_successive_populations_do_not_alias(self):
        rng = np.random.default_rng(42)
        first = initialize_population(5, 32, rng)
        other_rows = first[1:].copy()
        second = initialize_population(5, 32, rng)
        self.assertFalse(np.shares_memory(first, second))
        first[0] ^= 1
        np.testing.assert_array_equal(first[1:], other_rows)

    def test_reject_invalid_sizes_and_lengths(self):
        cases = ((0, 32), (1, 32), (-1, 32), (2.5, 32), (True, 32),
                 (2, 0), (2, -1), (2, 3.5), (2, True))
        for size, length in cases:
            with self.subTest(size=size, length=length):
                with self.assertRaises(ValueError):
                    initialize_population(size, length, np.random.default_rng(42))

    def test_reject_missing_or_legacy_generator(self):
        for rng in (None, 42, np.random.RandomState(42)):
            with self.subTest(rng=rng):
                with self.assertRaisesRegex(ValueError, "Generator"):
                    initialize_population(2, 32, rng)

    def test_accept_numpy_integer_sizes(self):
        population = initialize_population(np.int64(2), np.int64(3), np.random.default_rng(42))
        self.assertEqual(population.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
