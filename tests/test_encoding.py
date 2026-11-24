"""Bit-order, bounds, precision, and representation validation checks."""

import unittest

import numpy as np

from ga import BinaryEncoding


class BinaryEncodingTests(unittest.TestCase):
    def test_known_bit_order_and_coordinate_blocks(self):
        encoding = BinaryEncoding(((-2.0, 1.0), (10.0, 13.0)), bits_per_variable=2)
        chromosome = np.array([0, 1, 1, 0], dtype=np.uint8)
        np.testing.assert_array_equal(encoding.decode(chromosome), [-1.0, 12.0])
        np.testing.assert_array_equal(encoding.encode([-1.0, 12.0]), chromosome)

    def test_endpoints_are_exact(self):
        encoding = BinaryEncoding(((-0.1, 0.2), (-5.12, 5.12)))
        zeros = np.zeros(encoding.chromosome_length, dtype=np.uint8)
        ones = np.ones(encoding.chromosome_length, dtype=np.uint8)
        np.testing.assert_array_equal(encoding.decode(zeros), [-0.1, -5.12])
        np.testing.assert_array_equal(encoding.decode(ones), [0.2, 5.12])
        np.testing.assert_array_equal(encoding.encode([-0.1, -5.12]), zeros)
        np.testing.assert_array_equal(encoding.encode([0.2, 5.12]), ones)

    def test_small_grid_order_spacing_and_halfway_ties(self):
        encoding = BinaryEncoding(((0.0, 3.0),), bits_per_variable=2)
        grid = [encoding.decode([0, 0])[0], encoding.decode([0, 1])[0],
                encoding.decode([1, 0])[0], encoding.decode([1, 1])[0]]
        np.testing.assert_array_equal(grid, [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(encoding.resolution, [1.0])
        for point, bits in ((0.5, [0, 0]), (1.5, [1, 0]), (2.5, [1, 0])):
            with self.subTest(point=point):
                np.testing.assert_array_equal(encoding.encode([point]), bits)

    def test_decode_encode_round_trip_for_every_small_chromosome(self):
        encoding = BinaryEncoding(((-2.0, 2.0), (10.0, 20.0)), bits_per_variable=4)
        for index in range(256):
            bits = np.array([int(bit) for bit in f"{index:08b}"], dtype=np.uint8)
            with self.subTest(index=index):
                np.testing.assert_array_equal(encoding.encode(encoding.decode(bits)), bits)

    def test_quantization_error_is_at_most_half_a_grid_step(self):
        encoding = BinaryEncoding(((-5.12, 5.12), (-2.048, 2.048)))
        rng = np.random.default_rng(42)
        for point in rng.uniform([-5.12, -2.048], [5.12, 2.048], size=(100, 2)):
            decoded = encoding.decode(encoding.encode(point))
            self.assertTrue(np.all(np.abs(decoded - point) <= encoding.resolution / 2 + 1e-12))
            self.assertTrue(np.all(decoded >= [-5.12, -2.048]))
            self.assertTrue(np.all(decoded <= [5.12, 2.048]))

    def test_one_and_32_bit_coordinates(self):
        for bits in (1, 32):
            with self.subTest(bits=bits):
                encoding = BinaryEncoding(((0.0, 1.0),), bits_per_variable=bits)
                np.testing.assert_array_equal(encoding.decode(np.zeros(bits)), [0.0])
                np.testing.assert_array_equal(encoding.decode(np.ones(bits)), [1.0])
                chromosome = encoding.encode([0.75])
                self.assertEqual(chromosome.dtype, np.dtype(np.uint8))
                self.assertEqual(chromosome.shape, (bits,))
                np.testing.assert_array_equal(
                    encoding.encode(encoding.decode(chromosome)), chromosome
                )

    def test_default_grid_can_approximate_but_not_represent_zero(self):
        encoding = BinaryEncoding(((-5.12, 5.12),))
        point = encoding.decode(encoding.encode([0.0]))
        self.assertNotEqual(point[0], 0.0)
        self.assertLessEqual(abs(point[0]), encoding.resolution[0] / 2 + 1e-12)

    def test_bounds_inputs_and_outputs_are_independent(self):
        bounds = np.array([[0.0, 3.0]])
        encoding = BinaryEncoding(bounds, bits_per_variable=2)
        bounds[:] = 100
        chromosome = np.array([0, 1], dtype=np.uint8)
        chromosome.setflags(write=False)
        candidate = np.array([1.0])
        candidate.setflags(write=False)
        point = encoding.decode(chromosome)
        encoded = encoding.encode(candidate)
        self.assertEqual(point.dtype, np.dtype(np.float64))
        self.assertFalse(np.shares_memory(point, chromosome))
        self.assertFalse(np.shares_memory(encoded, candidate))
        point[:] = -100
        encoded[:] = 0
        resolution = encoding.resolution
        resolution[:] = 100
        np.testing.assert_array_equal(encoding.decode(chromosome), [1.0])
        np.testing.assert_array_equal(encoding.resolution, [1.0])

    def test_reject_invalid_bounds(self):
        cases = (
            (), (0, 1), ((0, 1, 2),), ((0, 0),), ((1, -1),),
            ((0, np.inf),), ((np.nan, 1),), ((-1e308, 1e308),),
            (("0", "1"),), ((0j, 1j),), ((0, 1), (0,)),
        )
        for bounds in cases:
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    BinaryEncoding(bounds)

    def test_reject_invalid_bit_counts(self):
        for bits in (0, -1, 33, 1.5, True, "16", None):
            with self.subTest(bits=bits):
                with self.assertRaises(ValueError):
                    BinaryEncoding(((0, 1),), bits_per_variable=bits)

    def test_reject_invalid_chromosomes_before_integer_conversion(self):
        encoding = BinaryEncoding(((0, 1),), bits_per_variable=2)
        cases = ([0], [0, 1, 0], [[0, 1]], [0, 256], [-1, 1], [0, 0.5],
                 [0, np.nan], [0, np.inf], ["0", "1"], [0j, 1j])
        for chromosome in cases:
            with self.subTest(chromosome=chromosome):
                with self.assertRaises(ValueError):
                    encoding.decode(chromosome)

    def test_reject_invalid_or_out_of_bounds_candidates(self):
        encoding = BinaryEncoding(((0, 1),))
        for candidate in ([], [0, 1], [[0]], 0, [-0.01], [1.01], [np.inf],
                          [np.nan], ["0"], [0j]):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    encoding.encode(candidate)


if __name__ == "__main__":
    unittest.main()
