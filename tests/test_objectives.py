"""Independent known-value checks and objective input-contract tests."""

import unittest

import numpy as np

from ga import himmelblau, rosenbrock, sphere, zero_count


OBJECTIVES = (sphere, rosenbrock, himmelblau, zero_count)
CONTINUOUS_OBJECTIVES = (sphere, rosenbrock, himmelblau)


class KnownObjectiveValuesTests(unittest.TestCase):
    def test_sphere_known_values(self):
        cases = (([0, 0], 0.0), ([1, 2], 5.0), ([-1, -2, -3], 14.0), ([3], 9.0))
        for point, expected in cases:
            with self.subTest(point=point):
                self.assertEqual(sphere(point), expected)

    def test_sphere_origin_in_multiple_dimensions(self):
        for dimensions in (1, 2, 5):
            with self.subTest(dimensions=dimensions):
                self.assertEqual(sphere(np.zeros(dimensions)), 0.0)

    def test_rosenbrock_known_values(self):
        # The 3D value checks that every adjacent coordinate pair is included.
        cases = (([0, 0], 1.0), ([-1, 1], 4.0), ([2, 2], 401.0), ([1, 2, 3], 201.0))
        for point, expected in cases:
            with self.subTest(point=point):
                self.assertEqual(rosenbrock(point), expected)

    def test_rosenbrock_minimum_in_multiple_dimensions(self):
        for dimensions in (2, 3, 5):
            with self.subTest(dimensions=dimensions):
                self.assertEqual(rosenbrock(np.ones(dimensions)), 0.0)

    def test_himmelblau_known_values(self):
        cases = (([3, 2], 0.0), ([0, 0], 170.0), ([1, 1], 106.0), ([-1, 2], 80.0))
        for point, expected in cases:
            with self.subTest(point=point):
                self.assertEqual(himmelblau(point), expected)

    def test_himmelblau_four_global_minima(self):
        # Three coordinates are rounded to six decimals, so use a tolerance.
        minima = (
            [3.0, 2.0],
            [-2.805118, 3.131312],
            [-3.779310, -3.283186],
            [3.584428, -1.848126],
        )
        for point in minima:
            with self.subTest(point=point):
                self.assertAlmostEqual(himmelblau(point), 0.0, delta=1e-9)

    def test_zero_count_known_values(self):
        cases = (
            ([0, 1, 0, 1, 0], 3.0),
            ([1, 1, 1], 0.0),
            ([0, 0, 0], 3.0),
            ([1], 0.0),
            (np.array([0, 1, 0], dtype=np.uint8), 2.0),
            (np.array([0.0, 1.0]), 1.0),
            (np.array([False, True, False]), 2.0),
        )
        for point, expected in cases:
            with self.subTest(point=point):
                self.assertEqual(zero_count(point), expected)


class ObjectiveContractTests(unittest.TestCase):
    def test_objectives_return_python_floats(self):
        for objective in OBJECTIVES:
            with self.subTest(objective=objective.__name__):
                self.assertIs(type(objective(np.array([0.0, 1.0]))), float)

    def test_sequences_and_numeric_array_dtypes(self):
        points = (
            [0, 1],
            (0.0, 1.0),
            np.array([0, 1], dtype=np.uint8),
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1], dtype=np.float32),
            np.array([0, 1], dtype=np.float64),
        )
        for objective in OBJECTIVES:
            expected = objective([0.0, 1.0])
            for point in points:
                with self.subTest(objective=objective.__name__, point=point):
                    self.assertEqual(objective(point), expected)

    def test_read_only_inputs_are_not_modified(self):
        for objective in OBJECTIVES:
            with self.subTest(objective=objective.__name__):
                point = np.array([0.0, 1.0])
                original = point.copy()
                point.setflags(write=False)
                objective(point)
                np.testing.assert_array_equal(point, original)

    def test_reject_empty_vectors(self):
        for objective in OBJECTIVES:
            with self.subTest(objective=objective.__name__):
                with self.assertRaisesRegex(ValueError, "coordinates"):
                    objective([])

    def test_reject_scalar_and_multidimensional_inputs(self):
        for objective in OBJECTIVES:
            for point in (1.0, np.array(1.0), [[0.0, 1.0]], np.zeros((2, 1))):
                with self.subTest(objective=objective.__name__, point=point):
                    with self.assertRaisesRegex(ValueError, "one-dimensional"):
                        objective(point)

    def test_reject_wrong_function_dimensions(self):
        cases = ((rosenbrock, [1.0]), (himmelblau, [3.0]), (himmelblau, [3.0, 2.0, 1.0]))
        for objective, point in cases:
            with self.subTest(objective=objective.__name__, point=point):
                with self.assertRaisesRegex(ValueError, "coordinates"):
                    objective(point)

    def test_reject_non_finite_coordinates(self):
        for objective in OBJECTIVES:
            for bad_value in (np.nan, np.inf, -np.inf):
                with self.subTest(objective=objective.__name__, value=bad_value):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        objective([0.0, bad_value])

    def test_reject_non_real_or_non_numeric_coordinates(self):
        invalid_points = (["0", "1"], [0j, 1j], [None, 1], [object(), 1])
        for objective in OBJECTIVES:
            for point in invalid_points:
                with self.subTest(objective=objective.__name__, point=point):
                    with self.assertRaisesRegex(ValueError, "real numeric"):
                        objective(point)

    def test_reject_ragged_vectors(self):
        for objective in OBJECTIVES:
            with self.subTest(objective=objective.__name__):
                with self.assertRaises(ValueError):
                    objective([[0], [0, 1]])

    def test_zero_count_rejects_non_binary_coordinates(self):
        for point in ([0, 2], [-1, 1], [0.0, 0.5]):
            with self.subTest(point=point):
                with self.assertRaisesRegex(ValueError, "binary"):
                    zero_count(point)

    def test_reject_objective_overflow_without_runtime_warnings(self):
        # The input is finite, but all three calculations exceed float64 range.
        for objective in CONTINUOUS_OBJECTIVES:
            with self.subTest(objective=objective.__name__):
                with np.errstate(all="raise"):
                    with self.assertRaisesRegex(ValueError, "cost is not finite"):
                        objective([1e308, 1e308])


if __name__ == "__main__":
    unittest.main()
