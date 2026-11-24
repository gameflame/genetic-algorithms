"""Representation-to-objective integration for continuous and discrete problems."""

import unittest

import numpy as np

from ga import initialize_population, make_benchmark_problem, make_discrete_problem


class ProblemSpecificationTests(unittest.TestCase):
    def test_default_benchmarks_decode_upper_bounds_and_evaluate(self):
        cases = (
            ("sphere", 5.12, 52.4288),
            ("rosenbrock", 2.048, 461.7603900416),
            ("himmelblau", 5.0, 890.0),
        )
        for name, upper, cost in cases:
            with self.subTest(name=name):
                problem = make_benchmark_problem(name)
                self.assertEqual(problem.chromosome_length, 32)
                point = problem.decode(np.ones(32, dtype=np.uint8))
                np.testing.assert_array_equal(point, [upper, upper])
                self.assertAlmostEqual(problem.objective(point), cost)

    def test_higher_dimensions_and_custom_precision(self):
        for name, dimensions in (("sphere", 1), ("sphere", 5), ("rosenbrock", 3)):
            with self.subTest(name=name, dimensions=dimensions):
                problem = make_benchmark_problem(name, dimensions=dimensions, bits_per_variable=8)
                self.assertEqual(problem.chromosome_length, dimensions * 8)
                point = problem.decode(np.zeros(problem.chromosome_length, dtype=np.uint8))
                self.assertEqual(point.shape, (dimensions,))
                self.assertTrue(np.isfinite(problem.objective(point)))

    def test_initial_populations_connect_to_all_objectives(self):
        problems = [make_benchmark_problem(name) for name in ("sphere", "rosenbrock", "himmelblau")]
        problems.append(make_discrete_problem())
        for problem in problems:
            with self.subTest(name=problem.name):
                population = initialize_population(5, problem.chromosome_length, np.random.default_rng(42))
                for chromosome in population:
                    cost = problem.objective(problem.decode(chromosome))
                    self.assertTrue(np.isfinite(cost))
                    self.assertGreaterEqual(cost, 0.0)

    def test_metadata_changes_do_not_modify_the_decoder(self):
        problem = make_benchmark_problem("sphere")
        problem.metadata["bounds"][0][0] = 999
        np.testing.assert_array_equal(problem.decode(np.zeros(32)), [-5.12, -5.12])

    def test_discrete_decoder_preserves_bits_without_aliasing(self):
        problem = make_discrete_problem(chromosome_length=4)
        chromosome = np.array([0, 1, 0, 1], dtype=np.uint8)
        chromosome.setflags(write=False)
        candidate = problem.decode(chromosome)
        np.testing.assert_array_equal(candidate, chromosome)
        self.assertEqual(candidate.dtype, np.dtype(np.float64))
        self.assertFalse(np.shares_memory(candidate, chromosome))
        self.assertEqual(problem.objective(candidate), 2.0)
        candidate[:] = 1
        self.assertEqual(problem.objective(candidate), 0.0)
        self.assertEqual(problem.objective(problem.decode(chromosome)), 2.0)

    def test_discrete_decoder_rejects_wrong_length_and_non_binary_values(self):
        problem = make_discrete_problem(chromosome_length=2)
        for chromosome in ([0], [0, 2], [[0, 1]], [0, np.nan]):
            with self.subTest(chromosome=chromosome):
                with self.assertRaises(ValueError):
                    problem.decode(chromosome)

    def test_reject_invalid_problem_settings(self):
        for name in ("unknown", "Sphere", [], None):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    make_benchmark_problem(name)
        for name, dimensions in (("sphere", 0), ("rosenbrock", 1), ("himmelblau", 3),
                                 ("sphere", 2.5), ("sphere", True)):
            with self.subTest(name=name, dimensions=dimensions):
                with self.assertRaises(ValueError):
                    make_benchmark_problem(name, dimensions=dimensions)
        for length in (0, -1, True, 1.5):
            with self.subTest(length=length):
                with self.assertRaises(ValueError):
                    make_discrete_problem(chromosome_length=length)


if __name__ == "__main__":
    unittest.main()
