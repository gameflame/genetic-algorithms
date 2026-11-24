"""Known population costs and decoder/objective boundary checks."""

import unittest

import numpy as np

from ga import (
    BinaryEncoding,
    ProblemSpec,
    evaluate_population,
    initialize_population,
    make_benchmark_problem,
    make_discrete_problem,
    sphere,
)


class PopulationEvaluationTests(unittest.TestCase):
    def test_discrete_costs_follow_population_row_order(self):
        problem = make_discrete_problem(chromosome_length=4)
        population = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1]], dtype=np.uint8)
        costs = evaluate_population(population, problem)
        np.testing.assert_array_equal(costs, [4.0, 0.0, 2.0])
        self.assertEqual(costs.shape, (3,))
        self.assertEqual(costs.dtype, np.dtype(np.float64))

    def test_custom_sphere_encoding_has_known_costs(self):
        encoding = BinaryEncoding(((0.0, 3.0), (0.0, 3.0)), bits_per_variable=2)
        problem = ProblemSpec("sphere-small-grid", 4, encoding.decode, sphere)
        costs = evaluate_population([[0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1]], problem)
        np.testing.assert_array_equal(costs, [0.0, 5.0, 18.0])

    def test_all_benchmark_factories_work_with_seeded_populations(self):
        for name in ("sphere", "rosenbrock", "himmelblau"):
            with self.subTest(name=name):
                problem = make_benchmark_problem(name)
                population = initialize_population(5, problem.chromosome_length, np.random.default_rng(42))
                costs = evaluate_population(population, problem)
                self.assertEqual(costs.shape, (5,))
                self.assertTrue(np.all(np.isfinite(costs)))
                self.assertTrue(np.all(costs >= 0))

    def test_single_offspring_batch_is_supported(self):
        costs = evaluate_population([[1, 0]], make_discrete_problem(chromosome_length=2))
        np.testing.assert_array_equal(costs, [1.0])

    def test_each_chromosome_is_decoded_and_evaluated_once(self):
        decoded_rows, evaluated_candidates = [], []

        def decode(chromosome):
            decoded_rows.append(chromosome.copy())
            return chromosome.astype(np.float64)

        def objective(candidate):
            evaluated_candidates.append(candidate.copy())
            return float(candidate.sum())

        population = np.array([[0, 1], [1, 1], [0, 0]], dtype=np.uint8)
        costs = evaluate_population(population, ProblemSpec("count-ones", 2, decode, objective))
        np.testing.assert_array_equal(decoded_rows, population)
        np.testing.assert_array_equal(evaluated_candidates, population)
        np.testing.assert_array_equal(costs, [1.0, 2.0, 0.0])

    def test_decoder_and_objective_cannot_modify_the_population(self):
        def decode(chromosome):
            chromosome[:] = 1
            return chromosome.astype(np.float64)

        def objective(candidate):
            candidate[:] = 0
            return 0.0

        population = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        original = population.copy()
        population.setflags(write=False)
        evaluate_population(population, ProblemSpec("mutating-callbacks", 2, decode, objective))
        np.testing.assert_array_equal(population, original)

    def test_reject_invalid_populations(self):
        problem = make_discrete_problem(chromosome_length=2)
        cases = ([], [0, 1], [[]], [[0]], [[0, 1, 0]], [[0, 1], [0]],
                 [[0, 256]], [[-1, 1]], [[0, 0.5]], [[0, np.nan]], [[0, np.inf]],
                 [["0", "1"]], [[0j, 1j]])
        for population in cases:
            with self.subTest(population=population):
                with self.assertRaises(ValueError):
                    evaluate_population(population, problem)

    def test_reject_invalid_problem_specifications(self):
        cases = (
            None,
            ProblemSpec("bad-length", 0, lambda x: x, sphere),
            ProblemSpec("bad-decoder", 2, None, sphere),
            ProblemSpec("bad-objective", 2, lambda x: x, None),
        )
        for problem in cases:
            with self.subTest(problem=problem):
                with self.assertRaises(ValueError):
                    evaluate_population([[0, 1]], problem)

    def test_reject_invalid_decoded_candidates(self):
        for decoded in ([], 1.0, [[1.0, 2.0]], [0.0, np.inf], [0.0, np.nan], ["0"], [1j]):
            with self.subTest(decoded=decoded):
                problem = ProblemSpec("bad-decoding", 2, lambda x: decoded, lambda x: 0.0)
                with self.assertRaisesRegex(ValueError, "population row 0"):
                    evaluate_population([[0, 1]], problem)

    def test_reject_non_scalar_or_invalid_objective_costs(self):
        for cost in (np.nan, np.inf, -np.inf, [1.0], [[1.0]], "1", 1j, None):
            with self.subTest(cost=cost):
                problem = ProblemSpec("bad-cost", 2, lambda x: x, lambda x: cost)
                with self.assertRaisesRegex(ValueError, "population row 0"):
                    evaluate_population([[0, 1]], problem)

    def test_error_identifies_later_population_row(self):
        problem = ProblemSpec("bad-second-row", 1, lambda x: x,
                              lambda x: 0.0 if x[0] == 0 else np.nan)
        with self.assertRaisesRegex(ValueError, "population row 1"):
            evaluate_population([[0], [1]], problem)


if __name__ == "__main__":
    unittest.main()
