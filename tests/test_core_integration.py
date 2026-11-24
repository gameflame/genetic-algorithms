"""Core review across configurations and failures, using test-only operators."""

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from ga import GAConfig, ProblemSpec, make_benchmark_problem, make_discrete_problem, run_ga
from tests.test_engine import copy_pair_test_double, identity_mutation_test_double
from tests.test_reproducibility import RecordingOperatorDoubles


class CoreIntegrationTests(TestCase):
    def test_configuration_matrix_preserves_core_invariants(self):
        # Cover the minimum sizes, odd/even populations, both tournament-size
        # extremes, supported precisions, higher dimensions, and 1-bit problems.
        cases = (
            (make_benchmark_problem("sphere", dimensions=1, bits_per_variable=1), 2, 1, 0, 0.0, 0.0),
            (make_benchmark_problem("sphere", dimensions=5, bits_per_variable=32), 3, 3, 3, 0.0, 1.0),
            (make_benchmark_problem("sphere"), 5, 1, 4, 1.0, 0.0),
            (make_benchmark_problem("rosenbrock", bits_per_variable=1), 2, 2, 3, 1.0, 1.0),
            (make_benchmark_problem("rosenbrock", dimensions=5, bits_per_variable=8), 7, 7, 3, 0.5, 0.025),
            (make_benchmark_problem("himmelblau", bits_per_variable=32), 4, 1, 3, 0.0, 0.0),
            (make_discrete_problem(chromosome_length=1), 2, 2, 3, 1.0, 0.0),
            (make_discrete_problem(chromosome_length=7), 5, 5, 3, 0.5, 1.0),
            (make_discrete_problem(chromosome_length=33), 7, 1, 3, 0.5, 1.0 / 33),
        )
        for problem, size, tournament, generations, crossover_rate, mutation_rate in cases:
            with self.subTest(problem=problem.name, length=problem.chromosome_length,
                              size=size, tournament=tournament):
                config = GAConfig(mutation_rate=mutation_rate, population_size=size,
                                  tournament_size=tournament, generations=generations,
                                  crossover_probability=crossover_rate, seed=0)
                operators = RecordingOperatorDoubles()
                objective = Mock(wraps=problem.objective)
                counted_problem = ProblemSpec(problem.name, problem.chromosome_length,
                                              problem.decode, objective, problem.metadata)
                result = run_ga(counted_problem, config, crossover=operators.crossover,
                                mutation=operators.mutation)
                self.assertEqual(result.evaluation_count, size + generations * (size - 1))
                self.assertEqual(objective.call_count, result.evaluation_count)
                self.assertEqual(len(result.history), generations + 1)
                self.assertEqual(result.best_chromosome.shape, (problem.chromosome_length,))
                self.assertEqual(result.best_chromosome.dtype, np.dtype(np.uint8))
                self.assertTrue(np.all((result.best_chromosome == 0) | (result.best_chromosome == 1)))
                np.testing.assert_array_equal(problem.decode(result.best_chromosome), result.best_solution)
                self.assertEqual(problem.objective(result.best_solution), result.best_cost)
                self.assertEqual(result.history[-1].best_so_far_cost, result.best_cost)
                self.assertTrue(all(a.best_so_far_cost >= b.best_so_far_cost
                                    for a, b in zip(result.history, result.history[1:])))
                if "bounds" in result.problem_metadata:
                    bounds = np.array(result.problem_metadata["bounds"])
                    self.assertTrue(np.all(result.best_solution >= bounds[:, 0]))
                    self.assertTrue(np.all(result.best_solution <= bounds[:, 1]))

    def test_default_population_and_generation_budget(self):
        # This is a stability check with copy/identity doubles, not a performance
        # comparison or evidence of the actual operators' optimization quality.
        problem = make_benchmark_problem("sphere")
        objective = Mock(wraps=problem.objective)
        counted_problem = ProblemSpec(problem.name, problem.chromosome_length,
                                      problem.decode, objective, problem.metadata)
        result = run_ga(counted_problem, GAConfig(mutation_rate=1 / 32),
                        crossover=copy_pair_test_double, mutation=identity_mutation_test_double)
        self.assertEqual(result.evaluation_count, 29800)
        self.assertEqual(objective.call_count, 29800)
        self.assertEqual(len(result.history), 301)
        self.assertEqual(result.history[-1].generation, 300)
        self.assertTrue(all(np.isfinite(record.mean_cost) for record in result.history))
        self.assertTrue(all(a.best_so_far_cost >= b.best_so_far_cost
                            for a, b in zip(result.history, result.history[1:])))
        self.assertEqual(problem.objective(result.best_solution), result.best_cost)

    def test_initial_evaluation_failure_identifies_generation_and_stops(self):
        problem = ProblemSpec("invalid-initial-cost", 2, lambda bits: bits,
                              lambda point: np.nan)
        config = GAConfig(mutation_rate=0.0, population_size=3, generations=1)
        crossover, mutation = Mock(), Mock()
        with self.assertRaisesRegex(ValueError, "Evaluation at generation 0:.*population row 0"):
            run_ga(problem, config, crossover=crossover, mutation=mutation)
        crossover.assert_not_called()
        mutation.assert_not_called()

    def test_offspring_evaluation_failure_identifies_generation_and_stops(self):
        def invalid_cost_for_offspring(point):
            return np.nan if np.all(point == 1) else 1.0

        problem = ProblemSpec("invalid-offspring-cost", 2, lambda bits: bits,
                              invalid_cost_for_offspring)
        config = GAConfig(mutation_rate=0.0, population_size=3, generations=2,
                          crossover_probability=1.0)
        mutation = Mock(side_effect=lambda child, rate, rng: np.ones_like(child))
        with patch("ga.engine.initialize_population", return_value=np.zeros((3, 2), dtype=np.uint8)):
            with self.assertRaisesRegex(ValueError, "Evaluation at generation 1:.*population row 0") as error:
                run_ga(problem, config, crossover=copy_pair_test_double, mutation=mutation)
        self.assertIsInstance(error.exception.__cause__, ValueError)
        self.assertEqual(mutation.call_count, 2)

    def test_a_failed_run_does_not_change_a_subsequent_seeded_run(self):
        problem = make_discrete_problem(chromosome_length=8)
        config = GAConfig(mutation_rate=0.125, population_size=5, generations=3)
        before_ops = RecordingOperatorDoubles()
        before = run_ga(problem, config, crossover=before_ops.crossover, mutation=before_ops.mutation)
        bad_mutation = Mock(return_value=np.full(8, 256))
        with self.assertRaises(ValueError):
            run_ga(problem, config, crossover=copy_pair_test_double, mutation=bad_mutation)
        after_ops = RecordingOperatorDoubles()
        after = run_ga(problem, config, crossover=after_ops.crossover, mutation=after_ops.mutation)
        np.testing.assert_array_equal(before.best_chromosome, after.best_chromosome)
        np.testing.assert_array_equal(before.best_solution, after.best_solution)
        self.assertEqual(before.history, after.history)
        self.assertEqual(before_ops.trace, after_ops.trace)

    def test_programming_errors_in_callbacks_are_not_hidden_or_retried(self):
        mutation = Mock(side_effect=TypeError("Test callback programming error"))
        with self.assertRaisesRegex(TypeError, "Test callback programming error"):
            run_ga(make_discrete_problem(), GAConfig(mutation_rate=0.125, generations=1),
                   crossover=copy_pair_test_double, mutation=mutation)
        self.assertEqual(mutation.call_count, 1)
