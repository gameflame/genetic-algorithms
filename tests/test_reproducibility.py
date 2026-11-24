"""Full-run replay and result integrity with random-consuming test fixtures.

These fixtures exercise the RNG contract without supplying the group's real
crossover or mutation implementations. Their outputs are not GA experiments.
"""

from copy import deepcopy
from dataclasses import fields, replace
import random
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ga import GAConfig, RunResult, make_benchmark_problem, make_discrete_problem, run_ga


class RecordingOperatorDoubles:
    """Record supplied-generator draws; internal trace state never affects output."""

    def __init__(self):
        self.trace = []

    def crossover(self, parent_a, parent_b, rng):
        self.trace.append(("crossover", tuple(float(value) for value in rng.random(3))))
        return parent_a.copy(), parent_b.copy()

    def mutation(self, child, mutation_rate, rng):
        # Random replacement is a test fixture, not a proposed mutation operator.
        output = rng.integers(0, 2, size=child.shape, dtype=np.uint8)
        self.trace.append(("mutation", mutation_rate, tuple(int(bit) for bit in output)))
        return output


def recorded_run(problem, config):
    operators = RecordingOperatorDoubles()
    result = run_ga(problem, config, crossover=operators.crossover, mutation=operators.mutation)
    return result, operators.trace


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.config = GAConfig(mutation_rate=0.125, population_size=7,
                               generations=4, crossover_probability=0.65, seed=42)

    def assert_same_result_except_time(self, first, second):
        # Compare every public result field, including any future metadata fields.
        for field in fields(RunResult):
            if field.name == "elapsed_seconds":
                continue
            left, right = getattr(first, field.name), getattr(second, field.name)
            with self.subTest(result_field=field.name):
                if isinstance(left, np.ndarray):
                    np.testing.assert_array_equal(left, right)
                else:
                    self.assertEqual(left, right)

    def test_same_seed_replays_complete_results_and_operator_draws(self):
        problems = [make_benchmark_problem(name) for name in ("sphere", "rosenbrock", "himmelblau")]
        problems.append(make_discrete_problem())
        for problem in problems:
            with self.subTest(problem=problem.name):
                first, first_trace = recorded_run(problem, self.config)
                second, second_trace = recorded_run(problem, self.config)
                self.assert_same_result_except_time(first, second)
                self.assertEqual(first_trace, second_trace)
                self.assertTrue(any(event[0] == "crossover" for event in first_trace))
                self.assertTrue(any(event[0] == "mutation" for event in first_trace))

    def test_unrelated_run_does_not_change_replay(self):
        problem = make_benchmark_problem("sphere")
        first, first_trace = recorded_run(problem, self.config)
        recorded_run(problem, replace(self.config, seed=99, generations=3))
        replay, replay_trace = recorded_run(problem, self.config)
        self.assert_same_result_except_time(first, replay)
        self.assertEqual(first_trace, replay_trace)

    def test_different_seeds_change_the_random_stream(self):
        problem = make_discrete_problem()
        _, first_trace = recorded_run(problem, self.config)
        _, second_trace = recorded_run(problem, replace(self.config, seed=43))
        self.assertNotEqual(first_trace, second_trace)
        # Different streams need not produce different final best solutions.

    def test_independence_from_and_preservation_of_global_random_state(self):
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        try:
            problem = make_benchmark_problem("sphere")
            first, first_trace = recorded_run(problem, self.config)
            np.random.seed(123)
            np.random.random(100)
            random.seed(456)
            random.random()
            disturbed_numpy_state = np.random.get_state()
            disturbed_python_state = random.getstate()
            second, second_trace = recorded_run(problem, self.config)
            self.assert_same_result_except_time(first, second)
            self.assertEqual(first_trace, second_trace)
            for actual, expected in zip(np.random.get_state(), disturbed_numpy_state):
                np.testing.assert_equal(actual, expected)
            self.assertEqual(random.getstate(), disturbed_python_state)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)

    def test_zero_generations_and_probability_endpoints_replay(self):
        problem = make_discrete_problem(chromosome_length=8)
        for generations, probability in ((0, 0.65), (3, 0.0), (3, 1.0)):
            with self.subTest(generations=generations, probability=probability):
                config = replace(self.config, generations=generations,
                                 crossover_probability=probability, population_size=8)
                first, first_trace = recorded_run(problem, config)
                second, second_trace = recorded_run(problem, config)
                self.assert_same_result_except_time(first, second)
                self.assertEqual(first_trace, second_trace)
                if generations == 0:
                    self.assertEqual(first_trace, [])

    def test_zero_and_large_integer_seeds_replay(self):
        problem = make_discrete_problem(chromosome_length=8)
        for seed in (0, 2**80 + 17):
            with self.subTest(seed=seed):
                config = replace(self.config, seed=seed, generations=2)
                first, first_trace = recorded_run(problem, config)
                second, second_trace = recorded_run(problem, config)
                self.assert_same_result_except_time(first, second)
                self.assertEqual(first_trace, second_trace)


class ResultIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.config = GAConfig(mutation_rate=0.125, population_size=6,
                               generations=3, crossover_probability=0.65)

    def test_counts_history_and_best_solution_match_actual_evaluations(self):
        problems = [make_benchmark_problem(name) for name in ("sphere", "rosenbrock", "himmelblau")]
        problems.append(make_discrete_problem())
        for problem in problems:
            with self.subTest(problem=problem.name):
                objective_spy = Mock(wraps=problem.objective)
                result, _ = recorded_run(replace(problem, objective=objective_spy), self.config)
                self.assertEqual(result.evaluation_count, objective_spy.call_count)
                self.assertEqual(result.evaluation_count, 21)
                self.assertEqual([entry.cumulative_evaluations for entry in result.history],
                                 [6, 11, 16, 21])
                self.assertEqual([entry.generation for entry in result.history], [0, 1, 2, 3])
                self.assertEqual(result.history[-1].best_so_far_cost, result.best_cost)
                self.assertEqual(problem.objective(result.best_solution), result.best_cost)
                np.testing.assert_array_equal(problem.decode(result.best_chromosome), result.best_solution)
                for entry in result.history:
                    self.assertTrue(np.isfinite(entry.mean_cost))
                    self.assertTrue(np.isfinite(entry.best_so_far_cost))
                    self.assertLessEqual(entry.best_so_far_cost, entry.mean_cost + 1e-12)

    def test_records_actual_rng_algorithm(self):
        # A different injected generator proves metadata is read from the run.
        rng = np.random.Generator(np.random.Philox(self.config.seed))
        with patch("ga.engine.np.random.default_rng", return_value=rng):
            result, _ = recorded_run(make_discrete_problem(), self.config)
        self.assertEqual(result.rng_algorithm, "Philox")

    def test_records_environment_problem_and_operator_metadata(self):
        problem = make_benchmark_problem("rosenbrock", dimensions=3, bits_per_variable=8)
        result, _ = recorded_run(problem, self.config)
        self.assertEqual(result.problem_name, "rosenbrock")
        self.assertEqual(result.problem_metadata["dimensions"], 3)
        self.assertEqual(result.problem_metadata["bits_per_variable"], 8)
        self.assertEqual(len(result.best_chromosome), 24)
        self.assertEqual(result.config, self.config)
        self.assertEqual(result.config.seed, 42)
        self.assertEqual(result.crossover_name, "crossover")
        self.assertEqual(result.mutation_name, "mutation")
        self.assertEqual(result.numpy_version, np.__version__)
        self.assertTrue(result.python_version)
        self.assertTrue(result.rng_algorithm)

    def test_elapsed_seconds_uses_the_clock_and_is_excluded_from_replay(self):
        problem = make_discrete_problem()
        with patch("ga.engine.perf_counter", side_effect=[100.0, 102.25]):
            first, _ = recorded_run(problem, self.config)
        with patch("ga.engine.perf_counter", side_effect=[200.0, 204.5]):
            second, _ = recorded_run(problem, self.config)
        self.assertEqual(first.elapsed_seconds, 2.25)
        self.assertEqual(second.elapsed_seconds, 4.5)
        self.assertEqual(first.history, second.history)
        np.testing.assert_array_equal(first.best_chromosome, second.best_chromosome)

    def test_mutating_one_result_does_not_affect_other_or_future_runs(self):
        problem = make_benchmark_problem("sphere")
        original_metadata = deepcopy(problem.metadata)
        first, _ = recorded_run(problem, self.config)
        second, _ = recorded_run(problem, self.config)
        second_chromosome = second.best_chromosome.copy()
        second_solution = second.best_solution.copy()
        self.assertFalse(np.shares_memory(first.best_chromosome, second.best_chromosome))
        self.assertFalse(np.shares_memory(first.best_solution, second.best_solution))
        first.best_chromosome[:] = 0
        first.best_solution[:] = 999
        first.problem_metadata["bounds"][0][0] = 999
        first.problem_metadata["resolution"][0] = 999
        np.testing.assert_array_equal(second.best_chromosome, second_chromosome)
        np.testing.assert_array_equal(second.best_solution, second_solution)
        self.assertEqual(problem.metadata, original_metadata)
        self.assertEqual(second.problem_metadata, original_metadata)
        replay, _ = recorded_run(problem, self.config)
        np.testing.assert_array_equal(replay.best_chromosome, second_chromosome)
        np.testing.assert_array_equal(replay.best_solution, second_solution)
        self.assertEqual(replay.history, second.history)


if __name__ == "__main__":
    unittest.main()
