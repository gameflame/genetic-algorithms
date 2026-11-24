"""Controlled engine tests; operator doubles here are not GA implementations."""

from dataclasses import replace
import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ga import (
    GAConfig,
    ProblemSpec,
    RunResult,
    evaluate_population,
    make_benchmark_problem,
    make_discrete_problem,
    run_ga,
    tournament_select,
)


def copy_pair_test_double(parent_a, parent_b, rng):
    return parent_a.copy(), parent_b.copy()


def identity_mutation_test_double(child, mutation_rate, rng):
    return child.copy()


def run_fixed_population(population, config, *, crossover=copy_pair_test_double,
                         mutation=identity_mutation_test_double, problem=None):
    problem = problem or make_discrete_problem(chromosome_length=population.shape[1])
    with patch("ga.engine.initialize_population", return_value=population.copy()):
        return run_ga(problem, config, crossover=crossover, mutation=mutation)


class GenerationLoopTests(unittest.TestCase):
    def setUp(self):
        # Costs are 1, 3, 4, 3. The first row is the unique starting elite.
        self.population = np.array(
            [[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.uint8
        )
        self.config = GAConfig(mutation_rate=0.125, population_size=4,
                               generations=3, tournament_size=2, crossover_probability=1.0)

    def test_zero_generations_evaluates_only_initial_population(self):
        crossover, mutation = Mock(), Mock()
        result = run_fixed_population(self.population, replace(self.config, generations=0),
                                      crossover=crossover, mutation=mutation)
        np.testing.assert_array_equal(result.best_chromosome, self.population[0])
        self.assertEqual(result.best_cost, 1.0)
        self.assertEqual(result.evaluation_count, 4)
        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.history[0].generation, 0)
        self.assertEqual(result.history[0].mean_cost, 2.75)
        crossover.assert_not_called()
        mutation.assert_not_called()

    def test_elite_survives_when_every_offspring_is_worse(self):
        def zero_offspring_test_double(child, mutation_rate, rng):
            return np.zeros_like(child)

        with patch("ga.engine.evaluate_population", wraps=evaluate_population) as evaluate:
            result = run_fixed_population(self.population, self.config,
                                          mutation=zero_offspring_test_double)
        self.assertEqual(result.best_cost, 1.0)
        np.testing.assert_array_equal(result.best_chromosome, self.population[0])
        self.assertEqual([record.generation for record in result.history], [0, 1, 2, 3])
        self.assertEqual([record.cumulative_evaluations for record in result.history], [4, 7, 10, 13])
        self.assertEqual([record.mean_cost for record in result.history], [2.75, 3.25, 3.25, 3.25])
        self.assertEqual([record.best_so_far_cost for record in result.history], [1.0] * 4)
        self.assertEqual(result.evaluation_count, 13)
        self.assertEqual([len(call.args[0]) for call in evaluate.call_args_list], [4, 3, 3, 3])

    def test_better_offspring_replaces_elite_and_becomes_best_result(self):
        def optimal_offspring_test_double(child, mutation_rate, rng):
            return np.ones_like(child)

        result = run_fixed_population(self.population, self.config,
                                      mutation=optimal_offspring_test_double)
        self.assertEqual(result.best_cost, 0.0)
        np.testing.assert_array_equal(result.best_chromosome, [1, 1, 1, 1])
        self.assertEqual([record.best_so_far_cost for record in result.history], [1, 0, 0, 0])
        # Reaching a known optimum does not bypass the configured generation limit.
        self.assertEqual(len(result.history), 4)

    def test_population_size_and_operator_counts_for_odd_even_and_minimum_sizes(self):
        for size in (2, 3, 4, 5):
            with self.subTest(size=size):
                initial = np.zeros((size, 4), dtype=np.uint8)
                config = replace(self.config, population_size=size,
                                 tournament_size=min(3, size), generations=2)
                crossover = Mock(side_effect=copy_pair_test_double)
                mutation = Mock(side_effect=identity_mutation_test_double)
                with patch("ga.engine.tournament_select", wraps=tournament_select) as select:
                    result = run_fixed_population(initial, config, crossover=crossover,
                                                  mutation=mutation)
                for call in select.call_args_list:
                    self.assertEqual(call.args[0].shape, (size, 4))
                    self.assertEqual(call.args[1].shape, (size,))
                self.assertEqual(mutation.call_count, 2 * (size - 1))
                self.assertEqual(crossover.call_count, 2 * math.ceil((size - 1) / 2))
                self.assertEqual(result.evaluation_count, size + 2 * (size - 1))

    def test_crossover_probability_zero_skips_crossover_but_still_calls_mutation(self):
        crossover = Mock(side_effect=AssertionError("Crossover should be skipped"))
        mutation = Mock(side_effect=identity_mutation_test_double)
        result = run_fixed_population(
            self.population, replace(self.config, crossover_probability=0.0, mutation_rate=0.0),
            crossover=crossover, mutation=mutation,
        )
        crossover.assert_not_called()
        self.assertEqual(mutation.call_count, 9)
        self.assertTrue(all(call.args[1] == 0.0 for call in mutation.call_args_list))
        self.assertEqual(result.best_cost, 1.0)

    def test_one_generator_is_shared_across_initialization_and_operators(self):
        crossover = Mock(side_effect=copy_pair_test_double)
        mutation = Mock(side_effect=identity_mutation_test_double)
        problem = make_discrete_problem(chromosome_length=4)
        with patch("ga.engine.initialize_population", return_value=self.population.copy()) as initialize:
            run_ga(problem, self.config, crossover=crossover, mutation=mutation)
        rng = initialize.call_args.args[2]
        self.assertIsInstance(rng, np.random.Generator)
        self.assertTrue(all(call.args[2] is rng for call in crossover.call_args_list))
        self.assertTrue(all(call.args[2] is rng for call in mutation.call_args_list))
        self.assertTrue(all(call.args[1] == 0.125 for call in mutation.call_args_list))

    def test_mean_cost_handles_large_finite_costs_and_zero_costs(self):
        for cost in (0.0, 1e308, -1e308):
            with self.subTest(cost=cost):
                problem = ProblemSpec("constant", 4, lambda bits: bits.astype(float), lambda x: cost)
                result = run_fixed_population(self.population, self.config, problem=problem)
                self.assertTrue(all(record.mean_cost == cost for record in result.history))


class OperatorBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.population = np.array([[1, 1, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.uint8)
        self.config = GAConfig(mutation_rate=0.1, population_size=3,
                               generations=1, tournament_size=2, crossover_probability=1.0)

    def test_parent_mutation_cannot_corrupt_population_or_elite(self):
        original = self.population.copy()

        def modifies_parents_test_double(parent_a, parent_b, rng):
            parent_a[:] = 0
            parent_b[:] = 0
            return parent_a, parent_b

        result = run_fixed_population(self.population, self.config,
                                      crossover=modifies_parents_test_double)
        np.testing.assert_array_equal(self.population, original)
        np.testing.assert_array_equal(result.best_chromosome, original[0])
        self.assertEqual(result.best_cost, 1.0)
        self.assertAlmostEqual(result.history[-1].mean_cost, 3.0)

    def test_crossover_aliases_are_copied_before_mutation(self):
        shared = np.array([0, 1, 0, 1], dtype=np.uint8)
        calls = 0

        def alias_pair_test_double(parent_a, parent_b, rng):
            return shared, shared

        def modifies_child_test_double(child, mutation_rate, rng):
            nonlocal calls
            child[:] = 1 if calls == 0 else 0
            calls += 1
            return child

        result = run_fixed_population(self.population, self.config,
                                      crossover=alias_pair_test_double,
                                      mutation=modifies_child_test_double)
        np.testing.assert_array_equal(shared, [0, 1, 0, 1])
        np.testing.assert_array_equal(result.best_chromosome, [1, 1, 1, 1])
        self.assertAlmostEqual(result.history[-1].mean_cost, 5.0 / 3.0)

    def test_reject_invalid_crossover_outputs(self):
        cases = (None, (), ([0, 0, 0, 0],), ([0] * 4,) * 3,
                 np.zeros((2, 4)), ([0, 0], [0] * 4), ([0, 0, 0, 256], [0] * 4))
        for output in cases:
            with self.subTest(output=output):
                crossover = Mock(return_value=output)
                with self.assertRaisesRegex(ValueError, "Crossover at generation 1"):
                    run_fixed_population(self.population, self.config, crossover=crossover)

    def test_reject_invalid_second_crossover_child_even_when_it_would_be_unused(self):
        initial = self.population[:2]
        config = replace(self.config, population_size=2)
        crossover = Mock(return_value=(np.zeros(4), [0, 0, 0, 256]))
        with self.assertRaisesRegex(ValueError, "Crossover at generation 1"):
            run_fixed_population(initial, config, crossover=crossover)

    def test_reject_invalid_mutation_outputs(self):
        for output in (None, [0], [[0] * 4], [0, 0, 0, 256], [0, 0, 0, np.nan]):
            with self.subTest(output=output):
                mutation = Mock(return_value=output)
                with self.assertRaisesRegex(ValueError, "Mutation at generation 1, offspring 0"):
                    run_fixed_population(self.population, self.config, mutation=mutation)

    def test_operator_value_errors_keep_their_cause_and_context(self):
        mutation = Mock(side_effect=ValueError("Test callback failure"))
        with self.assertRaisesRegex(ValueError, "offspring 0: Test callback failure") as error:
            run_fixed_population(self.population, self.config, mutation=mutation)
        self.assertIsInstance(error.exception.__cause__, ValueError)

    def test_operators_are_required_even_for_zero_generations(self):
        config = replace(self.config, generations=0)
        problem = make_discrete_problem(chromosome_length=4)
        for crossover, mutation in ((None, identity_mutation_test_double),
                                    (copy_pair_test_double, None)):
            with self.subTest(crossover=crossover, mutation=mutation):
                with patch("ga.engine.initialize_population") as initialize:
                    with self.assertRaisesRegex(ValueError, "supplied callables"):
                        run_ga(problem, config, crossover=crossover, mutation=mutation)
                    initialize.assert_not_called()


class EngineValidationAndResultTests(unittest.TestCase):
    def test_reject_invalid_configuration_before_initialization(self):
        baseline = GAConfig(mutation_rate=0.1, population_size=4, tournament_size=2)
        invalid = {
            "population_size": (0, 1, True, 4.5),
            "generations": (-1, True, 1.5),
            "tournament_size": (0, 5, True, 1.5),
            "seed": (-1, True, 2.5),
            "crossover_probability": (-0.1, 1.1, np.inf, np.nan, "0.5", [0.5]),
            "mutation_rate": (-0.1, 1.1, np.inf, np.nan, "0.5", [0.5]),
        }
        problem = make_discrete_problem()
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with patch("ga.engine.initialize_population") as initialize:
                        with self.assertRaises(ValueError):
                            run_ga(problem, replace(baseline, **{field: value}),
                                   crossover=copy_pair_test_double,
                                   mutation=identity_mutation_test_double)
                        initialize.assert_not_called()

    def test_reject_wrong_config_or_problem_type(self):
        for problem, config in ((make_discrete_problem(), None),
                                (None, GAConfig(mutation_rate=0.1))):
            with self.subTest(problem=problem, config=config):
                with self.assertRaises(ValueError):
                    run_ga(problem, config, crossover=copy_pair_test_double,
                           mutation=identity_mutation_test_double)

    def test_numpy_scalar_settings_are_normalized(self):
        config = GAConfig(mutation_rate=np.float32(0.125), population_size=np.int64(4),
                          tournament_size=np.int64(2), generations=np.int64(0), seed=np.int64(42))
        result = run_ga(make_discrete_problem(), config, crossover=copy_pair_test_double,
                        mutation=identity_mutation_test_double)
        self.assertIs(type(result.config.seed), int)
        self.assertIs(type(result.config.population_size), int)
        self.assertIs(type(result.config.mutation_rate), float)

    def test_all_benchmarks_return_consistent_results(self):
        for name in ("sphere", "rosenbrock", "himmelblau"):
            with self.subTest(name=name):
                problem = make_benchmark_problem(name)
                config = GAConfig(mutation_rate=0.0, population_size=5, generations=2)
                result = run_ga(problem, config, crossover=copy_pair_test_double,
                                mutation=identity_mutation_test_double)
                self.assertIsInstance(result, RunResult)
                self.assertEqual(problem.objective(result.best_solution), result.best_cost)
                self.assertEqual(result.evaluation_count, 13)
                self.assertEqual(len(result.history), 3)
                self.assertTrue(all(a.best_so_far_cost >= b.best_so_far_cost
                                    for a, b in zip(result.history, result.history[1:])))
                self.assertEqual(result.crossover_name, "copy_pair_test_double")
                self.assertEqual(result.mutation_name, "identity_mutation_test_double")
                self.assertEqual(result.numpy_version, np.__version__)
                self.assertTrue(result.python_version)
                self.assertTrue(np.isfinite(result.elapsed_seconds))
                self.assertGreaterEqual(result.elapsed_seconds, 0.0)

    def test_result_arrays_and_nested_metadata_are_independent(self):
        shared_point = np.array([0.0, 1.0])
        problem = ProblemSpec("shared-decode", 2, lambda bits: shared_point,
                              lambda point: float(point.sum()), {"nested": [1, 2]})
        config = GAConfig(mutation_rate=0.0, population_size=2, tournament_size=2, generations=1)
        initial = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        result = run_fixed_population(initial, config, problem=problem)
        self.assertFalse(np.shares_memory(result.best_solution, shared_point))
        self.assertFalse(np.shares_memory(result.best_chromosome, initial))
        result.best_solution[:] = -1
        result.best_chromosome[:] = 1
        result.problem_metadata["nested"][0] = 99
        np.testing.assert_array_equal(shared_point, [0.0, 1.0])
        np.testing.assert_array_equal(initial, [[0, 0], [1, 1]])
        self.assertEqual(problem.metadata["nested"], [1, 2])


if __name__ == "__main__":
    unittest.main()
