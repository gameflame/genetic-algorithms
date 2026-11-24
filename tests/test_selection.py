"""Tournament membership, minimization, ties, and parent-copy checks."""

import unittest

import numpy as np

from ga import evaluate_population, make_discrete_problem, tournament_select


class FixedTournamentGenerator(np.random.Generator):
    """Test-only generator controlling tournament membership and order."""

    def __init__(self, contestants):
        super().__init__(np.random.PCG64(42))
        self.contestants = np.array(contestants, dtype=np.int64)
        self.request = None

    def choice(self, a, size=None, replace=True):
        self.request = (a, size, replace)
        return self.contestants.copy()


class TournamentSelectionTests(unittest.TestCase):
    def setUp(self):
        self.population = np.array(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]],
            dtype=np.uint8,
        )

    def test_selects_lowest_cost_among_sampled_contestants(self):
        # Global best row 0 is excluded; sampled row 3 must win.
        rng = FixedTournamentGenerator([1, 3, 4])
        parent = tournament_select(self.population, [-10.0, 5.0, 20.0, -3.0, 2.0], 3, rng)
        np.testing.assert_array_equal(parent, self.population[3])
        self.assertEqual(rng.request, (5, 3, False))

    def test_full_population_tournament_selects_global_best(self):
        parent = tournament_select(
            self.population, [0.0, -1.0, -5.0, 2.0, 10.0], 5, np.random.default_rng(42)
        )
        np.testing.assert_array_equal(parent, self.population[2])

    def test_single_contestant_wins_regardless_of_global_rank(self):
        parent = tournament_select(
            self.population, [0.0, 1.0, 9.0, 2.0, 3.0], 1, FixedTournamentGenerator([2])
        )
        np.testing.assert_array_equal(parent, self.population[2])

    def test_ties_choose_first_sampled_contestant(self):
        parent = tournament_select(
            self.population, [8.0, 2.0, 6.0, 2.0, 2.0], 3, FixedTournamentGenerator([4, 1, 3])
        )
        np.testing.assert_array_equal(parent, self.population[4])

    def test_returned_parent_is_an_independent_uint8_copy(self):
        population = self.population.copy()
        population.setflags(write=False)
        costs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        costs.setflags(write=False)
        parent = tournament_select(population, costs, 5, np.random.default_rng(42))
        self.assertEqual(parent.dtype, np.dtype(np.uint8))
        self.assertFalse(np.shares_memory(parent, population))
        parent[:] = 1
        np.testing.assert_array_equal(population, self.population)
        np.testing.assert_array_equal(costs, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_seeded_selection_sequence_is_reproducible(self):
        first_rng, second_rng = np.random.default_rng(42), np.random.default_rng(42)
        costs = [5.0, 4.0, 3.0, 2.0, 1.0]
        first = [tournament_select(self.population, costs, 3, first_rng) for _ in range(10)]
        second = [tournament_select(self.population, costs, 3, second_rng) for _ in range(10)]
        np.testing.assert_array_equal(first, second)

    def test_evaluation_and_selection_work_together(self):
        population = [[0, 0, 0], [1, 0, 1], [1, 1, 1]]
        costs = evaluate_population(population, make_discrete_problem(chromosome_length=3))
        parent = tournament_select(population, costs, 3, np.random.default_rng(42))
        np.testing.assert_array_equal(parent, [1, 1, 1])

    def test_reject_invalid_population(self):
        for population in ([], [0, 1], [[0, 1]], [[], []], [[0, 1], [0]],
                           [[0, 256], [1, 0]], [[0, 0.5], [1, 0]]):
            with self.subTest(population=population):
                with self.assertRaises(ValueError):
                    tournament_select(population, [0.0, 1.0], 1, np.random.default_rng(42))

    def test_reject_wrong_shape_non_numeric_or_non_finite_costs(self):
        cases = ([0, 1], [[0, 1, 2, 3, 4]], 1.0, [0, 1, 2, 3, np.nan],
                 [0, 1, 2, 3, np.inf], [0, 1, 2, 3, -np.inf], ["0"] * 5, [1j] * 5)
        for costs in cases:
            with self.subTest(costs=costs):
                with self.assertRaises(ValueError):
                    tournament_select(self.population, costs, 3, np.random.default_rng(42))

    def test_reject_invalid_tournament_size(self):
        for size in (0, -1, 6, 2.5, True, "3"):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    tournament_select(self.population, [0, 1, 2, 3, 4], size, np.random.default_rng(42))

    def test_reject_missing_or_legacy_generator(self):
        for rng in (None, 42, np.random.RandomState(42)):
            with self.subTest(rng=rng):
                with self.assertRaisesRegex(ValueError, "Generator"):
                    tournament_select(self.population, [0, 1, 2, 3, 4], 3, rng)


if __name__ == "__main__":
    unittest.main()
