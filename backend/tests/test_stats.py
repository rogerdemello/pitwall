"""Tests for the statistics that gate published claims.

These functions decide whether a finding gets reported as real. Two of them were
consolidated out of scripts where they had drifted, and `pearson_p` in
particular used to return a Fisher z approximation from a function that computed
a t-statistic and threw it away - so the tests pin both the values and the
guard behaviour that the honesty of the evidence rests on.
"""

from __future__ import annotations

import os
import sys

import pytest
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import stats  # noqa: E402


class TestPearson:
    def test_perfect_correlation(self):
        assert stats.pearson_r([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0

    def test_perfect_anticorrelation(self):
        assert stats.pearson_r([1, 2, 3, 4], [8, 6, 4, 2]) == -1.0

    def test_constant_input_is_none_not_zero(self):
        """Undefined must not be reported as 'no correlation'."""
        assert stats.pearson_r([1, 1, 1, 1], [1, 2, 3, 4]) is None

    def test_too_few_points(self):
        assert stats.pearson_r([1], [2]) is None

    def test_p_matches_scipy(self):
        """The published corpus numbers, against the reference implementation."""
        for r, n in [(0.0428, 1155), (0.1572, 112), (-0.1307, 110), (0.62, 10)]:
            expected = float(2 * scipy_stats.t.sf(
                abs(r) * ((n - 2) / (1 - r * r)) ** 0.5, n - 2))
            assert stats.pearson_p(r, n) == pytest.approx(expected, abs=5e-5)

    def test_headline_null_is_not_significant(self):
        """r=0.043 on n=1155 must not come out significant."""
        assert stats.pearson_p(0.0428, 1155) > 0.05

    def test_small_n_large_r_is_not_significant(self):
        """r=0.62 from n=10 was a real false positive this project caught."""
        assert stats.pearson_p(0.62, 10) > 0.05

    def test_p_is_none_when_undefined(self):
        assert stats.pearson_p(None, 100) is None
        assert stats.pearson_p(0.5, 3) is None
        assert stats.pearson_p(1.0, 100) is None


class TestFisherZCI:
    def test_interval_contains_the_estimate(self):
        lo, hi = stats.fisher_z_ci(0.3, 100)
        assert lo < 0.3 < hi

    def test_null_result_interval_spans_zero(self):
        """The interval is what stops r=0.043 reading as a small positive effect."""
        lo, hi = stats.fisher_z_ci(0.0428, 1155)
        assert lo < 0 < hi

    def test_small_sample_interval_is_wide(self):
        """r=0.62 on n=10 and r=0.043 on n=1155 are not comparable claims."""
        small = stats.fisher_z_ci(0.62, 10)
        large = stats.fisher_z_ci(0.62, 1000)
        assert (small[1] - small[0]) > 4 * (large[1] - large[0])

    def test_undefined_is_none(self):
        assert stats.fisher_z_ci(1.0, 100) is None
        assert stats.fisher_z_ci(0.5, 3) is None


class TestSignTest:
    def test_even_split_is_not_significant(self):
        assert stats.sign_test_p(38, 80) > 0.5

    def test_published_corpus_value(self):
        assert stats.sign_test_p(38, 80) == pytest.approx(0.7376, abs=1e-3)

    def test_lopsided_split_is_significant(self):
        assert stats.sign_test_p(75, 80) < 0.001

    def test_symmetric_in_direction(self):
        assert stats.sign_test_p(20, 80) == stats.sign_test_p(60, 80)

    def test_zero_n(self):
        assert stats.sign_test_p(0, 0) is None


class TestBootstrapCI:
    def test_interval_brackets_the_mean(self):
        vals = list(range(100))
        lo, hi = stats.bootstrap_ci(vals, lambda s: sum(s) / len(s))
        assert lo < 49.5 < hi

    def test_is_deterministic(self):
        """An interval that moves between runs of the same script is not evidence."""
        vals = [1, 5, 2, 8, 3, 9, 4, 7]
        f = lambda s: sum(s) / len(s)  # noqa: E731
        assert stats.bootstrap_ci(vals, f) == stats.bootstrap_ci(vals, f)

    def test_too_few_values(self):
        assert stats.bootstrap_ci([1], lambda s: s[0]) is None

    def test_survives_a_statistic_that_sometimes_raises(self):
        def flaky(sample):
            if sample[0] == 0:
                raise ZeroDivisionError
            return 1 / sample[0]
        assert stats.bootstrap_ci([1, 2, 3, 4, 5], flaky) is not None


class TestProportionCI:
    def test_stays_inside_zero_one(self):
        """The normal approximation runs outside [0,1] at 9/10; Wilson does not."""
        lo, hi = stats.proportion_ci(9, 10)
        assert 0.0 <= lo <= hi <= 1.0

    def test_brackets_the_proportion(self):
        lo, hi = stats.proportion_ci(50, 100)
        assert lo < 0.5 < hi

    def test_all_successes_does_not_exceed_one(self):
        assert stats.proportion_ci(10, 10)[1] <= 1.0

    def test_zero_n(self):
        assert stats.proportion_ci(0, 0) is None


class TestKappa:
    def test_perfect_agreement(self):
        pairs = [("a", "a"), ("b", "b"), ("a", "a"), ("b", "b")]
        assert stats.cohens_kappa(pairs, ["a", "b"]) == 1.0

    def test_chance_agreement_is_near_zero(self):
        """Two raters who both over-produce one class agree often and inform nothing.

        This is exactly the convergent-validity case: the reference model put
        82% of radio clips into a single class.
        """
        pairs = [("a", "a")] * 80 + [("a", "b")] * 10 + [("b", "a")] * 10
        k = stats.cohens_kappa(pairs, ["a", "b"])
        assert abs(k) < 0.15

    def test_degenerate_single_class_is_none_not_one(self):
        pairs = [("a", "a")] * 50
        assert stats.cohens_kappa(pairs, ["a"]) is None

    def test_empty(self):
        assert stats.cohens_kappa([], ["a"]) is None

    @pytest.mark.parametrize("k,expected", [
        (None, "not computable"), (-0.01, "worse than chance"), (0.1, "slight"),
        (0.3, "fair"), (0.5, "moderate"), (0.7, "substantial"), (0.9, "almost perfect"),
    ])
    def test_bands(self, k, expected):
        assert stats.kappa_band(k) == expected

    def test_published_convergent_kappa_reads_as_worse_than_chance(self):
        assert stats.kappa_band(-0.0013) == "worse than chance"


class TestBonferroni:
    def test_divides_by_the_number_of_tests(self):
        assert stats.bonferroni_alpha(5) == 0.01

    def test_zero_tests_does_not_divide_by_zero(self):
        assert stats.bonferroni_alpha(0) == 0.05
