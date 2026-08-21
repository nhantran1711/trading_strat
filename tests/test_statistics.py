import numpy as np

from analysis.statistics import bootstrap_ci, multiple_testing_correction, permutation_test_two_groups, sign_flip_test


def test_bootstrap_ci_covers_true_mean_for_known_distribution():
    rng = np.random.default_rng(1)
    values = rng.normal(loc=0.5, scale=1.0, size=2000)
    result = bootstrap_ci(values, n_boot=2000, random_state=1)
    assert result.ci_low < 0.5 < result.ci_high
    assert abs(result.point_estimate - 0.5) < 0.1


def test_sign_flip_test_rejects_null_for_strong_positive_edge():
    rng = np.random.default_rng(2)
    returns = rng.normal(loc=2.0, scale=1.0, size=200)  # clearly, strongly positive
    result = sign_flip_test(returns, n_perm=5000, random_state=2)
    assert result.p_value < 0.01


def test_sign_flip_test_does_not_reject_null_for_pure_noise():
    rng = np.random.default_rng(3)
    returns = rng.normal(loc=0.0, scale=1.0, size=200)  # genuinely zero-mean
    result = sign_flip_test(returns, n_perm=5000, random_state=3)
    assert result.p_value > 0.05


def test_permutation_two_groups_detects_real_difference():
    rng = np.random.default_rng(4)
    a = rng.normal(1.0, 1.0, 150)
    b = rng.normal(0.0, 1.0, 150)
    result = permutation_test_two_groups(a, b, n_perm=5000, random_state=4)
    assert result.p_value < 0.01


def test_multiple_testing_correction_widens_significance_bar():
    # Several borderline p-values -- FDR correction should not let all of them
    # through as "significant" the way uncorrected alpha=0.05 would.
    p_values = [0.04, 0.03, 0.045, 0.02, 0.048]
    reject, adj_p = multiple_testing_correction(p_values, method="fdr_bh", alpha=0.05)
    assert all(adj_p >= np.array(p_values))  # adjustment never makes p-values smaller
