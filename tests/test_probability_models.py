import numpy as np

from unseen_awg.probability_models import gumbel_max_sample


def test_gumbel_max_sample_output_shape():
    rng = np.random.default_rng()
    unnormalized_logp = np.array([0.0, 1.0, 2.0])
    assert gumbel_max_sample(unnormalized_logp, rng, 1).shape == (1,)
    assert gumbel_max_sample(unnormalized_logp, rng, (2, 3)).shape == (2, 3)
    unnormalized_logp = np.array([0.0])
    assert gumbel_max_sample(unnormalized_logp, rng, 1).shape == (1,)
    unnormalized_logp = np.array(0.0)
    assert gumbel_max_sample(unnormalized_logp, rng, 1).shape == (1,)
    unnormalized_logp = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    assert gumbel_max_sample(unnormalized_logp, rng, 1).shape == (1,)


def test_gumbel_max_sample_output_type():
    rng = np.random.default_rng()
    unnormalized_logp = np.array([0.0, 1.0, 2.0])
    assert isinstance(gumbel_max_sample(unnormalized_logp, rng, 1), np.ndarray)
    assert gumbel_max_sample(unnormalized_logp, rng, 1).dtype == np.int64


def test_gumbel_max_sample_output_range():
    rng = np.random.default_rng()
    unnormalized_logp = np.array([0.0, 1.0, 2.0])
    result = gumbel_max_sample(unnormalized_logp, rng, 1000)
    assert np.all(result >= 0)
    assert np.all(result < len(unnormalized_logp))


def test_gumbel_max_sample_deterministic():
    rng = np.random.default_rng(0)
    unnormalized_logp = np.array([-np.inf, 2.0])
    result = gumbel_max_sample(unnormalized_logp, rng, 1000)
    assert np.array_equal(result, np.full(shape=(1000,), fill_value=1, dtype=np.int_))
