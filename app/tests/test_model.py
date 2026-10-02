import os
import sys

import numpy as np
import pytest

# make `app/` importable when pytest is run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logistic_regression import LogisticRegression  # noqa: E402

N_FEATURES, N_CLASSES = 6, 4


@pytest.fixture(scope="module")
def model():
    # tiny synthetic dataset -> test needs no network, no MLflow, no CSV
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, N_FEATURES))
    y = rng.integers(0, N_CLASSES, 200)
    return LogisticRegression(k=N_CLASSES, alpha=0.1, max_iter=50).fit(X, y)


def test_model_takes_expected_input(model):
    # correct number of features is accepted ...
    model.predict(np.zeros((3, N_FEATURES)))
    # ... and a wrong number of features is rejected
    with pytest.raises(ValueError):
        model.predict(np.zeros((3, N_FEATURES + 1)))


def test_output_has_expected_shape(model):
    X = np.random.default_rng(1).normal(size=(5, N_FEATURES))
    preds, probs = model.predict(X), model.predict_proba(X)
    assert preds.shape == (5,)                      # one class per row
    assert probs.shape == (5, N_CLASSES)            # one probability per class
    assert np.allclose(probs.sum(axis=1), 1.0)      # probabilities sum to 1
    assert set(preds) <= set(range(N_CLASSES))      # classes are 0..3
