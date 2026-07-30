import numpy as np
from scipy.sparse import csr_matrix

from SIDISH.VAE import _as_dense_array


def test_as_dense_array_accepts_numpy_array():
    expression = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = _as_dense_array(expression)

    np.testing.assert_array_equal(result, expression)


def test_as_dense_array_converts_sparse_matrix():
    expression = csr_matrix([[1.0, 0.0], [0.0, 4.0]])

    result = _as_dense_array(expression)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [[1.0, 0.0], [0.0, 4.0]])
