# Helper functions for SVD (SK-LEARN)
import numpy as np
import numbers
from scipy import linalg

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def safe_sparse_dot(a, b, dense_output=False):
    """
    Dot product that handle the sparse matrix case correctly
    """
    from scipy import sparse
    if sparse.issparse(a) or sparse.issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)

def qr_economic(A, **kwargs):
    """
    Compat function for the QR-decomposition in economic mode
    Scipy 0.9 changed the keyword econ=True to mode='economic'
    """
    import scipy.linalg
    # trick: triangular solve has introduced in 0.9
    if hasattr(scipy.linalg, 'solve_triangular'):
        return scipy.linalg.qr(A, mode='economic', **kwargs)
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return scipy.linalg.qr(A, econ=True, **kwargs)

def randomized_range_finder(A, size, n_iter, random_state=None, n_iterations=None):
    """
    Computes an orthonormal matrix whose range approximates the range of A.
    """
    random_state = check_random_state(random_state)
    R = random_state.normal(size=(A.shape[1], size))
    Y = safe_sparse_dot(A, R)
    del R
    for i in xrange(n_iter):
        Y = safe_sparse_dot(A, safe_sparse_dot(A.T, Y))
    Q, R = qr_economic(Y)
    return Q

def svd_flip(u, v):
    """Sign correction to ensure deterministic output from SVD

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u, v: arrays
        The output of `linalg.svd` or `sklearn.utils.extmath.randomized_svd`,
        with matching inner dimensions so one can compute `np.dot(u * s, v)`.

    Returns
    -------
    u_adjusted, s, v_adjusted: arrays with the same dimensions as the input.

    """
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, xrange(u.shape[1])])
    u *= signs
    v *= signs[:, np.newaxis]
    return u, v

def svd(M, n_components, n_oversamples=10, n_iter=5, transpose='auto', flip_sign=True, random_state=0, n_iterations=None):
    """
    Equivalent to scikit-learn Truncated SVD
    """
    if n_components >= M.shape[1]:
        raise ValueError("n_components must be < n_features;"
            " got %d >= %d" % (n_components, M.shape[1]))

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape   
    if transpose == 'auto' and n_samples > n_features:
        transpose = True
    if transpose:
        M = M.T
    Q = randomized_range_finder(M, n_random, n_iter, random_state)
    B = safe_sparse_dot(Q.T, M)
    Uhat, s, V = linalg.svd(B, full_matrices=False)
    del B
    U = np.dot(Q, Uhat)
    if flip_sign:
        U, V = svd_flip(U, V)
    if transpose:
        U, Sigma, VT =  V[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        U, Sigma, VT =  U[:, :n_components], s[:n_components], V[:n_components, :]
    Sigma = np.diag(Sigma)
    return np.dot(U, Sigma.T), VT
    #return U, np.dot(Sigma.T, VT)


