"""Test distance matrix calculations using CUDA/Numba."""
# import os
# from importlib import reload

from numpy.testing import assert_allclose
import numpy as np
import pytest

from scipy.spatial.distance import euclidean, cdist
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance

from dist_matrix.utils.Timer import Timer
from numba.cuda.testing import unittest, CUDATestCase

# from ElMD import ElMD

from dist_matrix.njit_dist_matrix_full import dist_matrix as cpu_dist_matrix
from dist_matrix.cuda_dist_matrix_full import dist_matrix as gpu_dist_matrix

# os.environ["COLUMNS"] = str(cols)
# os.environ["USE_64"] = "0"
# os.environ["INLINE"] = "never"
# os.environ["FASTMATH"] = "1"
# os.environ["TARGET"] = "cuda"

# from mat_discover.ElM2D import cuda_dist_matrix  # noqa

# # to overwrite env vars (source: https://stackoverflow.com/a/1254379/13697228)
# reload(cuda_dist_matrix)
# gpu_dist_matrix = cuda_dist_matrix.dist_matrix
# else:
# from mat_discover.ElM2D import njit_dist_matrix  # noqa

# # to overwrite env vars (source: https://stackoverflow.com/a/1254379/13697228)
# reload(njit_dist_matrix)
# cpu_dist_matrix = njit_dist_matrix.dist_matrix

# cols = len(ElMD(metric="mod_petti").periodic_tab)  # just for generating test data
cols = 100

verbose_test = True

def generate_test_data(rows, cols):
    """
    Generate seeded test values and weights for two distributions.

    Returns
    -------
    U : 2D array
        Values of first distribution.
    V : 2D array
        Values of second distribution.
    U_weights : 2D array
        Weights of first distribution.
    V_weights : 2D array
        Weights of second distribution.

    """
    np.random.seed(42)
    [U, V, U_weights, V_weights] = [
        np.random.rand(rows, cols) for i in range(4)
    ]
    return U, V, U_weights, V_weights
        
def my_wasserstein_distance(u_uw, v_vw):
    """
    Return Earth Mover's distance using concatenated values and weights.

    Parameters
    ----------
    u_uw : 1D numeric array
        Horizontally stacked values and weights of first distribution.
    v_vw : TYPE
        Horizontally stacked values and weights of second distribution.

    Returns
    -------
    distance : numeric scalar
        Earth Mover's distance given two distributions.

    """
    # split into values and weights
    n = len(u_uw)
    i = n // 2
    u = u_uw[0:i]
    uw = u_uw[i:n]
    v = v_vw[0:i]
    vw = v_vw[i:n]
    # calculate distance
    distance = scipy_wasserstein_distance(u, v, u_weights=uw, v_weights=vw)
    return distance
        
def join_wasserstein(U, V, Uw, Vw):
    """
    Horizontally stack values and weights for each distribution.

    Weights are added as additional columns to values.

    Example:
        u_uw, v_vw = join_wasserstein(u, v, uw, vw)
        d = my_wasserstein_distance(u_uw, v_vw)
        cdist(u_uw, v_vw, metric=my_wasserstein_distance)

    Parameters
    ----------
    u : 1D or 2D numeric array
        First set of distribution values.
    v : 1D or 2D numeric array
        Second set of values of distribution values.
    uw : 1D or 2D numeric array
        Weights for first distribution.
    vw : 1D or 2D numeric array
        Weights for second distribution.

    Returns
    -------
    u_uw : 1D or 2D numeric array
        Horizontally stacked values and weights of first distribution.
    v_vw : TYPE
        Horizontally stacked values and weights of second distribution.

    """
    U_Uw = np.concatenate((U, Uw), axis=1)
    V_Vw = np.concatenate((V, Vw), axis=1)
    return U_Uw, V_Vw





@pytest.mark.parametrize("metric", ["euclidean", "wasserstein"])  
@pytest.mark.parametrize("sets", ["single", "two"])  
@pytest.mark.parametrize("sparsity", ["dense", "sparse"])  
@pytest.mark.parametrize("target", ["cuda", "cpu"])  
def test_dist_matrix(sets, metric, target, sparsity):
    """
    Loop through distance metrics and perform unit tests.

    The four test cases are:
        pairwise distances within a single set of vectors
        pairwise distances between two sets of vectors
        sparse pairwise distances within a single set of vectors
        sparse pairwise distances between two sets of vectors

    The ground truth for Euclidean comes from cdist.
    The ground truth for Earth Mover's distance (1-Wasserstein) is via
    a scipy.stats function.

    Helper functions are used to generate test data and support the use of
    Wasserstein distances in cdist.

    Returns
    -------
    None.

    """

    testQ = sets == "single"
    my_dist_matrix = gpu_dist_matrix if target == "cuda" else cpu_dist_matrix
    tol = 1e-5

    # large # of rows with testQ == True is slow due to use of
    # cdist and non-jitted scipy-wasserstein
    rows = 6 if testQ else 100

    print("====[PARTIAL TEST WITH {} ROWS]====".format(rows))
    U, V, U_weights, V_weights = generate_test_data(rows, cols)
    Utest, Vtest, Uwtest, Vwtest = [
        x[0:6] for x in [U, V, U_weights, V_weights]
    ]  # noqa

    print(f"[{target.upper()}_{metric.upper()}_{sparsity.upper()}_{sets.upper()}]")
    
    # test and production data
    pairs = np.array([(0, 1), (1, 2), (2, 3)]) if sparsity == "sparse" else None

    # compile
    if testQ:
        V = None
        Vtest = None
        Vwtest = None
        V_weights = None

    # Compile
    my_dist_matrix(
        Utest,
        V=Vtest,
        U_weights=Uwtest,
        V_weights=Vwtest,
        pairs=pairs,
        metric=metric,
    )

    with Timer("one set" if testQ else "two set"):
        dist_matrix = my_dist_matrix(
            U,
            V=V,
            U_weights=U_weights,
            V_weights=V_weights,
            pairs=pairs,
            metric=metric,
        )

    if metric == "euclidean":
        dist_func = euclidean
        U_Uw, V_Vw = U, V
    elif metric == "wasserstein":
        dist_func = my_wasserstein_distance
        if testQ:
            U_Uw, _ = join_wasserstein(U, U, U_weights, U_weights)
        else:
            U_Uw, V_Vw = join_wasserstein(U, V, U_weights, V_weights)

    if sparsity == "dense":
        if testQ:
            with Timer("one set check (cdist)"):
                dist_matrix_check = cdist(U_Uw, U_Uw, metric=dist_func)
        else:
            with Timer("two set check (cdist)"):
                dist_matrix_check = cdist(U_Uw, V_Vw, metric=dist_func)
    else:
        if testQ:
            dist_matrix_check = [dist_func(U_Uw[i], U_Uw[j]) for i, j in pairs]
        else:
            dist_matrix_check = [dist_func(U_Uw[i], V_Vw[j]) for i, j in pairs]

    # check results
    assert_allclose(
        dist_matrix,
        np.asarray(dist_matrix_check),
        rtol=tol,
        err_msg="distance matrix inaccurate".format(
            target, metric
        ),
    )  # noqa

if __name__ == "__main__":
    unittest.main()


# TF = [
#     np.allclose(one_set, one_set_check),
#     np.allclose(two_set, two_set_check),
#     np.allclose(one_set_sparse, one_sparse_check),
#     np.allclose(two_set_sparse, two_sparse_check)
#     ]

# from numba import cuda

# os.environ["NUMBA_DISABLE_JIT"] = "1"

# use_cuda = cuda.is_available()
# if use_cuda:

# settings = {
#     "INLINE": "never",
#     "FASTMATH": True,
#     "COLUMNS": cols,
#     "USE_64": False,
#     "TARGET": "cuda",
# }

# with open("dist_matrix_settings.json", "w") as f:
#     json.dump(settings, f)
