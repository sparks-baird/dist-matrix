# Distance Matrices
Fast Numba-enabled CPU and GPU computations of 1D Earth Mover's ([`scipy.stats.wasserstein_distance`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)) and Euclidean distances. `10000 x 10000` weighted Wasserstein distance matrix in ~15 s on GPU.

# Installation

| conda | pip |
| ---- | --- |
| `conda install -c sgbaird dist_matrix` | `pip install dist_matrix` |

To best reflect the development workflow, you can clone the repository and install via [`flit`](https://flit.readthedocs.io/en/latest/):
```python
git clone https://github.com/sparks-baird/dist-matrix.git
cd dist-matrix
conda install flit # alternatively, `pip install -e .`
flit install --pth-file # --pth-file flag is a Windows-compatible local installation; you can edit the source without reinstalling
```

# Usage

You can compute distance matrices (more efficient per distance calculation) or access the lower-level single distance calculation.

## Distance Matrices
There is a GPU version (`dist_matrix.cuda_dist_matrix_full`) as well as a CPU version (`dist_matrix.njit_dist_matrix_full`).

```python
import numpy as np
from dist_matrix.cuda_dist_matrix_full import dist_matrix as gpu_dist_matrix
# from dist_matrix.njit_dist_matrix_full import dist_matrix as cpu_dist_matrix
n_features = 10
n_rows = 100
U, V, U_weights, V_weights = np.random.rand(4, n_rows, n_features)
distances = gpu_dist_matrix(
    U,
    V=V,
    U_weights=U_weights, # optional
    V_weights=V_weights, # optional
    metric="wasserstein", # "euclidean"
)
```

## Single Distance Calculations
See [metrics.py](dist_matrix/utils/metrics.py). Note that these lower-level functions are not GPU-accelerated.

```python
import numpy as np
from dist_matrix.utils.metrics import wasserstein_distance, euclidean_distance
n_features = 10
u, v, u_weights, v_weights = np.random.rand(4, n_features)
presorted, cumweighted, prepended = [False, False, False]
em_dist = wasserstein_distance(u, v, u_weights, v_weights, presorted, cumweighted, prepended)
eucl_dist = euclidean_distance(u, v)
```

# See Also
Element Mover's Distances via [chem_wasserstein](https://github.com/sparks-baird/chem_wasserstein) (based on [ElM2D](https://github.com/lrcfmd/ElM2D))
