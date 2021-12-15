# Distance Matrices
Fast Numba-enabled CPU and GPU computations of Earth Mover's ([`scipy.stats.wasserstein_distance`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)) and Euclidean distances. `10000 x 10000` weighted Wasserstein distance matrix in ~15 s on GPU.

# Installation
| conda | pip | source |
| ---- | --- | --- |
| `conda install -c sgbaird dist_matrix` | `pip install dist_matrix` | `git clone <url>; pip install -e .`

where `<url>` is https://github.com/sparks-baird/dist-matrix.git.

# Usage
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

# See Also
Element Mover's Distances via [chem_wasserstein](https://github.com/sparks-baird/chem_wasserstein) (based on [ElM2D](https://github.com/lrcfmd/ElM2D))
