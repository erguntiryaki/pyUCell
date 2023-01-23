import numpy as np
from numba import njit, prange


@njit(parallel=True)
def rank_sparse_matrix(csr_indptr, csr_indices, csr_data, shape):
    n_cells = shape[0]
    n_genes = shape[1]
    ranked_mat = np.empty(shape)
    for row in prange(n_cells):
        start_idx = csr_indptr[row + 1]
        step_size = csr_indptr[row + 1] - csr_indptr[row]
        n_zeros = n_genes - step_size
        base_rank = (sum(range(n_genes + 1)) - sum(range(step_size + 1))) / n_zeros

        arr = csr_data[start_idx: start_idx + step_size]
        sorter = np.argsort(arr, kind='quicksort')
        inv = np.empty(sorter.size).astype(np.intp)
        inv[sorter] = np.arange(sorter.size).astype(np.intp)
        arr = arr[sorter]
        obs = np.hstack((np.array([True]), arr[1:] != arr[:-1]))
        count = np.hstack((np.nonzero(obs)[0], np.array([len(obs)])))
        dense = obs.cumsum()[inv]
        result = .5 * (count[dense] + count[dense - 1] + 1)
        result = len(result) + 1 - result

        ranked_vec = np.full((n_genes,), fill_value=base_rank)
        ranked_vec[csr_indices[start_idx: start_idx + step_size]] = result

        ranked_mat[row, :] = ranked_vec
    return ranked_mat


@njit(parallel=True)
def calculate_u_score(rank_mat_subset, max_rank):
    n_cells = rank_mat_subset.shape[0]
    n_signature = rank_mat_subset.shape[1]
    score_vec = np.empty(shape=(n_cells,))
    for row in prange(n_cells):
        u_val = sum([i - (n_signature * (n_signature + 1)) / 2 for i in rank_mat_subset[row, :]])
        auc = 1 - (u_val / (n_signature * max_rank))
        score_vec[row] = auc
    return score_vec


def score_genes_ucell(adata, signature, max_rank=1500, score_name='ucell_score',  copy=False):
    adata = adata.copy() if copy else adata
    idx = [adata.var.index.to_list().index(s) for s in signature]
    rnk = rank_sparse_matrix(adata.X.indptr, adata.X.indices, adata.X.data, adata.shape)
    rnk = rnk[:, idx]
    rnk[rnk > max_rank] = max_rank + 1
    result = calculate_u_score(rnk, max_rank=max_rank)
    adata.obs[score_name] = result
    return adata if copy else None
