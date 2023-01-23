import numpy as np
from numba import njit


@njit
def minimal_ranker(arr, ascending=False):
    arr = np.ravel(arr)
    sorter = np.argsort(arr, kind='quicksort')

    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    arr = arr[sorter]
    obs = np.hstack((np.array([True]), arr[1:] != arr[:-1]))

    count = np.hstack((np.nonzero(obs)[0], np.array([len(obs)])))
    dense = obs.cumsum()[inv]
    result = .5 * (count[dense] + count[dense - 1] + 1)

    if ascending:
        return result
    else:
        return len(result) + 1 - result


@njit
def get_ranks_of_zeros(n_genes, n_nonzero):
    n_zeros = n_genes - n_nonzero
    base_rank = (sum(range(n_genes + 1)) - sum(range(n_nonzero + 1))) / n_zeros
    new_arr = np.full((n_genes,), fill_value=base_rank)
    return new_arr


@njit
def insert_ranks_of_nonzero(base_rank_arr, nonzero_arr, csr_indices):
    rnk = minimal_ranker(nonzero_arr)
    base_rank_arr[csr_indices] = rnk


def rank_sparse(sp_arr):
    new_arr = get_ranks_of_zeros(sp_arr.shape[1], sp_arr.nnz)
    insert_ranks_of_nonzero(new_arr, sp_arr[:, sp_arr.indices].toarray(), sp_arr.indices)
    return new_arr


def _calculate_u_score(vec, max_rank, n_signature, idx):
    rnk = rank_sparse(vec)
    rnk[rnk > max_rank] = max_rank + 1
    rnk = rnk[idx]
    u_val = sum([i - (n_signature * (n_signature + 1)) / 2 for i in rnk])
    auc = 1 - (u_val / (n_signature * max_rank))
    return auc


def score_genes_ucell(adata, signature, max_rank=1500, score_name='ucell_score',  copy=False):
    adata = adata.copy() if copy else adata
    n_signature = len(signature)
    idx = [adata.var.index.to_list().index(s) for s in signature]

    res = map(lambda vec: _calculate_u_score(vec, max_rank=max_rank, n_signature=n_signature, idx=idx), [vec for vec in adata.X])
    adata.obs[score_name] = list(res)
    return adata if copy else None
