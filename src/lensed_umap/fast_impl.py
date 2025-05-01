"""This module contains (private) numba accelerated helper functions."""

import numba as nb
import numpy as np


@nb.njit(
    nb.types.void(nb.int32[::1]),
    locals={"c": nb.int32, "v": nb.int32},
    cache=True,
    inline="always",
)
def _inplace_cumsum_shift(arr):
    """
    Transforms the given array to contain the cumulative sums starting at 0.
    """
    c = 0
    for idx, v in enumerate(arr):
        arr[idx] = c
        c += v


@nb.njit(
    nb.types.Tuple((nb.float32[::1], nb.int32[::1], nb.int32[::1]))(
        nb.float32[::1], nb.int32[::1], nb.int32[::1], nb.int32[::1], nb.boolean
    ),
    locals={
        "row": nb.int32,
        "col": nb.int32,
        "count": nb.int32,
        "value": nb.float32,
        "max_segment": nb.int32,
        "diff": nb.int32,
    },
    fastmath=True,
    cache=True,
)
def _apply_lens_filter(data, indices, indptr, binned_lens, circular):
    """
    Filters the given sparse csr matrix by removing all edges between
    points that do not have neighbouring bin values.
    """
    # Allocate new graph
    num_stored = len(data)
    new_indptr = np.zeros_like(indptr)
    new_indices = np.empty(num_stored, dtype=np.int32)
    new_data = np.empty(num_stored, dtype=np.float32)
    count = 0

    # Process edges
    max_segment = binned_lens.max()
    for row in range(len(indptr) - 1):
        start_idx = indptr[row]
        end_idx = indptr[row + 1]

        parent_bin = binned_lens[row]
        for col, value in zip(indices[start_idx:end_idx], data[start_idx:end_idx]):
            child_bin = binned_lens[col]
            ## Keep only if within or adjacent segment
            diff = abs(parent_bin - child_bin)
            if diff <= 1 or (circular and diff == max_segment):
                new_indptr[row] += 1
                new_data[count] = value
                new_indices[count] = col
                count += 1

    _inplace_cumsum_shift(new_indptr)
    return (new_data[:count], new_indices[:count], new_indptr)


@nb.njit(
    fastmath=True,
    parallel=True,
    cache=True,
    locals={
        "idx": nb.int32,
        "row": nb.int32,
        "col": nb.int32,
    },
)
def _extract_local_lens_edges(indices, indptr, data, values, metric_fn, n_neighbors):
    """
    Extract's each point's lens-distance-smallest `n_neighbor` edges from the UMAP graph.
    Provides indices into CSR graph's rows! Not into the data itself!

    Function cannot be typed, creating a compile on import, as it takes a function argument.
    This results in a JIT-compile on every (first) call.
    """
    # Allocate heaps
    remaining_neighbors = np.full((len(indptr) - 1, n_neighbors), -1, dtype=np.int32)
    remaining_lens_dists = np.full(
        (len(indptr) - 1, n_neighbors), np.inf, dtype=np.float32
    )
    remaining_edge_dists = np.full(
        (len(indptr) - 1, n_neighbors), np.inf, dtype=np.float32
    )

    # Process rows
    for row in nb.prange(len(indptr) - 1):
        start_idx = indptr[row]
        end_idx = indptr[row + 1]

        # Extract row vectors
        row_indices = indices[start_idx:end_idx]
        row_strengths = data[start_idx:end_idx]

        # Compute lens distances
        for idx, col in enumerate(row_indices):
            heap_push(
                remaining_neighbors[row, :],
                remaining_lens_dists[row, :],
                remaining_edge_dists[row, :],
                idx,
                metric_fn(values[row, :], values[col, :]),
                1 - row_strengths[idx],
            )

    return remaining_neighbors


@nb.njit(
    (nb.int32[::1], nb.float32[::1], nb.float32[::1], nb.int32, nb.float32, nb.float32),
    fastmath=True,
    cache=True,
    locals={
        "size": nb.types.intp,
        "i": nb.types.uint16,
        "ic1": nb.types.uint16,
        "ic2": nb.types.uint16,
        "i_swap": nb.types.uint16,
    },
)
def heap_push(indices, lens_dists, edge_dists, new_idx, new_lens_d, new_edge_d):
    """Maintain heap sorted by lens distance with ties broken on edge distance."""
    # Skip if the new distances are too large.
    if new_lens_d > lens_dists[0] or (
        new_lens_d == lens_dists[0] and new_edge_d > edge_dists[0]
    ):
        return

    # Set new value as largest element
    indices[0] = new_idx
    lens_dists[0] = new_lens_d
    edge_dists[0] = new_edge_d

    # Descend the heap
    i = 0
    size = lens_dists.shape[0]
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= size:
            break
        elif ic2 >= size:
            # ic1 greater than new
            if lens_dists[ic1] > new_lens_d or (
                lens_dists[ic1] == new_lens_d and edge_dists[ic1] > new_edge_d
            ):
                i_swap = ic1
            else:
                break
        # ic1 greater equal than ic2
        elif lens_dists[ic1] > lens_dists[ic2] or (
            lens_dists[ic1] == lens_dists[ic2] and edge_dists[ic1] >= edge_dists[ic2]
        ):
            # new smaller than ic1
            if new_lens_d < lens_dists[ic1] or (
                lens_dists[ic1] == new_lens_d and new_edge_d < edge_dists[ic1]
            ):
                i_swap = ic1
            else:
                break
        else:
            # new smaller than ic2
            if new_lens_d < lens_dists[ic2] or (
                lens_dists[ic2] == new_lens_d and new_edge_d < edge_dists[ic2]
            ):
                i_swap = ic2
            else:
                break

        indices[i] = indices[i_swap]
        lens_dists[i] = lens_dists[i_swap]
        edge_dists[i] = edge_dists[i_swap]

        i = i_swap

    indices[i] = new_idx
    lens_dists[i] = new_lens_d
    edge_dists[i] = new_edge_d


@nb.njit(
    nb.types.Tuple((nb.float32[::1], nb.int32[::1], nb.int32[::1]))(
        nb.types.List(nb.types.DictType(nb.int32, nb.float32))
    ),
    locals={
        "row": nb.int32,
        "count": nb.int32,
        "col": nb.int32,
    },
    parallel=True,  # TODO Check benefit
    fastmath=True,
    cache=True,
)
def _convert_to_csr(dok_matrix):
    # Construct new indptr (sequential)
    indptr = np.empty(len(dok_matrix) + 1, dtype=np.int32)
    indptr[0] = 0
    for row in range(len(dok_matrix)):
        indptr[row + 1] = indptr[row] + len(dok_matrix[row])

    # Fill new indices & data (parallel)
    data = np.empty(indptr[-1], dtype=np.float32)
    indices = np.empty(indptr[-1], dtype=np.int32)
    for row in nb.prange(len(dok_matrix)):
        # Extract indptr
        start_idx = indptr[row]
        end_idx = indptr[row + 1]

        # Extract rows
        row_data = data[start_idx:end_idx]
        row_indices = indices[start_idx:end_idx]

        # Fill values (in order)
        keys = sorted(dok_matrix[row].keys())
        for count, key in enumerate(keys):
            row_indices[count] = key
            row_data[count] = dok_matrix[row][key]

    return (data, indices, indptr)


@nb.njit(
    nb.types.Tuple((nb.float32[::1], nb.int32[::1], nb.int32[::1]))(
        nb.float32[::1], nb.int32[::1], nb.int32[::1], nb.int32[:, ::1]
    ),
    locals={
        "row": nb.int32,
        "col": nb.int32,
    },
    fastmath=True,
    cache=True,
)
def _apply_local_mask_filter(data, indices, indptr, knn_indices):
    """Filters the graph keeping only the detected undirected shortest lens-distance edges."""
    # Construct symmetric dok matrix
    dok_matrix = [
        {np.int32(0): np.float32(0) for _ in range(0)} for _ in range(len(indptr) - 1)
    ]
    for row in range(len(indptr) - 1):
        # Extract indptr
        start_idx = indptr[row]
        end_idx = indptr[row + 1]

        # Extract row values
        row_data = data[start_idx:end_idx]
        row_indices = indices[start_idx:end_idx]

        # Extract existing neighbour indices
        columns = knn_indices[row]
        last_idx = len(columns) - 1
        while last_idx >= 0 and columns[last_idx] == -1:
            last_idx -= 1

        # Fill the matrix
        for col in knn_indices[row][: last_idx + 1]:
            dok_matrix[row][row_indices[col]] = row_data[col]
            dok_matrix[row_indices[col]][row] = row_data[col]

    return _convert_to_csr(dok_matrix)


@nb.njit(
    nb.types.Tuple((nb.float32[::1], nb.int32[::1], nb.int32[::1]))(
        nb.float32[::1],
        nb.int32[::1],
        nb.int32[::1],
        nb.int32[::1],
        nb.int32[::1],
        nb.types.UniTuple(nb.int64, 2),
    ),
    locals={
        "idx": nb.int32,
        "row": nb.int32,
    },
    cache=True,
    fastmath=True,
)
def _apply_matrix_mask(data_1, indices_1, indptr_1, indices_2, indptr_2, shape):
    """Keep undirected edges of csr matrix 1 that also exist in csr matrix 2."""
    # Quick stop if either one has no values
    if indptr_1[-1] == 0 or indptr_2[-1] == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.zeros(shape[0] + 1, dtype=np.int32),
        )

    # Construct symmetric dok matrix
    dok_matrix = [
        {np.int32(0): np.float32(0) for _ in range(0)} for _ in range(shape[0])
    ]
    for row in range(shape[0]):
        start_idx1 = indptr_1[row]
        end_idx1 = indptr_1[row + 1]

        start_idx2 = indptr_2[row]
        end_idx2 = indptr_2[row + 1]

        # Keep edge only when both columns are equal
        while start_idx1 < end_idx1 and start_idx2 < end_idx2:
            if indices_1[start_idx1] == indices_2[start_idx2]:
                dok_matrix[row][indices_1[start_idx1]] = data_1[start_idx1]
                dok_matrix[indices_1[start_idx1]][row] = data_1[start_idx1]
                start_idx1 += 1
                start_idx2 += 1
            elif indices_1[start_idx1] < indices_2[start_idx2]:
                start_idx1 += 1
            else:
                start_idx2 += 1

    return _convert_to_csr(dok_matrix)
