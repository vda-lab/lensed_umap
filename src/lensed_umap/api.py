"""This module contains the public API for lensed UMAP."""

import numpy as np
import numba as nb
import pandas as pd
from copy import copy
from warnings import warn

from scipy.sparse import csr_array
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state

from umap import UMAP
from umap.umap_ import reset_local_connectivity as umap_reset_connectivity
import umap.distances as dist

from typing import Optional, Callable, Union    
from numpy.typing import ArrayLike

from .fast_impl import (
    _apply_lens_filter,
    _extract_local_lens_edges,
    _apply_local_mask_filter,
    _apply_matrix_mask,
)


def apply_lens(
    projector: UMAP,
    values: ArrayLike,
    resolution: int = 5,
    circular: bool = False,
    discretization: str = "regular",
    skip_embedding: bool = False,
    reset_local_connectivity: bool = False,
    ret_bins: bool = False,
    **kwargs,
):
    """
    Filters edges from a UMAP model that cross-over segments in discretised
    lens dimensions.

    Discretises the given lenses into segments. Only edges within one segment,
    or to a direct neighbour segment are kept.

    Parameters
    ----------
    projector : UMAP
        A fitted UMAP object to apply the lens to. Only the computed graph is
        needed, so some compute time can be safed by specifying
        'transform_mode="graph"' when constructing the UMAP object.
    values : array_like
        The 1 dimensional values to serve as lens dimension.
    resolution : int, default=5
        The number of segments to create in the lens.
    circular : bool, default=False
        A flag indicating wether the extrema should be seen as neighbours.
    discretization : str, default='regular'
        A flag indicating which discretization strategy to apply:

        * 'regular' -> each segment contains the same value-range.
        * 'balanced' -> each segment contains the same number of points.
    skip_embedding : bool, default=False
        A flag indicating whether to skip computing the embedding.
    reset_local_connectivity : bool, default=False
        A flag to enable normalising edge weights after applying the lens.

        UMAP's embedding assumes each point has at least one fully connected
        edge. This assumption can be enforced by enabling this flag. Embeddings
        tend to vary more smoothly when this setting is enabled.
    ret_bins : bool, default=False
        A flag to return an array with each points' bin id.
    **kwargs : dict, otpional
        UMAP arguments to overwrite projector's values. These can overwrite
        layout parameters to fine-tune the filtered embedding.

    Returns
    -------
    clone : UMAP
        A clone of projector with filtered `graph_` and updated `embedding_`.

        Other attributes are soft-copies, modifying them also modifies the input
        object's attributes!
    """
    # Validations
    if not hasattr(projector, "graph_"):
        raise AttributeError("Projector has not been fit yet.")
    N = projector._raw_data.shape[0]
    if len(values) != N:
        raise ValueError("Lens does not have a single value for every data point.")
    values = np.asarray(values)
    if len(values.shape) > 1:
        raise ValueError("Lens has more than one dimension.")
    if discretization not in ("regular", "balanced"):
        raise ValueError("Invalid discretization strategy given")
    if resolution <= 2:
        raise ValueError("Resolution too low.")

    # Process lens segments
    if discretization == "regular":
        binned_lens = pd.cut(
            pd.Series(values), bins=resolution, labels=np.arange(resolution)
        ).cat.codes.values.astype(np.int32)
    else:  # 'balanced'
        order = np.argsort(values)
        points_per_segment = N // resolution
        binned_lens = np.empty_like(values, dtype=np.int32)
        binned_lens[order[: (resolution * points_per_segment)]] = np.repeat(
            np.arange(resolution), points_per_segment
        )
        binned_lens[order[(resolution * points_per_segment) :]] = resolution - 1

    # Shallow-copy the projector with new embedding and graph and overwrite kwargs
    clone = _clone_projector(projector, **kwargs)

    # Actually perform the filtering...
    clone.graph_ = csr_array(
        _apply_lens_filter(
            clone.graph_.data,
            clone.graph_.indices,
            clone.graph_.indptr,
            binned_lens,
            circular,
        ),
        shape=(N, N),
    )
    if reset_local_connectivity:
        umap_reset_connectivity(clone.graph_, True)

    # Update embedding or remove related attributes
    _update_embedding(clone, skip_embedding)

    if ret_bins:
        return clone, binned_lens
    return clone


def apply_mask(
    projector: UMAP,
    masker: UMAP,
    skip_embedding: bool = False,
    reset_local_connectivity: bool = False,
    **kwargs,
):
    """
    Filters edges from a UMAP model that do not exist in the mask-model.

    Parameters
    ----------
    projector : UMAP
        A fitted UMAP object to apply the lens to. Only the computed graph is
        needed, so some compute time can be safed by specifying
        'transform_mode="graph"' when constructing the UMAP object.
    masker : UMAP
        A fitted UMAP object to serve as mask for the lens. Only the computed
        graph is needed, so some compute time can be safed by specifying
        'transform_mode="graph"' when constructing the UMAP object.
    skip_embedding : bool, default=False
        A flag indicating whether to skip computing the embedding.
    reset_local_connectivity : bool, default=False
        A flag to enable normalising edge weights after applying the lens.

        UMAP's embedding assumes each point has at least one fully connected
        edge. This assumption can be enforced by enabling this flag. Embeddings
        tend to vary more smoothly when this setting is enabled.
    **kwargs : dict, optional
        UMAP arguments to overwrite projector's values. These can overwrite
        layout parameters to fine-tune the filtered embedding.

    Returns
    -------
    clone : UMAP
        A clone of projector with filtered `graph_` and updated `embedding_`.

        Other attributes are soft-copies, modifying them also modifies the input
        object's attributes!
    """
    if not hasattr(projector, "graph_"):
        raise AttributeError("Projector has not been fit yet.")
    if not hasattr(masker, "graph_"):
        raise AttributeError("Masker has not been fit yet.")
    N = projector._raw_data.shape[0]
    if masker._raw_data.shape[0] != N:
        raise ValueError("Masker does not have the same size as projector.")

    # Shallow-copy the projector with new embedding and graph and overwrite kwargs
    clone = _clone_projector(projector, **kwargs)

    # Apply mask
    clone.graph_ = csr_array(
        _apply_matrix_mask(
            clone.graph_.data,
            clone.graph_.indices,
            clone.graph_.indptr,
            masker.graph_.indices,
            masker.graph_.indptr,
            clone.graph_.shape,
        ),
        shape=(N, N),
    )
    if reset_local_connectivity:
        umap_reset_connectivity(clone.graph_, True)

    # Update embedding or remove related attributes
    _update_embedding(clone, skip_embedding)

    return clone


def apply_local_mask(
    projector: UMAP,
    values: ArrayLike,
    metric: Union[str, Callable] = "euclidean",
    metric_kwds: Optional[dict] = None,
    n_neighbors: int = 5,
    skip_embedding: bool = False,
    reset_local_connectivity: bool = False,
    **kwargs,
):
    """
    Filters the given UMAP model by keeping only each points' `n_neighbors` closest edges by
    the metric over the lens values.

    Computes the distance for each edge in the UMAP model, and keeps the `n_neigbhors` closest ones.

    Parameters
    ----------
    projector : UMAP
        A fitted UMAP object to apply the lens to. Only the computed graph is
        needed, so some compute time can be safed by specifying
        'transform_mode="graph"' when constructing the UMAP object.
    values : array_like
        The values to serve as lens dimensions, which are used to compute lens-distances.
    metric : string or function, default='euclidean'
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:

        * euclidean (or l2)
        * manhattan (or l1)
        * cityblock
        * braycurtis
        * canberra
        * chebyshev
        * correlation
        * cosine
        * dice
        * hamming
        * jaccard
        * kulsinski
        * ll_dirichlet
        * mahalanobis
        * matching
        * minkowski
        * rogerstanimoto
        * russellrao
        * seuclidean
        * sokalmichener
        * sokalsneath
        * sqeuclidean
        * yule
        * wminkowski

        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.
    metric_kwds : dict, default={}
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance. Should be in calling order.
    n_neighbors : int, default=10
        The number of edges to keep for each point. Should be lower than the value
        used on Projector.
    skip_embedding : bool, default=False
        A flag indicating whether to skip computing the embedding.
    reset_local_connectivity : bool, default=False
        A flag to enable normalising edge weights after applying the lens.

        UMAP's embedding assumes each point has at least one fully connected
        edge. This assumption can be enforced by enabling this flag. Embeddings
        tend to vary more smoothly when this setting is enabled.
    **kwargs : dict, optional
        UMAP arguments to overwrite projector's values. These can overwrite
        layout parameters to fine-tune the filtered embedding.

    Returns
    -------
    clone : UMAP
        A clone of projector with filtered `graph_` and updated `embedding_`.

        Other attributes are soft-copies, modifying them also modifies the input
        object's attributes!
    """
    # Validations
    if not hasattr(projector, "graph_"):
        raise AttributeError("Projector has not been fit yet.")
    N = projector._raw_data.shape[0]
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] != N:
        raise ValueError("Lens does not have a single value for every data point.")
    if len(values.shape) == 1:
        values = values[None].T
    if n_neighbors >= projector.n_neighbors:
        raise ValueError(
            "n_neighbours too large, should be lower than used for `projector`."
        )

    # Extract metric function
    if callable(metric):
        _metric_fn = metric
    elif metric in dist.named_distances:
        _metric_fn = dist.named_distances[metric]
    else:
        raise ValueError("metric is not a function nor a recognized name")

    # Fill in keywords
    if metric_kwds is None:
        metric_fn = _metric_fn
    elif isinstance(metric_kwds, dict):
        metric_vals = tuple(metric_kwds.values())

        @nb.njit(nb.float32(nb.float32[::1], nb.float32[::1]), fastmath=True)
        def partial_fn(x, y):
            return _metric_fn(x, y, *metric_vals)

        metric_fn = partial_fn
    else:
        raise ValueError("metric_kwds is not a dictionary.")

    # Shallow-copy the projector with new embedding and graph and overwrite kwargs
    clone = _clone_projector(projector, **kwargs)

    # Extract smallest `n_neighbor` edges
    knn_indices = _extract_local_lens_edges(
        clone.graph_.indices, clone.graph_.indptr, values, metric_fn, n_neighbors
    )

    # Filter graph to keep only the extracted edges
    clone.graph_ = csr_array(
        _apply_local_mask_filter(
            clone.graph_.data,
            clone.graph_.indices,
            clone.graph_.indptr,
            knn_indices,
        ),
        shape=(N, N),
    )
    if reset_local_connectivity:
        umap_reset_connectivity(clone.graph_, True)

    # Update embedding or remove related attributes
    _update_embedding(clone, skip_embedding)

    return clone


def embed_graph(
    projector: UMAP,
    repulsion_strengths: Optional[list[float]] = None,
    epoch_sequence: Optional[list[int]] = None,
):
    """
    (Re)-Computes UMAP embedding of (fitted) UMAP projector. Uses
    the current embedding as initialisation if one is present.

    Parameters
    ----------
    projector : UMAP
        A fitted UMAP object (at least with transform_mode='graph'). The
        `embedding_` field will be created or updated if present.
    repulsion_strengths : list of float, optional
        Runs embedding procedure with given repulsion strengths in sequence,
        allowing fine-grained repulsion ramps that can improve convergence.
        Defaults to the projector's `repulsion_strength` attribute.
    epoch_sequence : list of int, optional
        The number of epochs to run each repulsion strength. If not given, uses
        the projector's `n_epochs` attribute.
    Returns
    -------
    projector : UMAP
        Projector with filled `embedding_` field.

        This return value is provided to chain functions but can be ignored as
        the given UMAP object is updated inplace.
    """
    # Fill default parameters
    if repulsion_strengths is None:
        repulsion_strengths = [projector.repulsion_strength]
    if epoch_sequence is None:
        epoch_sequence = [projector.n_epochs]
    elif len(epoch_sequence) > len(repulsion_strengths):
        warn("More epochs than repulsion strengths given. Ignoring extra epochs.")

    # Run embedding for each repulsion strength
    disconnected_vertices = np.array(projector.graph_.sum(axis=1)).flatten() == 0
    for i, repulsion in enumerate(repulsion_strengths):
        # Apply repulsion strength
        if len(epoch_sequence) > i:
            projector.n_epochs = epoch_sequence[i]
        projector.repulsion_strength = repulsion
        
        # Extract initialisation
        if not hasattr(projector, "embedding_"):
            init = projector.init
        else:
            init = projector.embedding_
            # Fill in reasonable far-away coordinates for disconnected vertices
            # So they don't introduce repulsion in an occupied region of the
            # embedding
            init[np.any(np.isnan(init), axis=1), :] = np.array(
                np.repeat([-8.0], init.shape[1])
            )
        
        # Run embedding procedure
        projector.embedding_, aux_data = projector._fit_embed_data(
            projector._raw_data,
            projector.n_epochs,
            init,
            check_random_state(projector.random_state),
        )
        
        # Assign any points that are fully disconnected from our manifold(s) to
        # have embedding coordinates of np.nan. These will be filtered by our
        # plotting functions automatically.
        if len(disconnected_vertices) > 0:
            projector.embedding_[disconnected_vertices] = np.full(
                projector.n_components, np.nan
            )
        
        # Store Densmap radii if needed
        if projector.output_dens:
            projector.rad_orig_ = aux_data["rad_orig"]
            projector.rad_emb_ = aux_data["rad_emb"]

    return projector


def tile_components(
    projector: UMAP,
    component_labels: ArrayLike,
    secondary_axis: Optional[ArrayLike] = None,
    padding: int = 20,
):
    """
    Tiles connected components along the vertical axis. Ensures the within-component
    structure remains visible. Maintains component order (as indicated by their)
    label value and centers components horizontally (after PCA transformation).

    Parameters
    ----------
    projector : UMAP
        A fitted UMAP object with embedding computed.

        The embedding is updated inplace.
    component_labels : array_like
        An integer label (zero to N) indicating which connected component each point
        belongs to.
    secondary_axis : array_like
        A feature to align along the horizontal axis. Components are flipped to have
        the highest correlation between their x-coordinate and the given feature.
    padding : int, default=20
        A padding percentage to separate each component.

    Returns
    -------
    projector : UMAP
        The fitted UMAP object with updated embedding.

        This return value is provided to chain functions but can be ignored as
        the given UMAP object is updated inplace.
    """
    y_offset = 0
    pad_factor = padding / 100 + 1
    nan_mask = np.any(np.isnan(projector.embedding_), axis=1)
    for c in np.unique(component_labels):
        # Extract component
        mask = (component_labels == c) & (~nan_mask)
        X = projector.embedding_[mask]
        X = PCA(n_components=2).fit_transform(X)

        # Bounding box-shift
        ix, iy = np.min(X, axis=0)
        ax, ay = np.max(X, axis=0)
        X -= np.array([[(ax + ix) / 2, iy - y_offset]])
        y_offset += pad_factor * (ay - iy)

        # Flip secondary axis?
        if secondary_axis is not None:
            corr = np.correlate(secondary_axis[mask], X[:, 0])
            if corr < 0:
                X[:, 0] *= -1

        # Store the updated coordinates
        projector.embedding_[mask, :] = X

    return projector


def extract_embedding(projector: UMAP):
    """
    Returns the first two embedding coordinates as individual variables.

    Parameters
    ----------
    projector : UMAP
        A fitted UMAP object to extract the embedding from.

        The returned arrays are views into `embedding_`.

    Returns
    -------
    x, y : array_like
        Views with the embedded coordinates.

        Changes to the returned variables also change the original
        UMAP object's `embedding_` field.
    """
    XY = projector.embedding_
    return XY[:, 0], XY[:, 1]


def _clone_projector(projector: UMAP, **kwargs):
    """
    Shallow-copy the projector with overwrite kwargs.
    """
    clone = copy(projector)
    for key, value in kwargs.items():
        if hasattr(clone, key):
            setattr(clone, key, value)
        else:
            warn(f"{key} attribute does not exist on UMAP object. Value not set.")
    return clone


def _update_embedding(clone: UMAP, skip_embedding: bool):
    """
    Either updates the embedding or removes related attributes when this step is skipped.
    """
    # Recompute layout
    if not skip_embedding:
        embed_graph(clone)
    else:
        # Delete attributes related to the embedding
        del clone.embedding_
        if clone.output_dens:
            del clone.rad_orig_
            del clone.rad_emb_
