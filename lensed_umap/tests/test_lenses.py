"""
API Tests for lensed UMAP.
"""

import pytest
import numpy as np
import numba as nb
from copy import copy
from umap import UMAP
from sklearn.datasets import make_blobs
from sklearn.utils._testing import assert_raises
from lensed_umap import (
    apply_lens,
    apply_mask,
    apply_local_mask,
    embed_graph,
    extract_embedding,
)

blobs, blob_labels = make_blobs(
    n_samples=50,
    centers=[(-0.75, 2.25), (2.0, -0.5)],
    cluster_std=0.2,
    random_state=3,
)
projector = UMAP().fit(blobs)
empty_projector = UMAP()
lens = np.random.uniform(size=blobs.shape[0])
masker = UMAP(n_neighbors=20).fit(lens[None].T)
half_masker = UMAP(n_neighbors=20).fit(lens[:25][None].T)
local_lens = np.random.uniform(size=(blobs.shape[0], 2))


def test_extract_embedding():
    x, y = extract_embedding(projector)
    assert np.all(x == projector.embedding_[:, 0])
    assert np.all(y == projector.embedding_[:, 1])


def test_embed_graph():
    # No missing values
    embed_graph(projector)
    assert hasattr(projector, "embedding_")

    # Missing values
    clone = copy(projector)
    clone.embedding_ = clone.embedding_.copy()
    clone.embedding_[0, 0] = np.nan
    clone.embedding_[0, 1] = np.nan
    embed_graph(clone)
    assert hasattr(clone, "embedding_")

    # No embedding present
    del clone.embedding_
    embed_graph(clone)
    assert hasattr(clone, "embedding_")


def test_apply_lens_inputs_invalid():
    # Invalid values
    assert_raises(AttributeError, lambda: apply_lens(empty_projector, lens))
    assert_raises(ValueError, lambda: apply_lens(projector, []))
    assert_raises(
        ValueError, lambda: apply_lens(projector, lens, discretization="wrong")
    )
    assert_raises(ValueError, lambda: apply_lens(projector, lens[None]))
    assert_raises(ValueError, lambda: apply_lens(projector, lens, resolution=2))
    assert_raises(ValueError, lambda: apply_lens(projector, lens, resolution=1))
    assert_raises(ValueError, lambda: apply_lens(projector, lens, resolution=0))
    assert_raises(ValueError, lambda: apply_lens(projector, lens, resolution=-1))
    with pytest.warns(UserWarning):
        apply_lens(projector, lens, i_dont_exist=2)


def test_apply_lens_inputs_valid_1():
    l = apply_lens(projector, lens)
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)


def test_apply_lens_inputs_valid_2():
    l = apply_lens(projector, lens.tolist())
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)


def test_apply_lens_inputs_valid_3():
    l = apply_lens(projector, lens, resolution=10)
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)


def test_apply_lens_inputs_valid_4():
    l = apply_lens(projector, lens, resolution=10, circular=True)
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)


def test_apply_lens_inputs_valid_5():
    l = apply_lens(projector, lens, discretization="balanced")
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)


def test_apply_lens_inputs_valid_6():
    l = apply_lens(
        projector, lens, resolution=10, circular=True, discretization="balanced"
    )
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)


def test_apply_lens_inputs_valid_7():
    l = apply_lens(projector, lens, skip_embedding=True)
    assert not hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)


def test_apply_lens_inputs_valid_8():
    l, bins = apply_lens(projector, lens, ret_bins=True)
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)
    assert len(bins) == blobs.shape[0]


def test_apply_lens_inputs_valid_9():
    l = apply_lens(projector, lens, min_dist=0.2)
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)


def test_apply_mask_inputs_invalid():
    # Invalid values
    assert_raises(AttributeError, lambda: apply_mask(empty_projector, masker))
    assert_raises(AttributeError, lambda: apply_mask(projector, empty_projector))
    assert_raises(ValueError, lambda: apply_mask(projector, half_masker))
    with pytest.warns(UserWarning):
        apply_mask(projector, masker, i_dont_exist=2)


def test_apply_mask_inputs_valid_1():
    l = apply_mask(projector, masker)
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)


def test_apply_mask_inputs_valid_2():
    l = apply_mask(projector, masker, min_dist=0.2)
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)


def test_apply_mask_inputs_valid_3():
    l = apply_mask(projector, masker, skip_embedding=True)
    assert not hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)


def test_apply_local_mask_inputs_invalid():
    # Invalid values
    assert_raises(AttributeError, lambda: apply_local_mask(empty_projector, local_lens))
    assert_raises(ValueError, lambda: apply_local_mask(projector, []))
    assert_raises(
        ValueError,
        lambda: apply_local_mask(projector, local_lens, metric="wrong"),
    )
    assert_raises(
        ValueError,
        lambda: apply_local_mask(projector, local_lens, metric_kwds=()),
    )
    assert_raises(
        ValueError,
        lambda: apply_local_mask(projector, local_lens, metric_kwds="something else"),
    )
    assert_raises(
        ValueError,
        lambda: apply_local_mask(projector, local_lens, metric_kwds=90),
    )
    assert_raises(
        nb.core.errors.TypingError,
        lambda: apply_local_mask(projector, local_lens, metric_kwds={"p": 1}),
    )
    assert_raises(
        ValueError,
        lambda: apply_local_mask(
            projector, local_lens, n_neighbors=projector.n_neighbors
        ),
    )
    assert_raises(
        ValueError,
        lambda: apply_local_mask(
            projector, local_lens, n_neighbors=projector.n_neighbors + 1
        ),
    )
    with pytest.warns(UserWarning):
        apply_local_mask(projector, local_lens, i_dont_exist=2)


def test_apply_local_mask_inputs_valid_1():
    l = apply_local_mask(projector, local_lens)
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)


def test_apply_local_mask_inputs_valid_2():
    l = apply_local_mask(projector, local_lens, min_dist=0.2)
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)


def test_apply_local_mask_inputs_valid_3():
    l = apply_local_mask(projector, local_lens, skip_embedding=True)
    assert not hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)


def test_apply_local_mask_inputs_valid_4():
    l = apply_local_mask(projector, local_lens, metric="l2")
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)


def test_apply_local_mask_inputs_valid_5():
    @nb.njit()
    def fn(x, y):
        return np.min(x - y)

    l = apply_local_mask(projector, local_lens, metric=fn)
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)


def test_apply_local_mask_inputs_valid_6():
    l = apply_local_mask(
        projector, local_lens, metric="minkowski", metric_kwds={"p": 3}
    )
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)


def test_apply_local_mask_inputs_valid_7():
    l = apply_local_mask(projector, local_lens, n_neighbors=7)
    assert hasattr(l, "embedding_")
    assert hasattr(l, "graph_")
    assert len(l.graph_.indices) < len(projector.graph_.indices)
    assert np.any(l.embedding_ != projector.embedding_)
