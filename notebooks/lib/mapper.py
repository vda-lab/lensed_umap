import numpy as np
import networkx as nx
import lensed_umap as lu
from gtda.mapper import FirstHistogramGap
from gtda.mapper import OneDimensionalCover


def compute_mapper_embedding(
    distances,
    binary_filter,
    real_filter,
    kind="balanced",
    resolution=10,
    overlap=1 / 2,
    max_clusters_per_interval=4,
    layout_program="neato",
):
    """
    Computes a Mapper network with force-directed layout using a binary filter
    and real-valued filter. The FirstHistogramGap clustering algorithm is
    hardcoded.
    """
    # Compute interval clusters
    cover = OneDimensionalCover(
        kind=kind, n_intervals=resolution, overlap_frac=overlap
    ).fit(real_filter)
    real_pullbacks = [
        np.where((real_filter >= begin) & (real_filter <= end))[0]
        for begin, end in zip(cover.left_limits_, cover.right_limits_)
    ]
    pullback_sets = [
        pts[binary_filter[pts] == target]
        for pts in real_pullbacks
        for target in [True, False]
    ]
    distances_sets = [distances[:, pts][pts, :] for pts in pullback_sets]
    label_sets = [
        compute_clusters_in_interval(dists, max_clusters_per_interval)
        for dists in distances_sets
    ]

    # Convert to networkx graph
    nodes, node_attrs, edges, edge_attrs = to_nodes_and_edges(
        pullback_sets, label_sets, cover.left_limits_, cover.right_limits_
    )
    g = to_nx_graph(nodes, node_attrs, edges, edge_attrs)

    # Compute the layout
    pos = nx.nx_agraph.graphviz_layout(g, prog=layout_program)
    coords = np.fromiter(pos.values(), dtype="f4,f4").view(np.float32).reshape(-1, 2)
    tiled = lu.tile_components(
        Embedder(coords),
        np.asarray(
            [binary_filter[node_attrs[node]["points"]].any() for node in pos.keys()]
        ),
        np.asarray(
            [real_filter[node_attrs[node]["points"]].mean() for node in pos.keys()]
        ),
        padding=5,
    ).embedding_
    for i, n in enumerate(pos.keys()):
        g.nodes[n]["pos"] = tiled[i, :]

    return g, cover.left_limits_, cover.right_limits_


def compute_clusters_in_interval(dist, max_clusters_per_interval):
    if dist.shape[0] == 0:
        return np.zeros(dist.shape[0])
    c = FirstHistogramGap(
        n_bins_start=20,
        affinity="precomputed",
        max_fraction=min(1, max_clusters_per_interval / dist.shape[0]),
    ).fit(dist)
    if np.all(c.labels_ == -1):
        return np.zeros(dist.shape[0])
    return c.labels_


def to_nodes_and_edges(pullback_sets, label_sets, starts, ends):
    cnt = 0

    def inc():
        nonlocal cnt
        cnt += 1
        return cnt

    node_attrs = {
        inc(): {
            "level": int(i),
            "start": float(start),
            "end": float(end),
            "size": int(np.sum(label == j)),
            "points": pts[label == j].astype(int).tolist(),
        }
        for i, (pts, label, start, end) in enumerate(
            zip(
                pullback_sets,
                label_sets,
                np.concatenate((starts, starts)),
                np.concatenate((ends, ends)),
            )
        )
        for j in np.unique(label)
        if j >= 0
    }
    nodes = np.arange(1, len(node_attrs), dtype=int).tolist()
    edges = []
    edge_attrs = {}

    for i in node_attrs.keys():
        for j in node_attrs.keys():
            if i == j:
                continue
            overlap = len(
                np.intersect1d(node_attrs[i]["points"], node_attrs[j]["points"])
            )
            if overlap > 0:
                edges += [(i, j)]
                edge_attrs[(i, j)] = {"weight": float(overlap)}

    return nodes, node_attrs, edges, edge_attrs


def to_nx_graph(nodes, node_attrs, edges, edge_attrs):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    nx.set_node_attributes(g, node_attrs)
    nx.set_edge_attributes(g, edge_attrs)
    return g


class Embedder:
    def __init__(self, embedding):
        self.embedding_ = embedding
