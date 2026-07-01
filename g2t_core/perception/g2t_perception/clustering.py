"""Euclidean clustering for 2-D point clouds.

We use a KD-Tree based fixed-radius (eps) neighbour query — equivalent to
DBSCAN with ``min_samples = 1`` — because in the trailer-side-wall use
case spurious points are already few (median filter + ROI) and a simple
connected-components grouping suffices.

Rationale (see ``docs/methodology.md``):
  * Full DBSCAN with ``min_samples > 1`` over-prunes the corner points of
    the trailer rectangles, where local density drops because two surfaces
    meet at an angle.
  * Our downstream RANSAC line fit imposes a far stronger inlier criterion
    than density-based clustering would.
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.spatial import cKDTree


def euclidean_clusters(points: np.ndarray,
                       eps: float = 0.15,
                       min_points: int = 8,
                       max_points: int = 1000) -> List[np.ndarray]:
    """Connected-components clustering with a fixed radius.

    Returns a list of arrays, each of shape ``(k_i, 2)``, sorted by
    descending cardinality.
    """
    if points.shape[0] == 0:
        return []
    tree = cKDTree(points)
    visited = np.zeros(points.shape[0], dtype=bool)
    clusters: List[np.ndarray] = []
    for seed in range(points.shape[0]):
        if visited[seed]:
            continue
        # BFS from `seed`
        queue = [seed]
        comp = []
        while queue:
            i = queue.pop()
            if visited[i]:
                continue
            visited[i] = True
            comp.append(i)
            neigh = tree.query_ball_point(points[i], r=eps)
            for j in neigh:
                if not visited[j]:
                    queue.append(j)
        if min_points <= len(comp) <= max_points:
            clusters.append(points[np.array(comp)])
    clusters.sort(key=lambda a: -a.shape[0])
    return clusters
