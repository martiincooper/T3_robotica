"""Geometric estimation of G2T articulation angles from 2-D LiDAR scans.

The pipeline (see ``docs/methodology.md`` for derivations):

    raw scan -> preprocess -> ROI -> Euclidean clusters -> RANSAC line fit
             -> trailer identification -> articulation-angle estimation
             -> (optional) 1-D Kalman smoothing

Each module is implemented as a stateless function or thin class so that
single-scan results are easy to unit-test and to reproduce.
"""

from .pipeline import GeometricEstimator, EstimationResult  # noqa: F401
