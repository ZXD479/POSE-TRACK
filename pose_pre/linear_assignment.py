"""
Modified code from https://github.com/nwojke/deep_sort
"""

from __future__ import absolute_import

import numpy as np


def linear_assignment(cost_matrix):
  try:
    from sklearn.utils.linear_assignment_ import linear_assignment
    return linear_assignment(cost_matrix)
  except ImportError:
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


INFTY_COST = 1e+5

def min_cost_matching(
        distance_metric, max_distance, tracks, detections, appce, ious_dists=None, track_indices=None,
        detection_indices=None):

    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices, [] 

    cost_matrix_a = distance_metric(tracks, detections, track_indices, detection_indices, appce, ious_dists)
    cost_matrix   = cost_matrix_a

    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    # cost_matrix                             = np.log(cost_matrix)
    # max_distance                            = np.log(max_distance)
    
    indices = linear_assignment(cost_matrix)  

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
            
    return matches, unmatched_tracks, unmatched_detections, cost_matrix

def matching_simple(distance_metric, max_distance, cascade_depth, tracks, detections, appce, ious_dists=None, track_indices=None, detection_indices=None):
    
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices

    matches, _, unmatched_detections, cost = min_cost_matching(distance_metric, max_distance, tracks, detections, appce, ious_dists, track_indices, unmatched_detections)

    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections, cost
