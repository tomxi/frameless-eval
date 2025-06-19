from typing import List, Tuple, Callable, Union
import mir_eval
import numpy as np

__all__ = ["meet", "align_hier", "make_common_itvls", "labels_at_ts"]

def meet(gridded_hier_labels: List[np.ndarray], compare_func: Callable = np.equal, mono: bool = False) -> np.ndarray:
    """
    Compute the meet matrix for a hierarchy of labels.
    hier_labels.shape = (depth, n_seg). output shape = (n_seg, n_seg)
    compare_func needs to support numpy broadcasting.
    """
    hier_labels = np.array(gridded_hier_labels)
    # Using broadcasting to compute the outer comparison for each level.
    meet_per_level = compare_func(hier_labels[:, :, None], hier_labels[:, None, :])
    max_depth = meet_per_level.shape[0]

    # Create an array representing the level numbers (starting from 1)
    level_indices = np.arange(1, max_depth + 1)[:, None, None]
    if not mono:
        # Deepest level where the labels meet
        depths = np.max(meet_per_level * level_indices, axis=0)
    else:
        # Shallowest level where the labels stops meeting
        depths = max_depth - np.max(
            ~meet_per_level * np.flip(level_indices, axis=0), axis=0
        )

    return depths.astype(int)

def align_hier(ref_itvls: List[np.ndarray], ref_labels: List[List[str]], est_itvls: List[np.ndarray], est_labels: List[List[str]]) -> Tuple[List[np.ndarray], List[List[str]], List[np.ndarray], List[List[str]]]:
    # First, find the maximum length of the reference
    _, t_end = mir_eval.hierarchy._hierarchy_bounds(ref_itvls)

    # Pre-process the intervals to match the range of the reference,
    # and start at 0
    new_ref_itvls, new_ref_labels = mir_eval.hierarchy._align_intervals(
        ref_itvls, ref_labels, t_min=0.0, t_max=None
    )
    new_est_itvls, new_est_labels = mir_eval.hierarchy._align_intervals(
        est_itvls, est_labels, t_min=0.0, t_max=t_end
    )
    return new_ref_itvls, new_ref_labels, new_est_itvls, new_est_labels

def labels_at_ts(hier_itvls: list, hier_labels: list, ts: np.ndarray):
    """
    get label at ts for all levels in a hierarchy
    """
    results = []
    for itvls, labs in zip(hier_itvls, hier_labels):
        result = _label_at_ts(itvls, labs, ts)
        results.append(result)
    return results

def make_common_itvls(
    hier_itvls1: List[np.ndarray],
    hier_labels1: List[List[str]],
    hier_itvls2: List[np.ndarray],
    hier_labels2: List[List[str]],
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Label condition is a element of labels2,
    we slice the itvls and labels to get a new sub segmentation
    based on the conditioned on labels2 = label_condition.
    """
    # Strategy: build a new set of common_intervals, and labels of equal length, then do array indexing.
    # Merge boundaries and compute segment durations.
    common_bs = _common_boundaries(hier_itvls1 + hier_itvls2)
    common_itvls = mir_eval.util.boundaries_to_intervals(common_bs)

    # Find the label at each common boundary.
    gridded_labels1 = labels_at_ts(hier_itvls1, hier_labels1, common_bs[:-1])
    gridded_labels2 = labels_at_ts(hier_itvls2, hier_labels2, common_bs[:-1])
    return common_itvls, gridded_labels1, gridded_labels2

def _common_boundaries(list_of_itvls: List[np.ndarray]) -> np.ndarray:
    # Get the boundaries of both sets of intervals
    bs = set()
    for itvls in list_of_itvls:
        bs = bs.union(set(mir_eval.util.intervals_to_boundaries(itvls)))
    return np.array(sorted(bs))

def _label_at_ts(itvls: np.ndarray, labels: Union[List[str], np.ndarray], ts: np.ndarray) -> np.ndarray:
    """
    Assign labels to timestamps using interval boundaries.

    Parameters
    ----------
    itvls : np.ndarray
        An array of shape (n, 2) representing intervals.
    labels : array_like
        An array-like object of labels corresponding to each interval.
    ts : array_like
        Timestamps to be labeled.

    Returns
    -------
    np.ndarray
        An array of labels corresponding to each timestamp in ts.
    """
    boundaries = mir_eval.util.intervals_to_boundaries(itvls)
    labels = np.asarray(labels)
    extended = np.concatenate([labels, [labels[-1]]])  # Extend last label
    return extended[np.searchsorted(boundaries, np.atleast_1d(ts), side="right") - 1]
