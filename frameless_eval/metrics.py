import itertools
from typing import Tuple, Union, List
import numpy as np
from scipy import stats
from scipy.sparse import coo_matrix
import pandas as pd
from mir_eval.util import f_measure

from frameless_eval.utils import meet, align_hier, make_common_itvls


def pairwise(
    ref_itvls: np.ndarray, 
    ref_labels: List[str], 
    est_itvls: np.ndarray, 
    est_labels: List[str], 
    beta: float = 1.0
) -> Tuple[float, float, float]:
    # make sure est the same lenght as ref
    aligned_hiers = align_hier([ref_itvls], [ref_labels], [est_itvls], [est_labels])
    # Make common grid
    common_itvls, ref_labs, est_labs = make_common_itvls(*aligned_hiers)

    # Get the segment durations and use as weights
    seg_dur = np.diff(common_itvls, axis=1).flatten()
    # Build label agreement maps (meet matrix)
    meet_ref = meet(ref_labs)
    meet_est = meet(est_labs)
    meet_both = meet_ref * meet_est

    seg_pair_size = np.outer(seg_dur, seg_dur)
    ref_sig_pair_size = np.sum(meet_ref * seg_pair_size)
    est_sig_pair_size = np.sum(meet_est * seg_pair_size)
    intersection_size = np.sum(meet_both * seg_pair_size)

    precision = intersection_size / est_sig_pair_size
    recall = intersection_size / ref_sig_pair_size
    return precision, recall, f_measure(precision, recall, beta=beta)


def vmeasure(
    ref_itvls: np.ndarray, 
    ref_labels: List[str], 
    est_itvls: np.ndarray, 
    est_labels: List[str], 
    beta: float = 1.0
) -> Tuple[float, float, float]:
    """
    V-measure

    Args:
        ref_itvls (_type_): _description_
        ref_labels (_type_): _description_
        est_itvls (_type_): _description_
        est_labels (_type_): _description_
        beta (float, optional): _description_. Defaults to 1.0.

    Returns:
        precision, recall, f_measure
    """
    
    # make sure est the same lenght as ref
    aligned_hiers = align_hier([ref_itvls], [ref_labels], [est_itvls], [est_labels])
    # Make common grid
    common_itvls, ref_labs, est_labs = make_common_itvls(*aligned_hiers)
    # Get the segment durations
    seg_dur = np.diff(common_itvls, axis=1).flatten()

    # Get the contingency matrix and normalize
    contingency, _, _ = _weighted_contingency(ref_labs, est_labs, seg_dur)
    contingency = contingency / np.sum(seg_dur)

    # Compute the marginals
    p_est = contingency.sum(axis=0)
    p_ref = contingency.sum(axis=1)

    # H(true | prediction) = sum_j P[estimated = j] *
    # sum_i P[true = i | estimated = j] log P[true = i | estimated = j]
    # entropy sums over axis=0, which is true labels

    true_given_est = p_est.dot(stats.entropy(contingency, base=2))
    pred_given_ref = p_ref.dot(stats.entropy(contingency.T, base=2))

    # Normalize conditional entropy by marginal entropy
    z_ref = stats.entropy(p_ref, base=2)
    z_est = stats.entropy(p_est, base=2)
    r = (1.0 - true_given_est / z_ref) if z_ref > 0 else 0
    p = (1.0 - pred_given_ref / z_est) if z_est > 0 else 0
    return p, r, f_measure(p, r, beta=beta)

def _weighted_contingency(
    ref_labels: np.ndarray, 
    est_labels: np.ndarray, 
    durations: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a weighted contingency matrix, used for V-measure.
    Each cell (i,j) sums the durations for frames with ref label i and est label j.
    """
    # Ensure all inputs are numpy arrays for consistency with NumPy 2.x
    ref_labels = np.asarray(ref_labels).flatten()
    est_labels = np.asarray(est_labels).flatten()
    durations = np.asarray(durations).flatten()
    
    # Check that all arrays have the same length
    if not (len(ref_labels) == len(est_labels) == len(durations)):
        raise ValueError(f"Array length mismatch: ref_labels={len(ref_labels)}, "
                        f"est_labels={len(est_labels)}, durations={len(durations)}")
    
    # Handle empty arrays
    if len(ref_labels) == 0:
        return np.zeros((0, 0), dtype=np.float64), np.array([]), np.array([])
    
    ref_classes, ref_idx = np.unique(ref_labels, return_inverse=True)
    est_classes, est_idx = np.unique(est_labels, return_inverse=True)
    
    # Double-check that the indices have the same length as durations (NumPy 2.x safety)
    if not (len(ref_idx) == len(est_idx) == len(durations)):
        raise ValueError(f"Index length mismatch after np.unique: ref_idx={len(ref_idx)}, "
                        f"est_idx={len(est_idx)}, durations={len(durations)}")

    contingency = coo_matrix(
        (durations, (ref_idx, est_idx)),
        shape=(len(ref_classes), len(est_classes)),
        dtype=np.float64,
    ).toarray()
    return contingency, ref_classes, est_classes

def _triplet_stats(
    ref_itvls: List[np.ndarray], 
    ref_labels: List[List[str]], 
    est_itvls: List[np.ndarray], 
    est_labels: List[List[str]], 
    transitive: bool = True
) -> pd.DataFrame:
    # align hierarchies and build common grid
    aligned = align_hier(ref_itvls, ref_labels, est_itvls, est_labels)
    common_itvls, ref_labs, est_labs = make_common_itvls(*aligned)

    # segment durations
    seg_dur = np.diff(common_itvls, axis=1).flatten()
    dur_sq = np.sum(seg_dur) ** 2

    # meet matrices
    ref_meet = meet(ref_labs)
    est_meet = meet(est_labs)

    # triplet counts per segment
    triplet_cap, triplet_ref = _segment_triplet_components(
        ref_meet, est_meet, seg_dur, transitive=transitive
    )
    _, triplet_est = _segment_triplet_components(
        est_meet, ref_meet, seg_dur, transitive=transitive
    )

    # assemble dataframe
    return pd.DataFrame(
        {
            "dur": seg_dur,
            "cap": triplet_cap / dur_sq,
            "ref": triplet_ref / dur_sq,
            "est": triplet_est / dur_sq,
        }
    )


def lmeasure(
    ref_itvls: List[np.ndarray], 
    ref_labels: List[List[str]], 
    est_itvls: List[np.ndarray], 
    est_labels: List[List[str]], 
    beta: float = 1.0
) -> Tuple[float, float, float]:
    # Now lmeasure_components returns a DataFrame with columns: dur, cap, ref, est
    l_parts = _triplet_stats(
        ref_itvls, ref_labels, est_itvls, est_labels, transitive=True
    )

    # segment durations and normalized weights
    durations = l_parts["dur"]
    weights = durations / durations.sum()

    # per‐segment precision & recall
    seg_precision = l_parts["cap"] / l_parts["est"]
    seg_precision[l_parts["est"] == 0] = 1
    seg_recall = l_parts["cap"] / l_parts["ref"]
    seg_recall[l_parts["ref"] == 0] = 1

    # weighted-average precision & recall
    precision = seg_precision.dot(weights)
    recall = seg_recall.dot(weights)

    return precision, recall, f_measure(precision, recall, beta=beta)


def _segment_triplet_components(
    meet_ref: np.ndarray, 
    meet_est: np.ndarray, 
    seg_dur: np.ndarray, 
    transitive: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    # Get out two list: weighted intersection and total of ref and est
    # given a segment idx, their relevance against each other segment.
    intersections = []
    normalizers = []
    for seg_idx in range(len(seg_dur)):
        ref_rel_against_seg_i = meet_ref[seg_idx, :]
        est_rel_against_seg_i = meet_est[seg_idx, :]

        # use count inversions to get normalizer and number of inversions
        inversions, normalizer = _compare_segment_rankings(
            ref_rel_against_seg_i,
            est_rel_against_seg_i,
            ref_w=seg_dur,
            est_w=seg_dur,
            transitive=transitive,
        )
        intersections.append(normalizer - inversions)
        normalizers.append(normalizer)

    return np.array(intersections), np.array(normalizers)


def _count_weighted_inversions(
    a: np.ndarray, 
    wa: np.ndarray, 
    b: np.ndarray, 
    wb: np.ndarray
) -> float:
    """
    Count weighted inversions between two arrays.
    An inversion is any pair (i, j) with a[i] >= b[j],
    contributing wa[i] * wb[j] to the sum.
    """
    ua, inv_a = np.unique(a, return_inverse=True)
    wa_sum = np.bincount(inv_a, weights=wa)
    ub, inv_b = np.unique(b, return_inverse=True)
    wb_sum = np.bincount(inv_b, weights=wb)

    inversions = 0.0
    i = 0
    j = 0
    while i < len(ua) and j < len(ub):
        if ua[i] < ub[j]:
            i += 1
        else:
            inversions += np.sum(wa_sum[i:]) * wb_sum[j]
            j += 1
    return inversions


def _compare_segment_rankings(
    ref: np.ndarray, 
    est: np.ndarray, 
    ref_w: Union[np.ndarray, None] = None, 
    est_w: Union[np.ndarray, None] = None, 
    transitive: bool = True
) -> Tuple[float, float]:
    """
    Compute weighted ranking disagreements between two lists.

    Parameters
    ----------
    ref : np.ndarray, shape=(n,)
        Reference ranked list.
    est : np.ndarray, shape=(n,)
        Estimated ranked list.
    ref_w : np.ndarray, shape=(n,), optional
        Weights for ref (default: ones).
    est_w : np.ndarray, shape=(n,), optional
        Weights for est (default: ones).
    transitive : bool, optional
        If True, compare all pairs of distinct ref levels;
        if False, compare only adjacent levels.

    Returns
    -------
    inversions : float
        Weighted inversion count: sum_{(i,j) in pairs} [inversions between est slices].
    normalizer : float
        Total weighted number of pairs considered.
    """
    n = len(ref)
    if ref_w is None:
        ref_w = np.ones(n)
    if est_w is None:
        est_w = np.ones(n)

    # The algo starts by sorting everything by ref's values for easy inversion counting.
    idx = np.argsort(ref)
    ref, est = ref[idx], est[idx]
    ref_w, est_w = ref_w[idx], est_w[idx]

    # Get the unique levels of values and their positions in the sorted array
    levels, pos = np.unique(ref, return_index=True)
    pos = list(pos) + [len(ref)]

    # For each group of segments that has the same level/label value, we get the summed weights.
    level_groups = {
        lvl: slice(start, end) for lvl, start, end in zip(levels, pos[:-1], pos[1:])
    }
    ref_level_weights = {lvl: np.sum(ref_w[level_groups[lvl]]) for lvl in levels}

    if transitive:
        level_pairs = itertools.combinations(levels, 2)
    else:
        level_pairs = [(levels[i], levels[i + 1]) for i in range(len(levels) - 1)]

    # Create two independent iterators over level_pairs.
    level_pairs, level_pairs_counts = itertools.tee(level_pairs)
    normalizer = float(
        sum(ref_level_weights[i] * ref_level_weights[j] for i, j in level_pairs_counts)
    )
    if normalizer == 0:
        return 0.0, 0.0

    # We already sorted by ref array, so we count inversions now.
    inversions = sum(
        _count_weighted_inversions(
            est[level_groups[l1]],
            est_w[level_groups[l1]],
            est[level_groups[l2]],
            est_w[level_groups[l2]],
        )
        for l1, l2 in level_pairs
    )
    return inversions, float(normalizer)
