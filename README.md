# frameless_eval

WIP repo accompanying the following paper:
> Lose the Frames: Event-based Metrics for Efficient Music Structure Analysis Evaluations
> Qingyang Xi, Brian McFee. International Society for Music Information Retrieval (ISMIR) conference. 2025.


It currently hosts frameless versions of the following metrics:
- `pairwise`: Pairwise Clustering Score
- `vmeasure`: V-Measure (Normalized Mutual Information)
- `lmeasure`: L-Measure (Hierarchical Label Metric)

They have the same signature as `mir_eval`'s metrics, so you can use them as drop-in replacements.
See example usage in the notebook.

## Installation

```bash
conda env create -f environment.yml
conda activate frameless-eval
pip install -e .
```




