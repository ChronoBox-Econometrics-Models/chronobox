"""Dataset loading utilities.

This module is kept for backward compatibility.
The main implementation is in chronobox.datasets.__init__.
"""

from __future__ import annotations

from chronobox.datasets import DATASET_CATALOG, list_datasets, load_dataset

__all__ = ["DATASET_CATALOG", "list_datasets", "load_dataset"]
