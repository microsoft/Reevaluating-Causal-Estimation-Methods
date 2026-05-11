"""Helpers for loading pre-computed pickled outputs.

The repository ships with a ``saved_outputs/`` directory containing the
propensity score arrays and result DataFrames produced for the paper.  These
let users reproduce every figure without re-running the (expensive) AutoML
fitting loop.

Notes
-----
* Only **arrays** (propensity scores, optimal trim threshold) and **result
  DataFrames** are loadable across environments. The fitted model objects in
  the original Colab notebooks were pickled with custom wrapper classes that
  live on ``__main__`` and are not portable; if you need the models, set
  ``RERUN=True`` in the notebook and refit them.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Optional

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "saved_outputs"


def cache_path(filename: str, cache_dir: Optional[os.PathLike] = None) -> Path:
    """Return the absolute path to ``filename`` inside the cache directory."""
    base = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return base / filename


def load_pickle(filename: str, cache_dir: Optional[os.PathLike] = None) -> Any:
    """Load a pickle file from the cache directory.

    Raises ``FileNotFoundError`` if the file is missing so the caller can
    decide to refit instead.
    """
    path = cache_path(filename, cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"Cached output not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_or_compute(filename, compute_fn, rerun: bool = False,
                    cache_dir: Optional[os.PathLike] = None,
                    save: bool = False) -> Any:
    """Load ``filename`` from the cache, or fall back to ``compute_fn()``.

    Parameters
    ----------
    filename : str
        Name of the pickle inside ``saved_outputs/``.
    compute_fn : callable
        Zero-argument callable that produces the object when no cache is used.
    rerun : bool
        If True, ignore the cache and always recompute.
    cache_dir : path-like, optional
        Override the default ``saved_outputs/`` location.
    save : bool
        If True (and recomputing), save the result back to the cache directory.
    """
    if not rerun:
        try:
            return load_pickle(filename, cache_dir)
        except FileNotFoundError:
            pass  # fall through to recompute

    obj = compute_fn()

    if save:
        path = cache_path(filename, cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    return obj
