"""
Runtime extension that lets the UNCHANGED paper code (xad/, experiments/)
run on the extra datasets prepared under data/prepared/.

Nothing in xad/ or experiments/ is edited. `install()` (called from
datasets/register/sitecustomize.py when XAD_EXTRA=1) does three
things in-process:

  1. registers one `PreparedLoader` per prepared dataset in
     xad.data_loaders.DataLoaderFactory,
  2. adds its windowing/calibration settings to
     experiments.pipeline.DATASET_CONFIG (read from that dataset's meta.json),
  3. widens argparse `--dataset` choices so the exp_*.py scripts accept the
     new names.

Prepared format (written by datasets/prepare.py), per dataset:
  prepared/<name>/data.npz   X (T, D) float32, y (T,) int8 point labels,
                             split (T,) int8 {0 train, 1 val, 2 test},
                             seg (T,) int32 recording id
  prepared/<name>/meta.json  feature_names, config, split policy, provenance

The split is fixed at preparation time and stored per row, so every seed and
every experiment sees the identical train/val/test partition. Within a split,
each contiguous run of one recording is windowed on its own, so no window
straddles a recording or split boundary (same guarantee as
BaseDataLoader._load_and_split_temporal). The scaler is fit on train rows
only.
"""

import json
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PREPARED = os.path.join(REPO, 'data', 'prepared')


def available() -> List[str]:
    if not os.path.isdir(PREPARED):
        return []
    return sorted(d for d in os.listdir(PREPARED)
                  if os.path.isfile(os.path.join(PREPARED, d, 'meta.json'))
                  and os.path.isfile(os.path.join(PREPARED, d, 'data.npz')))


def read_meta(name: str) -> dict:
    with open(os.path.join(PREPARED, name, 'meta.json')) as f:
        return json.load(f)


def _make_loader_class(name: str):
    from xad.data_loaders import BaseDataLoader

    class PreparedLoader(BaseDataLoader):
        dataset_name = name

        def __init__(self, data_path: Optional[str] = None, sequence_length: int = 100,
                     scaler_type: str = 'robust', random_state: int = 42,
                     window_overlap: float = 0.5):
            super().__init__(data_path or os.path.join(PREPARED, name), sequence_length,
                             scaler_type, random_state, window_overlap)
            self.meta = read_meta(name)
            self.feature_names = list(self.meta['feature_names'])

        def _arrays(self):
            z = np.load(os.path.join(PREPARED, name, 'data.npz'))
            return (z['X'].astype(np.float32), z['y'].astype(np.int64),
                    z['split'].astype(np.int8), z['seg'].astype(np.int64))

        def _runs(self, split_id: int, X, y, split, seg) -> List[Tuple[np.ndarray, np.ndarray]]:
            """Contiguous (recording, split) runs, in stored order."""
            idx = np.flatnonzero(split == split_id)
            if len(idx) == 0:
                return []
            breaks = np.flatnonzero((np.diff(idx) != 1) | (np.diff(seg[idx]) != 0)) + 1
            return [(X[r], y[r]) for r in np.split(idx, breaks)]

        def load_data(self):
            X, y, _, _ = self._arrays()
            return X, y

        def load_segments(self):
            X, y, split, seg = self._arrays()
            return [run for s in (0, 1, 2) for run in self._runs(s, X, y, split, seg)]

        def load_and_split(self, test_size: float = 0.2, val_size: float = 0.2,
                           normalize: bool = True, temporal_split: bool = True,
                           stratify: bool = True):
            # test_size/val_size are ignored on purpose: the partition is the
            # one fixed in meta.json['split_policy'] at preparation time.
            if not temporal_split:
                raise ValueError(f'{name}: only the fixed temporal split is supported')
            X, y, split, seg = self._arrays()
            parts = {s: self._runs(s, X, y, split, seg) for s in (0, 1, 2)}

            if normalize:
                self.scaler = self._get_scaler()
                self.scaler.fit(X[split == 0])
                transform = self.scaler.transform
            else:
                transform = lambda a: a  # noqa: E731

            def windows_for(runs):
                xs, ys = [], []
                for d, l in runs:
                    if len(d) < self.sequence_length:
                        continue
                    W, wl = self.create_sequences(transform(d).astype(np.float32), l)
                    if len(W) > 0:
                        xs.append(W)
                        ys.append(wl)
                if not xs:
                    return np.empty((0, self.sequence_length, X.shape[1]), np.float32), np.empty((0,))
                return np.concatenate(xs), np.concatenate(ys)

            out = [windows_for(parts[s]) for s in (0, 1, 2)]
            (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = out
            for nm, yy in (('train', y_tr), ('val', y_va), ('test', y_te)):
                if len(yy) == 0:
                    raise ValueError(f'{name}: empty {nm} split after windowing')
                if nm == 'test' and yy.mean() == 0.0:
                    warnings.warn(f'{name}: test split has no anomalous windows')
            return X_tr, X_va, X_te, y_tr, y_va, y_te

    PreparedLoader.__name__ = f'Prepared_{name}'
    return PreparedLoader


_INSTALLED = False


def install(verbose: bool = False) -> List[str]:
    global _INSTALLED
    names = available()
    if _INSTALLED:
        return names

    import argparse
    _orig_add_argument = argparse._ActionsContainer.add_argument

    def add_argument(self, *args, **kwargs):
        if '--dataset' in args and kwargs.get('choices') and 'skab' in kwargs['choices']:
            kwargs['choices'] = list(kwargs['choices']) + [n for n in names if n not in kwargs['choices']]
        return _orig_add_argument(self, *args, **kwargs)

    argparse._ActionsContainer.add_argument = add_argument

    from xad.data_loaders import DataLoaderFactory
    from experiments import pipeline as pl
    for n in names:
        DataLoaderFactory.register_loader(n, _make_loader_class(n))
        cfg = read_meta(n)['config']
        pl.DATASET_CONFIG[n] = {
            'sequence_length': int(cfg['sequence_length']),
            'window_overlap': float(cfg['window_overlap']),
            'calibration_window': int(cfg['calibration_window']),
            'smoothness_weight': float(cfg.get('smoothness_weight', 0.0)),
        }
    _INSTALLED = True
    if verbose:
        print(f'[xad_extra] registered {len(names)} extra datasets: {names}')
    return names
