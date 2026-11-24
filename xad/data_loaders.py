"""
Supported Datasets:
    - SKAB: Skoltech Anomaly Benchmark for industrial time series
    - SWaT: Secure Water Treatment testbed dataset
    - Custom: Flexible loader

Usage:
    # Load SKAB
    loader = DataLoaderFactory.create('skab', data_path='./SKAB/')
    X_train, X_test, y_train, y_test = loader.load_and_split()
    
    # Load SWaT
    loader = DataLoaderFactory.create('swat', data_path='./SWaT/')
    X_train, X_test, y_train, y_test = loader.load_and_split()
    
    # Load custom dataset
    loader = DataLoaderFactory.create('custom', data_path='./my_data/', 
                                       label_column='anomaly')
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List, Dict, Union, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings


@dataclass
class DatasetInfo:
    """Container for dataset information."""
    name: str
    n_samples: int
    n_features: int
    sequence_length: int
    n_anomalies: int
    anomaly_ratio: float
    feature_names: List[str]


class BaseDataLoader(ABC):
    """
    Abstract base class for dataset loaders.
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 50,
        scaler_type: str = 'robust',
        random_state: int = 42,
        window_overlap: float = 0.0
    ):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.scaler_type = scaler_type
        self.random_state = random_state
        self.window_overlap = window_overlap
        self.scaler: Optional[Any] = None
        self.feature_names: List[str] = []
        
    def _get_scaler(self):
        if self.scaler_type == 'robust':
            return RobustScaler()
        elif self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
    
    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pass

    def load_segments(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Return the dataset as a list of independent (data, labels) segments,
        each internally chronological. Used by `_load_and_split_temporal` to
        assign whole segments to train/val/test rather than slicing
        positions in a single concatenated array - correct for datasets that
        actually are one continuous recording (the default here: wraps
        `load_data()` as a single segment) as well as for datasets made of
        several independent recordings (override this, see
        `SKABDataLoader.load_segments`).
        """
        data, labels = self.load_data()
        if labels is None:
            labels = np.zeros(len(data))
        return [(data, labels)]

    def create_sequences(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        overlap: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create overlapping sequences from time series data.

        Args:
            data: 2D array  (n_timesteps, n_features)
            labels: 1D array of labels (optional)
            overlap: fraction of overlap between consecutive windows (e.g. 0.8
                for SKAB, 0.5 for SWaT). Defaults to
                self.window_overlap. Stride = sequence_length * (1 - overlap).

        Returns:
            sequences: 3D array of shape (n_sequences, sequence_length, n_features)
            sequence_labels: 1D array where label is 1 if any anomaly in window
        """
        overlap = self.window_overlap if overlap is None else overlap
        stride = max(1, int(round(self.sequence_length * (1 - overlap))))

        sequences = []
        sequence_labels = []

        for i in range(0, len(data) - self.sequence_length + 1, stride):
            sequences.append(data[i:i + self.sequence_length])

            if labels is not None:
                # Label sequence as anomaly if any point in window is anomaly
                seq_label = int(labels[i:i + self.sequence_length].any())
                sequence_labels.append(seq_label)

        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels) if labels is not None else None

        return sequences, sequence_labels

    def load_and_preprocess(
        self,
        normalize: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Kept for backward compatibility / exploratory use. Fits the scaler on
        the *entire* series before windowing. `load_and_split` (the path used
        by all experiments) does NOT use this — it fits the scaler on the
        training block only, to avoid leaking val/test statistics into
        preprocessing.
        """
        data, labels = self.load_data()

        if normalize:
            self.scaler = self._get_scaler()
            data = self.scaler.fit_transform(data)

        sequences, sequence_labels = self.create_sequences(data, labels)

        return sequences, sequence_labels

    def load_and_split(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        normalize: bool = True,
        temporal_split: bool = True,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split into train/val/test.

        temporal_split=True (default, and the only methodologically valid
        choice for a "temporal-aware" method): the *raw, pre-windowing*
        series is split chronologically into three contiguous blocks first
        (train block first, then val, then test), the scaler is fit on the
        train block only, and sliding windows are created independently
        within each block. This guarantees (a) no window straddles a
        train/val/test boundary, so overlapping windows (80% SKAB, 50% SWaT)
        cannot leak near-duplicate content across splits, (b) the scaler
        never sees val/test data, and (c) X_test (and X_val) preserve
        chronological order, which the sliding-window conformal predictor
        (Eq. 8) requires to make sense.

        temporal_split=False reproduces the old behavior (window first, then
        a random - optionally stratified - split of windows). This shuffles
        windows across splits and is kept only for backward compatibility /
        ablation; it should not be used to produce reported numbers.
        """
        if temporal_split:
            return self._load_and_split_temporal(test_size, val_size, normalize)
        return self._load_and_split_random(test_size, val_size, normalize, stratify)

    def _load_and_split_temporal(
        self,
        test_size: float,
        val_size: float,
        normalize: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split via `load_segments()` rather than a single concatenated array.

        For a genuinely single-recording dataset (one segment - the default,
        e.g. SWaT), this reduces to plain positional chronological slicing,
        same as before. For a multi-recording dataset (several segments,
        e.g. SKAB's ~35 independent experiment files), a single global
        chronological slice over the naive concatenation is not a meaningful
        timeline and can (and did, on real SKAB data) put an entire
        anomaly-free file's rows in the test split with nothing else,
        producing 0% test anomaly rate and undefined F1/ROC-AUC. Instead,
        whole segments are assigned to train/val/test (via a seeded shuffle
        of segment order, so the split is still deterministic and doesn't
        depend on filesystem file-listing order), and windows are created
        per segment so no window straddles a segment boundary either.
        """
        segments = self.load_segments()
        segments = [(d, (l if l is not None else np.zeros(len(d)))) for d, l in segments if len(d) > 0]
        if not segments:
            raise ValueError("No data segments available to split")

        if len(segments) == 1:
            data, labels = segments[0]
            n = len(data)
            n_test = int(round(n * test_size))
            n_val = int(round(n * val_size))
            n_train = n - n_val - n_test
            if n_train <= self.sequence_length:
                raise ValueError(
                    f"Training block ({n_train} timesteps) is too short for "
                    f"sequence_length={self.sequence_length}. Reduce test_size/val_size "
                    f"or sequence_length."
                )
            train_segs = [(data[:n_train], labels[:n_train])]
            val_segs = [(data[n_train:n_train + n_val], labels[n_train:n_train + n_val])]
            test_segs = [(data[n_train + n_val:], labels[n_train + n_val:])]
        else:
            rng = np.random.RandomState(self.random_state)
            order = rng.permutation(len(segments))
            total = sum(len(d) for d, _ in segments)
            target_train_end = total * (1 - test_size - val_size)
            target_val_end = target_train_end + total * val_size

            train_segs, val_segs, test_segs = [], [], []
            cum = 0
            for idx in order:
                d, l = segments[idx]
                if cum < target_train_end:
                    train_segs.append((d, l))
                elif cum < target_val_end:
                    val_segs.append((d, l))
                else:
                    test_segs.append((d, l))
                cum += len(d)

            for name, segs in (('train', train_segs), ('val', val_segs), ('test', test_segs)):
                if not segs:
                    raise ValueError(
                        f"Segment-based temporal split produced an empty '{name}' split "
                        f"across {len(segments)} segments (test_size={test_size}, "
                        f"val_size={val_size}). Try a different random_state, or "
                        f"test_size/val_size, or check for one oversized segment "
                        f"dominating the total timestep count."
                    )

        if normalize:
            self.scaler = self._get_scaler()
            self.scaler.fit(np.vstack([d for d, _ in train_segs]))
            transform = self.scaler.transform
        else:
            transform = lambda x: x  # noqa: E731

        def windows_for(segs):
            xs, ys = [], []
            for d, l in segs:
                if len(d) < self.sequence_length:
                    continue
                X, y = self.create_sequences(transform(d), l)
                if len(X) > 0:
                    xs.append(X)
                    ys.append(y)
            if not xs:
                n_features = segs[0][0].shape[1] if segs else 0
                return np.empty((0, self.sequence_length, n_features)), np.empty((0,))
            return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

        X_train, y_train = windows_for(train_segs)
        X_val, y_val = windows_for(val_segs)
        X_test, y_test = windows_for(test_segs)

        for name, y in (('train', y_train), ('val', y_val), ('test', y_test)):
            if len(y) > 0 and (y.mean() == 0.0 or y.mean() == 1.0):
                warnings.warn(
                    f"{self.__class__.__name__}: '{name}' split has a degenerate anomaly "
                    f"rate ({y.mean():.0%}) after temporal_split - detection metrics "
                    f"(F1/ROC-AUC) on this split will be meaningless (NaN/0). Consider a "
                    f"different random_state or test_size/val_size."
                )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _load_and_split_random(
        self,
        test_size: float,
        val_size: float,
        normalize: bool,
        stratify: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X, y = self.load_and_preprocess(normalize)

        stratify_labels = y if stratify and y is not None else None

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y if y is not None else np.zeros(len(X)),
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_labels
        )

        stratify_labels_val = y_train_val if stratify and y is not None else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size / (1 - test_size),
            random_state=self.random_state,
            stratify=stratify_labels_val
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def normal_only(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return only the normal (y==0) instances. Use this to filter detector
        TRAINING data and conformal CALIBRATION data - both require
        exchangeability with the normal population. Validation-for-model-
        selection and test data are deliberately not required to go through
        this filter: test data needs anomalies to evaluate detection at all,
        and this project uses the normal-filtered slice of the val split for
        both early-stopping monitoring and calibration (see experiments/pipeline.py).
        """
        mask = y == 0
        return X[mask], y[mask]
    
    def get_info(self) -> DatasetInfo:
        X, y = self.load_and_preprocess(normalize=False)
        
        return DatasetInfo(
            name=self.__class__.__name__,
            n_samples=len(X),
            n_features=X.shape[2],
            sequence_length=self.sequence_length,
            n_anomalies=int(y.sum()) if y is not None else 0,
            anomaly_ratio=float(y.mean()) if y is not None else 0.0,
            feature_names=self.feature_names
        )


class SKABDataLoader(BaseDataLoader):
    """
    Data loader for SKAB (Skoltech Anomaly Benchmark) dataset.
    
    SKAB is an industrial time series dataset for anomaly detection
    containing sensor readings from a testbed with anomaly labels.
    
    Expected directory structure:
        SKAB/
        ├── valve1/
        │   ├── file1.csv
        │   └── ...
        ├── valve2/
        │   └── ...
        └── other/
            └── ...
    """
    
    def __init__(
        self,
        data_path: str = './SKAB/',
        sequence_length: int = 50,
        scaler_type: str = 'robust',
        random_state: int = 42,
        exclude_normal: bool = False,
        window_overlap: float = 0.8
    ):
        super().__init__(data_path, sequence_length, scaler_type, random_state, window_overlap)
        self.exclude_normal = exclude_normal
    
    def load_segments(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Return one (data, labels) segment per CSV file, in a deterministic
        order. SKAB is not one continuous recording - each file is an
        independent experiment run (a distinct valve/pump/etc. session, or
        the dedicated anomaly-free run), typically starting normal and
        transitioning into (and sometimes back out of) an anomalous period.
        Concatenating all files into one array and then chronologically
        slicing that concatenation - which earlier versions of this loader
        did - is not a meaningful "timeline": it can, and did, put an entire
        anomaly-free file's rows in the test slice with zero other content,
        producing a test split with 0% anomalies. `_load_and_split_temporal`
        uses this method (not `load_data`) for SKAB so it can assign whole
        files to splits instead of slicing positions in an artificial
        concatenation - see `BaseDataLoader._load_and_split_temporal`.
        """
        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path does not exist: {self.data_path}")

        # Recursively find CSV files. Sorted for determinism: os.walk() does
        # not guarantee a stable file order across runs/filesystems, and the
        # file order determines which rows land in train/val/test.
        csv_files = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        csv_files.sort()

        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_path}")

        print(f"Found {len(csv_files)} CSV files in {self.data_path}")

        segments = []
        for filepath in csv_files:
            try:
                try:
                    df = pd.read_csv(filepath, sep=';', index_col='datetime', parse_dates=True)
                except:
                    try:
                        df = pd.read_csv(filepath, sep=',', index_col='datetime', parse_dates=True)
                    except:
                        try:
                            df = pd.read_csv(filepath, sep=';')
                        except:
                            df = pd.read_csv(filepath, sep=',')

                if df.empty:
                    continue

                data_columns = [
                    col for col in df.columns
                    if col not in ['anomaly', 'changepoint', 'datetime']
                ]

                if not data_columns:
                    continue

                if not self.feature_names:
                    self.feature_names = data_columns

                values = df[data_columns].values
                seg_labels = df['anomaly'].values if 'anomaly' in df.columns else np.zeros(len(values))
                segments.append((values, seg_labels))

            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
                continue

        if not segments:
            raise ValueError(f"No valid data loaded from {self.data_path}")

        return segments

    def load_data(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        segments = self.load_segments()
        combined_data = np.vstack([s[0] for s in segments])
        combined_labels = np.hstack([s[1] for s in segments])

        print(f"Loaded data shape: {combined_data.shape}")
        if combined_labels is not None:
            print(f"Anomaly ratio: {combined_labels.mean():.2%}")
        
        return combined_data, combined_labels


class SWaTDataLoader(BaseDataLoader):
    """
    Data loader for SWaT (Secure Water Treatment) dataset.
    
    SWaT is a benchmark dataset from a water treatment testbed
    with multiple attack scenarios.
    
    Expected file structure:
        SWaT/
        ├── SWaT_Dataset_Normal_v1.xlsx (or .csv)
        ├── SWaT_Dataset_Attack_v0.xlsx (or .csv)
        └── ...
    
    Or single combined file:
        SWaT/
        └── swat.csv
    """
    
    SENSOR_COLUMNS = [
        'FIT101', 'LIT101', 'MV101', 'P101', 'P102',
        'AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202',
        'P203', 'P204', 'P205', 'P206',
        'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304',
        'P301', 'P302',
        'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402', 'P403', 'P404',
        'UV401',
        'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503',
        'FIT504', 'P501', 'P502', 'PIT501', 'PIT502', 'PIT503',
        'FIT601', 'P601', 'P602', 'P603'
    ]
    
    def __init__(
        self,
        data_path: str = './SWaT/',
        sequence_length: int = 50,
        scaler_type: str = 'robust',
        random_state: int = 42,
        use_all_features: bool = True,
        window_overlap: float = 0.5
    ):
        super().__init__(data_path, sequence_length, scaler_type, random_state, window_overlap)
        self.use_all_features = use_all_features
    
    def load_data(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load SWaT data from files."""
        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path does not exist: {self.data_path}")
        

        files = os.listdir(self.data_path)
        

        combined_files = [f for f in files if 'swat' in f.lower() and f.endswith('.csv')]

        if combined_files:
            df = pd.read_csv(os.path.join(self.data_path, combined_files[0]))
            # Some SWaT distributions ship column headers with stray leading
            # whitespace (e.g. ' Timestamp', ' MV101') - strip it so column
            # names match the exclude-list checks below.
            df.columns = df.columns.str.strip()
        else:
            normal_files = [f for f in files if 'normal' in f.lower()]
            attack_files = [f for f in files if 'attack' in f.lower()]
            
            dfs = []
            
            for f in normal_files:
                filepath = os.path.join(self.data_path, f)
                if f.endswith('.xlsx'):
                    temp_df = pd.read_excel(filepath)
                else:
                    temp_df = pd.read_csv(filepath)
                temp_df['Normal/Attack'] = 'Normal'
                dfs.append(temp_df)
            
            for f in attack_files:
                filepath = os.path.join(self.data_path, f)
                if f.endswith('.xlsx'):
                    temp_df = pd.read_excel(filepath)
                else:
                    temp_df = pd.read_csv(filepath)
                dfs.append(temp_df)
            
            if not dfs:
                raise ValueError(f"No valid SWaT data files found in {self.data_path}")
            
            df = pd.concat(dfs, ignore_index=True)
        
        # Process labels
        if 'Normal/Attack' in df.columns:
            labels = (df['Normal/Attack'].str.lower() != 'normal').astype(int).values
        elif 'label' in df.columns:
            labels = df['label'].values
        elif 'Label' in df.columns:
            labels = df['Label'].values
        elif 'attack' in df.columns.str.lower():
            label_col = [c for c in df.columns if 'attack' in c.lower()][0]
            labels = df[label_col].values
        else:
            labels = None
            warnings.warn("No label column found in SWaT data")
        
        if self.use_all_features:
            exclude_cols = ['Normal/Attack', 'label', 'Label', 'Timestamp',
                          'timestamp', 'datetime', 'Date', 'Time']
            data_columns = [c for c in df.columns 
                          if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        else:
            data_columns = [c for c in self.SENSOR_COLUMNS if c in df.columns]
        
        if not data_columns:
            raise ValueError("No valid data columns found in SWaT dataset")
        
        self.feature_names = data_columns
        data = df[data_columns].values.astype(np.float32)
        
        data = np.nan_to_num(data, nan=0.0)
        
        print(f"Loaded SWaT data: {data.shape}")
        if labels is not None:
            print(f"Anomaly ratio: {np.mean(labels):.2%}")
        
        return data, labels


class CustomDataLoader(BaseDataLoader):
    """
    Data loader for custom datasets.
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 50,
        scaler_type: str = 'robust',
        random_state: int = 42,
        label_column: str = 'label',
        timestamp_column: Optional[str] = 'timestamp',
        feature_columns: Optional[List[str]] = None,
        separator: str = ',',
        window_overlap: float = 0.0
    ):
        super().__init__(data_path, sequence_length, scaler_type, random_state, window_overlap)
        self.label_column = label_column
        self.timestamp_column = timestamp_column
        self.feature_columns = feature_columns
        self.separator = separator
    
    def load_data(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load custom data from CSV files."""
        if os.path.isfile(self.data_path):
            csv_files = [self.data_path]
        else:
            csv_files = [
                os.path.join(self.data_path, f) 
                for f in os.listdir(self.data_path) 
                if f.endswith('.csv')
            ]
        
        if not csv_files:
            raise ValueError(f"No CSV files found at {self.data_path}")
        
        all_data = []
        all_labels = []
        
        for filepath in csv_files:
            df = pd.read_csv(filepath, sep=self.separator)
            
            # Determine feature columns
            if self.feature_columns:
                data_columns = [c for c in self.feature_columns if c in df.columns]
            else:
                exclude = [self.label_column, self.timestamp_column] if self.timestamp_column else [self.label_column]
                exclude = [c for c in exclude if c]
                data_columns = [
                    c for c in df.columns 
                    if c not in exclude and df[c].dtype in ['float64', 'int64', 'float32', 'int32']
                ]
            
            if not data_columns:
                continue
            
            if not self.feature_names:
                self.feature_names = data_columns
            
            all_data.append(df[data_columns].values)
            
            if self.label_column in df.columns:
                all_labels.append(df[self.label_column].values)
        
        data = np.vstack(all_data)
        labels = np.hstack(all_labels) if all_labels else None
        
        # Handle missing values
        data = np.nan_to_num(data, nan=0.0)
        
        return data, labels


class DataLoaderFactory:
    LOADERS = {
        'skab': SKABDataLoader,
        'swat': SWaTDataLoader,
        'custom': CustomDataLoader
    }
    
    @classmethod
    def create(
        cls,
        dataset_type: str,
        **kwargs
    ) -> BaseDataLoader:
        """
        Args:
            dataset_type: One of 'skab', 'swat', 'custom'
            **kwargs: Arguments passed to the data loader
        
        Returns:
            Configured data loader instance
        """
        dataset_type = dataset_type.lower()
        
        if dataset_type not in cls.LOADERS:
            raise ValueError(
                f"Unknown dataset type: {dataset_type}. "
                f"Available: {list(cls.LOADERS.keys())}"
            )
        
        return cls.LOADERS[dataset_type](**kwargs)
    
    @classmethod
    def register_loader(cls, name: str, loader_class: type) -> None:
        cls.LOADERS[name.lower()] = loader_class


def quick_load(
    dataset: str,
    data_path: str,
    sequence_length: int = 50,
    test_size: float = 0.2,
    val_size: float = 0.1,
    **kwargs
):
    loader = DataLoaderFactory.create(
        dataset,
        data_path=data_path,
        sequence_length=sequence_length,
        **kwargs
    )
    
    return loader.load_and_split(test_size=test_size, val_size=val_size)
