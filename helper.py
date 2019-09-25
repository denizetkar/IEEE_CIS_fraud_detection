import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset


class LargeTabularDataset(IterableDataset):
    def __init__(self, data_path, cont_cols, cat_cols, output_col):
        self.data_path = data_path
        # self.nb_samples = pd.read_csv(data_path, usecols=[0]).shape[0]
        with open(data_path) as f:
            self.nb_samples = max(sum(1 for line in f if line) - 1, 0)

        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.output_col = output_col

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = self.nb_samples
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(self.nb_samples / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.nb_samples)
        return LargeTabularDatesetIterator(self, iter_start, iter_end)


class LargeTabularDatesetIterator:

    def __init__(self, tabular_dataset, start_row, end_row):
        self._tabular_dataset = tabular_dataset
        self._iter_size = end_row - start_row
        self._index = 0
        self._pd_chunk_iter = pd.read_csv(
            self._tabular_dataset.data_path,
            skiprows=range(1, start_row + 1),
            chunksize=1)

    def __next__(self):
        if self._index < self._iter_size:
            x = next(self._pd_chunk_iter)
            cont_x = x[self._tabular_dataset.cont_cols].squeeze(axis=0).astype(np.float32).values
            cat_x = x[self._tabular_dataset.cat_cols].squeeze(axis=0).values
            # 'y' is a scalar
            y = x[self._tabular_dataset.output_col].squeeze(axis=0)
            self._index += 1
            return (cont_x, cat_x), y

        raise StopIteration


class BiDirectionalDict:

    def __init__(self, d=None):
        if d is None:
            d = {}
        self._forward_dict = {}
        self._backward_dict = {}
        self.update(d)

    def __len__(self):
        return len(self._forward_dict)

    def __str__(self):
        return 'FORWARD: ' + str(self._forward_dict) + '\nBACKWARD: ' + str(self._backward_dict)

    def add(self, first_key, second_key):
        if first_key in self._forward_dict:
            prev_second_key = self._forward_dict[first_key]
            if second_key != prev_second_key:
                del self._backward_dict[prev_second_key]
        if second_key in self._backward_dict:
            prev_first_key = self._backward_dict[second_key]
            if first_key != prev_first_key:
                del self._forward_dict[prev_first_key]

        self._forward_dict[first_key] = second_key
        self._backward_dict[second_key] = first_key

    def forward(self, first_key):
        if first_key not in self._forward_dict:
            return None
        return self._forward_dict[first_key]

    def backward(self, second_key):
        if second_key not in self._backward_dict:
            return None
        return self._backward_dict[second_key]

    def update(self, dictionary):
        for first_key, second_key in dictionary.items():
            self.add(first_key, second_key)


class NaN(float):

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, float('nan'))

    def __hash__(self):
        return np.nan.__hash__()

    def __eq__(self, other):
        return np.isnan(other)


nan = NaN()


# credit to @guiferviz for the memory reduction
def memory_usage_mb(df, *args, **kwargs):
    """Dataframe memory usage in MB. """
    return df.memory_usage(*args, **kwargs).sum() / 1024 ** 2


def reduce_memory_usage(df, deep=True, verbose=True):
    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        # Log the conversion performed.
        if verbose and best_type is not None and best_type != str(col_type):
            print(f"Column '{col}' converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"Memory usage decreased from"
              f" {start_mem:.2f}MB to {end_mem:.2f}MB"
              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")

    return df


def safe_del(var_list, local_context):
    for v in var_list:
        if v in local_context:
            del local_context[v]
