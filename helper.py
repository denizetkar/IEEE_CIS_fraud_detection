import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


class LargeTabularDataset(IterableDataset):
    def __init__(self, data_path, cont_cols, cat_cols, output_col, chunksize, shuffle=False, is_hdf=False):
        self.data_path = data_path
        if is_hdf:
            with pd.HDFStore(data_path) as store:
                self.nb_samples = store.get_storer('data').nrows
        else:
            # Assume it is csv
            # self.nb_samples = pd.read_csv(data_path, usecols=[0]).shape[0]
            with open(data_path) as f:
                self.nb_samples = max(sum(1 for line in f if line) - 1, 0)

        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.output_col = output_col
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.is_hdf = is_hdf

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

        if self._tabular_dataset.is_hdf:
            self._pd_chunk_iter = iter(
                pd.read_hdf(
                    self._tabular_dataset.data_path,
                    start=start_row, stop=end_row,
                    chunksize=self._tabular_dataset.chunksize))
        else:
            # Assume it is csv
            self._pd_chunk_iter = pd.read_csv(
                self._tabular_dataset.data_path,
                skiprows=range(1, start_row + 1),
                nrows=end_row - start_row,
                chunksize=self._tabular_dataset.chunksize)

    def __next__(self):
        x = next(self._pd_chunk_iter)
        if self._tabular_dataset.shuffle:
            x = x.sample(frac=1)
        cont_x = x[self._tabular_dataset.cont_cols].astype(np.float32).squeeze(axis=0).values
        cat_x = x[self._tabular_dataset.cat_cols].astype(np.int64).squeeze(axis=0).values
        # 'y' is a vector of scalar
        y = x[self._tabular_dataset.output_col].astype(np.int64).values

        return (cont_x, cat_x), y


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


class DataEpochGenerator:
    def __init__(self, chunk_loader, batch_size, epoch=100):
        self.chunk_loader = chunk_loader
        self.batch_size = batch_size
        self.epoch = epoch

    def __iter__(self):
        return data_epoch_generator(self.chunk_loader, self.batch_size, self.epoch)


def data_epoch_generator(chunk_loader, batch_size, epoch=100):
    for _ in range(epoch):
        for (cont_chunk, cat_chunk), target_chunk in chunk_loader:
            # Fix the shapes from (1 x N x F) -> (N x F) and (1 x N) -> (N)
            (cont_chunk, cat_chunk), target_chunk = \
                (cont_chunk.view(-1, cont_chunk.shape[-1]),
                 cat_chunk.view(-1, cat_chunk.shape[-1])), \
                target_chunk.view(target_chunk.shape[-1])
            # Read batches from chunks and yield them
            chunk_size = target_chunk.shape[0]
            start_index = 0
            while start_index < chunk_size:
                end_index = min(start_index + batch_size, chunk_size)
                yield (cont_chunk[start_index:end_index, :],
                       cat_chunk[start_index:end_index, :]), \
                      target_chunk[start_index:end_index]
                start_index = end_index


def train_eval_split_hdf5(train_eval_path, train_path, eval_path, train_ratio=0.9, processed_size=50000):
    with pd.HDFStore(train_eval_path) as store:
        nb_samples = store.get_storer('data').nrows
    nb_train_samples = round(int(nb_samples) * train_ratio)
    chunk_iter = iter(pd.read_hdf(train_eval_path, chunksize=processed_size))
    with pd.HDFStore(train_path, mode='w') as train_f, pd.HDFStore(eval_path, mode='w') as eval_f:
        for chunk in chunk_iter:
            chunk = chunk.sample(frac=1)
            last_train_index = round(train_ratio * len(chunk))
            train_f.append('data', chunk.iloc[:last_train_index, :], format='table', expectedrows=nb_train_samples)
            eval_f.append('data', chunk.iloc[last_train_index:, :], format='table',
                          expectedrows=nb_samples - nb_train_samples)


def l2_norm_model(model, step_count, step_per_update, update_per_verbose, *args, **kwargs):
    step_per_verbose = step_per_update * update_per_verbose
    if step_count % step_per_verbose == 0:
        norm = 0.0
        l2 = nn.MSELoss()
        for param in model.parameters():
            norm += l2(param, torch.zeros_like(param)).item()
        return 'L2 of model', norm
    return None
