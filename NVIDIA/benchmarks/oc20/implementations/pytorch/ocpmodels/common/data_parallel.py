"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import heapq
import logging
import math
from typing import Iterable, Iterator, List, Optional, TypeVar, Union

import numba
import numpy as np
import torch
import torch.distributed as dist
from ocpmodels.common import distutils
from torch.utils.data import Dataset, Sampler

T_co = TypeVar("T_co", covariant=True)


class DistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        return iter(indices)

    def __len__(self) -> int:
        return self.total_size

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class BatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance` check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integer value, " "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got " "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_replicas = dist.get_world_size()

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size * self.num_replicas:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // (self.batch_size * self.num_replicas)
        else:
            return (len(self.sampler) + self.batch_size * self.num_replicas - 1) // (
                self.batch_size * self.num_replicas
            )


@numba.njit
def balanced_partition(sizes, num_parts):
    """
    Greedily partition the given set by always inserting
    the largest element into the smallest partition.
    """
    sort_idx = np.argsort(-sizes)
    heap = []
    for idx in sort_idx[:num_parts]:
        heap.append((np.int64(sizes[idx]), [idx]))
    heapq.heapify(heap)
    for idx in sort_idx[num_parts:]:
        smallest_part = heapq.heappop(heap)
        new_size = smallest_part[0] + np.int64(sizes[idx])
        new_idx = smallest_part[1] + [idx]
        heapq.heappush(heap, (new_size, new_idx))
    idx_balanced = [part[1] for part in heap]
    return idx_balanced


class BalancedBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        device,
        mode="atoms",
        shuffle=True,
        drop_last=False,
        force_balancing=False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.mode = mode.lower()
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_replicas = distutils.get_world_size()
        self.rank = distutils.get_rank()
        self.group = dist.group.WORLD

        # Explicit init process groups
        one = torch.ones(1, device="cuda")
        dist.all_reduce(one, group=dist.group.WORLD)
        dist.all_reduce(one, group=self.group)

        self.balance_batches = self.num_replicas > 1
        if self.balance_batches:
            if not hasattr(dataset, "metadata_path") or not dataset.metadata_path.is_file():
                if force_balancing:
                    logging.warning(
                        f"No metadata file found at '{dataset.metadata_path}'. "
                        "BalancedBatchSampler has to load the data to "
                        "determine batch sizes, which incurs "
                        "significant overhead!"
                    )
                    self.sizes = None
                else:
                    logging.warning(
                        f"No metadata file found at '{dataset.metadata_path}'. "
                        "Batches will not be balanced, "
                        "which can incur significant overhead!"
                    )
                    self.balance_batches = False
                    self.sizes = None
            else:
                if self.mode == "atoms":
                    self.sizes = np.load(dataset.metadata_path)["natoms"]
                elif self.mode == "neighbors":
                    self.sizes = np.load(dataset.metadata_path)["neighbors"]
                else:
                    raise NotImplementedError(f"Unknown load balancing mode: {self.mode}")
        else:
            self.sizes = None

        self.single_sampler = DistributedSampler(
            dataset,
            num_replicas=distutils.get_world_size(),
            rank=distutils.get_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.batch_sampler = BatchSampler(
            self.single_sampler,
            batch_size,
            drop_last=drop_last,
        )

    def __len__(self):
        return len(self.batch_sampler)

    def set_epoch(self, epoch):
        self.single_sampler.set_epoch(epoch)

    def __iter__(self):
        for batch_idx in self.batch_sampler:
            if self.balance_batches:
                if self.sizes is None:
                    # Unfortunately, we need to load the data to know the image sizes
                    data_list = [self.dataset[idx] for idx in batch_idx]

                    if self.mode == "atoms":
                        sizes = [data.num_nodes for data in data_list]
                    elif self.mode == "neighbors":
                        sizes = [data.edge_index.shape[1] for data in data_list]
                    else:
                        raise NotImplementedError(f"Unknown load balancing mode: {self.mode}")
                else:
                    sizes = [self.sizes[idx] for idx in batch_idx]

                local_idx_balanced = balanced_partition(np.array(sizes), num_parts=self.num_replicas)
                # Since DistributedSampler pads the last batch this should always have an entry for each replica.
                yield torch.tensor(batch_idx)[local_idx_balanced[self.rank]]
            else:
                yield torch.tensor(batch_idx)[self.rank : len(batch_idx) : self.num_replicas]
