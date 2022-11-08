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

import bisect
import pickle
from pathlib import Path

import lmdb
import numpy as np
from ocpmodels.common.registry import registry
from torch.utils.data import Dataset


def connect_db(lmdb_path=None):
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    return env


def get_num_samples(config, mode):
    srcdir = Path(config["data"], config[f"{mode}_dataset"])
    db_paths = sorted(srcdir.glob("*.lmdb"))
    assert len(db_paths) > 0, f"No LMDBs found in {srcdir}"

    _keys, envs = [], []
    for db_path in db_paths:
        envs.append(connect_db(db_path))
        length = pickle.loads(envs[-1].begin().get("length".encode("ascii")))
        _keys.append(list(range(length)))

    keylens = [len(k) for k in _keys]
    num_samples = sum(keylens)
    return num_samples


@registry.register_dataset("trajectory_lmdb")
class TrajectoryLmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation trajectories.
    Useful for Structure to Energy & Force (S2EF) and Initial State to
    Relaxed State (IS2RS) tasks.

    Args:
        config (dict): Dataset configuration
        transform (callable, optional): Data transform function.
            (default: :obj:`None`)
    """

    def __init__(self, config, transform=None, mode="train"):
        super(TrajectoryLmdbDataset, self).__init__()
        self.config = config
        self.mode = mode

        if "src" in config:
            srcdir = Path(config["src"])
        else:
            srcdir = Path(self.config["data_target"], self.config[f"{mode}_dataset"])

        self.metadata_path = srcdir / "metadata.npz"

        db_paths = sorted(srcdir.glob("*.lmdb"))
        assert len(db_paths) > 0, f"No LMDBs found in {srcdir}"

        self._keys, self.envs = [], []
        for db_path in db_paths:
            self.envs.append(self.connect_db(db_path))
            length = pickle.loads(self.envs[-1].begin().get("length".encode("ascii")))
            self._keys.append(list(range(length)))

        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        self.transform = transform
        self.num_samples = sum(keylens)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Figure out which db this should be indexed from.
        db_idx = bisect.bisect(self._keylen_cumulative, idx)
        # Extract index of element within that db.
        el_idx = idx - self._keylen_cumulative[db_idx - 1] if db_idx > 0 else idx

        # Return features.
        datapoint_pickled = self.envs[db_idx].begin().get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
        data_object = pickle.loads(datapoint_pickled)
        data_object.id = f"{db_idx}_{el_idx}"

        # DISTANCE
        cell_offsets = data_object.cell_offsets.unsqueeze(1).float()
        cell = data_object.cell.expand(cell_offsets.shape[0], -1, -1)
        data_object.offsets = cell_offsets.bmm(cell).squeeze(1)

        # sort edges by source node
        _, idx = data_object.edge_index[0].sort()
        data_object.edge_index = data_object.edge_index[:, idx]
        data_object.offsets = data_object.offsets[idx]
        if hasattr(data_object, "distances"):
            data_object.distances = data_object.distances[idx]

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        for env in self.envs:
            env.close()
