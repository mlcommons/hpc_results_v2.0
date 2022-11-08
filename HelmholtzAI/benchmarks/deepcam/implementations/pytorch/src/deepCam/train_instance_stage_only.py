# The MIT License (MIT)
#
# Modifications Copyright (c) 2020-2022 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Basics
import os
import sys
import numpy as np
import datetime as dt
import subprocess as sp

# logging
# wandb
have_wandb = False
try:
    import wandb

    have_wandb = True
except ImportError:
    pass

# mlperf logger
import utils.mlperf_log_utils as mll

# Torch
import torch
import torch.optim as optim
from torch.autograd import Variable

# Custom
from driver import Trainer, train_epoch
from driver import Validator, validate
from utils import parser as prs
from utils import losses
from utils import optimizer_helpers as oh
from utils import bnstats as bns
from data import get_dataloaders, get_datashapes
from architecture import deeplab_xception

# DDP
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

# amp
import torch.cuda.amp as amp

# comm wrapper
from utils import comm


# main function
def main(pargs):
    print("Staging only script")
    # print(pargs)
    # this should be global
    global have_wandb

    # init distributed training
    mpi_comm, mpi_instance_comm, instance_id, comm_local_group = comm.init_split(
        pargs.wireup_method,
        pargs.training_instance_size,
        pargs.batchnorm_group_size,
        pargs.batchnorm_group_stride,
        verbose=True,
        directory=os.path.join(pargs.output_dir, "wireup")
    )

    # pytorch dist data
    comm_rank = comm.get_data_parallel_rank()
    comm_local_rank = comm.get_local_rank()
    comm_size = comm.get_data_parallel_size()
    comm_local_size = comm.get_local_size()
    comm_is_root = (comm_rank == comm.get_data_parallel_root())

    # print(f"GRANK: {comm.get_world_rank()}, RANK: {comm_rank}, SIZE: {comm_size}, IS_ROOT: {comm_is_root}")

    # set up logging
    pargs.logging_frequency = max([pargs.logging_frequency, 0])
    log_file = os.path.normpath(
        os.path.join(pargs.output_dir, pargs.run_tag + f"_{instance_id + 1}.log")
    )
    if comm_is_root:
        print(f"log file: {log_file}")
    logger = mll.mlperf_logger(
        log_file, "deepcam",
        "HelmholtzAI",
        mpi_comm.Get_size() // comm_local_size
        )
    logger.log_start(key="init_start", sync=True)
    logger.log_event(key="cache_clear")

    # set seed: make it different for each instance
    seed = pargs.seed + instance_id * 3
    logger.log_event(key="seed", value=seed)

    # stage data if requested
    num_instances = mpi_comm.Get_size() // mpi_instance_comm.Get_size()
    # be careful with the seed here, for the global shuffling we should use the same seed or otherwise we break correlation
    if pargs.h5stage:
        from data import stage_from_single_h5 as sd
    elif pargs.enable_gds:
        from data import stage_data_v4_oo as sd
    else:
        from data import stage_data_v2_oo as sd

    # check if I mode only:
    if pargs.stage_read_only and pargs.stage_verify:
        if comm_is_root:
            print(
                "WARNING: we cannot verify staged files if we are reading them! Disabling verification!"
            )
        pargs.stage_verify = False
    if comm_is_root:
        print("creating stager")
    stager = sd.FileStager(
        mpi_comm,
        num_instances,
        instance_id,
        mpi_instance_comm,
        comm_local_size,
        comm_local_rank,
        batch_size=pargs.stage_batch_size,
        num_read_workers=pargs.stage_num_read_workers,
        num_write_workers=pargs.stage_num_write_workers,
        stage_mode=pargs.stage_mode,
        verify=pargs.stage_verify,
        full_dataset_per_node=pargs.stage_full_data_per_node,
        use_direct_io=pargs.stage_use_direct_io,
        read_only=pargs.stage_read_only,
        seed=333
    )
    if comm_is_root:
        print("done creating stager")

    # prepare staging: add local rank if requested
    stage_dir_prefix = pargs.stage_dir_prefix.format(local_rank=comm_local_rank)
    # prepare(self, npy_dir_prefix, stage_dir_prefix, h5_dir_prefix)
    stager.prepare(
        pargs.data_dir_prefix,
        stage_dir_prefix,
        stage_filter_list=['validation', 'train']
    )

    # get sizes of dataset
    global_train_size = stager.file_stats['train']["num_files"]
    global_validation_size = stager.file_stats['validation']["num_files"]

    # we need to adjust a few parameters or otherwise the
    # sharding and shuffling will be wrong
    root_dir = os.path.join(stage_dir_prefix, f"instance{instance_id}")
    if not pargs.stage_full_data_per_node:
        pargs.shuffle_mode = "global"
        num_shards = comm_local_size
        shard_id = comm_local_rank
    else:
        num_shards = mpi_instance_comm.Get_size()
        shard_id = mpi_instance_comm.Get_rank()


    # Some setup
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda", comm_local_rank)
        torch.cuda.manual_seed(seed)
        # necessary for AMP to work
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = not pargs.disable_tuning
    else:
        device = torch.device("cpu")

    # set up directories
    output_dir = pargs.output_dir
    if comm_is_root:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)




    # perform a global barrier across all nodes
    mpi_comm.Barrier()
    logger.log_end(key="init_stop", sync=True)

    # start trining
    logger.log_start(key="run_start", sync=True)

    # stage the data or start prefetching
    logger.log_start(key="staging_start")
    stager.execute_stage()
    logger.log_end(key="staging_stop", sync=True)
    # mpi_comm.Barrier()
    # print('after barrier')
    # exit here if we only want to stage


if __name__ == "__main__":
    # get parsers
    parser = prs.get_parser()

    # add custom arguments
    parser.add_argument(
        "--training_instance_size", default=1, type=int,
        help="Determines how big the individual training instances are"
    )
    parser.add_argument(
        "--stage_dir_prefix", default=None, type=str, help="Prefix for where to stage the data"
    )
    parser.add_argument(
        "--stage_num_read_workers", default=1, type=int,
        help="Number of workers used for reading during staging"
    )
    parser.add_argument(
        "--stage_num_write_workers", default=1, type=int,
        help="Number of workers used for writing during staging"
    )
    parser.add_argument(
        "--stage_batch_size", default=-1, type=int, help="Batch size for data staging optimizations"
    )
    parser.add_argument(
        "--stage_mode", default="node", type=str, choices=["node", "instance", "global"],
        help="How to load the data from file system: shard files across nodes, across a single instances or across all instances"
    )
    parser.add_argument("--stage_verify", action='store_true')
    parser.add_argument(
        "--stage_only", action='store_true', help="Just perform data staging, don't run training"
    )
    parser.add_argument("--stage_full_data_per_node", action='store_true')
    parser.add_argument("--stage_use_direct_io", action='store_true')
    parser.add_argument("--stage_read_only", action='store_true')
    parser.add_argument("--stage_archives", action='store_true')
    # prepare(self, npy_dir_prefix, stage_dir_prefix, h5_dir_prefix)
    # parser.add_argument("--stage_npy_dir", type=str)
    # get arguments
    pargs = parser.parse_args()

    # run the stuff
    main(pargs)
