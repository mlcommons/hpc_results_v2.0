import os
from glob import glob
import itertools
import numpy as np
import concurrent.futures as cf
import time
import torch.cuda.nvtx as nvtx
from typing import Union, Tuple

from .common import get_datashapes
import h5py

from mpi4py import MPI
from mpi4py.util import dtlib

import io_helpers as ioh
import gc


def numpy_integrity_check(src_dir, dst_dir, files):
    checklist = []
    issue_found = False
    for fname in files:
        src_arr = np.load(os.path.join(src_dir, fname))
        dst_arr = np.load(os.path.join(dst_dir, fname))
        compare = np.allclose(
            src_arr, dst_arr,
            rtol=1e-07,
            atol=1e-08
        )
        src_nan = np.any(np.isnan(src_arr))
        dst_nan = np.any(np.isnan(dst_arr))

        issue_found = issue_found or not compare or src_nan or dst_nan

        checklist.append({"file": fname, "equal": compare, "src_nan": src_nan, "dst_nan": dst_nan})

    return checklist, issue_found


def get_shard(
        num_files: int,  # number of files to shard
        num_shards: int,  # number of shards
        shard_id: int,  # who am i? local shard number
        cycle_dist: int = 0,  # should the remainder be distributed between nodes?
        offset: int = 0,  # am i working on the global dataset? if not, give offset
        return_all_slices: bool = False,  # return just my slice or all slices
) -> Union[slice, Tuple[slice, list]]:
    # MODIFICATION: return a slice to get from the H5 file
    # shard files into bulk and remainder:
    num_files_per_shard = num_files // num_shards
    rem = num_files % num_shards

    shards = np.full((num_shards), num_files_per_shard)
    # # deal with remainder: roundrobin with some offset for even distribution
    cycle_offset = 0
    for idf in range(rem):  # 1 offset to keep 0 as 1st index
        shards[cycle_offset % num_shards] += 1
        cycle_offset += cycle_dist

    shards = np.concatenate(([0], np.cumsum(shards).tolist()))
    shards += offset
    if not return_all_slices:
        return slice(shards[shard_id], shards[shard_id + 1])
    else:
        all_slices = [slice(shards[s], shards[s + 1]) for s in range(num_shards)]
        return all_slices[shard_id], all_slices


def allgather_safe(comm, fdata):
    # total size
    comm_size = comm.Get_size()
    num_bytes = len(fdata)
    total_bytes = num_bytes * comm_size

    # chunk by ~1GB:
    gigabyte = 1024 * 1024 * 1024

    # determine number of chunks
    num_chunks = (total_bytes + gigabyte - 1) // gigabyte

    # determine local chunksize
    chunksize = (num_bytes + num_chunks - 1) // num_chunks

    # datatype stuff
    datatype = MPI.BYTE
    np_dtype = dtlib.to_numpy_dtype(datatype)

    # gather stuff
    # prepare buffers:
    sendbuff = np.frombuffer(memoryview(fdata), dtype=np_dtype, count=num_bytes)
    recvbuff = np.empty((comm_size * chunksize), dtype=np_dtype)
    resultbuffs = np.split(np.empty(num_bytes * comm_size, dtype=np_dtype), comm_size)

    # do subsequent gathers
    for i in range(0, num_chunks):
        # create buffer views
        start = i * chunksize
        end = min(start + chunksize, num_bytes)
        eff_bytes = end - start
        sendbuffv = sendbuff[start:end]
        recvbuffv = recvbuff[0:eff_bytes * comm_size]

        # perform allgather on views
        comm.Allgather([sendbuffv, datatype], [recvbuffv, datatype])

        # split result buffer for easier processing
        recvbuff_split = np.split(recvbuffv, comm_size)
        for j in range(comm_size):
            resultbuffs[j][start:end] = recvbuff_split[j][...]
    results = [x.tobytes() for x in resultbuffs]

    return results


def load_file(filename):
    nvtx.range_push("load_file")
    with open(filename, "rb") as f:
        token = f.read()
    nvtx.range_pop()

    return filename, token, len(token)


def save_file(ofname, target_dir, fdata):
    nvtx.range_push("save_file")
    ts = time.perf_counter()
    ofname = os.path.join(target_dir, ofname)
    # print(f"starting save for {ofname}")
    with open(ofname, "wb") as f:  # os.O_RDWR|os.O_CREAT|os.O_DIRECT) as f:
        np.save(f, fdata)
    # np.save(ofname, fdata)#, fix_imports=False)
    nvtx.range_pop()
    bts = fdata.nbytes
    # del fdata

    # ofname.close()
    # print(f"finished save: {time.perf_counter() - ts}")
    return bts


def save_file_direct(ofname, target_dir, fdata, blocksize=512, sync=True):
    # todo: keep this one?
    nvtx.range_push("save_file_direct")
    ofname = os.path.join(target_dir, ofname)
    fdata = str(fdata)
    wbytes = ioh.save_file_direct(ofname, fdata, blocksize, sync)
    nvtx.range_pop()
    return wbytes


global _open_file


# noinspection DuplicatedCode
class FileStager(object):
    @staticmethod
    def load_hdf5_range(chunk_sl: slice, shard_dict):
        nvtx.range_push("load_file")
        # data_shape = (N, 768, 1152, 16)
        # label_shape = (N, 768, 1152)

        # data = np.zeros((chunk_sl.stop - chunk_sl.start, 768, 1152, 16))
        # labels = np.zeros((chunk_sl.stop - chunk_sl.start, 768, 1152))
        # _open_file["data"].read_direct(data, np.s_[chunk_sl], np.s_[:])
        # _open_file["labels"].read_direct(labels, np.s_[chunk_sl], np.s_[:])

        data = _open_file["data"][chunk_sl]
        labels = _open_file["labels"][chunk_sl]
        nread = data.nbytes + labels.nbytes
        nvtx.range_pop()
        return data, labels, shard_dict, nread

    def __init__(
            self,
            global_comm,
            num_instances,
            instance_id,
            instance_comm,
            local_size,
            local_rank,
            batch_size=-1,
            num_read_workers=1,
            num_write_workers=1,
            stage_mode="global",
            verify=False,
            full_dataset_per_node=True,
            use_direct_io=False,
            read_only=False,
            seed=333
    ):
        # global chunking parameters
        self.global_comm = global_comm
        self.num_instances = num_instances
        self.instance_id = instance_id
        self.instance_comm = instance_comm
        self.local_size = local_size
        self.local_rank = local_rank

        # stage optimization info
        self.batch_size = batch_size
        self.num_read_workers = num_read_workers
        self.num_write_workers = num_write_workers
        self.stage_mode = stage_mode
        self.full_dataset_per_node = full_dataset_per_node
        self.use_direct_read = use_direct_io
        self.use_direct_write = use_direct_io
        self.read_only = read_only
        self.seed = seed

        # debug
        self.verify = verify
        # warning, this is slow!
        self.extended_verify = False

        # extract comm info
        self.gsize = self.global_comm.Get_size()
        self.grank = self.global_comm.Get_rank()
        self.isize = self.instance_comm.Get_size()
        self.irank = self.instance_comm.Get_rank()
        self.lsize = self.local_size
        self.lrank = self.local_rank

        self.global_node_id = self.grank // self.lsize

        # create new helper comms
        self.stage_comm = self.global_comm.Split(color=self.irank, key=self.instance_id)
        self.ssize = self.stage_comm.Get_size()  # rank of instance (instance id is rank)
        self.srank = self.stage_comm.Get_rank()
        # split the instance by nodes and create a comm with all matching local ranks by node
        self.num_nodes_per_instance = self.isize // self.lsize
        self.instance_node_id = self.irank // self.lsize
        self.instance_node_comm = self.instance_comm.Split(
            color=self.lrank, key=self.instance_node_id
        )
        # get a local communicator too
        self.local_comm = self.instance_comm.Split(color=(self.irank // self.lsize), key=self.lrank)

        # create stage executor
        self.read_executor = cf.ThreadPoolExecutor(max_workers=self.num_read_workers)
        self.write_executor = cf.ThreadPoolExecutor(max_workers=self.num_write_workers)

    # helper to prepare staging for the instance
    def _prepare_instance_stage(self, num_files, target_directory, idx_files, lab_idx_files):
        # files -> og is ALL FILES
        # new -> return the range of indexes of the data

        if (self.stage_mode == "node") and self.full_dataset_per_node:
            # in this case, only do the number of shards to load to a single node
            # files_slice, all_slices = get_shard(
            #     num_files, self.lsize, self.lrank, return_all_slices=True
            # )
            node_files_slice = slice(None)
        else:
            # here we need to make sure the data is evenly distributed
            # across nodes, otherwise one node might have longer epochs
            files_slice, all_slices = get_shard(
                num_files=num_files,
                num_shards=self.isize,
                shard_id=self.irank,
                cycle_dist=self.lsize,
                return_all_slices=True,
            )
            node_st_rank = self.grank % self.isize // self.lsize * self.lsize
            node_sp_rank = node_st_rank + self.lsize - 1
            node_files_slice = slice(
                all_slices[node_st_rank].start,
                all_slices[node_sp_rank].stop
            )
            # isize -> instance size, irank -> instance rank

        # create tags, ONLY WRITE ON LOCAL RANK 0
        # tag = os.path.basename(num_files[0]).split("-")[0]  # data/label
        # all_slices has slices for every rank
        fnames = []
        num_files_node = []
        for tag in ["data", "label"]:
            fnames.append(os.path.join(target_directory, f"files_{tag}.lst"))
            if self.lrank == 0:  # create a files_tag.lst on each node
                write_list = idx_files if tag == "data" else lab_idx_files
                write_list = write_list[node_files_slice]

                with open(fnames[-1], "w") as f:
                    f.write("\n".join(write_list))
                num_files_node.append(len(write_list))
        return fnames[0], fnames[1], node_files_slice

    # prepare staging the instance
    def prepare(self, h5_dir_prefix, stage_dir_prefix, stage_filter_list=None):
        # append instance ID to target dir
        target_directory = os.path.join(stage_dir_prefix, f"instance{self.instance_id}")

        if self.grank == 0:
            print("Copying stats.h5", flush=True)
            with open(os.path.join(h5_dir_prefix, "stats.h5"), "rb") as f:
                statsfile = f.read()
        else:
            statsfile = None

        # broadcast the statsfile
        statsfile = self.global_comm.bcast(statsfile, 0)

        # save it
        if self.lrank == 0:
            os.makedirs(target_directory, exist_ok=True)
            with open(os.path.join(target_directory, "stats.h5"), "wb") as f:
                f.write(statsfile)

        # iterate over staging filters -> train data, train labels, val....
        #   this changes for us, data and labels are together in each h5 file
        self.file_stats = {}
        for target_file in ["train.h5", "validation.h5"]:
            nvtx.range_push(f"stage {target_file}")
            # cut off '.h5' from name, target_directory is xxxx/[train, val]/
            train_val = target_file[:-3]

            if (self.grank == 0):
                print(f"Preparing file lists for {target_file}", flush=True)

            # get stage source
            stage_source = os.path.join(h5_dir_prefix, target_file)
            stage_target_directory = os.path.join(target_directory, train_val)
            # create target directory if not exist:
            if self.local_rank == 0:
                os.makedirs(stage_target_directory, exist_ok=True)

            # get file info to everybody
            # file info exists in {target_file} + .files.trimmed
            # to make trimmed, loop over existing and use:
            #   new_train_files.write(l.split('/')[-1] + "\n")

            # dont need the filenames anymore, only need unique names for each saved file
            # load the file object to get the number of files....hardcode instead...
            # get lists of file names in order of h5 file
            idx_file = os.path.join(h5_dir_prefix, target_file + ".files.trimmed")
            with open(idx_file, "r") as fidx:
                files_idx_names = fidx.read().splitlines()
            num_files = len(files_idx_names)

            idx_labs_file = os.path.join(h5_dir_prefix, target_file + ".labels.files.trimmed")
            with open(idx_labs_file, "r") as fidx:
                files_idx_names_labels = fidx.read().splitlines()

            # num_files //= 20

            # now stage the data so that each rank in each instance has the relevant data
            data_list_file, label_list_file, node_slice = self._prepare_instance_stage(
                num_files, stage_target_directory, idx_files=files_idx_names,
                lab_idx_files=files_idx_names_labels,
            )

            # updating file stats buffer
            self.file_stats[train_val] = {
                "num_files": num_files,  # TOTAL number of files
                "stage_source": stage_source,  # source dir
                "target_directory": stage_target_directory,  # where to stage it to
                "data_list_file": data_list_file,  # list of all files to be staged on this node
                "label_list_file": label_list_file,
                "node_slice": node_slice,  # list of all slices for this node
            }
            # print(f"{target_file} elements to stage: {node_slice}")
        return

    def _gather_and_save_files(self, load_queue, target_dir, data_files, label_files):
        # call allgather when a load finishes
        # add files to queue

        # out_filenames -> [data names, label names]
        rbytes, save_queue = 0, []

        data_shp, label_shp = get_datashapes()
        cnt = 0
        # print("len load_queue:", len(load_queue))
        wait_times = []
        send_times = []

        # open up all of the save file while waiting for the first files
        # for f in range(len(label_files)):
        #    ofname = os.path.join(target_dir, label_files[f])
        #    label_files[f] = open(ofname, "w")
        #    ofname = os.path.join(target_dir, data_files[f])
        #    data_files[f] = ofname  #open(ofname, "w")

        t0 = time.perf_counter()
        for res in cf.as_completed(load_queue):
            lpdata, lplabels, lpshard_dict, lp_bytes = res.result()
            time_waiting = time.perf_counter() - t0
            if len(lpdata) == 0 or len(lplabels) == 0:
                print(lpshard_dict["start"], "error here no data in lpdata!")
                continue
            st = lpshard_dict["start"]
            end = lpshard_dict["end"]
            # data_fnames = lpshard_dict["output_files_data"]
            # label_fnames = lpshard_dict["output_files_label"]
            num_elements = end - st
            if lpshard_dict["num_shards"] <= 1:  # remainder or full stage to node (no comm)
                rbytes += lpdata.nbytes + lplabels.nbytes
                for idx in range(num_elements):
                    # labels
                    save_queue.append(
                        self.write_executor.submit(
                            save_file, label_files[cnt], target_dir, lplabels[idx]
                        )
                    )
                    # data
                    save_queue.append(
                        self.write_executor.submit(
                            save_file, data_files[cnt], target_dir, lpdata[idx]
                        )
                    )
                    cnt += 1
                time_for_send = None
                # continue
                # del lplabels, lpdata
            else:
                # other batches which are sharded:
                # lpshard_dict -> the start and end elements here are from BEFORE the instance sharding
                t1 = time.perf_counter()
                data_buff = self.stage_comm.allgather(lpdata)
                label_buff = self.stage_comm.allgather(lplabels)
                time_for_send = time.perf_counter() - t1
                # TODO: fix this in the non-blocking case.
                #   this should be only what was READ, so its only what is on each process
                rbytes += data_buff[0].nbytes + label_buff[0].nbytes
                # wait for labels, issue them to the saves
                for i in range(len(label_buff)):
                    num_elements = label_buff[i].shape[0]
                    for idx in range(num_elements):
                        save_queue.append(
                            self.write_executor.submit(
                                save_file, label_files[cnt + idx], target_dir, label_buff[i][idx]
                            )
                        )
                    for idx in range(num_elements):
                        save_queue.append(
                            self.write_executor.submit(
                                save_file, data_files[cnt + idx], target_dir, data_buff[i][idx]
                            )
                        )
                    cnt += num_elements
            t0 = time.perf_counter()
            if self.grank == 0:
                #    print(f"to save: {len(save_queue)}, already read: {len(load_queue)}")
                send_times.append(time_for_send)
                wait_times.append(time_waiting)
        if self.grank == 0 and send_times[0] is not None:
            try:
                print(
                    f"Send times avg: {sum(send_times[:-1]) / (len(send_times) - 1)}, wait_times avg: {sum(wait_times[:-1]) / (len(wait_times) - 1)}"
                )
                print(f"send times: {send_times[:10]}, wait_times: {wait_times[:10]}")
            except:
                print(f"send times: {send_times[:10]}, wait_times: {wait_times[:10]}")

        return save_queue, rbytes

    def _stage_all_loads(
            self,
            stage_info_full: dict,
    ):
        queue = []
        for sharddict in stage_info_full:
            # sharddict = stage_info_full[shardkey]
            st = sharddict["start"]
            # if this is the remainder, load on all (num_shards == 1)
            if sharddict["num_shards"] > 1:
                shard_slice = get_shard(
                    num_files=sharddict["end"] - st,
                    shard_id=sharddict["shard_id"],
                    num_shards=sharddict["num_shards"],
                    offset=st,
                )
            else:
                shard_slice = slice(st, sharddict["end"])
            # print(shard_slice, sharddict["end"] - st)
            queue.append(
                self.read_executor.submit(
                    self.load_hdf5_range, shard_slice, sharddict
                )
            )
        # return 1, 1234
        return queue

    def _finalize_save_local(self, save_queue):
        nvtx.range_push("finalize_save_local")
        wbytes = 0
        times = []
        t0 = time.perf_counter()
        for handle in cf.as_completed(save_queue):
            fbytes = handle.result()
            times.append(time.perf_counter() - t0)
            wbytes += fbytes
            t0 = time.perf_counter()
        nvtx.range_pop()
        if self.grank == 0:
            print(f"write times: {times[:10]}, avg: {sum(times[:-1]) / (len(times) - 1)}")
        return wbytes

    def _stage_instance_data(self, stage_filter):
        # comm parameters
        # num_files = self.file_stats[stage_filter]["num_files"]
        stage_source = self.file_stats[stage_filter]["stage_source"]  # h5 file
        target_directory = self.file_stats[stage_filter]["target_directory"]

        # # list of files to be written on this node
        # fname = self.file_stats[stage_filter]["list_file"]
        # slice of h5 to be handled by this node
        node_slice = self.file_stats[stage_filter]["node_slice"]

        # shard locally .. why? -> in OG the files were gathered locally again
        # 2ND SHARD: files were 'addressed by node before, now back to local rank
        node_files = node_slice.stop - node_slice.start

        node_offset = node_slice.start
        local_rank_slice, all_node_slices = get_shard(
            num_files=node_files,
            num_shards=self.lsize,
            shard_id=self.lrank,
            offset=node_offset,
            return_all_slices=True,
        )
        # files_shard = get_shard(files_shard, self.lsize, self.lrank)

        # SHARD 3: shard across the instances
        # now, let's take care of the data: update the number of files because of remainder
        batch_size = self.batch_size if stage_filter == "train" else 1
        if self.stage_mode == "global":
            batch_size *= self.ssize
        # print("local_rank_slice", local_rank_slice, node_slice)
        loc_files_to_stage = local_rank_slice.stop - local_rank_slice.start
        num_batches_bulk = loc_files_to_stage // batch_size
        num_files_remainder = loc_files_to_stage - (num_batches_bulk * batch_size)

        # currently: node local files, need rank-local files!
        #   split them up here already, add the remainder later
        with open(self.file_stats[stage_filter]["data_list_file"], "r") as fidx:
            data_names = fidx.read().splitlines()
        with open(self.file_stats[stage_filter]["label_list_file"], "r") as fidx:
            label_names = fidx.read().splitlines()

        local_local_slice = slice(
            local_rank_slice.start - node_offset,
            local_rank_slice.stop - node_offset,
        )
        #  local_local_size is from 0 to the number of files that lrank should load
        rank_data_files = data_names[local_local_slice]
        rank_label_files = label_names[local_local_slice]
        # adjust the slices to have the remainders as well

        # create list of batch sizes, shard sizes, etc:
        # CHANGE FROM OG: adding 1 to end, will be used as slices

        # currently not using the correct file names in the output...not necessary, only need to
        #   ensure that the labels are the same
        # print(f"num_batches_bulk: {num_batches_bulk} num_files {node_files}")
        # num_shards = num_batches_bulk - 1
        # print("num_shards", self.ssize)
        stage_info = [{"start": local_rank_slice.start + (i * batch_size),
                       "end": local_rank_slice.start + ((i + 1) * batch_size),
                       "num_shards": int(self.ssize) if self.stage_mode == "global" else 1,
                       "shard_id": int(self.srank) if self.stage_mode == "global" else 0,
                       # "shard_num": i,
                       # "output_files_data": rank_data_files[i * batch_size: (i + 1) * batch_size],
                       # "output_files_label":
                       #     rank_label_files[i * batch_size: (i + 1) * batch_size],
                       }
                      for i in range(0, num_batches_bulk)]

        # file lists do not include the remainder here, they are only the batches
        #   need to slice from the data/label_names

        # deal with the remainder:
        remainder_start = num_batches_bulk * batch_size + local_rank_slice.start
        lcl_rem_st = num_batches_bulk * batch_size
        if self.stage_mode == "global":
            # see if we can squeeze in one more batch with reduced size
            eff_batchsize = (num_files_remainder // int(self.ssize)) * int(self.ssize)
            if eff_batchsize > 0:
                # num_shards += 1
                stage_info.append(
                    {"start": remainder_start,
                     "end": remainder_start + eff_batchsize,
                     "num_shards": int(self.ssize),
                     "shard_id": int(self.srank),
                     # "shard_num": num_shards,
                     # "output_files_data":
                     #    data_names[lcl_rem_st: lcl_rem_st + eff_batchsize],
                     # "output_files_label":
                     #    label_names[lcl_rem_st: lcl_rem_st + eff_batchsize],
                     }
                )
            remainder_start += eff_batchsize
            lcl_rem_st += eff_batchsize
            rank_data_files += data_names[lcl_rem_st: lcl_rem_st + eff_batchsize]
            rank_label_files += label_names[lcl_rem_st: lcl_rem_st + eff_batchsize]

        # remainder: any other remainders are handled by all procs
        # print(f"remainder?: {loc_files_to_stage}, {lcl_rem_st}")
        if (loc_files_to_stage - lcl_rem_st > 0):
            # num_shards += 1
            stage_info.append(
                {"start": remainder_start,
                 "end": remainder_start + (loc_files_to_stage - lcl_rem_st),
                 "num_shards": 1,
                 "shard_id": 0,
                 # "shard_num": num_shards,
                 # "output_files_data":
                 #    data_names[lcl_rem_st: ],
                 # "output_files_label":
                 #    label_names[lcl_rem_st: ],
                 }
            )
            rank_data_files += data_names[lcl_rem_st:]
            rank_label_files += label_names[lcl_rem_st:]

        # do the staging
        # load shards -> allgather each one (non-blocking) -> save to npy files
        # need to handle the remainder + allgather...
        global _open_file
        _open_file = h5py.File(stage_source, 'r', libver='latest', swmr=True)  # todo: test??

        # get each load section, shard across the instance dimension, submit loads to job queue
        load_queue = self._stage_all_loads(stage_info)

        # call allgather to join the loaded batches and start the saving
        #   get filenames -> in self.file_stats[stage_filter]["list_file"]

        # these lists have the list of all files to be stages TO MY NODE
        tt = time.perf_counter()
        save_queue, total_bytes_read = self._gather_and_save_files(
            load_queue,
            target_directory,
            data_files=rank_data_files,
            label_files=rank_label_files,
        )
        time_gather = time.perf_counter() - tt

        tt = time.perf_counter()
        if not self.read_only:
            total_bytes_write = self._finalize_save_local(save_queue)
        else:
            total_bytes_write = 0
        time_save = time.perf_counter() - tt

        # global barrier
        self.instance_comm.Barrier()
        _open_file.close()

        if self.grank == 0:
            print(f"Time to read from h5: {time_gather} \ttime to save: {time_save}")

        return total_bytes_read, total_bytes_write

    def execute_stage(self):
        # unit conversion
        unit_convert_gb = 1. / float(1024 * 1024 * 1024)

        # iterate over all the prepared stage filters
        # NEW: only staging 2 h5 files: train.h5 and validation.h5
        for stage_filter in ['train', 'validation']:
            nvtx.range_push(f"stage {stage_filter}")

            stage_start = time.perf_counter()
            total_read, total_write = self._stage_instance_data(stage_filter)
            stage_stop = time.perf_counter()
            self.global_comm.Barrier()

            # allreduce:
            total_read = self.global_comm.allreduce(total_read)
            total_write = self.global_comm.allreduce(total_write)

            # convert units
            total_read *= unit_convert_gb
            total_write *= unit_convert_gb

            # stage duration:
            stage_duration = stage_stop - stage_start

            ##if self.lrank == 0:
            stage_target_directory = self.file_stats[stage_filter]["target_directory"]
            # print(stage_target_directory, stage_filter)
            num_staged_files = len(os.listdir(stage_target_directory))
            num_staged_files = self.global_comm.allreduce(num_staged_files / self.lsize)

            # target_file_names = []
            # for tag in ["data", "label"]:
            #    fname = os.path.join(stage_target_directory, f"files_{tag}.lst")
            #    with open(fname, "r") as fidx:
            #        files_names = fidx.read().splitlines()
            #    target_file_names += files_names
            ## get a list of the directory now
            # staged_files = os.listdir(stage_target_directory)
            # if len(staged_files) != len(target_file_names):
            #    print(f"not all files staged for {stage_filter}!! {len(staged_files)} out of {len(target_file_names)}")

            # print(f"missing:

            # if num_staged_files != self.file_stats[stage_filter]["num_files"]:
            #    # if this

            if self.grank == 0:
                print(
                    f"""Staging {stage_filter} done.
                          Total number of files: {self.file_stats[stage_filter]["num_files"]} / {num_staged_files}.
                          Elapsed time {stage_duration:.2f}s.
                          Read {total_read:.2f} GB (bandwidth: {total_read / stage_duration:.2f} GB/s).
                          Write {total_write:.2f} GB (bandwidth: {total_write / stage_duration:.2f} GB/s).
                       """, flush=True
                )

            # verify staging results if requested
            if self.verify:
                nvtx.range_push(f"stage_verify")
                if self.lrank == 0:
                    stage_target_directory = self.file_stats[stage_filter]["target_directory"]
                    # print(stage_target_directory, stage_filter)
                    staged_files = os.listdir(stage_target_directory)
                    # print(staged_files)
                else:
                    staged_files = []

                if not self.full_dataset_per_node:
                    # if every node hosts a shard, we need to sum the results, if not we need to make sure everybody has the same
                    staged_files_full = self.instance_comm.allgather(staged_files)
                    staged_files_full = list(itertools.chain(*staged_files_full))
                else:
                    staged_files_full = list(staged_files)
                staged_num_files = len(staged_files_full)

                global_files_full = None
                # if self.irank == 0:
                # stage_source_directory = self.file_stats[stage_filter]["stage_source"]
                # global_files_full = glob(
                #    os.path.join(stage_source_directory, stage_filter)
                # )
                global_files_full = _open_file["labels"].shape[0]
                global_num_files = global_files_full
                # global_files_full = self.instance_comm.bcast(global_files_full, 0)
                # global_num_files = len(global_files_full)

                # strip off the directory
                # checkfiles1 = sorted([os.path.basename(x) for x in staged_files_full])
                # checkfiles2 = sorted([os.path.basename(x) for x in global_files_full])
                # print(f"expected {global_num_files} but got {staged_num_files}")
                # assert (staged_num_files == global_num_files), \
                #    f"Error, expected {global_num_files} but got {staged_num_files}"
                # assert (checkfiles1 == checkfiles2), f"Error, staged and unstaged files differ"

                if self.extended_verify:
                    if self.lrank == 0:
                        checks, issue = numpy_integrity_check(
                            self.file_stats[stage_filter]["stage_source"],
                            self.file_stats[stage_filter]["target_directory"],
                            [os.path.basename(x) for x in staged_files]
                        )
                        if issue:
                            print(
                                f"Instance {self.instance_id}, local rank {self.irank}: Verification Error. Results:",
                                checks, flush=True
                            )
                        else:
                            print(
                                f"Instance {self.instance_id}, local rank {self.irank}: Verification OK",
                                flush=True
                            )
                    self.instance_comm.Barrier()

                if self.irank == 0:
                    print(
                        f'Staged data for {stage_filter}: {staged_num_files}, expected: {global_num_files}',
                        flush=True
                    )
                nvtx.range_pop()

            # close range
            nvtx.range_pop()
        return
