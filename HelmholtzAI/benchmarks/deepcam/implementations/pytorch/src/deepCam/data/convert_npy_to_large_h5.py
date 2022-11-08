import os
import time

import h5py
from mpi4py import MPI
import numpy as np

import argparse as ap
import shutil

def write_to_h5_file(
        data_files,
        label_files,
        tfname,
        start_entry,
        all_data_shape,
        all_label_shape,
        tell_progress=10,
):
    if MPI.COMM_WORLD.rank == 0 and os.path.isfile(tfname):
        print("file exists")
        exit(1)
        os.remove(tfname)
    # MPI.COMM_WORLD.Barrier()
    print("creating file")
    chunk_size = tuple([1] + list(all_data_shape[1:]))
    with h5py.File(
        tfname,
        'w',
        driver='mpio',
        comm=MPI.COMM_WORLD,
        libver='latest',
        #rdcc_nbytes=np.prod(chunk_size) * 4,
        #rdcc_nslots=np.prod(chunk_size) * 400,
    ) as fi:
        print("creating dset", all_data_shape, chunk_size)
        dset = fi.create_dataset(
            'data',
            all_data_shape,
            dtype='f',
            #chunks=chunk_size,
        )
        print("creating lset")
        lset = fi.create_dataset('labels', all_label_shape, dtype='f')

        startt = time.time()
        for ii, (f, l) in enumerate(zip(data_files, label_files)):
            data = np.load(f)
            label = np.load(l)
            dset[start_entry + ii] = data
            lset[start_entry + ii] = label
            now = time.time()
            time_remaining = len(data_files) * (now - startt) / (ii + 1)
            if ii % tell_progress == 0:
                print(ii, time_remaining / 60, f, l)


def trim_filenames(filenames):
    for fi in range(len(filenames)):
        filenames[fi] = filenames[fi].split('/')[-1]
    return filenames


if __name__ == "__main__":
    parser = ap.ArgumentParser(description='MLPerf DeepCAM data conversion to large H5')
    parser.add_argument(
        '--preshuffle', type=bool, default=True,
        help='should the data be shuffled before its added to the output H5 files'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='seed for RNG'
    )
    parser.add_argument(
        '--system', type=str,
        help='which system are you using? (booster v horeka)',
        choices=['booster', 'horeka']
    )
    parser.add_argument(
        '--npy-files-dir', type=str, default="",
        help='directory with the numpy files. (i.e. /.../npy-files-dir/[train, validation])'
    )
    parser.add_argument(
        '--out-dir', type=str, default="",
        help='to put the output h5 files in'
    )
    parser.add_argument(
        '--trim-filenames', action="store_true",
        help="if true, trim filenames to be only 'dataXXXXX.npy' without the original path"
    )

    args = parser.parse_args()
    # print(args.accumulate(args.integers))

    preshuffle = args.preshuffle
    seed = args.seed

    if args.system == "booster":
        project_name = 'hai_mlperf'
        # Try to use CSCRATCH, fall back to SCRATCH.
        using_cscratch = ('CSCRATCH_' + project_name) in os.environ
        project_dir = os.getenv(
            'CSCRATCH_' + project_name,
            os.getenv('SCRATCH_' + project_name),
        )

        root_dir = os.path.join(project_dir, "deepcam_v1.0")
        out_dir = os.path.join(project_dir, "deepcam_hdf5")
    else:
        using_cscratch = False
        root_dir = args.npy_files_dir
        out_dir = args.out_dir

    if MPI.COMM_WORLD.rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    np.random.seed(seed)

    print("this is the h5py file we are using:", h5py.__file__)

    for data_subset in ["train", "validation"]:
        print(data_subset)

        hdf5_file_name = os.path.join(out_dir, f"{data_subset}.h5")
        files_file_name = os.path.join(out_dir, f"{data_subset}.h5.files")

        load_dir = os.path.join(root_dir, data_subset)
        files_to_load = os.listdir(load_dir)

        data_files = list(filter(
            lambda x: x.endswith("npy") and x.startswith("data"),
            files_to_load,
        ))
        data_files.sort()
        data_files = np.array(data_files)
        label_files = list(filter(
            lambda x: x.endswith("npy") and x.startswith("label"),
            files_to_load,
        ))
        label_files.sort()
        label_files = np.array(label_files)

        if using_cscratch and MPI.COMM_WORLD.rank == 0:
            # Initialize IME cache.
            for data_file in data_files:
                os.system(
                    'ime-ctl --prestage '
                    + os.path.join(load_dir, data_file)
                )
            for label_file in label_files:
                os.system(
                    'ime-ctl --prestage '
                    + os.path.join(load_dir, label_file)
                )
        # Make sure IME cache is initialized before continuing.
        MPI.COMM_WORLD.Barrier()

        perm = np.random.permutation(len(label_files))
        data_files = data_files[perm]
        label_files = label_files[perm]

        no_shards = MPI.COMM_WORLD.size
        data_files_filtered = data_files[:]
        label_files_filtered = label_files[:]
        data_files_shards = []
        label_files_shards = []
        for i in range(no_shards):
            shard_size = int(np.ceil(len(data_files_filtered) / no_shards))
            start = i * shard_size
            end = min((i + 1) * shard_size, len(data_files_filtered))
            data_files_shards.append(data_files_filtered[start:end])
            label_files_shards.append(label_files_filtered[start:end])

        start_entries = np.cumsum([len(x) for x in data_files_shards])
        start_entries = ([0] + list(start_entries))[:-1]

        data_shape = np.load(os.path.join(load_dir, data_files[0])).shape
        label_shape = np.load(os.path.join(load_dir, label_files[0])).shape
        all_data_shape = (len(data_files_filtered), *data_shape)
        all_label_shape = (len(data_files_filtered), *label_shape)

        rank = MPI.COMM_WORLD.rank
        my_data_files = [
            os.path.join(load_dir, f) for f in data_files_shards[rank]
        ]
        my_label_files = [
            os.path.join(load_dir, f) for f in label_files_shards[rank]
        ]
        start_entry = start_entries[rank]

        all_data_files = np.concatenate(MPI.COMM_WORLD.allgather(
            my_data_files))
        all_label_files = np.concatenate(MPI.COMM_WORLD.allgather(
            my_label_files))

        print(len(np.unique(all_data_files)), "unique files and",
              len(all_data_files), "total files.")
        if len(np.unique(all_data_files)) != len(all_data_files):
            print("There is an error with the file distribution")

        with open(files_file_name, "w") as files_file:
            files_file.write("\n".join(all_data_files))
        if args.trim_filenames:
            trimed_files = trim_filenames(all_data_files)
            trimmed_fname = os.path.join(out_dir, f"{data_subset}.h5.files.trimmed")

            with open(trimmed_fname, "w") as files_file:
                files_file.write("\n".join(trimed_files))
            # save labels too
            trimed_files = trim_filenames(all_label_files)
            trimmed_fname = os.path.join(out_dir, f"{data_subset}.h5.labels.files.trimmed")

            with open(trimmed_fname, "w") as files_file:
                files_file.write("\n".join(trimed_files))

        write_to_h5_file(
            [os.path.join(load_dir, f) for f in my_data_files],
            [os.path.join(load_dir, f) for f in my_label_files],
            hdf5_file_name,
            start_entry,
            all_data_shape,
            all_label_shape,
        )
        MPI.COMM_WORLD.Barrier()
        if using_cscratch and MPI.COMM_WORLD.rank == 0:
            # Synchronize file system with IME cache.
            os.system('ime-ctl --sync ' + hdf5_file_name)
            os.system('ime-ctl --sync ' + files_file_name)
        MPI.COMM_WORLD.Barrier()
    # copy stats.h5
    if rank == 0:
        shutil.copyfile(args.npy_files_dir + "stats.h5", out_dir)

    print('done')
