#!/bin/bash
# This file is the first things to be done with srun

ml --force purge

SRUN_PARAMS=(
  --mpi            pspmix
  --cpu-bind       none
  --label
)

cat "${CONFIG_FILE}"
echo "CONFIG_FILE ${CONFIG_FILE}"

export SLURM_CPU_BIND_USER_SET="none"
echo $CSCRATCH

#if [ -z $DATA_DIR_PREFIX ]; then
# export DATA_DIR_PREFIX="/p/cscratch/fs/hai_mlperf/deepcam_hdf5_2/"
export DATA_DIR_PREFIX="/p/scratch/hai_mlperf/deepcam_hdf5_2/"
#    export DATA_DIR_PREFIX="${CSCRATCH}/deepcam_hdf5/"
#fi

ime-ctl --prestage "${DATA_DIR_PREFIX}*"
ime-ctl --frag-stat "${DATA_DIR_PREFIX}train.h5"

export STAGE_DIR_PREFIX="/tmp"

if [ -z "${NINSTANCES}" ];
  then STRONGWEAK="strong";
elif [ "${NINSTANCES}" -gt "1" ];
  then STRONGWEAK="weak";
fi

if [ -z "${TRAINING_INSTANCE_SIZE}" ];
  then ngpus="$(( 4 * SLURM_NNODES ))";
else
  ngpus="${TRAINING_INSTANCE_SIZE}"
fi

#export DATA_DIR_PREFIX="/p/ime-scratch/fs/jb_benchmark/deepCam2/"
export WIREUP_METHOD="nccl-slurm"
export SEED="${RANDOM}"

base_dir="/p/project/hai_mlperf/coquelin1/optimized-hpc/HelmholtzAI/benchmarks/implementations/deepcam/"
export DEEPCAM_DIR="${base_dir}pytorch/src/deepCam/"
#"/opt/deepCam/"
SCRIPT_DIR="/p/project/hai_mlperf/coquelin1/optimized-hpc/HelmholtzAI/benchmarks/implementations/deepcam/pytorch/run_scripts/"

#SINGULARITY_FILE="${base_dir}docker/nvidia-optimized-image-2.sif"
#SINGULARITY_FILE="/p/project/jb_benchmark/nvidia_singularity_images/nvidia_deepcam_21.09-pmi2.sif"
#SINGULARITY_FILE="/p/project/jb_benchmark/MLPerf-1.0/mlperf-deepcam/docker/nvidia-optimized-image-2.sif"
SINGULARITY_FILE="/p/project/hai_mlperf/coquelin1/optimized-hpc/HelmholtzAI/benchmarks/implementations/deepcam/containers/deepcam2.sif"

# TODO: FIXME! abs path is best option
export OUTPUT_ROOT="${PWD}/../../../../../results/juwelsbooster_gpu_n${ngpus}_pytorch1.11/${STRONGWEAK}/deepcam/"
export OUTPUT_DIR="${OUTPUT_ROOT}"

#MASTER=$(echo "$SLURM_STEP_NODELIST" | cut -d "," -f 1)
#echo "Node list $SLURM_STEP_NODELIST"
#export MASTER="$(echo "$MASTER" | cut -d "." -f 1)i.juwels"
#echo "pinging $MASTER from $HOSTNAME"

export SINGULARITY_FILE
export UCX_MEMTYPE_CACHE=0
export NCCL_IB_TIMEOUT=20
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
#export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0

export HDF5_USE_FILE_LOCKING=0

# /p/project/jb_benchmark/kesselheim1/MLPerf/benchmarks-closed/deepcam/run_scripts/my_pytorch/distributed_c10d.py:/opt/conda/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py

echo "${SLURM_NODELIST}"

srun "${SRUN_PARAMS[@]}" apptainer exec --nv \
  --bind "${DATA_DIR_PREFIX}",${SCRIPT_DIR},${OUTPUT_ROOT},${base_dir} ${SINGULARITY_FILE} \
    bash -c "\
      export CUDA_VISIBLE_DEVICES='0,1,2,3';  \
      export PMIX_SECURITY_MODE='native'; \
      source ${CONFIG_FILE}; \
      export NCCL_DEBUG=INFO; \
      bash ../run_and_time.sh"
#       export NCCL_DEBUG=INFO; \
#      export NCCL_DEBUG_SUBSYS=INIT,GRAPH ; \


# MASTER=$(echo "$SLURM_STEP_NODELIST" | cut -d "," -f 1);
#    apptainer run --nv \
#  --bind "${DATA_DIR_PREFIX}",${SCRIPT_DIR},${OUTPUT_ROOT},${base_dir} ${SINGULARITY_FILE} \
#    bash -c "\
#      export CUDA_VISIBLE_DEVICES="0,1,2,3";  \
#      export PMIX_SECURITY_MODE="native";
#      export NCCL_DEBUG=INFO; \
#      export NCCL_DEBUG_SUBSYS=INIT,GRAPH ; \
#      source ${CONFIG_FILE}; \
#      bash ../run_and_time.sh"'

      
    #echo "Node list $SLURM_STEP_NODELIST";
    #export MASTER="$(scontrol show hostnames  $SLURM_STEP_NODELIST| head -n 1)i.juwels";
    #echo "pinging $MASTER from $HOSTNAME";
    #ping -c 1 $MASTER; 
      #export NCCL_DEBUG_SUBSYS=INIT,GRAPH ; \
