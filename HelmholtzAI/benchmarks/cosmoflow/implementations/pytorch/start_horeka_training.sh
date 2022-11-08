#!/bin/bash

ml purge
#ml devel/cuda/11.4 
#ml compiler/gnu/11 mpi/openmpi/4.1

# pmi2 cray_shasta
SRUN_PARAMS=(
  --mpi="pmi2"
  --label
  --cpu-bind="none"
  --gpu-bind="none"
  --cpus-per-task="19"
#  --bb="/mnt/odfs/${SLURM_JOB_ID}/stripe_1"
)
echo "${CONFIG_FILE}"
cat "${CONFIG_FILE}"
echo "${STAGE_ONLY}"
export STAGE_ONLY="${STAGE_ONLY}"
echo "${SLURM_NODELIST}"

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


#if [ "${STRONGWEAK}" == "strong" ];
#  then
#    ngpus="$(( SLURM_NNODES * 4 ))";
#    export TRAINING_INSTANCE_SIZE="${ngpus}"
#    # need to set up what is missing
#    export STAGE_BATCH_SIZE=8
#    export STAGE_MODE="global"
#    export STAGE_NUM_READ_WORKERS=2
#    export STAGE_NUM_WRITE_WORKERS=8
#    export NINSTANCES=1
#    echo "Stage info: ${STAGE_BATCH_SIZE}/${STAGE_NUM_READ_WORKERS}/${STAGE_NUM_WRITE_WORKERS}"
#
#elif [ "${STRONGWEAK}" == "weak" ];
#  then
#    ngpus="${TRAINING_INSTANCE_SIZE}";
#    # other new things should also be here??
#fi

export HDF5_USE_FILE_LOCKING=0
export SLURM_CPU_BIND_USER_SET="ldoms"

#export DATA_DIR_PREFIX="/hkfs/home/dataset/datasets/deepcam_npy/"
export DATA_DIR_PREFIX="/hkfs/work/workspace/scratch/qv2382-cosmoflow/data/gzipped"

export STAGE_DIR_PREFIX="$TMP"
#export DATA_CACHE_DIRECTORY="${TMP}"

export WIREUP_METHOD="nccl-slurm"

export SEED="${RANDOM}"
echo "run tag: ${SEED}"
export EXPID="${SEED}"


export HHAI_DIR="/hkfs/work/workspace/scratch/qv2382-mlperf_2022/optimized-hpc/HelmholtzAI/"
export COSMOFLOW_DIR="${HHAI_DIR}benchmarks/implementations/cosmoflow/pytorch/"

SINGULARITY_FILE="${HHAI_DIR}benchmarks/implementations/cosmoflow/mlperf-cosmo.sif"
echo "${SINGULARITY_FILE}"

export CONFIG_DIR="${COSMOFLOW_DIR}/configs"

export OUTPUT_ROOT="${HHAI_DIR}results/horeka_gpu_n${ngpus}_pytorch1.11/${STRONGWEAK}/cosmoflow/"
export OUTPUT_DIR="${OUTPUT_ROOT}"
echo "OUTPUT_ROOT=${HHAI_DIR}results/horeka_gpu_n${ngpus}_pytorch1.11/${STRONGWEAK}/cosmoflow/"

#echo "${CONFIG_FILE}"
#cat "${CONFIG_FILE}"

#export UCX_MEMTYPE_CACHE=0
export NCCL_IB_TIMEOUT=100
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0
#echo "${NCCL_SOCKET_NTHREADS}"
export NCCL_SOCKET_NTHREADS=8

export HDF5_USE_FILE_LOCKING=0

srun "${SRUN_PARAMS[@]}" singularity exec --nv \
  --bind "${DATA_DIR_PREFIX}":"/dataset","${HHAI_DIR}","${OUTPUT_ROOT}":"/results","${DATA_CACHE_DIRECTORY}","/scratch","$TMP","${STAGE_DIR_PREFIX}":"/staging_area" \
  ${SINGULARITY_FILE} \
    bash -c "\
      export PYTHONPATH='${PYTHONPATH}:/utils'; \
      export SLURM_CPU_BIND_USER_SET=\"none\"; \
      export HDF5_USE_FILE_LOCKING=FALSE; \
      ls /dataset; \
      bash run_and_time.sh"
