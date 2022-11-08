#!/bin/bash


# run benchmark
readonly global_rank=${SLURM_PROCID:-}
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"

SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$DGXNGPU}
CONFIG_FILE=${CONFIG_FILE}
INSTANCES=${INSTANCES:-1}



SEED=${SEED:-0}
DATAROOT="/data"

export NGPUS=$SLURM_NTASKS_PER_NODE
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}

GPUS=$(seq 0 $(($NGPUS - 1)) | tr "\n" "," | sed 's/,$//')
PARAMS=(
    hydra.run.dir="/results/${CONFIG_FILE}/${now:%Y-%m-%d}/${now:%H-%M-%S}"

    +mpi.local_size=${NGPUS}
    +mpi.local_rank=${local_rank}

    data.root_dir=${DATAROOT}
)

if [ -n "${SLURM_LOCALID-}" ]; then
  # Mode 1: Slurm launched a task for each GPU and set some envvars; nothing to do
  DISTRIBUTED=
else
  # Mode 2: Single-node Docker; need to launch tasks with mpirun
  DISTRIBUTED="mpirun --allow-run-as-root --bind-to none --np ${DGXNGPU}"
fi

BIND=''
cluster=''
if [[ "${DGXSYSTEM}" == DGX2* ]]; then
    cluster='circe'
fi
if [[ "${DGXSYSTEM}" == DGXA100* ]]; then
    cluster='selene'
fi
BIND="./bind.sh --cpu=exclusive --ib=single --cluster=${cluster} --"
export TEST_CONFIG="${CONFIG_FILE}"
export TEST_OVERRIDE="${PARAMS[@]}"
${DISTRIBUTED} ${BIND} python -m checks.data_staging -v