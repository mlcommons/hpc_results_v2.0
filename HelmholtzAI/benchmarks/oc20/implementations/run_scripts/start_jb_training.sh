#!/bin/bash

### Reset environment by unloading all modules
ml purge

# define srun parameters
SRUN_PARAMS=(
  --mpi            pspmix
  --cpu-bind       none
)

### Define environment variables
export SLURM_CPU_BIND_USER_SET="none"
export WIREUP_METHOD="nccl-slurm"
export UCX_MEMTYPE_CACHE=0
export NCCL_IB_TIMEOUT=20
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0
export PROCESS_GROUP_INIT_METHOD="BOOSTER"  # make distutils.py use the alternative init method

### Define paths
export DATA_DIR="/p/scratch/hai_mlperf/data_oc20"
export DATA_TARGET="/dev/shm"
SINGULARITY_FILE="/p/project/hai_mlperf/oc20_singularity_v4_2022_09_06.sif"
echo "SINGULARITY_FILE=${SINGULARITY_FILE}"

### Output the config file and its contents
echo "Config file: ${CONFIG_FILE}"
cat "${CONFIG_FILE}"

### Start NEXP training runs
for i in $(seq 1 "${NEXP}"); do
  export SEED="${RANDOM}"
  export EXP_ID=${i}
  echo "Beginning trial ${i} of ${NEXP} with seed ${SEED}"

  ### Start run_and_time.sh in the container with srun
  srun "${SRUN_PARAMS[@]}"  env PMIX_SECURITY_MODE=native singularity exec --nv \
    --bind "${DATA_DIR}":/data:ro,"${HHAI_DIR}","${OUTPUT_DIR}","${DATA_TARGET}" ${SINGULARITY_FILE} \
    bash -c "\
    export CUDA_VISIBLE_DEVICES='0,1,2,3';  \
    export PMIX_SECURITY_MODE='native'; \
    export NCCL_DEBUG=INFO; \
    export NCCL_DEBUG_SUBSYS=INIT,GRAPH ; \
    source ${CONFIG_FILE}; \
    export DATA_TARGET=${DATA_TARGET}; \
    export DATA_SRC=${DATA_DIR}; \
    cd ../pytorch; \
    bash ./run_and_time.sh"
done