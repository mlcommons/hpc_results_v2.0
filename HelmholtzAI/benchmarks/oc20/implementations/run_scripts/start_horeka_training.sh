#!/bin/bash

### Reset environment by unloading all modules
ml purge

# define srun parameters
SRUN_PARAMS=(
  --mpi="pmi2"
  --cpus-per-task="19"
  --kill-on-bad-exit="1"
)

### Define paths
export DATA_DIR="${OC20_WORKSPACE}/data/"
export DATA_TARGET="$TMP"
SINGULARITY_FILE="${OC20_WORKSPACE}/containers/oc20_singularity_v4_2022_09_06.sif"
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
  srun "${SRUN_PARAMS[@]}" singularity exec --nv \
    --bind "${DATA_DIR}":/data:ro,"${HHAI_DIR}","${OUTPUT_DIR}","${DATA_TARGET}" ${SINGULARITY_FILE} \
      bash -c "\
        source ${CONFIG_FILE}; \
        export DATA_TARGET=${DATA_TARGET};
        export NUM_COPY_JOBS=${NUM_COPY_JOBS:-4}; \
        cd ../pytorch; \
        bash ./run_and_time.sh"
done