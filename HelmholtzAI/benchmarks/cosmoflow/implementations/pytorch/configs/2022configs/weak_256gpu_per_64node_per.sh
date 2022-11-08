source $(dirname ${BASH_SOURCE[0]})/../config_DGXA100_common.sh

export STAGING_DIR="${TMP}"

## using the 64x8x1 config from nvidia, but with batch size 2! -> this is batch size 512 again
export CONFIG_FILE="submission_dgxa100_64x8x2_other.yaml"

## System run parms
export DGXNNODES="${SLURM_NNODES}"
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

WALLTIME_MINUTES=40
export WALLTIME=$(( 40 + (${NEXP:-1} * ${WALLTIME_MINUTES}) ))

