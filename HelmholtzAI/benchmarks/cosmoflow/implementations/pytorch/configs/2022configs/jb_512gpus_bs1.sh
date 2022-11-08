source $(dirname ${BASH_SOURCE[0]})/../config_DGXA100_common.sh

export STAGING_DIR="/tmp"

## using the 64x8x1 config from nvidia -> batch size is 512
export CONFIG_FILE="submission_dgxa100_64x8x1_other.yaml"

## System run parms
export DGXNNODES=128
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

WALLTIME_MINUTES=20
export WALLTIME=$(( 20 + (${NEXP:-1} * ${WALLTIME_MINUTES}) ))

