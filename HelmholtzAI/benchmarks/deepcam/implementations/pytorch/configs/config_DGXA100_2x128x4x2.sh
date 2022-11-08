#!/bin/bash

# data directory
source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh 

# hyperparameters
export LOCAL_BATCH_SIZE=2
export START_LR=0.0055
export OPTIMIZER="MixedPrecisionLAMB"
export LR_SCHEDULE_TYPE="multistep"
export LR_MILESTONES="800"
export LR_DECAY_RATE="0.1"
export LR_WARMUP_STEPS=400
export LR_WARMUP_FACTOR=1.
export WEIGHT_DECAY=0.01
export BATCHNORM_GROUP_SIZE=1

# data parameters
export SHUFFLE_MODE="global"
export DATA_FORMAT="dali-numpy"
#"dali-es"
export PRECISION_MODE="amp"
export LOCAL_VALIDATION_BATCH_SIZE=8


export TRAINING_INSTANCE_SIZE=128
# staging parameter
export STAGE_DIR_PREFIX="/tmp"
#dev/shm"
export STAGE_BATCH_SIZE=1
export STAGE_MODE="global"
export STAGE_VERIFY=0
export STAGE_FULL_DATA_PER_NODE=0
#export STAGE_USE_DIRECT_IO=1
export STAGE_NUM_READ_WORKERS=8
export STAGE_NUM_WRITE_WORKERS=4



# misc args
export ADDITIONAL_ARGS="--enable_jit --disable_comm_overlap --enable_graph --h5stage"

# system parameters
export DGXNNODES=64
WALLTIME_MINUTES=10
export WALLTIME=$(( 15 + (${NEXP} * ${WALLTIME_MINUTES}) ))
