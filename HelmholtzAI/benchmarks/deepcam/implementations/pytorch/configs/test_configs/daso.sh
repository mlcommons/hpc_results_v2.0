#!/bin/bash

# hyperparameters
export TRAINING_INSTANCE_SIZE=64

# last working one
#export LOCAL_BATCH_SIZE=2
#export START_LR=0.0055
#export OPTIMIZER="MixedPrecisionLAMB"
#export LR_SCHEDULE_TYPE="multistep"
#export LR_MILESTONES="800"
#export LR_DECAY_RATE="0.1"
#export LR_WARMUP_STEPS=400
#export LR_WARMUP_FACTOR=1.
#export WEIGHT_DECAY=0.01
#export BATCHNORM_GROUP_SIZE=1

# recommeneded working one
export LOCAL_BATCH_SIZE=2
export START_LR=0.00155
# 0.00155
export OPTIMIZER="MixedPrecisionLAMB"
export LR_SCHEDULE_TYPE="cosine_annealing"
export LR_T_MAX="9000"
export LR_ETA_MIN="0.0"
export LR_WARMUP_STEPS=0
export LR_WARMUP_FACTOR=1.
export WEIGHT_DECAY=0.01
export BATCHNORM_GROUP_SIZE=4


# data parameters
export SHUFFLE_MODE="global"
export DATA_FORMAT="dali-numpy"

export PRECISION_MODE="amp"
export LOCAL_VALIDATION_BATCH_SIZE=8

# staging parameter
export STAGE_DIR_PREFIX="${TMP}"
export STAGE_BATCH_SIZE=1
export STAGE_MODE="global"
export STAGE_VERIFY=0
export STAGE_FULL_DATA_PER_NODE=0
#export STAGE_USE_DIRECT_IO=1
export STAGE_NUM_READ_WORKERS=12
export STAGE_NUM_WRITE_WORKERS=4

# daso stuff
export DASO_WARMUP="100"
export DASO_COOLDOWN_THRESHOLD="0.80"
export DASO_BATCHES_BETWEEN=4
export DASO_STALENESS_ADJUSTMENT=2
export DASO_STABILITY_LEVEL="1.05"

# auxiliary parameters
export LOGGING_FREQUENCY=0

# misc args
export ADDITIONAL_ARGS="--enable_jit --enable_graph --disable_comm_overlap --h5stage"

## system parameters
#export DGXNGPU=8
#export DGXNNODES=64
#export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
#export WALLTIME=01:20:00
