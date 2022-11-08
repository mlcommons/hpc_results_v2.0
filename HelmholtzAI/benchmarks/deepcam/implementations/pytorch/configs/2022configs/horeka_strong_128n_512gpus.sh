#!/bin/bash

# hyperparameters
export LOCAL_BATCH_SIZE=2
export START_LR=0.004
export OPTIMIZER="MixedPrecisionLAMB"
export LR_SCHEDULE_TYPE="multistep"
export LR_MILESTONES="1100:4096"
export LR_DECAY_RATE="0.1"
export LR_WARMUP_STEPS=200
export LR_WARMUP_FACTOR=1.
export WEIGHT_DECAY=0.01
export BATCHNORM_GROUP_SIZE=1

# data parameters
export SHUFFLE_MODE="global"
export DATA_FORMAT="dali-numpy"
export PRECISION_MODE="amp"
export LOCAL_VALIDATION_BATCH_SIZE=8

# auxiliary parameters
export LOGGING_FREQUENCY=0

# run parameters
export NEXP="${NEXP:-10}"

# staging parameter -> unique for HoreKa!
export STAGE_DIR_PREFIX="${TMP}"
export STAGE_MODE="global"
export STAGE_VERIFY=0
export STAGE_FULL_DATA_PER_NODE=0

export STAGE_BATCH_SIZE=1
export STAGE_NUM_READ_WORKERS=6
export STAGE_NUM_WRITE_WORKERS=13

# misc args
export ADDITIONAL_ARGS="--enable_jit --disable_comm_overlap --enable_graph --h5stage"
