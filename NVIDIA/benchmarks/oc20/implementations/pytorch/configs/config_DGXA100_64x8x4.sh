#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh

# hyperparameters
export BATCH_SIZE=4
export LR_INITIAL=0.0016
export WARMUP_STEPS=3908
export LR_MILESTONES="23448 31264"
export NUM_INSTANCES=1

# system parameters
export DGXNNODES=64
WALLTIME_MINUTES=30
export WALLTIME=$(( 10 + (${NEXP} * ${WALLTIME_MINUTES}) ))