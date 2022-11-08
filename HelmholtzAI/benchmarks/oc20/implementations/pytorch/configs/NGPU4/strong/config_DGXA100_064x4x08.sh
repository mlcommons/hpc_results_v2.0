#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/../common.sh

# hyperparameters
export BATCH_SIZE=8
export LR_INITIAL=0.0016
export WARMUP_STEPS=3908
export LR_MILESTONES="24425 32241"
export NUM_INSTANCES=1

# system parameters
export DGXNNODES=64
