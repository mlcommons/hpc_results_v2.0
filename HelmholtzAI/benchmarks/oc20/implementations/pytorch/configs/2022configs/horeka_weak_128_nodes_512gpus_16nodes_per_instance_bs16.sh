#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/../NGPU4/common.sh

# hyperparameters
export BATCH_SIZE=16
export LR_INITIAL=0.0012
export WARMUP_STEPS=7816
export LR_MILESTONES="31264 46896"
export NUM_INSTANCES=8

# system parameters
export DGXNNODES=128

