#!/bin/bash

# hyperparameters
export BATCH_SIZE=32
export EVAL_BATCH_SIZE=32
export LR_INITIAL=0.0016
export WARMUP_STEPS=3908
export WARMUP_FACTOR=0.2
export LR_MILESTONES="24425 32241"
export NUM_INSTANCES=32

# system parameters
export DGXNGPU=8
export DGXNNODES=256
