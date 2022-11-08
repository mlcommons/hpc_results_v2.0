#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_data_selene.sh

export DGXNGPU=8
export LR_GAMMA=0.1
export WARMUP_FACTOR=0.2
export EVAL_BATCH_SIZE=64
export SBATCH_NETWORK=sharp
export EVAL_NODES=0
export MAX_EPOCHS=45
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
