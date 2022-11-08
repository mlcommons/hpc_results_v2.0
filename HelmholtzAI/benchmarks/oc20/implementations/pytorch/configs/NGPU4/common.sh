#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/../config_DGXA100_common.sh

# shared parameters for nodes with 4x A100 40 GB
export DGXNGPU=4
export EVAL_BATCH_SIZE=32 # 64 causes out of memory errors on 40 GB
