#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/common_batch_size_16.sh

export NUM_INSTANCES=16
export DGXNNODES=256
