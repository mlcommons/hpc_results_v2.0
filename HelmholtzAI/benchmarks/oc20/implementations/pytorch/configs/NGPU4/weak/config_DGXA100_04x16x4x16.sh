#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/common_batch_size_16.sh

export NUM_INSTANCES=4
export DGXNNODES=64
