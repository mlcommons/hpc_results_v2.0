#!/bin/bash

# data directory
source $(dirname ${BASH_SOURCE[0]})/config_data_selene.sh 

# this should never be exceeded by any benchmark
export MAX_EPOCHS=50

# auxiliary parameters
export LOGGING_FREQUENCY=0

# run parameters
export NEXP="${NEXP:-10}"

# system parameters
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
