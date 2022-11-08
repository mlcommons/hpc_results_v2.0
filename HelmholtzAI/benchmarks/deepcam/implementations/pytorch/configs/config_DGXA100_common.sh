#!/bin/bash

# data directory
source $(dirname ${BASH_SOURCE[0]})/config_data_selene.sh 

# auxiliary parameters
export LOGGING_FREQUENCY=0

# run parameters
export NEXP="${NEXP:-10}"

# system parameters
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
