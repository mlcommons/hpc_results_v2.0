#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

PARAMS=(
    --batch_size ${BATCH_SIZE}
    --eval_batch_size ${EVAL_BATCH_SIZE}
    --lr_initial ${LR_INITIAL}
    --warmup_steps ${WARMUP_STEPS}
    --warmup_factor ${WARMUP_FACTOR}
    --lr_milestones ${LR_MILESTONES}
    --lr_gamma ${LR_GAMMA}
    --instances ${NUM_INSTANCES}
    --nodes_for_eval ${EVAL_NODES}
    --max_epochs ${MAX_EPOCHS}
)

# bind command
BIND_CMD="./bind.sh --cluster=selene --ib=single --cpu=exclusive,nosmt"

# run command
${BIND_CMD} python main.py "${PARAMS[@]}"; ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="OC20_HPC"
echo "RESULT,$result_name,,$result,$USER,$start_fmt"
