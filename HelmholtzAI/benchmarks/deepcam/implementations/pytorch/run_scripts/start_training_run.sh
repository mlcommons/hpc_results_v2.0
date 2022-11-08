#!/bin/bash

# hooray for stack overflow...
while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Launcher for training + timing for DeepCam on either HoreKa or Juwels Booster"
      echo " "
      echo "[options] application [arguments]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-s, --system              the HPC machine to use [horeka, booster]"
      echo "-N, --nodes               number of nodes to compute on"
      echo "-t, --time                compute time limit"
      echo "-c, --config              config file to use"
      echo "--reservation             name of reservation"
      echo "--daso                    use daso method from heat"
      exit 0
      ;;
    -s|--system) shift; export TRAINING_SYSTEM=$1; shift; ;;
    -N|--nodes) shift; export SLURM_NNODES=$1; shift; ;;
    -t|--time) shift; export TIMELIMIT=$1; shift; ;;
    -w|--strongweak) shift; export STRONGWEAK=$1; shift; ;;
    -r|--reservation) shift; export RESERVATION=$1; shift; ;;
    --daso) shift; export USE_DASO=$1; shift; ;;
    -o|--stage-only) shift; export STAGE_ONLY="1"; shift; ;;
    -c|--config) shift; export CONFIG_FILE=$1; echo "set config file: ${CONFIG_FILE}"; shift; ;;
    *) break; ;;
  esac
done

if [ -z "${TIMELIMIT}" ]; then TIMELIMIT="00:10:00"; fi

echo "Job time limit: "${TIMELIMIT}

if [ -z "${CONFIG_FILE}" ]; then echo "CONFIG_FILE is not set! use -c FILEPATH"; exit 1; fi

source "${CONFIG_FILE}"
echo "strong/weak: ${STRONGWEAK} \tstage only? ${STAGE_ONLY}"
export STAGE_ONLY="${STAGE_ONLY}"

echo "${STRONGWEAK} scaling run"

SBATCH_PARAMS=(
  --nodes              "${SLURM_NNODES}"
  --tasks-per-node     "4"
  --time               "${TIMELIMIT}"
  --gres               "gpu:4"
  --job-name           "deepcam-mlperf"
)

if [ "${STRONGWEAK}" == "strong" ];
  then
    ngpus="$(( SLURM_NNODES * 4 ))";
    export TRAINING_INSTANCE_SIZE="${ngpus}"
    export NINSTANCES=1

elif [ "${STRONGWEAK}" == "weak" ];
  then
    ngpus="${TRAINING_INSTANCE_SIZE}";
    export NINSTANCES="$(( SLURM_NNODES * 4 / TRAINING_INSTANCE_SIZE ))"
fi

#export RESERVATION="maint-booster-2022-08-30"


export TRAINING_SYSTEM="${TRAINING_SYSTEM}"

if [ "$TRAINING_SYSTEM" == "juwelsbooster" ] || [ "$TRAINING_SYSTEM" == "booster" ] || [ "$TRAINING_SYSTEM" == "jb" ]
  then
    export OUTPUT_ROOT="${PWD}/../../../../../results/juwelsbooster_gpu_n${ngpus}_pytorch1.11/${STRONGWEAK}/deepcam/"
    export OUTPUT_DIR="${OUTPUT_ROOT}"
    mkdir -p "${OUTPUT_ROOT}"
    mkdir -p "${OUTPUT_ROOT}/slurm"
    echo "${OUTPUT_ROOT}"

	#"largebooster"
    export PYTORCH_KERNEL_CACHE_PATH="/dev/shm"

    SBATCH_PARAMS+=(
      --partition     "largebooster"
      --output        "${OUTPUT_DIR}slurm/JB-N-${SLURM_NNODES}-%j.out"
      --error         "${OUTPUT_DIR}slurm/JB-N-${SLURM_NNODES}-%j.err"
      --exclude	      "jwb[0031,0075,0063,0156,0389,0418,0458,0474,0511,0513,0897,1067]"
#      --cpu-freq="high"
#      --gpu-freq="high"
    )
    #if [ -z "$RESERVATION" ]; then
    #  SBATCH_PARAMS+=(
    #    --account       "hai_oc20"
    #  )
    #else
    SBATCH_PARAMS+=(
      --account       "hai_oc20"
 #     --reservation   "${RESERVATION}"
    )
    #fi

    echo sbatch "${SBATCH_PARAMS[@]}" start_jb_training.sh
    sbatch "${SBATCH_PARAMS[@]}" start_jb_training.sh

elif [ "$TRAINING_SYSTEM" == "horeka" ]
  then
    base_dir="/hkfs/work/workspace/scratch/qv2382-mlperf_2022/optimized-hpc/HelmholtzAI/"
    export OUTPUT_ROOT="${base_dir}results/horeka_gpu_n${ngpus}_pytorch1.11/${STRONGWEAK}/deepcam/"
    export OUTPUT_DIR="${OUTPUT_ROOT}"
    echo "${OUTPUT_ROOT}"

    mkdir -p "${OUTPUT_ROOT}"
    mkdir -p "${OUTPUT_ROOT}/slurm"

    export PYTORCH_KERNEL_CACHE_PATH="${TMP}"

    SBATCH_PARAMS+=(
      --partition     "accelerated"
      --output        "${OUTPUT_DIR}slurm/HoreKa-N-${SLURM_NNODES}-%j.out"
      --error         "${OUTPUT_DIR}slurm/HoreKa-N-${SLURM_NNODES}-%j.err"
      #--exclude       "hkn[0526]"
      -A 	      "haicore-project-scc"
#      --cpu-freq="high"
#      --gpu-freq="high"
#      --constraint="BEEOND"
      --reservation  "MLperfSC"
    )
    sbatch "${SBATCH_PARAMS[@]}" start_horeka_training.sh
else
  echo "must specify system that we are running on!"
  exit 128
fi
