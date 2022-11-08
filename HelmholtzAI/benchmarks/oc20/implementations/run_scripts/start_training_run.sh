#!/bin/bash

### Parse CLI arguments
while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Launcher for training + timing for OC20 on either HoreKa or Juwels Booster"
      echo " "
      echo "[options] application [arguments]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-s, --system              the HPC machine to use [horeka, booster]"
      echo "-N, --nodes               number of nodes to compute on"
      echo "-t, --time                compute time limit, default 10 min"
      echo "-c, --config              config file to use"
      echo "-r, --runs                number of experiment runs, default 1"
      echo "-i, --instances           number of instances (>1 means weak scaling), default 1"
      echo "-r|--reservation          which reservation to use (if any)"
      exit 0
      ;;
    -s|--system) shift; export TRAINING_SYSTEM=$1; shift; ;;
    -N|--nodes) shift; export SLURM_NNODES=$1; shift; ;;
    -t|--time) shift; export TIMELIMIT=$1; shift; ;;
    -r|--runs) shift; export NEXP=$1; shift; ;;
    -c|--config) shift; export CONFIG_FILE=$1; shift; ;;
    -i|--instances) shift; export NINSTANCES=$1; shift; ;;
    -r|--reservation) shift; export RESERVATION=$1; shift; ;;
    *) break; ;;
  esac
done

# set defaults
export DATESTAMP=${DATESTAMP:-"result_$(date +'%F--%H-%M-%S-%N')"}
export TIMELIMIT=${TIMELIMIT:-"00:10:00"}
export NEXP=${NEXP:-1}
export NINSTANCES=${NINSTANCES:-1}

# print parameters
echo "System: "${TRAINING_SYSTEM}
echo "Number of nodes: "${SLURM_NNODES}
echo "Job time limit: "${TIMELIMIT}
echo "Config file: "${CONFIG_FILE}
echo "Number of runs: "${NEXP}
echo "Date stamp: "${DATESTAMP}
echo "Reservation: "${RESERVATION:-"none"}

# abort if no config file is given
if [ -z "${CONFIG_FILE}" ]; then echo "CONFIG_FILE is not set! use -c FILEPATH"; exit 1; fi


### Set sbatch parameters
SBATCH_PARAMS=(
  --nodes              "${SLURM_NNODES}"
  --tasks-per-node     "4"
  --time               "${TIMELIMIT}"
  --gres               "gpu:4"
  --job-name           "oc20-mlperf"
)
if [[ -n $RESERVATION ]]; then SBATCH_PARAMS+=(--reservation ${RESERVATION}); fi

### Set strong vs weak scaling and the number of GPUs, used to define the output path
# weak if more than one training instance, otherwise strong
if [ "${NINSTANCES}" -gt "1" ];
  then 
    export STRONGWEAK="weak";
    export ngpus="$(( SLURM_NNODES * 4 / NINSTANCES))"; # In weak scaling: # of GPUs per instance
else
  export STRONGWEAK="strong";
  export ngpus="$(( SLURM_NNODES * 4 ))"; # In strong scaling: # nodes * # GPUs per node (4)
fi

### Set system specific paths and parameters and start training
if [ "$TRAINING_SYSTEM" == "juwelsbooster" ] || [ "$TRAINING_SYSTEM" == "booster" ] || [ "$TRAINING_SYSTEM" == "jb" ]
  then
    base_dir="${PWD}/../../../../"
    export HHAI_DIR="/p/project/hai_mlperf/john2/optimized-hpc/HelmholtzAI/"
    export OUTPUT_DIR="${base_dir}results/juwelsbooster_gpu_n${ngpus}_pytorch1.11/${STRONGWEAK}/oc20/"
    mkdir -p "${OUTPUT_DIR}/slurm"
   
    echo "OUTPUT_DIR=${OUTPUT_DIR}"

    ACCOUNT=${ACCOUNT:-"hai_cosmo"}
    echo "Using account ${ACCOUNT}"
    SBATCH_PARAMS+=(
      --partition     "maintbooster"
      --output        "${OUTPUT_DIR}slurm/JB-N-${SLURM_NNODES}-%j.out"
      --error         "${OUTPUT_DIR}slurm/JB-N-${SLURM_NNODES}-%j.err"
      --account       "${ACCOUNT}"
      --exclude       "jwb[0031,0075,0063,0156,0389,0418,0458,0474,0511,0513,0897,1067,0449,0834]"
    )
    sbatch "${SBATCH_PARAMS[@]}" start_jb_training.sh

elif [ "$TRAINING_SYSTEM" == "horeka" ]
  then
    export OC20_WORKSPACE="/hkfs/work/workspace/scratch/bk6983-mlperf_oc20/"
    export HHAI_DIR="${OC20_WORKSPACE}/optimized-hpc/HelmholtzAI/"
    export OUTPUT_DIR="${HHAI_DIR}results/horeka_gpu_n${ngpus}_pytorch1.11/${STRONGWEAK}/oc20/"
    mkdir -p "${OUTPUT_DIR}/slurm"

    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    
    SBATCH_PARAMS+=(
      --partition     "accelerated"
      --output        "${OUTPUT_DIR}slurm/HoreKa-N-${SLURM_NNODES}-%j.out"
      --error         "${OUTPUT_DIR}slurm/HoreKa-N-${SLURM_NNODES}-%j.err"
      --account 	    "haicore-project-scc"
    )
    sbatch "${SBATCH_PARAMS[@]}" start_horeka_training.sh
else
  echo "must specify system that we are running on! give as first unnamed parameter"
  exit 128
fi
