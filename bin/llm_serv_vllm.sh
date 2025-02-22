# file: llm_serv_vllm.sh
# date: 2025-02-06


set -x

CURR_DIR=$(pwd)
WORKSPACE="./_llm_serv_vllm"
PYTHON=$(which python3)
PORT=6789
HF_TOKEN=""
MODEL="google/gemma-2-2b-it"
CUDA_VISIBLE_DEVICES=3,4
MAX_MODEL_LEN=5000
TENSOR_PARALLEL_SIZE=2
GPU_MEM_UTILIZATION=0.5
DTYPE=bfloat16
VLLM_ALLOW_LONG_MAX_MODEL_LEN=0



function init() {
  cd ${CURR_DIR}
  mkdir -p ${WORKSPACE}
}


function build() {
  cd ${CURR_DIR}
  cd ${WORKSPACE}
  ${PYTHON} -m venv ./_pyenv --copies
  ./_pyenv/bin/pip install vllm
}


function start() {
  cd ${CURR_DIR}
  cd ${WORKSPACE}
  ./_pyenv/bin/huggingface-cli login --token ${HF_TOKEN}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
  VLLM_ALLOW_LONG_MAX_MODEL_LEN=${VLLM_ALLOW_LONG_MAX_MODEL_LEN} \
  ./_pyenv/bin/vllm serve ${MODEL} \
    --dtype ${DTYPE} \
    --max_model_len ${MAX_MODEL_LEN} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --gpu-memory-utilization ${GPU_MEM_UTILIZATION} \
    --port ${PORT} \
    #--enable-prefix-caching 
}


function main() {
  #init
  #build
  start
}


main
