# file: llm_serv_vllm.sh
# date: 2025-02-06


set -x

CURR_DIR=$(pwd)
WORKSPACE="./_llm_serv_vllm"
PYTHON=$(which python3)
PORT=6789
VLLM_VERSION="0.7.3"
HF_TOKEN=""
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
CUDA_VISIBLE_DEVICES=3
MAX_MODEL_LEN=20000
TENSOR_PARALLEL_SIZE=1
GPU_MEM_UTILIZATION=0.5
DTYPE=bfloat16
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1



function init() {
  cd ${CURR_DIR}
  mkdir -p ${WORKSPACE}
}


function build() {
  cd ${CURR_DIR}
  cd ${WORKSPACE}
  ${PYTHON} -m venv ./_pyenv --copies
  ./_pyenv/bin/pip install vllm==${VLLM_VERSION}
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
    --enable-prefix-caching 
}


function main() {
  init
  build
  start
}


main
