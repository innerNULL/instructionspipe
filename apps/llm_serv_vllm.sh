# file: llm_serv_vllm.sh
# date: 2025-02-06


set -x

source $1

function init() {
  cd ${CURR_DIR}
  mkdir -p ${WORKSPACE}
}


function build() {
  cd ${CURR_DIR}
  cd ${WORKSPACE}
  ${PYTHON} -m venv ./_pyenv --copies
  ./_pyenv/bin/pip install vllm==${VLLM_VERSION} bitsandbytes hf-xet
}


function start() {
  cd ${CURR_DIR}
  cd ${WORKSPACE}
  ./_pyenv/bin/huggingface-cli login --token ${HF_TOKEN}
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=${VLLM_ALLOW_LONG_MAX_MODEL_LEN} \
  #./_pyenv/bin/vllm serve ${MODEL} \
  #  --tokenizer ${TOKENIZER} \
  #  --dtype ${DTYPE} \
  #  --max_model_len ${MAX_MODEL_LEN} \
  #  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  #  --gpu-memory-utilization ${GPU_MEM_UTILIZATION} \
  #  --port ${PORT} \
  #  --enable-prefix-caching \
  #  --enable-lora \
  #  --max_lora_rank ${MAX_LORA_RANK} \
  #  --lora-modules "{\"name\": \"${ADAPTER_NAME}\", \"path\": \"${ADAPTER_CKPT}\", \"base_model_name\": \"${ADAPTER_BASEMODEL}\"}"
  command="./_pyenv/bin/vllm serve"
  command="${command} ${MODEL}"
  command="${command} --tokenizer ${TOKENIZER}"
  command="${command} --dtype ${DTYPE}"
  command="${command} --max_model_len ${MAX_MODEL_LEN}"     
  command="${command} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"    
  command="${command} --pipeline-parallel-size  ${PIPELINE_PARALLEL_SIZE}"
  command="${command} --gpu-memory-utilization ${GPU_MEM_UTILIZATION}"   
  command="${command} --served-model-name ${SERVED_MODEL_NAME}"
  command="${command} --port ${PORT}"  
  command="${command} --enable-prefix-caching"
  gen_conf="{\"max_new_tokens\":${MAX_NEW_TOKENS}}"
  command="${command} --override-generation-config ${gen_conf}"
  if [ "${BITSANDBYTES_QUANTIZATION}" = "true" ]; then 
    command="${command} --quantization bitsandbytes --load-format bitsandbytes"
  fi
  if [[ -n "${ROPE_TYPE}" && -n "${ROPE_FACTOR}" ]]; then 
    rope_conf="{\"rope_type\":\"${ROPE_TYPE}\",\"factor\":${ROPE_FACTOR}}"
    command="${command} --rope-scaling ${rope_conf}"
  fi
  if [[ -n "${ADAPTER_NAME}" && -n "${ADAPTER_CKPT}" ]]; then
    command="${command} --enable-lora"     
    command="${command} --max_lora_rank ${MAX_LORA_RANK}"  
    lora_module="{\"name\":\"${ADAPTER_NAME}\",\"path\":\"${ADAPTER_CKPT}\",\"base_model_name\":\"${ADAPTER_BASEMODEL}\"}"
    command="${command} --lora-modules ${lora_module}"          
  fi
  echo ${command}
  ${command}
}


function main() {
  init
  build
  start
}


main
